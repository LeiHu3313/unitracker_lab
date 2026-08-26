"""Strict multi-clip G1 motion command with reference-state initialization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import quat_error_magnitude

from ..contracts import (
    FUTURE_REFERENCE_FRAMES,
    G1_ALL_JOINT_NAMES,
    G1_CONTROLLED_JOINT_NAMES,
    G1_LOCKED_WRIST_JOINT_NAMES,
    G1_LOCKED_WRIST_POSITIONS,
    G1_ROOT_BODY_NAME,
    G1_TRACKING_BODY_NAMES,
    reference_promotion_mask,
)
from ..motion_schema import load_and_validate_motion_dataset
from .adaptive_sampling import AdaptiveTrackingSampler

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _root_local_positions(
    body_pos_w: torch.Tensor, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor
) -> torch.Tensor:
    """Express body positions in each pose's own root frame."""

    vectors_w = body_pos_w - root_pos_w[:, None, :]
    roots = root_quat_w[:, None, :].expand(-1, vectors_w.shape[1], -1)
    return math_utils.quat_apply_inverse(roots.reshape(-1, 4), vectors_w.reshape(-1, 3)).reshape_as(vectors_w)


def _root_relative_quaternions(body_quat_w: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
    """Express body orientations relative to each pose's own root."""

    roots = root_quat_w[:, None, :].expand_as(body_quat_w)
    return math_utils.quat_mul(math_utils.quat_inv(roots.reshape(-1, 4)), body_quat_w.reshape(-1, 4)).reshape_as(
        body_quat_w
    )


class MotionCommand(CommandTerm):
    """Own 50-Hz clips with a ``k -> k+1`` body goal and a five-frame joint command."""

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        body_ids, body_names = self.robot.find_bodies(cfg.body_names, preserve_order=True)
        joint_ids, joint_names = self.robot.find_joints(cfg.controlled_joint_names, preserve_order=True)
        locked_ids, locked_names = self.robot.find_joints(cfg.locked_joint_names, preserve_order=True)
        all_joint_ids, all_joint_names = self.robot.find_joints(G1_ALL_JOINT_NAMES, preserve_order=True)
        if tuple(body_names) != tuple(cfg.body_names):
            raise ValueError(f"Live G1 tracking bodies do not match contract: {body_names}")
        if tuple(joint_names) != tuple(cfg.controlled_joint_names):
            raise ValueError(f"Live G1 controlled joints do not match contract: {joint_names}")
        if tuple(locked_names) != tuple(cfg.locked_joint_names):
            raise ValueError(f"Live G1 locked wrist joints do not match contract: {locked_names}")
        if tuple(all_joint_names) != G1_ALL_JOINT_NAMES:
            raise ValueError(f"Live G1 physical joints do not match contract: {all_joint_names}")

        self._body_ids = torch.as_tensor(body_ids, dtype=torch.long, device=self.device)
        self._controlled_joint_ids = torch.as_tensor(joint_ids, dtype=torch.long, device=self.device)
        self._locked_joint_ids = torch.as_tensor(locked_ids, dtype=torch.long, device=self.device)
        self._all_joint_ids = torch.as_tensor(all_joint_ids, dtype=torch.long, device=self.device)
        self._root_body_index = cfg.body_names.index(cfg.root_body_name)
        self._root_body_id = int(self._body_ids[self._root_body_index].item())
        self._non_root_body_indices = torch.tensor(
            [index for index in range(len(cfg.body_names)) if index != self._root_body_index],
            dtype=torch.long,
            device=self.device,
        )
        self._locked_positions = torch.as_tensor(cfg.locked_joint_positions, dtype=torch.float32, device=self.device)

        arrays = load_and_validate_motion_dataset(cfg.motion_file)
        self.motion_path = arrays.path
        self.motion_sha256 = arrays.sha256
        self.motion_fps = arrays.fps
        self.motion_paths = arrays.paths
        self.num_motions = arrays.num_motions
        self.frame_count = arrays.frame_count
        self._clip_starts = torch.as_tensor(arrays.clip_starts, dtype=torch.long, device=self.device)
        self._clip_lengths = torch.as_tensor(arrays.clip_lengths, dtype=torch.long, device=self.device)
        self._clip_ends = self._clip_starts + self._clip_lengths
        self._adaptive_sampler = AdaptiveTrackingSampler(
            self._clip_starts,
            self._clip_lengths,
            fps=self.motion_fps,
            bin_duration_s=cfg.adaptive_bin_duration_s,
            uniform_ratio=cfg.adaptive_uniform_ratio,
            ema_alpha=cfg.adaptive_ema_alpha,
            tracking_error_weight=cfg.adaptive_tracking_error_weight,
            tracking_error_clip=cfg.adaptive_tracking_error_clip,
            priority_epsilon=cfg.adaptive_priority_epsilon,
            device=self.device,
        )
        self._joint_pos = torch.as_tensor(arrays.joint_pos, device=self.device)
        self._joint_vel = torch.as_tensor(arrays.joint_vel, device=self.device)
        self._body_pos_w = torch.as_tensor(arrays.body_pos_w, device=self.device)
        self._body_quat_w = torch.as_tensor(arrays.body_quat_w, device=self.device)
        self._body_lin_vel_w = torch.as_tensor(arrays.body_lin_vel_w, device=self.device)
        self._body_ang_vel_w = torch.as_tensor(arrays.body_ang_vel_w, device=self.device)

        self.motion_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_index = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.just_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for name in (
            "body_position_error",
            "body_orientation_error",
            "body_local_position_error",
            "body_local_orientation_error",
            "joint_position_error",
            "joint_velocity_error",
            "body_linear_velocity_error",
            "body_angular_velocity_error",
            "adaptive_sampling_entropy",
            "adaptive_sampling_top_probability",
            "adaptive_tracking_error_mean",
            "adaptive_failure_rate_mean",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Five-frame joint command used by the teacher oracle and diagnostics."""

        return self.future_ref_joint_command

    @property
    def active_clip_end(self) -> torch.Tensor:
        """Exclusive end index for each environment's currently sampled clip."""

        return self._clip_ends[self.motion_id]

    @property
    def current_ref_joint_pos(self) -> torch.Tensor:
        return self._joint_pos[self.phase_index]

    @property
    def current_ref_joint_vel(self) -> torch.Tensor:
        return self._joint_vel[self.phase_index]

    @property
    def current_ref_body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[self.phase_index] + self._env.scene.env_origins[:, None, :]

    @property
    def current_ref_body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[self.phase_index]

    @property
    def current_ref_body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[self.phase_index]

    @property
    def current_ref_body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[self.phase_index]

    @property
    def target_ref_joint_pos(self) -> torch.Tensor:
        return self._joint_pos[self.target_index]

    @property
    def target_ref_joint_vel(self) -> torch.Tensor:
        return self._joint_vel[self.target_index]

    @property
    def future_ref_joint_indices(self) -> torch.Tensor:
        """Return reference indices ``t, ..., t+4``, clamped to each clip end."""

        offsets = torch.arange(FUTURE_REFERENCE_FRAMES, device=self.device, dtype=torch.long)
        indices = self.phase_index[:, None] + offsets[None, :]
        return torch.minimum(indices, (self.active_clip_end - 1)[:, None])

    @property
    def future_ref_joint_command(self) -> torch.Tensor:
        """Reference ``[q_t, ..., q_t+4, 0.05*dq_t, ..., 0.05*dq_t+4]`` command."""

        indices = self.future_ref_joint_indices
        joint_pos = self._joint_pos[indices].flatten(1)
        joint_vel = (0.05 * self._joint_vel[indices]).flatten(1)
        return torch.cat((joint_pos, joint_vel), dim=-1)

    @property
    def target_ref_body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[self.target_index] + self._env.scene.env_origins[:, None, :]

    @property
    def target_ref_body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[self.target_index]

    @property
    def target_ref_body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[self.target_index]

    @property
    def target_ref_body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[self.target_index]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self._body_ids]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self._body_ids]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self._body_ids]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self._body_ids]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self._root_body_id]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self._root_body_id]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self._controlled_joint_ids]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel[:, self._controlled_joint_ids]

    @property
    def robot_all_joint_pos(self) -> torch.Tensor:
        """Physical 29-DoF joint state in the checkpoint contract order."""

        return self.robot.data.joint_pos[:, self._all_joint_ids]

    @property
    def robot_all_joint_vel(self) -> torch.Tensor:
        """Physical 29-DoF joint velocity in the checkpoint contract order."""

        return self.robot.data.joint_vel[:, self._all_joint_ids]

    @property
    def robot_all_default_joint_pos(self) -> torch.Tensor:
        return self.robot.data.default_joint_pos[:, self._all_joint_ids]

    @property
    def locked_joint_ids(self) -> torch.Tensor:
        return self._locked_joint_ids

    @property
    def controlled_joint_ids(self) -> torch.Tensor:
        return self._controlled_joint_ids

    def _update_metrics(self) -> None:
        target_root_pos_w = self.target_ref_body_pos_w[:, self._root_body_index]
        target_root_quat_w = self.target_ref_body_quat_w[:, self._root_body_index]
        robot_body_pos_local = _root_local_positions(
            self.robot_body_pos_w[:, self._non_root_body_indices], self.robot_root_pos_w, self.robot_root_quat_w
        )
        target_body_pos_local = _root_local_positions(
            self.target_ref_body_pos_w[:, self._non_root_body_indices], target_root_pos_w, target_root_quat_w
        )
        robot_body_quat_local = _root_relative_quaternions(
            self.robot_body_quat_w[:, self._non_root_body_indices], self.robot_root_quat_w
        )
        target_body_quat_local = _root_relative_quaternions(
            self.target_ref_body_quat_w[:, self._non_root_body_indices], target_root_quat_w
        )
        self.metrics["body_position_error"] = torch.linalg.vector_norm(
            self.target_ref_body_pos_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["body_orientation_error"] = quat_error_magnitude(
            self.target_ref_body_quat_w, self.robot_body_quat_w
        ).mean(dim=-1)
        self.metrics["body_local_position_error"] = torch.linalg.vector_norm(
            target_body_pos_local - robot_body_pos_local, dim=-1
        ).mean(dim=-1)
        self.metrics["body_local_orientation_error"] = quat_error_magnitude(
            target_body_quat_local, robot_body_quat_local
        ).mean(dim=-1)
        self.metrics["joint_position_error"] = (
            torch.square(self.target_ref_joint_pos - self.robot_joint_pos).mean(dim=-1).sqrt()
        )
        self.metrics["joint_velocity_error"] = (
            torch.square(self.target_ref_joint_vel - self.robot_joint_vel).mean(dim=-1).sqrt()
        )
        self.metrics["body_linear_velocity_error"] = torch.linalg.vector_norm(
            self.target_ref_body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["body_angular_velocity_error"] = torch.linalg.vector_norm(
            self.target_ref_body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)
        if self.cfg.sampling_mode == "adaptive":
            valid = self._env.episode_length_buf > 0
            root_position_error = torch.linalg.vector_norm(target_root_pos_w - self.robot_root_pos_w, dim=-1)
            root_orientation_error = quat_error_magnitude(target_root_quat_w, self.robot_root_quat_w)
            tracking_error = torch.stack(
                (
                    self.metrics["body_local_position_error"] / 0.30,
                    self.metrics["body_local_orientation_error"] / 0.40,
                    self.metrics["body_linear_velocity_error"] / 1.00,
                    self.metrics["body_angular_velocity_error"] / 3.14,
                    root_position_error / 0.30,
                    root_orientation_error / 0.40,
                ),
                dim=-1,
            ).mean(dim=-1)
            self._adaptive_sampler.record_tracking_errors(
                self.motion_id[valid], self.target_index[valid], tracking_error[valid]
            )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        if self.cfg.sampling_mode == "eval":
            motion_id = torch.remainder(env_ids, self.num_motions)
            phase = self._clip_starts[motion_id]
        elif self.cfg.sampling_mode == "uniform":
            # First sample clips uniformly, then sample a valid k uniformly in
            # each clip. Long clips therefore do not dominate short clips.
            motion_id = torch.randint(0, self.num_motions, (env_ids.numel(),), device=self.device)
            valid_phase_counts = self._clip_lengths[motion_id] - 1
            phase_offset = torch.floor(torch.rand(env_ids.numel(), device=self.device) * valid_phase_counts).to(
                dtype=torch.long
            )
            phase = self._clip_starts[motion_id] + phase_offset
        elif self.cfg.sampling_mode == "adaptive":
            self._record_adaptive_outcomes(env_ids)
            _, motion_id, phase = self._adaptive_sampler.sample(env_ids.numel())
        else:
            raise ValueError(f"Unknown G1 RSI sampling_mode={self.cfg.sampling_mode!r}")
        self.motion_id[env_ids] = motion_id
        self.phase_index[env_ids] = phase
        self.target_index[env_ids] = phase + 1
        self.just_reset[env_ids] = True
        self._write_reference_state_to_sim(env_ids)

    def _write_reference_state_to_sim(self, env_ids: torch.Tensor) -> None:
        phase = self.phase_index[env_ids]
        body_pos = self._body_pos_w[phase]
        body_quat = self._body_quat_w[phase]
        body_lin_vel = self._body_lin_vel_w[phase]
        body_ang_vel = self._body_ang_vel_w[phase]
        root_pos = body_pos[:, self._root_body_index] + self._env.scene.env_origins[env_ids]
        root_state = torch.cat(
            (
                root_pos,
                body_quat[:, self._root_body_index],
                body_lin_vel[:, self._root_body_index],
                body_ang_vel[:, self._root_body_index],
            ),
            dim=-1,
        )

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:, self._controlled_joint_ids] = self._joint_pos[phase]
        joint_vel[:, self._controlled_joint_ids] = self._joint_vel[phase]
        joint_pos[:, self._locked_joint_ids] = self._locked_positions
        joint_vel[:, self._locked_joint_ids] = 0.0

        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # The 23-D ActionTerm never touches wrist targets, so seed every implicit
        # target explicitly on reset and leave the six wrist buffers at zero.
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        # Position-control PD uses a zero velocity target. The reference qdot is
        # written into simulator state and observations, not left as a persistent
        # actuator set-point after RSI.
        self.robot.set_joint_velocity_target(torch.zeros_like(joint_vel), env_ids=env_ids)

    def _record_adaptive_outcomes(self, env_ids: torch.Tensor) -> None:
        """Record the prior episode before command reset clears termination state."""

        completed = self._env.episode_length_buf[env_ids] > 0
        if not bool(completed.any()):
            return
        completed_env_ids = env_ids[completed]
        # ``terminated`` excludes time-outs. Continuous tracking error is
        # recorded every step; this path adds only actual tracking failures.
        failures = self._env.termination_manager.terminated[completed_env_ids]
        self._adaptive_sampler.record_failures(
            self.motion_id[completed_env_ids], self.target_index[completed_env_ids], failures
        )

    def apply_motion_cache_swap_if_pending_barrier(self) -> bool:
        """Synchronize adaptive tracking feedback once per PPO rollout."""

        if self.cfg.sampling_mode != "adaptive":
            return False
        updated = self._adaptive_sampler.apply_pending_feedback()
        if not updated:
            return False
        probabilities = self._adaptive_sampler.sampling_probabilities()
        entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
        if self._adaptive_sampler.num_bins > 1:
            entropy = entropy / torch.log(torch.tensor(float(self._adaptive_sampler.num_bins), device=self.device))
        self.metrics["adaptive_sampling_entropy"][:] = entropy
        self.metrics["adaptive_sampling_top_probability"][:] = probabilities.max()
        self.metrics["adaptive_tracking_error_mean"][:] = self._adaptive_sampler.tracking_errors.mean()
        self.metrics["adaptive_failure_rate_mean"][:] = self._adaptive_sampler.failure_rates.mean()
        return True

    def _update_command(self) -> None:
        # A terminated env is RSI-reset inside env.step() before command.compute().
        # Its episode length is zero and must retain k -> k+1 for the returned obs.
        promote = reference_promotion_mask(self.just_reset, self._env.episode_length_buf)
        self.phase_index[promote] = self.target_index[promote]
        clip_last = self.active_clip_end[promote] - 1
        self.target_index[promote] = torch.minimum(self.phase_index[promote] + 1, clip_last)
        self.just_reset[:] = False

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis and not hasattr(self, "current_body_visualizers"):
            self.current_body_visualizers = [
                VisualizationMarkers(
                    self.cfg.current_body_visualizer_cfg.replace(
                        prim_path=f"/Visuals/G1Teacher/current_bodies/{body_name}"
                    )
                )
                for body_name in self.cfg.body_names
            ]
            self.target_body_visualizers = [
                VisualizationMarkers(
                    self.cfg.target_body_visualizer_cfg.replace(
                        prim_path=f"/Visuals/G1Teacher/target_bodies/{body_name}"
                    )
                )
                for body_name in self.cfg.body_names
            ]
        if hasattr(self, "current_body_visualizers"):
            for visualizer in (*self.current_body_visualizers, *self.target_body_visualizers):
                visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event) -> None:
        if not self.robot.is_initialized:
            return
        for body_index in range(len(self.cfg.body_names)):
            self.current_body_visualizers[body_index].visualize(
                translations=self.robot_body_pos_w[:, body_index],
            )
            self.target_body_visualizers[body_index].visualize(
                translations=self.target_ref_body_pos_w[:, body_index],
            )


@configclass
class MotionCommandCfg(CommandTermCfg):
    class_type: type = MotionCommand
    asset_name: str = "robot"
    motion_file: str = MISSING
    root_body_name: str = G1_ROOT_BODY_NAME
    body_names: list[str] = list(G1_TRACKING_BODY_NAMES)
    controlled_joint_names: list[str] = list(G1_CONTROLLED_JOINT_NAMES)
    locked_joint_names: list[str] = list(G1_LOCKED_WRIST_JOINT_NAMES)
    locked_joint_positions: list[float] = list(G1_LOCKED_WRIST_POSITIONS)
    sampling_mode: str = "adaptive"
    adaptive_bin_duration_s: float = 0.25
    adaptive_uniform_ratio: float = 0.5
    adaptive_ema_alpha: float = 0.01
    adaptive_tracking_error_weight: float = 0.25
    adaptive_tracking_error_clip: float = 5.0
    adaptive_priority_epsilon: float = 0.1
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False
    current_body_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/G1Teacher/current_bodies/body",
        markers={
            "body": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 1.0)),
            )
        },
    )
    target_body_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/G1Teacher/target_bodies/body",
        markers={
            "body": sim_utils.SphereCfg(
                radius=0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.1)),
            )
        },
    )
