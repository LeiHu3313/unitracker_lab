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
    G1_ALL_JOINT_NAMES,
    G1_CONTROLLED_JOINT_NAMES,
    G1_MOTION_BODY_NAMES,
    G1_OBSERVATION_BODY_NAMES,
    G1_REWARD_BODY_NAMES,
    G1_ROOT_BODY_NAME,
    TEACHER_ACTION_HISTORY_OFFSETS,
    TEACHER_REFERENCE_OFFSETS,
    TEACHER_STATE_HISTORY_OFFSETS,
    TRACKING_REWARD_SPECS,
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
    """Own 50-Hz clips and the temporal state needed by the teacher observation."""

    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        body_ids, body_names = self.robot.find_bodies(cfg.motion_body_names, preserve_order=True)
        joint_ids, joint_names = self.robot.find_joints(cfg.controlled_joint_names, preserve_order=True)
        all_joint_ids, all_joint_names = self.robot.find_joints(G1_ALL_JOINT_NAMES, preserve_order=True)
        if tuple(body_names) != tuple(cfg.motion_body_names):
            raise ValueError(f"Live G1 motion bodies do not match contract: {body_names}")
        if tuple(joint_names) != tuple(cfg.controlled_joint_names):
            raise ValueError(f"Live G1 controlled joints do not match contract: {joint_names}")
        if tuple(all_joint_names) != G1_ALL_JOINT_NAMES:
            raise ValueError(f"Live G1 physical joints do not match contract: {all_joint_names}")

        self._body_ids = torch.as_tensor(body_ids, dtype=torch.long, device=self.device)
        self._controlled_joint_ids = torch.as_tensor(joint_ids, dtype=torch.long, device=self.device)
        self._all_joint_ids = torch.as_tensor(all_joint_ids, dtype=torch.long, device=self.device)
        if cfg.root_body_name not in cfg.motion_body_names:
            raise ValueError(f"Root body {cfg.root_body_name!r} is not in the motion body contract.")
        if len(set(cfg.observation_body_names)) != len(cfg.observation_body_names):
            raise ValueError("Observation body names must be unique.")
        missing_observation_bodies = [
            name for name in cfg.observation_body_names if name not in cfg.motion_body_names
        ]
        if missing_observation_bodies:
            raise ValueError(f"Observation bodies are missing from motion bodies: {missing_observation_bodies}")
        if len(set(cfg.adaptive_tracking_body_names)) != len(cfg.adaptive_tracking_body_names):
            raise ValueError("Adaptive tracking body names must be unique.")
        missing_adaptive_bodies = [
            name for name in cfg.adaptive_tracking_body_names if name not in cfg.motion_body_names
        ]
        if missing_adaptive_bodies:
            raise ValueError(f"Adaptive tracking bodies are missing from motion bodies: {missing_adaptive_bodies}")
        if not 0.0 <= cfg.adaptive_failure_rewind_probability <= 1.0:
            raise ValueError("adaptive_failure_rewind_probability must be in [0, 1].")
        rewind_min, rewind_max = cfg.adaptive_rewind_steps_range
        if rewind_min < 0 or rewind_max <= rewind_min:
            raise ValueError("adaptive_rewind_steps_range must be a non-negative, non-empty integer range.")
        if cfg.adaptive_rewind_tail_frames < 0:
            raise ValueError("adaptive_rewind_tail_frames must be non-negative.")
        self._root_body_index = cfg.motion_body_names.index(cfg.root_body_name)
        self._root_body_id = int(self._body_ids[self._root_body_index].item())
        self._observation_body_indices = torch.tensor(
            [cfg.motion_body_names.index(name) for name in cfg.observation_body_names],
            dtype=torch.long,
            device=self.device,
        )
        self._adaptive_tracking_body_indices = torch.tensor(
            [cfg.motion_body_names.index(name) for name in cfg.adaptive_tracking_body_names],
            dtype=torch.long,
            device=self.device,
        )
        self._non_root_body_indices = torch.tensor(
            [index for index in range(len(cfg.motion_body_names)) if index != self._root_body_index],
            dtype=torch.long,
            device=self.device,
        )
        self.reference_robot: Articulation | None = env.scene.articulations.get("reference_robot")
        if self.reference_robot is not None:
            reference_joint_ids, reference_joint_names = self.reference_robot.find_joints(
                list(G1_ALL_JOINT_NAMES), preserve_order=True
            )
            if tuple(reference_joint_names) != G1_ALL_JOINT_NAMES:
                raise ValueError(f"Visual reference G1 joints do not match contract: {reference_joint_names}")
            self._reference_joint_ids = torch.as_tensor(reference_joint_ids, dtype=torch.long, device=self.device)

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

        # Ring buffers are updated once from ``_update_command`` after each
        # control step.  They belong to the command rather than individual
        # observation terms, so actor and critic always read the same history.
        self._state_history_length = max(TEACHER_STATE_HISTORY_OFFSETS) + 1
        self._action_history_length = max(TEACHER_ACTION_HISTORY_OFFSETS) + 1
        self._state_history_head = 0
        self._action_history_head = 0
        self._root_ang_vel_history_b = torch.zeros(self.num_envs, self._state_history_length, 3, device=self.device)
        self._projected_gravity_history_b = torch.zeros_like(self._root_ang_vel_history_b)
        self._joint_pos_history = torch.zeros(
            self.num_envs, self._state_history_length, len(G1_ALL_JOINT_NAMES), device=self.device
        )
        self._joint_vel_history = torch.zeros_like(self._joint_pos_history)
        self._raw_action_history = torch.zeros(
            self.num_envs, self._action_history_length, len(G1_CONTROLLED_JOINT_NAMES), device=self.device
        )

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
        """Reference joint-position trajectory used by teacher diagnostics."""

        return self.reference_joint_pos_window

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

    def reference_indices(self, offsets: tuple[int, ...] | list[int]) -> torch.Tensor:
        """Return ``phase + offsets`` clamped independently to active clips."""

        offset_tensor = torch.as_tensor(offsets, device=self.device, dtype=torch.long)
        indices = self.phase_index[:, None] + offset_tensor[None, :]
        return torch.maximum(
            torch.minimum(indices, (self.active_clip_end - 1)[:, None]), self._clip_starts[self.motion_id, None]
        )

    @property
    def reference_joint_pos_window(self) -> torch.Tensor:
        """Eight-frame MimicLite-style reference joint-position trajectory."""

        return self._joint_pos[self.reference_indices(TEACHER_REFERENCE_OFFSETS)]

    @property
    def reference_body_pos_w_window(self) -> torch.Tensor:
        indices = self.reference_indices(TEACHER_REFERENCE_OFFSETS)
        return self._body_pos_w[indices] + self._env.scene.env_origins[:, None, None, :]

    @property
    def reference_body_quat_w_window(self) -> torch.Tensor:
        return self._body_quat_w[self.reference_indices(TEACHER_REFERENCE_OFFSETS)]

    @property
    def reference_body_lin_vel_w_window(self) -> torch.Tensor:
        return self._body_lin_vel_w[self.reference_indices(TEACHER_REFERENCE_OFFSETS)]

    @property
    def reference_body_ang_vel_w_window(self) -> torch.Tensor:
        return self._body_ang_vel_w[self.reference_indices(TEACHER_REFERENCE_OFFSETS)]

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
    def robot_all_applied_torque(self) -> torch.Tensor:
        return self.robot.data.applied_torque[:, self._all_joint_ids]

    @property
    def controlled_joint_ids(self) -> torch.Tensor:
        return self._controlled_joint_ids

    @property
    def observation_body_indices(self) -> torch.Tensor:
        """Indices into the 17-body motion state selected for actor/critic input."""

        return self._observation_body_indices

    @property
    def root_body_index(self) -> int:
        """Index of pelvis/root in the complete 17-body motion state."""

        return self._root_body_index

    def reference_body_state(
        self, offsets: tuple[int, ...] | list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return complete motion-body reference state at clip-clamped offsets."""

        indices = self.reference_indices(offsets)
        return (
            self._body_pos_w[indices] + self._env.scene.env_origins[:, None, None, :],
            self._body_quat_w[indices],
            self._body_lin_vel_w[indices],
            self._body_ang_vel_w[indices],
        )

    def teacher_state_history(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the ordered current/past robot and raw-action histories."""

        state_indices = (
            self._state_history_head
            + torch.as_tensor(TEACHER_STATE_HISTORY_OFFSETS, device=self.device, dtype=torch.long)
        ) % self._state_history_length
        action_indices = (
            self._action_history_head
            + torch.as_tensor(TEACHER_ACTION_HISTORY_OFFSETS, device=self.device, dtype=torch.long)
        ) % self._action_history_length
        return (
            self._root_ang_vel_history_b[:, state_indices].flatten(1),
            self._projected_gravity_history_b[:, state_indices].flatten(1),
            self._joint_pos_history[:, state_indices].flatten(1),
            self._joint_vel_history[:, state_indices].flatten(1),
            self._raw_action_history[:, action_indices].flatten(1),
        )

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
            adaptive_body_indices = self._adaptive_tracking_body_indices
            adaptive_robot_pos_local = _root_local_positions(
                self.robot_body_pos_w[:, adaptive_body_indices], self.robot_root_pos_w, self.robot_root_quat_w
            )
            adaptive_target_pos_local = _root_local_positions(
                self.target_ref_body_pos_w[:, adaptive_body_indices], target_root_pos_w, target_root_quat_w
            )
            adaptive_robot_quat_local = _root_relative_quaternions(
                self.robot_body_quat_w[:, adaptive_body_indices], self.robot_root_quat_w
            )
            adaptive_target_quat_local = _root_relative_quaternions(
                self.target_ref_body_quat_w[:, adaptive_body_indices], target_root_quat_w
            )
            adaptive_body_position_error = torch.linalg.vector_norm(
                adaptive_target_pos_local - adaptive_robot_pos_local, dim=-1
            ).mean(dim=-1)
            adaptive_body_orientation_error = quat_error_magnitude(
                adaptive_target_quat_local, adaptive_robot_quat_local
            ).mean(dim=-1)
            adaptive_body_linear_velocity_error = torch.linalg.vector_norm(
                self.target_ref_body_lin_vel_w[:, adaptive_body_indices]
                - self.robot_body_lin_vel_w[:, adaptive_body_indices],
                dim=-1,
            ).mean(dim=-1)
            adaptive_body_angular_velocity_error = torch.linalg.vector_norm(
                self.target_ref_body_ang_vel_w[:, adaptive_body_indices]
                - self.robot_body_ang_vel_w[:, adaptive_body_indices],
                dim=-1,
            ).mean(dim=-1)
            tracking_error = torch.stack(
                (
                    adaptive_body_position_error / TRACKING_REWARD_SPECS["body_position"]["sigma"],
                    adaptive_body_orientation_error / TRACKING_REWARD_SPECS["body_orientation"]["sigma"],
                    adaptive_body_linear_velocity_error / TRACKING_REWARD_SPECS["body_linear_velocity"]["sigma"],
                    adaptive_body_angular_velocity_error / TRACKING_REWARD_SPECS["body_angular_velocity"]["sigma"],
                    root_position_error / TRACKING_REWARD_SPECS["torso_position"]["sigma"],
                    root_orientation_error / TRACKING_REWARD_SPECS["torso_orientation"]["sigma"],
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
            # MimicLite-style failure recovery: re-enter the same reference
            # with its preceding motion context instead of always spawning in
            # an unrelated hard phase.  Time-outs and near-end failures still
            # use the global adaptive distribution.
            old_motion_id = self.motion_id[env_ids]
            failed_phase = self.target_index[env_ids]
            failed = self._env.termination_manager.terminated[env_ids]
            far_from_clip_end = failed_phase < (
                self._clip_ends[old_motion_id] - self.cfg.adaptive_rewind_tail_frames
            )
            rewind_mask = failed & far_from_clip_end
            rewind_mask &= torch.rand(env_ids.numel(), device=self.device) < self.cfg.adaptive_failure_rewind_probability
            rewind_steps = torch.randint(
                self.cfg.adaptive_rewind_steps_range[0],
                self.cfg.adaptive_rewind_steps_range[1],
                (env_ids.numel(),),
                device=self.device,
            )
            rewind_phase = torch.maximum(
                failed_phase - rewind_steps,
                self._clip_starts[old_motion_id],
            )
            motion_id = torch.where(rewind_mask, old_motion_id, motion_id)
            phase = torch.where(rewind_mask, rewind_phase, phase)
        else:
            raise ValueError(f"Unknown G1 RSI sampling_mode={self.cfg.sampling_mode!r}")
        self.motion_id[env_ids] = motion_id
        self.phase_index[env_ids] = phase
        self.target_index[env_ids] = phase + 1
        self.just_reset[env_ids] = True
        self._write_reference_state_to_sim(env_ids)
        self._seed_teacher_history(env_ids)
        self._write_visual_reference_state_to_sim(env_ids)

    def _seed_teacher_history(self, env_ids: torch.Tensor) -> None:
        """Seed reset histories from the RSI reference, with zero prior action."""

        phase = self.phase_index[env_ids]
        root_quat_w = self._body_quat_w[phase, self._root_body_index]
        root_ang_vel_w = self._body_ang_vel_w[phase, self._root_body_index]
        gravity_w = torch.zeros_like(root_ang_vel_w)
        gravity_w[:, 2] = -1.0
        root_ang_vel_b = math_utils.quat_apply_inverse(root_quat_w, root_ang_vel_w)
        gravity_b = math_utils.quat_apply_inverse(root_quat_w, gravity_w)
        joint_pos = self._joint_pos[phase] - self.robot.data.default_joint_pos[env_ids][:, self._all_joint_ids]
        joint_vel = self._joint_vel[phase]
        self._root_ang_vel_history_b[env_ids] = root_ang_vel_b[:, None, :]
        self._projected_gravity_history_b[env_ids] = gravity_b[:, None, :]
        self._joint_pos_history[env_ids] = joint_pos[:, None, :]
        self._joint_vel_history[env_ids] = joint_vel[:, None, :]
        self._raw_action_history[env_ids] = 0.0

    def _push_teacher_history(self) -> None:
        """Append the actual post-step state and raw action for every environment."""

        self._state_history_head = (self._state_history_head - 1) % self._state_history_length
        self._action_history_head = (self._action_history_head - 1) % self._action_history_length
        root_quat_w = self.robot_root_quat_w
        gravity_w = torch.zeros_like(self.robot_root_pos_w)
        gravity_w[:, 2] = -1.0
        self._root_ang_vel_history_b[:, self._state_history_head] = math_utils.quat_apply_inverse(
            root_quat_w, self.robot_body_ang_vel_w[:, self._root_body_index]
        )
        self._projected_gravity_history_b[:, self._state_history_head] = math_utils.quat_apply_inverse(
            root_quat_w, gravity_w
        )
        self._joint_pos_history[:, self._state_history_head] = self.robot_all_joint_pos - self.robot_all_default_joint_pos
        self._joint_vel_history[:, self._state_history_head] = self.robot_all_joint_vel
        self._raw_action_history[:, self._action_history_head] = self._env.action_manager.action

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

        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # Seed all 29 implicit position targets from the reference RSI state.
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        # Position-control PD uses a zero velocity target. The reference qdot is
        # written into simulator state and observations, not left as a persistent
        # actuator set-point after RSI.
        self.robot.set_joint_velocity_target(torch.zeros_like(joint_vel), env_ids=env_ids)

    def _write_visual_reference_state_to_sim(self, env_ids: torch.Tensor) -> None:
        """Render the ``k+1`` target on the optional non-physical green G1."""

        if self.reference_robot is None:
            return
        phase = self.target_index[env_ids]
        body_pos = self._body_pos_w[phase]
        body_quat = self._body_quat_w[phase]
        body_lin_vel = self._body_lin_vel_w[phase]
        body_ang_vel = self._body_ang_vel_w[phase]
        root_state = self.reference_robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = body_pos[:, self._root_body_index] + self._env.scene.env_origins[env_ids]
        root_state[:, 3:7] = body_quat[:, self._root_body_index]
        root_state[:, 7:10] = body_lin_vel[:, self._root_body_index]
        root_state[:, 10:13] = body_ang_vel[:, self._root_body_index]
        joint_pos = self.reference_robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:, self._reference_joint_ids] = self._joint_pos[phase]
        joint_vel[:, self._reference_joint_ids] = self._joint_vel[phase]
        self.reference_robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.reference_robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.reference_robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.reference_robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)

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
        self._push_teacher_history()
        promote = reference_promotion_mask(self.just_reset, self._env.episode_length_buf)
        self.phase_index[promote] = self.target_index[promote]
        clip_last = self.active_clip_end[promote] - 1
        self.target_index[promote] = torch.minimum(self.phase_index[promote] + 1, clip_last)
        self.just_reset[:] = False
        self._write_visual_reference_state_to_sim(torch.arange(self.num_envs, device=self.device))

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis and not hasattr(self, "current_body_visualizers"):
            self.current_body_visualizers = [
                VisualizationMarkers(
                    self.cfg.current_body_visualizer_cfg.replace(
                        prim_path=f"/Visuals/G1Teacher/current_bodies/{body_name}"
                    )
                )
                for body_name in self.cfg.motion_body_names
            ]
            self.target_body_visualizers = [
                VisualizationMarkers(
                    self.cfg.target_body_visualizer_cfg.replace(
                        prim_path=f"/Visuals/G1Teacher/target_bodies/{body_name}"
                    )
                )
                for body_name in self.cfg.motion_body_names
            ]
        if hasattr(self, "current_body_visualizers"):
            for visualizer in (*self.current_body_visualizers, *self.target_body_visualizers):
                visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event) -> None:
        if not self.robot.is_initialized:
            return
        for body_index in range(len(self.cfg.motion_body_names)):
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
    motion_body_names: list[str] = list(G1_MOTION_BODY_NAMES)
    observation_body_names: list[str] = list(G1_OBSERVATION_BODY_NAMES)
    controlled_joint_names: list[str] = list(G1_CONTROLLED_JOINT_NAMES)
    sampling_mode: str = "adaptive"
    adaptive_bin_duration_s: float = 0.25
    adaptive_uniform_ratio: float = 0.5
    adaptive_ema_alpha: float = 0.01
    adaptive_tracking_error_weight: float = 0.25
    adaptive_tracking_error_clip: float = 5.0
    adaptive_priority_epsilon: float = 0.1
    adaptive_tracking_body_names: list[str] = list(G1_REWARD_BODY_NAMES)
    adaptive_failure_rewind_probability: float = 0.8
    adaptive_rewind_steps_range: tuple[int, int] = (25, 125)
    adaptive_rewind_tail_frames: int = 50
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
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.85, 0.25)),
            )
        },
    )
