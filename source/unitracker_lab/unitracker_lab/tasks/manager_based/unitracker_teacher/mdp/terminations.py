"""G1 Stage-1 failure and motion-boundary termination terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command(env: ManagerBasedRLEnv, command_name: str) -> MotionCommand:
    return env.command_manager.get_term(command_name)


def motion_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    return command.target_index >= command.active_clip_end - 1


def projected_gravity_tracking_failure(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Terminate when pelvis projected gravity differs too far from the reference."""

    command = _command(env, command_name)
    gravity_w = command.robot.data.GRAVITY_VEC_W
    root_id = command.cfg.motion_body_names.index(command.cfg.root_body_name)
    reference_gravity_root = math_utils.quat_apply_inverse(command.target_ref_body_quat_w[:, root_id], gravity_w)
    robot_gravity_root = math_utils.quat_apply_inverse(command.robot_root_quat_w, gravity_w)
    return torch.linalg.vector_norm(robot_gravity_root - reference_gravity_root, dim=-1) > threshold


def pelvis_position_tracking_failure(
    env: ManagerBasedRLEnv, command_name: str, body_name: str, threshold: float
) -> torch.Tensor:
    """Terminate only when pelvis height drifts too far from its reference."""

    command = _command(env, command_name)
    body_id = command.cfg.motion_body_names.index(body_name)
    height_error = torch.abs(
        command.target_ref_body_pos_w[:, body_id, 2] - command.robot_body_pos_w[:, body_id, 2]
    )
    return height_error > threshold


def _body_id(command: MotionCommand, body_name: str) -> int:
    try:
        return command.cfg.motion_body_names.index(body_name)
    except ValueError as exc:
        raise ValueError(f"Tracking body {body_name!r} is not configured for the motion command.") from exc


def _body_ids(command: MotionCommand, body_names: list[str]) -> list[int]:
    return [_body_id(command, body_name) for body_name in body_names]


def _yaw_quat(quat_w: torch.Tensor) -> torch.Tensor:
    """Return the WXYZ quaternion containing only world yaw."""

    w, x, y, z = quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    result = torch.zeros_like(quat_w)
    result[..., 0] = torch.cos(0.5 * yaw)
    result[..., 3] = torch.sin(0.5 * yaw)
    return result


def _yaw_local_body_state(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    anchor_pos_w: torch.Tensor,
    anchor_quat_w: torch.Tensor,
    ground_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match MimicLite's torso-ground, yaw-only local body transform."""

    anchor_ground_w = anchor_pos_w.clone()
    anchor_ground_w[:, 2] = ground_z
    anchor_yaw_w = _yaw_quat(anchor_quat_w)
    anchor_yaw_bodies_w = anchor_yaw_w[:, None, :].expand(-1, body_pos_w.shape[1], -1)
    local_pos = math_utils.quat_apply_inverse(
        anchor_yaw_bodies_w.reshape(-1, 4),
        (body_pos_w - anchor_ground_w[:, None, :]).reshape(-1, 3),
    ).reshape_as(body_pos_w)
    local_quat = math_utils.quat_mul(
        math_utils.quat_inv(anchor_yaw_bodies_w.reshape(-1, 4)),
        body_quat_w.reshape(-1, 4),
    ).reshape_as(body_quat_w)
    return local_pos, local_quat


class _ConsecutiveTrackingFailure(ManagerTermBase):
    """Base for MimicLite-style failures sustained for ``min_steps`` calls."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._consecutive_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._consecutive_steps[env_ids] = 0

    def _update(self, error: torch.Tensor, threshold: float, min_steps: int) -> torch.Tensor:
        if min_steps <= 0:
            raise ValueError(f"min_steps must be positive, got {min_steps}.")
        exceeded = error >= threshold
        self._consecutive_steps = torch.where(exceeded, self._consecutive_steps + 1, 0)
        return self._consecutive_steps >= min_steps


class root_position_tracking_failure(_ConsecutiveTrackingFailure):
    """Terminate after sustained world-frame torso position error."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        body_name: str,
        threshold: float,
        min_steps: int,
    ) -> torch.Tensor:
        command = _command(env, command_name)
        body_id = _body_id(command, body_name)
        error = torch.linalg.vector_norm(
            command.target_ref_body_pos_w[:, body_id] - command.robot_body_pos_w[:, body_id], dim=-1
        )
        return self._update(error, threshold, min_steps)


class root_orientation_tracking_failure(_ConsecutiveTrackingFailure):
    """Terminate after sustained world-frame torso orientation error."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        body_name: str,
        threshold: float,
        min_steps: int,
    ) -> torch.Tensor:
        command = _command(env, command_name)
        body_id = _body_id(command, body_name)
        error = quat_error_magnitude(
            command.target_ref_body_quat_w[:, body_id], command.robot_body_quat_w[:, body_id]
        )
        return self._update(error, threshold, min_steps)


class body_position_tracking_failure(_ConsecutiveTrackingFailure):
    """Terminate after sustained torso-yaw-local body position error."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        body_names: list[str],
        anchor_body_name: str,
        threshold: float,
        min_steps: int,
    ) -> torch.Tensor:
        command = _command(env, command_name)
        body_ids = _body_ids(command, body_names)
        anchor_id = _body_id(command, anchor_body_name)
        ground_z = env.scene.env_origins[:, 2]
        robot_pos_local, _ = _yaw_local_body_state(
            command.robot_body_pos_w[:, body_ids],
            command.robot_body_quat_w[:, body_ids],
            command.robot_body_pos_w[:, anchor_id],
            command.robot_body_quat_w[:, anchor_id],
            ground_z,
        )
        ref_pos_local, _ = _yaw_local_body_state(
            command.target_ref_body_pos_w[:, body_ids],
            command.target_ref_body_quat_w[:, body_ids],
            command.target_ref_body_pos_w[:, anchor_id],
            command.target_ref_body_quat_w[:, anchor_id],
            ground_z,
        )
        error = torch.linalg.vector_norm(ref_pos_local - robot_pos_local, dim=-1).max(dim=1).values
        return self._update(error, threshold, min_steps)


class body_orientation_tracking_failure(_ConsecutiveTrackingFailure):
    """Terminate after sustained torso-yaw-local body orientation error."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        body_names: list[str],
        anchor_body_name: str,
        threshold: float,
        min_steps: int,
    ) -> torch.Tensor:
        command = _command(env, command_name)
        body_ids = _body_ids(command, body_names)
        anchor_id = _body_id(command, anchor_body_name)
        ground_z = env.scene.env_origins[:, 2]
        _, robot_quat_local = _yaw_local_body_state(
            command.robot_body_pos_w[:, body_ids],
            command.robot_body_quat_w[:, body_ids],
            command.robot_body_pos_w[:, anchor_id],
            command.robot_body_quat_w[:, anchor_id],
            ground_z,
        )
        _, ref_quat_local = _yaw_local_body_state(
            command.target_ref_body_pos_w[:, body_ids],
            command.target_ref_body_quat_w[:, body_ids],
            command.target_ref_body_pos_w[:, anchor_id],
            command.target_ref_body_quat_w[:, anchor_id],
            ground_z,
        )
        error = quat_error_magnitude(ref_quat_local, robot_quat_local).max(dim=1).values
        return self._update(error, threshold, min_steps)
