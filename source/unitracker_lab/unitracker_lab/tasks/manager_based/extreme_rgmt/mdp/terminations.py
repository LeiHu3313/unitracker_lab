"""G1 Stage-1 failure and motion-boundary termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

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
    root_id = command.cfg.body_names.index(command.cfg.root_body_name)
    reference_gravity_root = math_utils.quat_apply_inverse(command.target_ref_body_quat_w[:, root_id], gravity_w)
    robot_gravity_root = math_utils.quat_apply_inverse(command.robot_root_quat_w, gravity_w)
    return torch.linalg.vector_norm(robot_gravity_root - reference_gravity_root, dim=-1) > threshold


def pelvis_position_tracking_failure(
    env: ManagerBasedRLEnv, command_name: str, body_name: str, threshold: float
) -> torch.Tensor:
    """Terminate only when pelvis height drifts too far from its reference."""

    command = _command(env, command_name)
    body_id = command.cfg.body_names.index(body_name)
    height_error = torch.abs(
        command.target_ref_body_pos_w[:, body_id, 2] - command.robot_body_pos_w[:, body_id, 2]
    )
    return height_error > threshold
