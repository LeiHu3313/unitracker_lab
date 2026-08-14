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


def fall_from_projected_gravity(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command = _command(env, command_name)
    gravity_w = command.robot.data.GRAVITY_VEC_W
    gravity_root = math_utils.quat_apply_inverse(command.robot_root_quat_w, gravity_w)
    return torch.amax(torch.abs(gravity_root[:, :2]), dim=-1) > threshold


def mean_body_tracking_failure(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command = _command(env, command_name)
    distance = torch.linalg.vector_norm(command.target_ref_body_pos_w - command.robot_body_pos_w, dim=-1)
    return distance.mean(dim=-1) > threshold
