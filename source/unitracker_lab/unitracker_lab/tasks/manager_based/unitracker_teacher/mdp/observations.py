"""716-D privileged oracle observation for the G1 Stage-1 teacher."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

from ..contracts import ORACLE_OBSERVATION_DIM
from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command(env: ManagerBasedRLEnv, command_name: str) -> MotionCommand:
    return env.command_manager.get_term(command_name)


def _rotate_inverse(root_quat_w: torch.Tensor, vectors_w: torch.Tensor) -> torch.Tensor:
    root = root_quat_w[:, None, :].expand(*vectors_w.shape[:-1], 4)
    return math_utils.quat_apply_inverse(root.reshape(-1, 4), vectors_w.reshape(-1, 3)).reshape_as(vectors_w)


def _relative_quat(root_quat_w: torch.Tensor, body_quat_w: torch.Tensor) -> torch.Tensor:
    root = root_quat_w[:, None, :].expand_as(body_quat_w)
    return math_utils.quat_mul(math_utils.quat_inv(root.reshape(-1, 4)), body_quat_w.reshape(-1, 4)).reshape_as(
        body_quat_w
    )


def _quat_error(current_quat_w: torch.Tensor, target_quat_w: torch.Tensor) -> torch.Tensor:
    return math_utils.quat_mul(
        math_utils.quat_inv(current_quat_w.reshape(-1, 4)), target_quat_w.reshape(-1, 4)
    ).reshape_as(current_quat_w)


def _rot6d(quaternions: torch.Tensor) -> torch.Tensor:
    """Encode the first two rotation-matrix columns in a continuous 6-D form."""

    matrix = math_utils.matrix_from_quat(quaternions.reshape(-1, 4))
    return matrix[..., :2].reshape(*quaternions.shape[:-1], 6)


def teacher_oracle_observation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return current privileged state plus the explicit next-frame tracking goal."""

    command = _command(env, command_name)
    root_pos_w = command.robot_root_pos_w
    root_quat_w = command.robot_root_quat_w
    robot_body_pos_w = command.robot_body_pos_w
    robot_body_quat_w = command.robot_body_quat_w

    current_body_pos_local = _rotate_inverse(root_quat_w, robot_body_pos_w - root_pos_w[:, None, :])
    current_body_ori_local = _rot6d(_relative_quat(root_quat_w, robot_body_quat_w))
    current_body_lin_vel_local = _rotate_inverse(root_quat_w, command.robot_body_lin_vel_w)
    current_body_ang_vel_local = _rotate_inverse(root_quat_w, command.robot_body_ang_vel_w)

    target_pos_error_local = _rotate_inverse(root_quat_w, command.target_ref_body_pos_w - robot_body_pos_w)
    target_ori_error = _rot6d(_quat_error(robot_body_quat_w, command.target_ref_body_quat_w))
    target_lin_vel_error_local = _rotate_inverse(
        root_quat_w, command.target_ref_body_lin_vel_w - command.robot_body_lin_vel_w
    )
    target_ang_vel_error_local = _rotate_inverse(
        root_quat_w, command.target_ref_body_ang_vel_w - command.robot_body_ang_vel_w
    )
    target_body_pos_relative_root = _rotate_inverse(root_quat_w, command.target_ref_body_pos_w - root_pos_w[:, None, :])
    target_body_ori_relative_root = _rot6d(_relative_quat(root_quat_w, command.target_ref_body_quat_w))

    previous_action = env.action_manager.action
    blocks = (
        current_body_pos_local.flatten(1),
        current_body_ori_local.flatten(1),
        current_body_lin_vel_local.flatten(1),
        current_body_ang_vel_local.flatten(1),
        command.robot_joint_pos,
        command.robot_joint_vel,
        previous_action,
        target_pos_error_local.flatten(1),
        command.target_ref_joint_pos - command.robot_joint_pos,
        target_ori_error.flatten(1),
        target_lin_vel_error_local.flatten(1),
        target_ang_vel_error_local.flatten(1),
        target_body_pos_relative_root.flatten(1),
        target_body_ori_relative_root.flatten(1),
    )
    observation = torch.cat(blocks, dim=-1)
    if observation.shape[-1] != ORACLE_OBSERVATION_DIM:
        raise RuntimeError(
            f"G1 oracle observation ABI violation: got {observation.shape[-1]}, expected {ORACLE_OBSERVATION_DIM}."
        )
    return observation
