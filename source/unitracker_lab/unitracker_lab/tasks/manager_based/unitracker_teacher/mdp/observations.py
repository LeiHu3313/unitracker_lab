"""Fully privileged MimicLite-style observations for the 29-DoF G1 teacher."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

from ..contracts import (
    TEACHER_OBSERVATION_DIM,
    TEACHER_REFERENCE_OBSERVATION_DIM,
    TEACHER_STATE_OBSERVATION_DIM,
    TEACHER_TRACKING_FEEDBACK_OFFSETS,
)
from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command(env: ManagerBasedRLEnv, command_name: str) -> MotionCommand:
    return env.command_manager.get_term(command_name)


def _rot6d(quaternions: torch.Tensor) -> torch.Tensor:
    """Encode the first two rotation-matrix columns in a continuous 6-D form."""

    matrix = math_utils.matrix_from_quat(quaternions.reshape(-1, 4))
    return matrix[..., :2].reshape(*quaternions.shape[:-1], 6)


def _yaw_quat(quat_w: torch.Tensor) -> torch.Tensor:
    """Return the WXYZ quaternion containing only a pose's world yaw."""

    w, x, y, z = quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    yaw_quat = torch.zeros_like(quat_w)
    yaw_quat[..., 0] = torch.cos(0.5 * yaw)
    yaw_quat[..., 3] = torch.sin(0.5 * yaw)
    return yaw_quat


def _rotate_inverse_broadcast(quat_w: torch.Tensor, vectors_w: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors by inverse quaternion, broadcasting body/time axes."""

    while quat_w.ndim < vectors_w.ndim:
        quat_w = quat_w.unsqueeze(-2)
    quat_w = quat_w.expand(*vectors_w.shape[:-1], 4)
    return math_utils.quat_apply_inverse(quat_w.reshape(-1, 4), vectors_w.reshape(-1, 3)).reshape_as(vectors_w)


def _relative_quat_broadcast(parent_quat_w: torch.Tensor, child_quat_w: torch.Tensor) -> torch.Tensor:
    """Return child orientation in parent axes, broadcasting body/time axes."""

    while parent_quat_w.ndim < child_quat_w.ndim:
        parent_quat_w = parent_quat_w.unsqueeze(-2)
    parent_quat_w = parent_quat_w.expand_as(child_quat_w)
    return math_utils.quat_mul(
        math_utils.quat_inv(parent_quat_w.reshape(-1, 4)), child_quat_w.reshape(-1, 4)
    ).reshape_as(child_quat_w)


def _ground_anchor(position_w: torch.Tensor, env_origins: torch.Tensor) -> torch.Tensor:
    """Place an anchor on each environment's ground plane without changing x/y."""

    anchor = position_w.clone()
    ground_z = env_origins[:, 2]
    while ground_z.ndim < anchor.ndim - 1:
        ground_z = ground_z.unsqueeze(-1)
    anchor[..., 2] = ground_z
    return anchor


def _yaw_local_body_position(
    body_pos_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    env_origins: torch.Tensor,
) -> torch.Tensor:
    """Body position in that pose's pelvis-ground, yaw-only coordinate frame."""

    anchor_w = _ground_anchor(root_pos_w, env_origins)
    # ``body_pos_w`` can add a trajectory or body axis after the environment
    # axis, e.g. [N, 8, 3] for root references and [N, 2, 14, 3] for
    # tracking feedback.  Insert singleton axes before xyz so the same root
    # ground anchor is broadcast over each of those axes.
    while anchor_w.ndim < body_pos_w.ndim:
        anchor_w = anchor_w.unsqueeze(-2)
    return _rotate_inverse_broadcast(
        _yaw_quat(root_quat_w), body_pos_w - anchor_w
    )


def _yaw_local_body_orientation(body_quat_w: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
    """Body orientation relative to that pose's yaw-only pelvis axes."""

    return _relative_quat_broadcast(_yaw_quat(root_quat_w), body_quat_w)


def teacher_state_observation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return current/history robot state plus full privileged actuator feedback (677-D)."""

    command = _command(env, command_name)
    history = command.teacher_state_history()
    root_pos_w = command.robot_root_pos_w
    root_quat_w = command.robot_root_quat_w
    observation_body_indices = command.observation_body_indices
    body_pos_b = _rotate_inverse_broadcast(
        root_quat_w,
        command.robot_body_pos_w[:, observation_body_indices]
        - _ground_anchor(root_pos_w, env.scene.env_origins)[:, None, :],
    )
    body_lin_vel_b = _rotate_inverse_broadcast(
        root_quat_w, command.robot_body_lin_vel_w[:, observation_body_indices]
    )
    observation = torch.cat(
        (
            *history,
            body_pos_b.flatten(1),
            body_lin_vel_b.flatten(1),
            # MimicLite's applied action is kept in policy-action units.  This
            # task has no delay/filter, therefore it equals the raw action.
            env.action_manager.action,
            command.robot_all_applied_torque,
        ),
        dim=-1,
    )
    if observation.shape[-1] != TEACHER_STATE_OBSERVATION_DIM:
        raise RuntimeError(
            f"G1 teacher_state ABI violation: got {observation.shape[-1]}, expected {TEACHER_STATE_OBSERVATION_DIM}."
        )
    return observation


def teacher_reference_observation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return reference trajectory and current-state tracking feedback (748-D)."""

    command = _command(env, command_name)
    root_index = command.root_body_index
    observation_body_indices = command.observation_body_indices
    robot_root_pos_w = command.robot_root_pos_w
    robot_root_quat_w = command.robot_root_quat_w
    ref_body_pos_w = command.reference_body_pos_w_window
    ref_body_quat_w = command.reference_body_quat_w_window
    ref_root_pos_w = ref_body_pos_w[:, :, root_index]
    ref_root_quat_w = ref_body_quat_w[:, :, root_index]

    reference_root_pos_yaw_local = _yaw_local_body_position(
        ref_root_pos_w,
        command.current_ref_body_pos_w[:, root_index],
        command.current_ref_body_quat_w[:, root_index],
        env.scene.env_origins,
    )
    reference_root_ori_robot_b = _rot6d(_relative_quat_broadcast(robot_root_quat_w, ref_root_quat_w))
    reference_root_pos_robot_b = _rotate_inverse_broadcast(robot_root_quat_w, ref_root_pos_w - robot_root_pos_w[:, None, :])

    feedback_all_pos_w, feedback_all_quat_w, feedback_all_lin_vel_w, feedback_all_ang_vel_w = (
        command.reference_body_state(TEACHER_TRACKING_FEEDBACK_OFFSETS)
    )
    feedback_root_pos_w = feedback_all_pos_w[:, :, root_index]
    feedback_root_quat_w = feedback_all_quat_w[:, :, root_index]
    feedback_body_pos_w = feedback_all_pos_w[:, :, observation_body_indices]
    feedback_body_quat_w = feedback_all_quat_w[:, :, observation_body_indices]
    feedback_body_lin_vel_w = feedback_all_lin_vel_w[:, :, observation_body_indices]
    feedback_body_ang_vel_w = feedback_all_ang_vel_w[:, :, observation_body_indices]

    robot_body_pos_yaw_local = _yaw_local_body_position(
        command.robot_body_pos_w[:, observation_body_indices], robot_root_pos_w, robot_root_quat_w, env.scene.env_origins
    )
    ref_body_pos_yaw_local = _yaw_local_body_position(
        feedback_body_pos_w, feedback_root_pos_w, feedback_root_quat_w, env.scene.env_origins
    )
    robot_body_ori_yaw_local = _yaw_local_body_orientation(
        command.robot_body_quat_w[:, observation_body_indices], robot_root_quat_w
    )
    ref_body_ori_yaw_local = _yaw_local_body_orientation(feedback_body_quat_w, feedback_root_quat_w)
    body_pos_error = ref_body_pos_yaw_local - robot_body_pos_yaw_local[:, None]
    body_ori_error = _rot6d(_relative_quat_broadcast(robot_body_ori_yaw_local[:, None], ref_body_ori_yaw_local))
    body_lin_vel_error_w = feedback_body_lin_vel_w - command.robot_body_lin_vel_w[:, None, observation_body_indices]
    body_ang_vel_error_w = feedback_body_ang_vel_w - command.robot_body_ang_vel_w[:, None, observation_body_indices]

    observation = torch.cat(
        (
            reference_root_pos_yaw_local.flatten(1),
            reference_root_ori_robot_b.flatten(1),
            command.reference_joint_pos_window.flatten(1),
            reference_root_pos_robot_b.flatten(1),
            body_pos_error.flatten(1),
            body_ori_error.flatten(1),
            body_lin_vel_error_w.flatten(1),
            body_ang_vel_error_w.flatten(1),
        ),
        dim=-1,
    )
    if observation.shape[-1] != TEACHER_REFERENCE_OBSERVATION_DIM:
        raise RuntimeError(
            "G1 teacher_reference ABI violation: "
            f"got {observation.shape[-1]}, expected {TEACHER_REFERENCE_OBSERVATION_DIM}."
        )
    return observation


def teacher_observation_dim() -> int:
    """Expose the actor/critic concatenated dimension for static checks."""

    return TEACHER_OBSERVATION_DIM
