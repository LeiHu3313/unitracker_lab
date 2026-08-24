"""789-D privileged oracle observation for the G1 Stage-1 teacher."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

from ..contracts import FOOT_CONTACT_FORCE_THRESHOLD_N, G1_FOOT_BODY_NAMES, ORACLE_OBSERVATION_DIM
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


def _foot_contact_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the current left/right foot contact mode in the contract order."""

    sensor = env.scene["contact_forces"]
    body_ids, body_names = sensor.find_bodies(list(G1_FOOT_BODY_NAMES), preserve_order=True)
    if tuple(body_names) != G1_FOOT_BODY_NAMES:
        raise RuntimeError(f"Live G1 contact bodies do not match contract: {body_names}")
    contact_force = torch.linalg.vector_norm(sensor.data.net_forces_w_history[:, 0, body_ids], dim=-1)
    return (contact_force > FOOT_CONTACT_FORCE_THRESHOLD_N).to(dtype=torch.float32)


def teacher_oracle_observation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return current state, a ``t -> t+1`` body goal, and five reference joint frames.

    The pelvis is represented once as a root block.  All remaining tracking
    bodies are root-local, so global root x/y translation and absolute yaw do
    not enter the policy ABI.  The next-frame body-pose goals compare the
    robot and reference in their respective pelvis frames, so they represent
    relative configuration rather than accumulated world-space drift.
    """

    command = _command(env, command_name)
    root_pos_w = command.robot_root_pos_w
    root_quat_w = command.robot_root_quat_w
    # The hard body-order contract puts pelvis/root at index zero.  Excluding
    # it from body blocks avoids constant root-local position/orientation terms.
    non_root = slice(1, None)
    robot_body_pos_w = command.robot_body_pos_w[:, non_root]
    robot_body_quat_w = command.robot_body_quat_w[:, non_root]
    robot_body_lin_vel_w = command.robot_body_lin_vel_w[:, non_root]
    robot_body_ang_vel_w = command.robot_body_ang_vel_w[:, non_root]
    target_body_pos_w = command.target_ref_body_pos_w[:, non_root]
    target_body_quat_w = command.target_ref_body_quat_w[:, non_root]
    target_body_lin_vel_w = command.target_ref_body_lin_vel_w[:, non_root]
    target_body_ang_vel_w = command.target_ref_body_ang_vel_w[:, non_root]

    root_height = root_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]
    gravity_w = torch.zeros_like(root_pos_w)
    gravity_w[:, 2] = -1.0
    projected_gravity = math_utils.quat_apply_inverse(root_quat_w, gravity_w)
    root_lin_vel_local = math_utils.quat_apply_inverse(root_quat_w, command.robot_body_lin_vel_w[:, 0])
    root_ang_vel_local = math_utils.quat_apply_inverse(root_quat_w, command.robot_body_ang_vel_w[:, 0])

    current_body_pos_local = _rotate_inverse(root_quat_w, robot_body_pos_w - root_pos_w[:, None, :])
    current_body_ori_local_quat = _relative_quat(root_quat_w, robot_body_quat_w)
    current_body_ori_local = _rot6d(current_body_ori_local_quat)
    current_body_lin_vel_local = _rotate_inverse(root_quat_w, robot_body_lin_vel_w)
    current_body_ang_vel_local = _rotate_inverse(root_quat_w, robot_body_ang_vel_w)

    target_root_pos_w = command.target_ref_body_pos_w[:, 0]
    target_root_quat_w = command.target_ref_body_quat_w[:, 0]
    target_root_lin_vel_w = command.target_ref_body_lin_vel_w[:, 0]
    target_root_ang_vel_w = command.target_ref_body_ang_vel_w[:, 0]
    torso_body_index = command.cfg.body_names.index("torso_link")
    target_torso_pos_w = command.target_ref_body_pos_w[:, torso_body_index]
    robot_torso_pos_w = command.robot_body_pos_w[:, torso_body_index]
    target_root_height_error = target_root_pos_w[:, 2:3] - root_pos_w[:, 2:3]
    target_root_ori_error = _rot6d(_quat_error(root_quat_w, target_root_quat_w))
    target_root_lin_vel_error_local = math_utils.quat_apply_inverse(
        root_quat_w, target_root_lin_vel_w - command.robot_body_lin_vel_w[:, 0]
    )
    target_root_ang_vel_error_local = math_utils.quat_apply_inverse(
        root_quat_w, target_root_ang_vel_w - command.robot_body_ang_vel_w[:, 0]
    )
    target_torso_pos_error_local = math_utils.quat_apply_inverse(root_quat_w, target_torso_pos_w - robot_torso_pos_w)

    target_body_pos_local = _rotate_inverse(
        target_root_quat_w, target_body_pos_w - target_root_pos_w[:, None, :]
    )
    target_body_ori_local_quat = _relative_quat(target_root_quat_w, target_body_quat_w)
    target_body_pos_error_local = target_body_pos_local - current_body_pos_local
    target_body_ori_error = _rot6d(_quat_error(current_body_ori_local_quat, target_body_ori_local_quat))
    target_body_lin_vel_error_local = _rotate_inverse(root_quat_w, target_body_lin_vel_w - robot_body_lin_vel_w)
    target_body_ang_vel_error_local = _rotate_inverse(root_quat_w, target_body_ang_vel_w - robot_body_ang_vel_w)

    previous_action = env.action_manager.action
    blocks = (
        root_height,
        projected_gravity,
        root_lin_vel_local,
        root_ang_vel_local,
        current_body_pos_local.flatten(1),
        current_body_ori_local.flatten(1),
        current_body_lin_vel_local.flatten(1),
        current_body_ang_vel_local.flatten(1),
        command.robot_all_joint_pos - command.robot_all_default_joint_pos,
        command.robot_all_joint_vel,
        _foot_contact_mask(env),
        previous_action,
        target_root_height_error,
        target_root_ori_error,
        target_root_lin_vel_error_local,
        target_root_ang_vel_error_local,
        target_torso_pos_error_local,
        target_body_pos_error_local.flatten(1),
        target_body_ori_error.flatten(1),
        target_body_lin_vel_error_local.flatten(1),
        target_body_ang_vel_error_local.flatten(1),
        command.future_ref_joint_command,
    )
    observation = torch.cat(blocks, dim=-1)
    if observation.shape[-1] != ORACLE_OBSERVATION_DIM:
        raise RuntimeError(
            f"G1 oracle observation ABI violation: got {observation.shape[-1]}, expected {ORACLE_OBSERVATION_DIM}."
        )
    return observation
