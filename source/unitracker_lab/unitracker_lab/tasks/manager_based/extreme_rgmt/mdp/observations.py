"""Paper-aligned actor and asymmetric-critic observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

from ..contracts import (
    ACTION_DIM,
    CRITIC_PRIVILEGED_DIM,
    PROPRIOCEPTION_DIM,
    REFERENCE_TOKEN_DIM,
    REFERENCE_WINDOW_LENGTH,
)
from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command(env: ManagerBasedRLEnv, command_name: str) -> MotionCommand:
    return env.command_manager.get_term(command_name)


def _rotate_inverse(quaternions: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    return math_utils.quat_apply_inverse(quaternions.reshape(-1, 4), vectors.reshape(-1, 3)).reshape_as(vectors)


def _relative_quat(root_quat_w: torch.Tensor, body_quat_w: torch.Tensor) -> torch.Tensor:
    root = root_quat_w[:, None, :].expand_as(body_quat_w)
    return math_utils.quat_mul(math_utils.quat_inv(root.reshape(-1, 4)), body_quat_w.reshape(-1, 4)).reshape_as(
        body_quat_w
    )


def _rot6d(quaternions: torch.Tensor) -> torch.Tensor:
    matrix = math_utils.matrix_from_quat(quaternions.reshape(-1, 4))
    return matrix[..., :, :2].transpose(-1, -2).reshape(*quaternions.shape[:-1], 6)


def _noise_like(values: torch.Tensor, magnitude: float, active: torch.Tensor) -> torch.Tensor:
    noise = (2.0 * torch.rand_like(values) - 1.0) * magnitude
    shape = (active.shape[0],) + (1,) * (values.ndim - 1)
    return values + noise * active.reshape(shape)


def proprioception(env: ManagerBasedRLEnv, command_name: str, enable_noise: bool = True) -> torch.Tensor:
    """Eq. (1): projected gravity, base angular velocity, q-q0, and qdot."""

    command = _command(env, command_name)
    root_quat_w = command.robot_root_quat_w
    gravity_w = torch.zeros_like(command.robot_root_pos_w)
    gravity_w[:, 2] = -1.0
    gravity = math_utils.quat_apply_inverse(root_quat_w, gravity_w)
    angular_velocity = math_utils.quat_apply_inverse(root_quat_w, command.robot_body_ang_vel_w[:, 0])
    joint_position = command.robot_all_joint_pos - command.robot_all_default_joint_pos
    joint_velocity = command.robot_all_joint_vel
    if enable_noise:
        active = command.acquisition_mask
        gravity = _noise_like(gravity, 0.05, active)
        angular_velocity = _noise_like(angular_velocity, 0.2, active)
        joint_position = _noise_like(joint_position, 0.01, active)
        joint_velocity = _noise_like(joint_velocity, 0.5, active)
    observation = torch.cat((gravity, angular_velocity, joint_position, joint_velocity), dim=-1)
    if observation.shape[-1] != PROPRIOCEPTION_DIM:
        raise RuntimeError(f"Proprioception width is {observation.shape[-1]}; expected {PROPRIOCEPTION_DIM}.")
    return observation


def previous_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    action = env.action_manager.action
    if action.shape[-1] != ACTION_DIM:
        raise RuntimeError(f"Previous-action width is {action.shape[-1]}; expected {ACTION_DIM}.")
    return action


def reference_window(env: ManagerBasedRLEnv, command_name: str, enable_noise: bool = True) -> torch.Tensor:
    """Eq. (2): 21 reference tokens ``[v, omega, gravity, q]``."""

    command = _command(env, command_name)
    root_quat_w = command.reference_window_root_quat_w
    linear_velocity = _rotate_inverse(root_quat_w, command.reference_window_root_lin_vel_w)
    angular_velocity = _rotate_inverse(root_quat_w, command.reference_window_root_ang_vel_w)
    gravity_w = torch.zeros_like(linear_velocity)
    gravity_w[..., 2] = -1.0
    gravity = _rotate_inverse(root_quat_w, gravity_w)
    joint_position = command.reference_window_joint_pos
    if enable_noise:
        active = command.acquisition_mask
        linear_velocity = _noise_like(linear_velocity, 0.5, active)
        angular_velocity = _noise_like(angular_velocity, 0.52, active)
        gravity = _noise_like(gravity, 0.05, active)
        joint_position = _noise_like(joint_position, 0.1, active)
    tokens = torch.cat((linear_velocity, angular_velocity, gravity, joint_position), dim=-1)
    if tokens.shape[-2:] != (REFERENCE_WINDOW_LENGTH, REFERENCE_TOKEN_DIM):
        raise RuntimeError(
            f"Reference window shape is {tokens.shape[-2:]}; "
            f"expected {(REFERENCE_WINDOW_LENGTH, REFERENCE_TOKEN_DIM)}."
        )
    return tokens.flatten(1)


def critic_privileged_state(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reference height plus robot link pose and base velocity available only in simulation."""

    command = _command(env, command_name)
    root_pos_w = command.robot_root_pos_w
    root_quat_w = command.robot_root_quat_w
    body_pos_local = _rotate_inverse(
        root_quat_w[:, None, :].expand(-1, command.robot_body_pos_w.shape[1], -1),
        command.robot_body_pos_w - root_pos_w[:, None, :],
    )
    body_orientation_local = _rot6d(_relative_quat(root_quat_w, command.robot_body_quat_w))
    robot_base_linear_velocity = math_utils.quat_apply_inverse(
        command.robot_root_quat_w, command.robot_body_lin_vel_w[:, 0]
    )
    reference_base_height = command.current_ref_body_pos_w[:, 0, 2:3] - env.scene.env_origins[:, 2:3]
    privileged = torch.cat(
        (
            reference_base_height,
            body_pos_local.flatten(1),
            body_orientation_local.flatten(1),
            robot_base_linear_velocity,
        ),
        dim=-1,
    )
    if privileged.shape[-1] != CRITIC_PRIVILEGED_DIM:
        raise RuntimeError(
            f"Critic privileged width is {privileged.shape[-1]}; expected {CRITIC_PRIVILEGED_DIM}."
        )
    return privileged
