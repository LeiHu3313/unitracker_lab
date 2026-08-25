"""Whole-body tracking rewards and physical regularization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command(env: ManagerBasedRLEnv, command_name: str) -> MotionCommand:
    return env.command_manager.get_term(command_name)


def _exp_mean_square(error: torch.Tensor, sigma: float, dims: tuple[int, ...]) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError("Tracking reward sigma must be positive.")
    return torch.exp(-torch.square(error).mean(dim=dims) / (sigma * sigma))


def _body_id(command: MotionCommand, body_name: str) -> int:
    try:
        return command.cfg.body_names.index(body_name)
    except ValueError as exc:
        raise ValueError(f"Tracking body {body_name!r} is not configured for the motion command.") from exc


def _root_local_positions(
    body_pos_w: torch.Tensor, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor
) -> torch.Tensor:
    """Express a batch of world positions in its corresponding root frame."""

    vectors_w = body_pos_w - root_pos_w[:, None, :]
    roots = root_quat_w[:, None, :].expand(-1, vectors_w.shape[1], -1)
    return math_utils.quat_apply_inverse(roots.reshape(-1, 4), vectors_w.reshape(-1, 3)).reshape_as(vectors_w)


def _root_relative_quaternions(body_quat_w: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
    """Return body orientations relative to the corresponding root orientation."""

    roots = root_quat_w[:, None, :].expand(-1, body_quat_w.shape[1], -1)
    return math_utils.quat_mul(
        math_utils.quat_inv(roots.reshape(-1, 4)), body_quat_w.reshape(-1, 4)
    ).reshape_as(body_quat_w)


def _root_relative_position_error(command: MotionCommand, body_ids: list[int]) -> torch.Tensor:
    """Reference and robot positions in their own root-local frames."""

    robot_local = _root_local_positions(
        command.robot_body_pos_w[:, body_ids], command.robot_root_pos_w, command.robot_root_quat_w
    )
    reference_local = _root_local_positions(
        command.target_ref_body_pos_w[:, body_ids],
        command.target_ref_body_pos_w[:, 0],
        command.target_ref_body_quat_w[:, 0],
    )
    return reference_local - robot_local


def global_body_orientation_tracking_exp(
    env: ManagerBasedRLEnv, command_name: str, body_name: str, sigma: float
) -> torch.Tensor:
    """Track one body orientation in world coordinates as a soft global anchor."""

    command = _command(env, command_name)
    body_id = _body_id(command, body_name)
    error = quat_error_magnitude(command.target_ref_body_quat_w[:, body_id], command.robot_body_quat_w[:, body_id])
    return torch.exp(-torch.square(error) / (sigma * sigma))


def body_position_tracking_exp(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    """Track non-root body layout in root-local coordinates, not world position."""

    command = _command(env, command_name)
    body_ids = list(range(1, len(command.cfg.body_names)))
    return _exp_mean_square(_root_relative_position_error(command, body_ids), sigma, (1, 2))


def body_orientation_tracking_exp(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    """Track non-root body orientation relative to each pose's root orientation."""

    command = _command(env, command_name)
    robot_relative = _root_relative_quaternions(command.robot_body_quat_w[:, 1:], command.robot_root_quat_w)
    reference_relative = _root_relative_quaternions(
        command.target_ref_body_quat_w[:, 1:], command.target_ref_body_quat_w[:, 0]
    )
    error = quat_error_magnitude(reference_relative, robot_relative)
    return _exp_mean_square(error, sigma, (1,))


def body_linear_velocity_tracking_exp(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    command = _command(env, command_name)
    return _exp_mean_square(command.target_ref_body_lin_vel_w - command.robot_body_lin_vel_w, sigma, (1, 2))


def body_angular_velocity_tracking_exp(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    command = _command(env, command_name)
    return _exp_mean_square(command.target_ref_body_ang_vel_w - command.robot_body_ang_vel_w, sigma, (1, 2))


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Unscheduled action-rate penalty used by the active Extreme-RGMT task."""

    return torch.square(env.action_manager.action - env.action_manager.prev_action).sum(dim=1)


def controlled_joint_position_limit_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize excursions beyond the articulation's soft joint limits."""

    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    below_limit = torch.clamp(limits[..., 0] - joint_pos, min=0.0)
    above_limit = torch.clamp(joint_pos - limits[..., 1], min=0.0)
    return (below_limit + above_limit).sum(dim=1)


def foot_slip_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Unscheduled foot-slip penalty used by the active Extreme-RGMT task."""

    sensor = env.scene[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    contact_force = torch.linalg.vector_norm(sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids], dim=-1)
    planar_speed_sq = torch.square(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]).sum(dim=-1)
    return (planar_speed_sq * (contact_force > threshold)).mean(dim=-1)
