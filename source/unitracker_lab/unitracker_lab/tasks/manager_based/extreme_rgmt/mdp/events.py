"""Small Extreme-RGMT-specific domain-randomization terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .actions import ReferenceResidualJointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_motor_zero_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    offset_range: tuple[float, float],
    action_name: str = "joint_pos",
) -> None:
    """Add a persistent per-environment calibration offset to PD targets."""

    action: ReferenceResidualJointPositionAction = env.action_manager.get_term(action_name)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=action.device)
    action.motor_zero_offset[env_ids].uniform_(*offset_range)


def randomize_motor_strength(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Scale implicit-actuator effort limits independently per environment."""

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)
    for actuator in asset.actuators.values():
        limits = actuator.effort_limit_sim[env_ids].clone()
        limits *= torch.empty_like(limits).uniform_(*scale_range)
        actuator.effort_limit_sim[env_ids] = limits
        actuator.effort_limit[env_ids] = limits
        asset.write_joint_effort_limit_to_sim(limits, joint_ids=actuator.joint_indices, env_ids=env_ids)
