"""Reference-relative 29-DoF action defined by Extreme-RGMT Eq. (3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ReferenceResidualJointPositionAction(JointPositionAction):
    """Apply ``q_target = q_reference + residual`` to all configured joints."""

    cfg: ReferenceResidualJointPositionActionCfg

    def __init__(self, cfg: ReferenceResidualJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        if cfg.use_default_offset:
            raise ValueError("Reference-residual action requires use_default_offset=False.")
        super().__init__(cfg, env)
        self.motor_zero_offset = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        command: MotionCommand = self._env.command_manager.get_term(self.cfg.command_name)
        if tuple(self._joint_names) != tuple(command.cfg.controlled_joint_names):
            raise RuntimeError("Action and motion-command joint orders differ.")
        self._processed_actions = command.current_ref_joint_pos + self._raw_actions * self._scale
        self._processed_actions += self.motor_zero_offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@configclass
class ReferenceResidualJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ReferenceResidualJointPositionAction
    command_name: str = "motion"
    use_default_offset: bool = False
