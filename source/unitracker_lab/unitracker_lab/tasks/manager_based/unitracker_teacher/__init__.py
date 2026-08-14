"""Gym registrations for the G1-only UniTracker Stage-1 teacher."""

import gymnasium as gym

from .agents import UnitrackerTeacherPPORunnerCfg
from .unitracker_teacher_env_cfg import UnitrackerTeacherEnvCfg, UnitrackerTeacherPlayEnvCfg

gym.register(
    id="Unitracker_Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UnitrackerTeacherEnvCfg,
        "rsl_rl_cfg_entry_point": UnitrackerTeacherPPORunnerCfg,
    },
)

gym.register(
    id="Unitracker_Teacher-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UnitrackerTeacherPlayEnvCfg,
        "rsl_rl_cfg_entry_point": UnitrackerTeacherPPORunnerCfg,
    },
)

__all__ = ["UnitrackerTeacherEnvCfg", "UnitrackerTeacherPlayEnvCfg", "UnitrackerTeacherPPORunnerCfg"]
