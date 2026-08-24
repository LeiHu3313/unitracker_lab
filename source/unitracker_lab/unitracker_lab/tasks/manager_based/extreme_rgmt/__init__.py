"""Gym registration for the standalone Extreme-RGMT reproduction."""

import gymnasium as gym

from .agents import ExtremeRGMTBasePPORunnerCfg, ExtremeRGMTExpansionPPORunnerCfg
from .extreme_rgmt_env_cfg import ExtremeRGMTEnvCfg

gym.register(
    id="Extreme-RGMT-Base-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ExtremeRGMTEnvCfg,
        "rsl_rl_cfg_entry_point": ExtremeRGMTBasePPORunnerCfg,
    },
)

gym.register(
    id="Extreme-RGMT-Expansion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ExtremeRGMTEnvCfg,
        "rsl_rl_cfg_entry_point": ExtremeRGMTExpansionPPORunnerCfg,
    },
)

__all__ = ["ExtremeRGMTEnvCfg", "ExtremeRGMTBasePPORunnerCfg", "ExtremeRGMTExpansionPPORunnerCfg"]
