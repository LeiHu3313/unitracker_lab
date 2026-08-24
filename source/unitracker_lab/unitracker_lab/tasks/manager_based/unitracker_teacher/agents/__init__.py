"""RSL-RL agent configuration for the G1 teacher."""

from .extreme_rgmt_ppo_cfg import UnitrackerExtremeRGMTPPORunnerCfg
from .rsl_rl_ppo_cfg import UnitrackerTeacherPPORunnerCfg

__all__ = ["UnitrackerTeacherPPORunnerCfg", "UnitrackerExtremeRGMTPPORunnerCfg"]
