"""Opt-in Isaac Sim reset/step and policy-construction smoke test."""

from __future__ import annotations

import os

import pytest

_HAS_MOTIONS = bool(
    os.getenv("EXTREME_RGMT_MASTERED_MOTION") and os.getenv("EXTREME_RGMT_CHALLENGING_MOTION")
)


@pytest.mark.skipif(not _HAS_MOTIONS, reason="set both EXTREME_RGMT_*_MOTION inputs for the Isaac Sim smoke")
def test_two_role_env_reset_step_and_policy_construction():
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    app = launcher.app
    try:
        import gymnasium as gym
        import torch
        import unitracker_lab.tasks  # noqa: F401
        from rsl_rl.runners import OnPolicyRunner
        from unitracker_lab.tasks.manager_based.extreme_rgmt.agents import ExtremeRGMTExpansionPPORunnerCfg
        from unitracker_lab.tasks.manager_based.extreme_rgmt.extreme_rgmt_env_cfg import ExtremeRGMTEnvCfg

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        env_cfg = ExtremeRGMTEnvCfg()
        env_cfg.scene.num_envs = 2
        env_cfg.commands.motion.mastered_motion_file = os.environ["EXTREME_RGMT_MASTERED_MOTION"]
        env_cfg.commands.motion.challenging_motion_file = os.environ["EXTREME_RGMT_CHALLENGING_MOTION"]
        env_cfg.commands.motion.debug_vis = False
        env = gym.make("Extreme-RGMT-Expansion-v0", cfg=env_cfg)
        try:
            env = RslRlVecEnvWrapper(env, clip_actions=1.0)
            observations = env.get_observations()
            assert observations["policy"].shape == (2, 789)
            observations, _, _, _ = env.step(torch.zeros((2, 23), device=env.unwrapped.device))
            assert observations["policy"].shape == (2, 789)
            runner_cfg = ExtremeRGMTExpansionPPORunnerCfg()
            runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
            assert runner is not None
        finally:
            env.close()
    finally:
        app.close()
