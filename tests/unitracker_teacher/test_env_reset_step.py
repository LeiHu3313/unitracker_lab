"""Opt-in Isaac Sim reset/step and policy-construction smoke test."""

from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(not os.getenv("UNITRACKER_G1_MOTION"), reason="set UNITRACKER_G1_MOTION for Isaac Sim smoke")
def test_one_env_reset_step_and_policy_construction():
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    app = launcher.app
    try:
        import gymnasium as gym
        import torch
        import unitracker_lab.tasks  # noqa: F401
        from rsl_rl.runners import OnPolicyRunner
        from unitracker_lab.tasks.manager_based.unitracker_teacher.agents import UnitrackerTeacherPPORunnerCfg
        from unitracker_lab.tasks.manager_based.unitracker_teacher.unitracker_teacher_env_cfg import (
            UnitrackerTeacherPlayEnvCfg,
        )

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        env_cfg = UnitrackerTeacherPlayEnvCfg()
        env_cfg.scene.num_envs = 1
        env_cfg.commands.motion.motion_file = os.environ["UNITRACKER_G1_MOTION"]
        env_cfg.commands.motion.debug_vis = False
        env = gym.make("Unitracker_Teacher-Play-v0", cfg=env_cfg)
        try:
            env = RslRlVecEnvWrapper(env, clip_actions=1.0)
            observations = env.get_observations()
            assert observations["teacher"].shape == (1, 789)
            observations, _, _, _ = env.step(torch.zeros((1, 23), device=env.unwrapped.device))
            assert observations["teacher"].shape == (1, 789)
            command = env.unwrapped.command_manager.get_term("motion")
            wrist_targets = command.robot.data.joint_pos_target[:, command.locked_joint_ids]
            assert torch.allclose(wrist_targets, torch.zeros_like(wrist_targets))
            runner_cfg = UnitrackerTeacherPPORunnerCfg()
            runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
            assert runner is not None
        finally:
            env.close()
    finally:
        app.close()
