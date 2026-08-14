"""Play a G1 Stage-1 teacher checkpoint on deterministic reference clips."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "rsl_rl"))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--motion", required=True, type=Path, help="Prepared G1 NPZ, recursive directory, or .txt/.lst manifest."
)
parser.add_argument(
    "--checkpoint_path", type=Path, default=None, help="Exact model .pt; otherwise resolve load_run/checkpoint."
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=0, help="Exit after N steps; 0 runs while the app is open.")
parser.add_argument("--task", default="Unitracker_Teacher-Play-v0", choices=["Unitracker_Teacher-Play-v0"])
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=1000)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import unitracker_lab.tasks  # noqa: F401, E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from unitracker_lab.tasks.manager_based.unitracker_teacher.motion_schema import (  # noqa: E402
    load_and_validate_motion_dataset,
)

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    motion = load_and_validate_motion_dataset(args_cli.motion)
    motion_path = motion.path
    motion_count = motion.num_motions
    motion_frame_count = motion.frame_count
    motion_fps = motion.fps
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_file = str(motion_path)
    env_cfg.commands.motion.debug_vis = not args_cli.headless
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    if args_cli.checkpoint_path is not None:
        checkpoint = args_cli.checkpoint_path.expanduser().resolve()
    else:
        log_root = _REPO_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name
        checkpoint = Path(get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint))
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")
    print(f"[INFO] Motion input: {motion_path}")
    print(f"[INFO] Motion clips/frames/FPS: {motion_count}/{motion_frame_count}/{motion_fps:g}")
    print(f"[INFO] Checkpoint: {checkpoint}")

    # Avoid retaining the preflight arrays while MotionCommand loads the data.
    del motion

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    try:
        if args_cli.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(checkpoint.parent / "videos" / "play"),
                step_trigger=lambda step: step == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        observations = env.get_observations()
        step = 0
        with torch.inference_mode():
            while simulation_app.is_running() and (args_cli.steps <= 0 or step < args_cli.steps):
                actions = policy(observations)
                observations, _, _, _ = env.step(actions)
                step += 1
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
