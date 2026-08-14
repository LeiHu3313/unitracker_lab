"""Train the G1-only UniTracker Stage-1 privileged tracking teacher."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "rsl_rl"))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--motion",
    required=True,
    type=Path,
    help="Prepared 50-Hz G1 NPZ, recursive directory, or .txt/.lst manifest.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Override the production default of 8192 envs.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override the 30000-iteration PPO baseline.")
parser.add_argument("--seed", type=int, default=None, help="Training seed; -1 samples a seed.")
parser.add_argument("--task", default="Unitracker_Teacher-v0", choices=["Unitracker_Teacher-v0"])
parser.add_argument("--video", action="store_true", help="Record periodic rollout videos.")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
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
from unitracker_lab.assets.g1 import G1_ACTION_SCALE  # noqa: E402
from unitracker_lab.tasks.manager_based.unitracker_teacher.contracts import contract_dict  # noqa: E402
from unitracker_lab.tasks.manager_based.unitracker_teacher.motion_schema import (  # noqa: E402
    load_and_validate_motion_dataset,
)

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False


def _dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_run_contracts(params_dir: Path, motion_summary: dict[str, object], motion_paths: tuple[Path, ...]) -> None:
    contract = contract_dict()
    contract["action_scale"] = G1_ACTION_SCALE
    _dump_json(params_dir / "g1_contract.json", contract)
    _dump_json(params_dir / "motion_data_resolved.json", motion_summary)
    (params_dir / "used_motions.txt").write_text("".join(f"{path}\n" for path in motion_paths), encoding="utf-8")
    asset_manifest_path = (
        _REPO_ROOT / "source" / "unitracker_lab" / "unitracker_lab" / "assets" / "g1" / "asset_manifest.json"
    )
    _dump_json(params_dir / "asset_manifest.json", json.loads(asset_manifest_path.read_text(encoding="utf-8")))
    _dump_json(
        params_dir / "reference_project.json",
        {
            "path": "/home/hul/whole_body_tracking",
            "commit": "cd65172032893724b445448818c34165846d847d",
            "role": "G1 asset/PD/action-scale and tracking implementation reference",
        },
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    motion = load_and_validate_motion_dataset(args_cli.motion)
    motion_path = motion.path
    motion_summary = motion.summary()
    motion_paths = motion.paths
    motion_count = motion.num_motions
    motion_frame_count = motion.frame_count
    motion_fps = motion.fps
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        if args_cli.num_envs <= 0:
            raise ValueError("--num_envs must be positive.")
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        if args_cli.max_iterations <= 0:
            raise ValueError("--max_iterations must be positive.")
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    env_cfg.commands.motion.motion_file = str(motion_path)
    env_cfg.seed = agent_cfg.seed

    log_root = _REPO_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name
    log_root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = log_root / run_name
    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] G1 motion input: {motion_path}")
    print(f"[INFO] Motion clips/frames/FPS: {motion_count}/{motion_frame_count}/{motion_fps:g}")
    print(f"[INFO] Environments: {env_cfg.scene.num_envs}")
    print(f"[INFO] 23-D action scales: {G1_ACTION_SCALE}")
    print(f"[INFO] Logging: {log_dir}")

    dump_yaml(str(params_dir / "env.yaml"), env_cfg)
    dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)
    _write_run_contracts(params_dir, motion_summary, motion_paths)

    # MotionCommand loads the dataset onto its target device during gym.make().
    # Release this preflight copy first so large datasets are not resident twice.
    del motion

    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    try:
        if args_cli.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(log_dir / "videos" / "train"),
                step_trigger=lambda step: step % args_cli.video_interval == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)
        if resume_path is not None:
            print(f"[INFO] Resuming checkpoint: {resume_path}")
            runner.load(resume_path)
        started = time.time()
        # RSI owns phase randomization. Random episode-length initialization would
        # introduce a second, unrelated time sampler.
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
        print(f"[INFO] Training completed in {time.time() - started:.1f}s")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
