"""Train Extreme-RGMT Stage II with PACE and STAR from a Stage-I checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "rsl_rl"))
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mastered-motion", required=True, type=Path, help="Frozen mastered .lst/.txt or dataset path.")
parser.add_argument(
    "--challenging-motion", required=True, type=Path, help="Frozen challenging .lst/.txt or dataset path."
)
parser.add_argument(
    "--base-checkpoint", required=True, type=Path, help="Stage-I model checkpoint used for pi and pi_ref."
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--task", default="Unitracker_ExtremeRGMT-v0", choices=["Unitracker_ExtremeRGMT-v0"])
parser.add_argument("--distributed", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.distributed:
    args_cli.device = f"cuda:{os.getenv('LOCAL_RANK', '0')}"
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import unitracker_lab.tasks  # noqa: F401, E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from unitracker_lab.tasks.manager_based.unitracker_teacher.contracts import contract_dict  # noqa: E402
from unitracker_lab.tasks.manager_based.unitracker_teacher.motion_schema import (  # noqa: E402
    load_and_validate_motion_dataset,
    merge_validated_motion_datasets,
)

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    mastered_path = args_cli.mastered_motion.expanduser().resolve()
    challenging_path = args_cli.challenging_motion.expanduser().resolve()
    checkpoint = args_cli.base_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Stage-I base checkpoint does not exist: {checkpoint}")

    mastered = load_and_validate_motion_dataset(mastered_path)
    challenging = load_and_validate_motion_dataset(challenging_path)
    merged = merge_validated_motion_datasets(mastered, challenging, source_label="extreme-rgmt-preflight")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # --resume has a different meaning here: Stage II always initializes from
    # --base-checkpoint and deliberately does not restore the Stage-I optimizer.
    if agent_cfg.resume:
        raise ValueError("Do not use --resume for Stage-II initialization; pass --base-checkpoint explicitly.")
    if args_cli.num_envs is not None:
        if args_cli.num_envs < 2:
            raise ValueError("PACE requires --num_envs >= 2.")
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        if args_cli.max_iterations <= 0:
            raise ValueError("--max_iterations must be positive.")
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        env_cfg.seed = agent_cfg.seed + app_launcher.local_rank
        agent_cfg.seed = env_cfg.seed
    elif args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
        env_cfg.seed = agent_cfg.seed

    env_cfg.commands.motion.motion_file = str(mastered_path)
    env_cfg.commands.motion.mastered_motion_file = str(mastered_path)
    env_cfg.commands.motion.challenging_motion_file = str(challenging_path)
    env_cfg.commands.motion.acquisition_fraction = 0.8

    is_main = not args_cli.distributed or app_launcher.global_rank == 0
    log_dir: Path | None = None
    if is_main:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            run_name += f"_{agent_cfg.run_name}"
        log_dir = _REPO_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name / run_name
        params_dir = log_dir / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(str(params_dir / "env.yaml"), env_cfg)
        dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)
        _dump_json(params_dir / "g1_contract.json", contract_dict())
        _dump_json(
            params_dir / "extreme_rgmt_contract.json",
            {
                "paper": "arXiv:2607.20110v1",
                "scope": "Stage-II PACE/STAR on the repository's 23-DoF privileged-teacher base",
                "base_checkpoint": str(checkpoint),
                "base_checkpoint_sha256": _sha256(checkpoint),
                "mastered": mastered.summary(),
                "challenging": challenging.summary(),
                "merged_sha256": merged.sha256,
                "valid_sample_definition": "rollout transitions whose done flag is false",
                "acquisition_fraction": 0.8,
                "pace": {"lambda_base": 0.3, "kappa": 5.0, "rho_ref": 0.6, "beta": 0.99},
                "star": {"topk_fraction": 0.05, "resample_fraction": 0.25, "difficulty_threshold": 1.0},
            },
        )
        (params_dir / "mastered_resolved.lst").write_text(
            "".join(f"{path}\n" for path in mastered.paths), encoding="utf-8"
        )
        (params_dir / "challenging_resolved.lst").write_text(
            "".join(f"{path}\n" for path in challenging.paths), encoding="utf-8"
        )
        print(f"[INFO] Extreme-RGMT Stage II log: {log_dir}")
        print(f"[INFO] Mastered/challenging clips: {mastered.num_motions}/{challenging.num_motions}")
        print(f"[INFO] Base checkpoint: {checkpoint}")

    del merged, mastered, challenging
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=str(log_dir) if log_dir is not None else None, device=agent_cfg.device
        )
        runner.add_git_repo_to_log(__file__)
        print("[INFO] Initializing pi_theta and frozen pi_ref from Stage-I checkpoint; optimizer is reset.")
        runner.load(str(checkpoint), load_optimizer=False, map_location=agent_cfg.device, restore_iteration=False)
        started = time.time()
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
        print(f"[INFO] Stage-II training completed in {time.time() - started:.1f}s")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
