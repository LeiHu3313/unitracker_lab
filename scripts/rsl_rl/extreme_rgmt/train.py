"""Train either stage of the standalone Extreme-RGMT reproduction."""

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
parser.add_argument("--stage", required=True, choices=("base", "expansion"))
parser.add_argument("--motion", type=Path, help="Full Stage-I motion dataset.")
parser.add_argument("--mastered-motion", type=Path, help="Frozen Stage-II mastered motion set.")
parser.add_argument("--challenging-motion", type=Path, help="Frozen Stage-II challenging motion set.")
parser.add_argument("--base-checkpoint", type=Path, help="Stage-I checkpoint used for pi_theta and pi_ref.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--distributed", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.task = "Extreme-RGMT-Base-v0" if args_cli.stage == "base" else "Extreme-RGMT-Expansion-v0"
if args_cli.distributed:
    args_cli.device = f"cuda:{os.getenv('LOCAL_RANK', '0')}"
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import unitracker_lab.tasks  # noqa: F401, E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from unitracker_lab.tasks.manager_based.extreme_rgmt.contracts import contract_dict  # noqa: E402
from unitracker_lab.tasks.manager_based.extreme_rgmt.motion_schema import (  # noqa: E402
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


def _resolve_stage_inputs(env_cfg: ManagerBasedRLEnvCfg) -> tuple[dict[str, object], Path | None]:
    if args_cli.stage == "base":
        if args_cli.motion is None:
            raise ValueError("Stage I requires --motion.")
        motion = load_and_validate_motion_dataset(args_cli.motion)
        env_cfg.commands.motion.motion_file = str(motion.path)
        return {"full": motion.summary(), "full_paths": motion.paths}, None

    missing = [
        name
        for name, value in (
            ("--mastered-motion", args_cli.mastered_motion),
            ("--challenging-motion", args_cli.challenging_motion),
            ("--base-checkpoint", args_cli.base_checkpoint),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"Stage II requires: {', '.join(missing)}")
    checkpoint = args_cli.base_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Stage-I checkpoint does not exist: {checkpoint}")
    mastered = load_and_validate_motion_dataset(args_cli.mastered_motion)
    challenging = load_and_validate_motion_dataset(args_cli.challenging_motion)
    merged = merge_validated_motion_datasets(mastered, challenging, source_label="extreme-rgmt-preflight")
    env_cfg.commands.motion.motion_file = str(mastered.path)
    env_cfg.commands.motion.mastered_motion_file = str(mastered.path)
    env_cfg.commands.motion.challenging_motion_file = str(challenging.path)
    return {
        "mastered": mastered.summary(),
        "challenging": challenging.summary(),
        "mastered_paths": mastered.paths,
        "challenging_paths": challenging.paths,
        "merged_sha256": merged.sha256,
    }, checkpoint


def _write_run_contracts(
    params_dir: Path,
    env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    inputs: dict[str, object],
    checkpoint: Path | None,
) -> None:
    dump_yaml(str(params_dir / "env.yaml"), env_cfg)
    dump_yaml(str(params_dir / "agent.yaml"), agent_cfg)
    _dump_json(params_dir / "g1_contract.json", contract_dict())
    method = {
        "paper": "arXiv:2607.20110v1",
        "stage": args_cli.stage,
        "scope": "standalone clean-room reproduction",
        "inputs": {key: value for key, value in inputs.items() if not key.endswith("_paths")},
    }
    if checkpoint is not None:
        method.update({
            "base_checkpoint": str(checkpoint),
            "base_checkpoint_sha256": _sha256(checkpoint),
            "valid_sample_definition": "rollout transitions whose done flag is false",
            "acquisition_fraction": 0.8,
            "pace": {"lambda_base": 0.3, "kappa": 5.0, "rho_ref": 0.6, "beta": 0.99},
            "star": {"topk_fraction": 0.05, "resample_fraction": 0.25, "difficulty_threshold": 1.0},
        })
    _dump_json(params_dir / "extreme_rgmt_contract.json", method)
    for key, value in inputs.items():
        if key.endswith("_paths"):
            (params_dir / f"{key.removesuffix('_paths')}_resolved.lst").write_text(
                "".join(f"{path}\n" for path in value), encoding="utf-8"
            )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    inputs, checkpoint = _resolve_stage_inputs(env_cfg)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if agent_cfg.resume:
        raise ValueError("This standalone launcher requires explicit stage initialization; --resume is unsupported.")
    if args_cli.num_envs is not None:
        minimum = 2 if args_cli.stage == "expansion" else 1
        if args_cli.num_envs < minimum:
            raise ValueError(f"{args_cli.stage} requires --num_envs >= {minimum}.")
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

    is_main = not args_cli.distributed or app_launcher.global_rank == 0
    log_dir: Path | None = None
    if is_main:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            run_name += f"_{agent_cfg.run_name}"
        log_dir = _REPO_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name / run_name
        params_dir = log_dir / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        _write_run_contracts(params_dir, env_cfg, agent_cfg, inputs, checkpoint)
        print(f"[INFO] Extreme-RGMT {args_cli.stage} log: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=str(log_dir) if log_dir is not None else None, device=agent_cfg.device
        )
        runner.add_git_repo_to_log(__file__)
        if checkpoint is not None:
            print("[INFO] Initializing pi_theta and frozen pi_ref from the Stage-I checkpoint.")
            runner.load(str(checkpoint), load_optimizer=False, map_location=agent_cfg.device, restore_iteration=False)
        started = time.time()
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
        print(f"[INFO] Training completed in {time.time() - started:.1f}s")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
