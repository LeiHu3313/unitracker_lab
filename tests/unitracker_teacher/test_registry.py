from pathlib import Path


def test_only_g1_teacher_ids_are_registered_in_source():
    source = (
        Path(__file__).resolve().parents[2]
        / "source/unitracker_lab/unitracker_lab/tasks/manager_based/unitracker_teacher/__init__.py"
    ).read_text(encoding="utf-8")
    assert 'id="Unitracker_Teacher-v0"' in source
    assert 'id="Unitracker_Teacher-Play-v0"' in source
    task_dir = (
        Path(__file__).resolve().parents[2]
        / "source/unitracker_lab/unitracker_lab/tasks/manager_based/unitracker_teacher"
    )
    assert {path.name for path in task_dir.iterdir() if path.is_dir()} <= {"agents", "mdp", "__pycache__"}


def test_teacher_training_entry_point_has_distributed_rank_and_logging_guards():
    source = (Path(__file__).resolve().parents[2] / "scripts/rsl_rl/teacher/train_teacher.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument("--distributed"' in source
    assert "args_cli.device = f\"cuda:{os.getenv('LOCAL_RANK', '0')}\"" in source
    assert 'env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"' in source
    assert "seed = agent_cfg.seed + app_launcher.local_rank" in source
    assert "is_main_process = not args_cli.distributed or app_launcher.global_rank == 0" in source
    assert "log_dir=str(log_dir) if log_dir is not None else None" in source

    launcher = (Path(__file__).resolve().parents[2] / "scripts/rsl_rl/teacher/train_teacher.sh").read_text(
        encoding="utf-8"
    )
    assert 'if [[ "${argument}" == "--distributed" ]]; then' in launcher
    assert "torch.distributed.run" in launcher
    assert "--standalone" in launcher
    assert 'nproc_per_node="${NUM_GPUS}"' in launcher


def test_teacher_adaptive_sampling_and_runner_sync_hook_are_wired():
    root = Path(__file__).resolve().parents[2]
    command_source = (
        root / "source/unitracker_lab/unitracker_lab/tasks/manager_based/unitracker_teacher/mdp/commands.py"
    ).read_text(encoding="utf-8")
    runner_source = (root / "rsl_rl/rsl_rl/runners/on_policy_runner.py").read_text(encoding="utf-8")
    assert 'sampling_mode: str = "adaptive"' in command_source
    assert "AdaptiveTrackingSampler" in command_source
    assert "_record_adaptive_outcomes" in command_source
    assert "record_tracking_errors" in command_source
    assert 'self.metrics["body_local_position_error"] / 0.30' in command_source
    assert 'self.metrics["body_local_orientation_error"] / 0.40' in command_source
    assert "apply_motion_cache_swap_if_pending_barrier" in command_source
    assert "_call_env_motion_cache_barrier(self.env)" in runner_source
