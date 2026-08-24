from pathlib import Path


def test_only_extreme_rgmt_task_is_registered():
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "source/unitracker_lab/unitracker_lab/tasks/manager_based/extreme_rgmt/__init__.py"
    ).read_text(encoding="utf-8")
    assert 'id="Extreme-RGMT-Base-v0"' in source
    assert 'id="Extreme-RGMT-Expansion-v0"' in source
    assert "Unitracker_Teacher" not in source


def test_training_entry_point_exposes_two_explicit_stages():
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/rsl_rl/extreme_rgmt/train.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--stage", required=True' in source
    assert 'parser.add_argument("--motion"' in source
    assert 'parser.add_argument("--mastered-motion"' in source
    assert 'parser.add_argument("--challenging-motion"' in source
    assert 'parser.add_argument("--base-checkpoint"' in source
    assert 'args_cli.task = "Extreme-RGMT-Base-v0"' in source
    assert 'else "Extreme-RGMT-Expansion-v0"' in source
    assert "Stage I requires --motion" in source
    assert "Stage II requires:" in source
    assert "load_optimizer=False" in source
    assert "restore_iteration=False" in source
    assert "extreme_rgmt_contract.json" in source


def test_adaptive_sampling_and_runner_metadata_hooks_are_wired():
    root = Path(__file__).resolve().parents[2]
    command_source = (
        root / "source/unitracker_lab/unitracker_lab/tasks/manager_based/extreme_rgmt/mdp/commands.py"
    ).read_text(encoding="utf-8")
    runner_source = (root / "rsl_rl/rsl_rl/runners/on_policy_runner.py").read_text(encoding="utf-8")
    assert "AdaptiveEloSampler" in command_source
    assert "get_training_transition_metadata" in command_source
    assert "_get_env_training_transition_metadata(self.env)" in runner_source
