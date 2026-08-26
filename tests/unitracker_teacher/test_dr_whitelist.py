from pathlib import Path


def test_main_task_contains_only_approved_randomization():
    root = Path(__file__).resolve().parents[2]
    env_source = (
        root
        / "source/unitracker_lab/unitracker_lab/tasks/manager_based/unitracker_teacher/unitracker_teacher_env_cfg.py"
    ).read_text(encoding="utf-8")
    asset_source = (root / "source/unitracker_lab/unitracker_lab/assets/g1/g1.py").read_text(encoding="utf-8")
    for forbidden in (
        "apply_external_force",
        "randomize_actuator_gains",
        "torque_noise",
        "rough_terrain",
    ):
        assert forbidden not in env_source
    assert "push_by_setting_velocity" in env_source
    assert 'interval_range_s=PUSH_EVENT_SPECS["interval_range_s"]' in env_source
    assert "self.events.external_push = None" in env_source
    assert "min_delay" not in asset_source
    assert "max_delay" not in asset_source
