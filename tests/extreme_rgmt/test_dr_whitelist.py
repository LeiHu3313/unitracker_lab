from pathlib import Path


def test_main_task_contains_only_asset_property_randomization():
    root = Path(__file__).resolve().parents[2]
    env_source = (
        root
        / "source/unitracker_lab/unitracker_lab/tasks/manager_based/extreme_rgmt/extreme_rgmt_env_cfg.py"
    ).read_text(encoding="utf-8")
    asset_source = (root / "source/unitracker_lab/unitracker_lab/assets/g1/g1.py").read_text(encoding="utf-8")
    for forbidden in (
        "push_by_setting_velocity",
        "apply_external_force",
        "randomize_actuator_gains",
        "torque_noise",
        "rough_terrain",
    ):
        assert forbidden not in env_source
    assert "min_delay" not in asset_source
    assert "max_delay" not in asset_source
