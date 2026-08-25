from pathlib import Path


def test_environment_wires_paper_table_ii_randomization_terms():
    root = Path(__file__).resolve().parents[2]
    source = (
        root
        / "source/unitracker_lab/unitracker_lab/tasks/manager_based/extreme_rgmt/extreme_rgmt_env_cfg.py"
    ).read_text(encoding="utf-8")
    for required in (
        'ASSET_DR_RANGES["ground_friction"]',
        'ASSET_DR_RANGES["added_base_mass_kg"]',
        'ASSET_DR_RANGES["base_com_x_m"]',
        "randomize_motor_strength",
        "randomize_actuator_gains",
        "randomize_motor_zero_offset",
        "randomize_joint_parameters",
        "push_by_setting_velocity",
    ):
        assert required in source


def test_unpublished_push_magnitude_is_marked_as_reproduction_choice():
    root = Path(__file__).resolve().parents[2]
    source = (
        root
        / "source/unitracker_lab/unitracker_lab/tasks/manager_based/extreme_rgmt/extreme_rgmt_env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "paper publishes the interval but not the impulse magnitude" in source
