from contracts import (
    ACTION_DIM,
    ASSET_DR_RANGES,
    FUTURE_REFERENCE_FRAMES,
    G1_ALL_JOINT_NAMES,
    G1_CONTROLLED_JOINT_NAMES,
    G1_LOCAL_FIVE_POINT_BODY_NAMES,
    G1_LOCKED_WRIST_JOINT_NAMES,
    G1_NON_ROOT_TRACKING_BODY_NAMES,
    G1_TRACKING_BODY_NAMES,
    ORACLE_OBSERVATION_DIM,
    REWARD_CURRICULUM,
    TERMINATION_SPECS,
    TRACKING_REWARD_SPECS,
    contract_dict,
)


def test_g1_hard_contract_counts_and_uniqueness():
    assert ACTION_DIM == 23
    assert len(G1_CONTROLLED_JOINT_NAMES) == 23
    assert len(G1_LOCKED_WRIST_JOINT_NAMES) == 6
    assert len(G1_ALL_JOINT_NAMES) == len(set(G1_ALL_JOINT_NAMES)) == 29
    assert len(G1_TRACKING_BODY_NAMES) == len(set(G1_TRACKING_BODY_NAMES)) == 16
    assert G1_TRACKING_BODY_NAMES[1:] == G1_NON_ROOT_TRACKING_BODY_NAMES
    assert len(G1_NON_ROOT_TRACKING_BODY_NAMES) == 15
    assert G1_LOCAL_FIVE_POINT_BODY_NAMES == (
        "torso_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_rubber_hand",
        "right_rubber_hand",
    )
    assert ORACLE_OBSERVATION_DIM == 789
    assert FUTURE_REFERENCE_FRAMES == 5
    assert contract_dict()["future_reference_frames"] == 5


def test_reward_keeps_torso_as_a_soft_global_anchor():
    assert TRACKING_REWARD_SPECS["torso_position"] == {"weight": 2.0, "sigma": 0.30}
    assert TRACKING_REWARD_SPECS["torso_orientation"] == {"weight": 2.0, "sigma": 0.40}
    assert TRACKING_REWARD_SPECS["torso_linear_velocity"] == {"weight": 1.0, "sigma": 1.00}
    assert TRACKING_REWARD_SPECS["torso_angular_velocity"] == {"weight": 2.0, "sigma": 2.50}
    assert "base_position" not in TRACKING_REWARD_SPECS


def test_reward_curriculum_is_disabled_for_the_active_teacher_task():
    assert REWARD_CURRICULUM is None


def test_relative_tracking_reward_matches_the_selected_teacher_baseline():
    assert TRACKING_REWARD_SPECS["body_position"] == {"weight": 2.0, "sigma": 0.30}
    assert TRACKING_REWARD_SPECS["body_orientation"] == {"weight": 1.0, "sigma": 0.40}
    assert TRACKING_REWARD_SPECS["body_linear_velocity"] == {"weight": 1.0, "sigma": 1.00}
    assert TRACKING_REWARD_SPECS["body_angular_velocity"] == {"weight": 1.0, "sigma": 2.50}
    assert TRACKING_REWARD_SPECS["joint_position"] == {"weight": 0.5, "sigma": 0.25}
    assert TRACKING_REWARD_SPECS["joint_velocity"] == {"weight": 0.5, "sigma": 2.50}


def test_reference_relative_termination_thresholds_are_configured():
    assert TERMINATION_SPECS == {
        "projected_gravity": {"threshold": 0.8},
        "pelvis_position": {"threshold": 0.4},
    }


def test_startup_asset_randomization_stays_in_the_small_overfit_range():
    assert ASSET_DR_RANGES == {
        "static_friction": (0.8, 1.2),
        "dynamic_friction": (0.8, 1.2),
        "restitution": (0.0, 0.15),
        "torso_pelvis_com_x": (-0.01, 0.01),
        "torso_pelvis_com_yz": (-0.01, 0.01),
        "link_mass_scale": (0.95, 1.05),
    }
