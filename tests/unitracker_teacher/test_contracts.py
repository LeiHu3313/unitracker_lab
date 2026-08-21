from contracts import (
    ACTION_DIM,
    G1_ALL_JOINT_NAMES,
    G1_CONTROLLED_JOINT_NAMES,
    G1_LOCKED_WRIST_JOINT_NAMES,
    G1_LOCAL_FIVE_POINT_BODY_NAMES,
    G1_NON_ROOT_TRACKING_BODY_NAMES,
    G1_TRACKING_BODY_NAMES,
    G1_TERMINATION_KEY_BODY_NAMES,
    ORACLE_OBSERVATION_DIM,
    REWARD_CURRICULUM,
    TERMINATION_SPECS,
    TRACKING_REWARD_SPECS,
)


def test_g1_hard_contract_counts_and_uniqueness():
    assert ACTION_DIM == 23
    assert len(G1_CONTROLLED_JOINT_NAMES) == 23
    assert len(G1_LOCKED_WRIST_JOINT_NAMES) == 6
    assert len(G1_ALL_JOINT_NAMES) == len(set(G1_ALL_JOINT_NAMES)) == 29
    assert len(G1_TRACKING_BODY_NAMES) == len(set(G1_TRACKING_BODY_NAMES)) == 16
    assert G1_NON_ROOT_TRACKING_BODY_NAMES == G1_TRACKING_BODY_NAMES[1:]
    assert len(G1_NON_ROOT_TRACKING_BODY_NAMES) == 15
    assert G1_LOCAL_FIVE_POINT_BODY_NAMES == (
        "torso_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_rubber_hand",
        "right_rubber_hand",
    )
    assert G1_TERMINATION_KEY_BODY_NAMES == (
        "pelvis",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_rubber_hand",
        "right_rubber_hand",
    )
    assert ORACLE_OBSERVATION_DIM == 588


def test_local_five_point_reward_matches_paper_settings():
    assert TRACKING_REWARD_SPECS["local_five_point_position"] == {"weight": 2.0, "sigma": 0.12}


def test_base_reward_anchor_is_global_position_and_orientation():
    assert TRACKING_REWARD_SPECS["base_position"] == {"weight": 1.0, "sigma": 0.30}
    assert TRACKING_REWARD_SPECS["base_orientation"] == {"weight": 1.0, "sigma": 0.40}
    assert not any(name.startswith("root_") for name in TRACKING_REWARD_SPECS)
    assert "torso_position" not in TRACKING_REWARD_SPECS


def test_reward_curriculum_is_disabled_for_the_active_teacher_task():
    assert REWARD_CURRICULUM is None


def test_torso_orientation_reward_matches_beyondmimic_anchor_settings():
    assert TRACKING_REWARD_SPECS["torso_orientation"] == {"weight": 1.0, "sigma": 0.40}


def test_reference_relative_termination_thresholds_are_configured():
    assert TERMINATION_SPECS == {
        "projected_gravity": {"threshold": 0.8},
        "key_body_height": {"threshold": 0.4},
        "pelvis_position": {"threshold": 0.4},
    }
