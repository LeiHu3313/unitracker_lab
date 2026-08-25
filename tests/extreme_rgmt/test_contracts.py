from contracts import (
    ACTION_DIM,
    ASSET_DR_RANGES,
    CONTROL_DECIMATION,
    CONTROL_FREQUENCY_HZ,
    CRITIC_OBSERVATION_DIM,
    G1_ALL_JOINT_NAMES,
    G1_BODY_JOINT_NAMES,
    G1_CONTROLLED_JOINT_NAMES,
    G1_NON_ROOT_TRACKING_BODY_NAMES,
    G1_TRACKING_BODY_NAMES,
    G1_WRIST_JOINT_NAMES,
    PHYSICS_DT,
    POLICY_OBSERVATION_DIM,
    PROPRIOCEPTION_DIM,
    REFERENCE_TOKEN_DIM,
    REFERENCE_WINDOW_LENGTH,
    TERMINATION_SPECS,
    TRACKING_REWARD_SPECS,
    contract_dict,
)


def test_g1_hard_contract_counts_and_uniqueness():
    assert ACTION_DIM == 29
    assert len(G1_CONTROLLED_JOINT_NAMES) == 29
    assert len(G1_BODY_JOINT_NAMES) == 23
    assert len(G1_WRIST_JOINT_NAMES) == 6
    assert len(G1_ALL_JOINT_NAMES) == len(set(G1_ALL_JOINT_NAMES)) == 29
    assert len(G1_TRACKING_BODY_NAMES) == len(set(G1_TRACKING_BODY_NAMES)) == 16
    assert G1_TRACKING_BODY_NAMES[1:] == G1_NON_ROOT_TRACKING_BODY_NAMES
    assert len(G1_NON_ROOT_TRACKING_BODY_NAMES) == 15
    assert PROPRIOCEPTION_DIM == 64
    assert REFERENCE_TOKEN_DIM == 38
    assert REFERENCE_WINDOW_LENGTH == 21
    assert POLICY_OBSERVATION_DIM == 1728
    assert CRITIC_OBSERVATION_DIM == 1876
    assert contract_dict()["history_length"] == 10
    assert contract_dict()["reference_window_length"] == 21
    assert PHYSICS_DT == 0.002
    assert CONTROL_DECIMATION == 10
    assert CONTROL_FREQUENCY_HZ == 50.0


def test_reward_matches_paper_table_i():
    assert TRACKING_REWARD_SPECS["anchor_orientation"] == {"weight": 0.5, "sigma": 0.40}


def test_relative_tracking_reward_matches_the_selected_extreme_rgmt_baseline():
    assert TRACKING_REWARD_SPECS["body_position"] == {"weight": 1.0, "sigma": 0.30}
    assert TRACKING_REWARD_SPECS["body_orientation"] == {"weight": 1.0, "sigma": 0.40}
    assert TRACKING_REWARD_SPECS["body_linear_velocity"] == {"weight": 1.0, "sigma": 1.00}
    assert TRACKING_REWARD_SPECS["body_angular_velocity"] == {"weight": 1.0, "sigma": 2.50}


def test_reference_relative_termination_thresholds_are_configured():
    assert TERMINATION_SPECS == {
        "projected_gravity": {"threshold": 0.8},
        "pelvis_position": {"threshold": 0.4},
    }


def test_startup_asset_randomization_stays_in_the_small_overfit_range():
    assert ASSET_DR_RANGES == {
        "ground_friction": (0.1, 1.75),
        "added_base_mass_kg": (-3.0, 6.0),
        "base_com_x_m": (-0.025, 0.025),
        "base_com_yz_m": (-0.05, 0.05),
        "motor_strength_scale": (0.8, 1.2),
        "pd_gain_scale": (0.8, 1.2),
        "motor_zero_offset_rad": (-0.01, 0.01),
        "joint_armature_scale": (1.0, 1.05),
        "external_push_interval_s": (1.0, 3.0),
    }
