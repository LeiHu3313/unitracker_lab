from contracts import POLICY_OBSERVATION_BLOCK_DIMS, POLICY_OBSERVATION_DIM, policy_observation_slices


def test_policy_layout_is_contiguous_and_789_dimensional():
    slices = policy_observation_slices()
    cursor = 0
    for name, width in POLICY_OBSERVATION_BLOCK_DIMS.items():
        assert slices[name] == (cursor, cursor + width)
        cursor += width
    assert cursor == POLICY_OBSERVATION_DIM == 789


def test_policy_state_goal_boundary():
    slices = policy_observation_slices()
    assert slices["current_all_joint_pos_rel_default"] == (235, 264)
    assert slices["current_foot_contact_mask"] == (293, 295)
    assert slices["previous_action"] == (295, 318)
    assert slices["next_root_height_error"] == (318, 319)
    assert slices["next_torso_pos_error_local"] == (331, 334)
    assert slices["next_non_root_body_pos_error_local"] == (334, 379)
    assert slices["future_controlled_joint_pos_vel_command"] == (559, 789)
