from contracts import ORACLE_OBSERVATION_BLOCK_DIMS, ORACLE_OBSERVATION_DIM, oracle_observation_slices


def test_oracle_layout_is_contiguous_and_588_dimensional():
    slices = oracle_observation_slices()
    cursor = 0
    for name, width in ORACLE_OBSERVATION_BLOCK_DIMS.items():
        assert slices[name] == (cursor, cursor + width)
        cursor += width
    assert cursor == ORACLE_OBSERVATION_DIM == 588


def test_oracle_state_goal_boundary():
    slices = oracle_observation_slices()
    assert slices["previous_action"] == (281, 304)
    assert slices["next_root_height_error"] == (304, 305)
    assert slices["next_non_root_body_pos_error_local"] == (317, 362)
    assert slices["next_joint_vel_error"] == (565, 588)
