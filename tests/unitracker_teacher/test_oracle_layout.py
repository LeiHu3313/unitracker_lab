from contracts import ORACLE_OBSERVATION_BLOCK_DIMS, ORACLE_OBSERVATION_DIM, oracle_observation_slices


def test_oracle_layout_is_contiguous_and_716_dimensional():
    slices = oracle_observation_slices()
    cursor = 0
    for name, width in ORACLE_OBSERVATION_BLOCK_DIMS.items():
        assert slices[name] == (cursor, cursor + width)
        cursor += width
    assert cursor == ORACLE_OBSERVATION_DIM == 716


def test_oracle_state_goal_boundary():
    slices = oracle_observation_slices()
    assert slices["previous_action"] == (286, 309)
    assert slices["next_body_pos_error_local"] == (309, 357)
