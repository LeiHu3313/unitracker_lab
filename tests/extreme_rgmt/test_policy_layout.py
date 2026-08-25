from contracts import POLICY_OBSERVATION_BLOCK_DIMS, POLICY_OBSERVATION_DIM, policy_observation_slices


def test_policy_layout_is_contiguous_and_paper_aligned():
    slices = policy_observation_slices()
    cursor = 0
    for name, width in POLICY_OBSERVATION_BLOCK_DIMS.items():
        assert slices[name] == (cursor, cursor + width)
        cursor += width
    assert cursor == POLICY_OBSERVATION_DIM == 1728


def test_policy_history_and_reference_boundaries():
    slices = policy_observation_slices()
    assert slices["proprioception_history"] == (0, 640)
    assert slices["previous_action_history"] == (640, 930)
    assert slices["reference_window"] == (930, 1728)
