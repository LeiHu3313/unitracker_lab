import numpy as np
from contracts import reference_promotion_mask


def test_reset_inside_step_does_not_promote_reference():
    just_reset = np.asarray([True, True, False, False])
    episode_length = np.asarray([0, 1, 0, 3])
    promote = reference_promotion_mask(just_reset, episode_length)
    np.testing.assert_array_equal(promote, [False, True, True, True])


def test_normal_step_promotes_k_to_k_plus_one():
    phase = np.asarray([3, 8])
    target = phase + 1
    promote = reference_promotion_mask(np.asarray([False, False]), np.asarray([7, 11]))
    phase[promote] = target[promote]
    target[promote] = phase[promote] + 1
    np.testing.assert_array_equal(phase, [4, 9])
    np.testing.assert_array_equal(target, [5, 10])
