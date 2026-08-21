from contracts import REGULARIZATION_REWARD_WEIGHTS
from curriculum import linear_curriculum_scale


def test_optional_regularization_curriculum_helper_boundaries_and_midpoint():
    assert linear_curriculum_scale(0, 2000, 10000) == 0.0
    assert linear_curriculum_scale(2000, 2000, 10000) == 0.0
    assert linear_curriculum_scale(6000, 2000, 10000) == 0.5
    assert linear_curriculum_scale(10000, 2000, 10000) == 1.0
    assert linear_curriculum_scale(30000, 2000, 10000) == 1.0


def test_one_shot_termination_penalty_cancels_reward_manager_dt():
    step_dt = 0.02
    raw_failure_indicator = 1.0 / step_dt
    early_termination_weight = REGULARIZATION_REWARD_WEIGHTS["early_termination"]
    assert early_termination_weight == -50.0
    assert early_termination_weight * step_dt * raw_failure_indicator == -50.0
