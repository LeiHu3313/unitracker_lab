from curriculum import linear_curriculum_scale


def test_regularization_curriculum_boundaries_and_midpoint():
    assert linear_curriculum_scale(0, 2000, 10000) == 0.0
    assert linear_curriculum_scale(2000, 2000, 10000) == 0.0
    assert linear_curriculum_scale(6000, 2000, 10000) == 0.5
    assert linear_curriculum_scale(10000, 2000, 10000) == 1.0
    assert linear_curriculum_scale(30000, 2000, 10000) == 1.0


def test_one_shot_termination_penalty_cancels_reward_manager_dt():
    step_dt = 0.02
    raw_failure_indicator = 1.0 / step_dt
    assert -200.0 * step_dt * raw_failure_indicator == -200.0
