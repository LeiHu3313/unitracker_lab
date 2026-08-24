from __future__ import annotations

import torch
from rsl_rl.pace_star import (
    difficulty_conditioned_advantages,
    pace_consolidation_weight,
    rollout_fragment_ids,
    select_star_pool,
)


def test_pace_weight_stays_at_base_below_reference_and_caps_at_one():
    ema, weight = pace_consolidation_weight(0.2, 0.6, beta=0.5)
    assert ema == 0.4
    assert weight == 0.3

    ema, weight = pace_consolidation_weight(1.0, 0.8, beta=0.0)
    assert ema == 1.0
    assert weight == 1.0


def test_difficulty_conditioned_advantages_normalize_groups_separately():
    raw = torch.tensor([[[1.0], [10.0]], [[3.0], [14.0]], [[99.0], [99.0]]])
    acquisition = torch.tensor([[[True], [True]], [[True], [True]], [[False], [False]]])
    difficulty = torch.tensor([[[2.0], [1.0]], [[2.0], [0.5]], [[1.0], [1.0]]])

    normalized, high = difficulty_conditioned_advantages(raw, acquisition, difficulty)

    assert torch.allclose(normalized[:2, 0, 0], torch.tensor([-1.0, 1.0]))
    assert torch.allclose(normalized[:2, 1, 0], torch.tensor([-1.0, 1.0]))
    assert torch.equal(normalized[2], torch.zeros_like(normalized[2]))
    assert high.squeeze(-1).tolist() == [[True, False], [True, False], [False, False]]


def test_fragment_ids_are_split_by_done_and_never_cross_environments():
    dones = torch.tensor(
        [
            [[False], [False]],
            [[True], [False]],
            [[False], [False]],
            [[False], [True]],
        ]
    )
    fragments = rollout_fragment_ids(dones).squeeze(-1)
    assert fragments[:, 0].tolist() == [0, 0, 1, 1]
    assert fragments[:, 1].tolist() == [4, 4, 4, 4]


def test_star_selects_top_fragment_per_bin_and_returns_all_fragment_transitions():
    raw = torch.tensor([[[1.0], [8.0]], [[1.0], [8.0]], [[9.0], [2.0]], [[9.0], [2.0]]])
    acquisition = torch.ones_like(raw, dtype=torch.bool)
    difficulty = torch.full_like(raw, 2.0)
    bins = torch.tensor([[[0], [1]], [[0], [1]], [[0], [1]], [[0], [1]]])
    dones = torch.tensor(
        [
            [[True], [False]],
            [[False], [False]],
            [[False], [False]],
            [[False], [False]],
        ]
    )

    pool, weights = select_star_pool(raw, acquisition, difficulty, bins, dones, topk_fraction=0.05)

    # Bin 0 chooses env0's second fragment (time 1..3); bin 1 chooses env1's
    # sole fragment (all four transitions). Flattening is time-major.
    assert set(pool.tolist()) == {1, 2, 3, 4, 5, 6, 7}
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert torch.all(weights > 0.0)


def test_star_empty_high_difficulty_group_is_a_noop():
    shape = (2, 2, 1)
    pool, weights = select_star_pool(
        torch.ones(shape),
        torch.ones(shape, dtype=torch.bool),
        torch.ones(shape),
        torch.zeros(shape, dtype=torch.long),
        torch.zeros(shape, dtype=torch.bool),
    )
    assert pool.numel() == weights.numel() == 0
