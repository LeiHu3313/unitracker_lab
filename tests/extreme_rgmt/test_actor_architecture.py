from __future__ import annotations

import torch
from rsl_rl.modules import ExtremeRGMTActor, ExtremeRGMTActorCritic, FiniteScalarQuantizer
from tensordict import TensorDict


def test_paper_actor_splits_histories_and_reference_window():
    actor = ExtremeRGMTActor(num_actions=29, actor_hidden_dims=(64, 32), history_num_layers=1)
    observation = torch.arange(2 * 1728, dtype=torch.float32).reshape(2, 1728)

    state, action, reference = actor.split_observation(observation)

    assert state.shape == (2, 10, 64)
    assert action.shape == (2, 10, 29)
    assert reference.shape == (2, 21, 38)
    assert state[0, 0, 0] == 0
    assert action[0, 0, 0] == 640
    assert reference[0, 0, 0] == 930


def test_fsq_uses_discrete_grid_with_straight_through_gradients():
    values = torch.tensor([-3.0, -0.2, 0.3, 2.0], requires_grad=True)
    quantized = FiniteScalarQuantizer(8)(values)
    grid_indices = 0.5 * (quantized.detach() + 1.0) * 7

    assert torch.allclose(grid_indices, grid_indices.round())
    quantized.sum().backward()
    assert values.grad is not None
    assert torch.all(values.grad > 0)


def test_actor_critic_consumes_1728_policy_and_1876_critic_dimensions():
    observations = TensorDict(
        {"policy": torch.randn(2, 1728), "critic": torch.randn(2, 1876)}, batch_size=[2]
    )
    module = ExtremeRGMTActorCritic(
        observations,
        {"policy": ["policy"], "critic": ["critic"]},
        29,
        actor_hidden_dims=(64, 32),
        critic_hidden_dims=(64, 32),
        history_num_layers=1,
    )

    actions = module.act_inference(observations)
    values = module.evaluate(observations)

    assert actions.shape == (2, 29)
    assert values.shape == (2, 1)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(values).all()
