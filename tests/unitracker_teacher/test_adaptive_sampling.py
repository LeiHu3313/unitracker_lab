import torch

from adaptive_sampling import AdaptiveEloSampler


def test_fixed_second_windows_preserve_clip_boundaries_and_rsi_first_half():
    sampler = AdaptiveEloSampler([0, 120], [120, 60], fps=50.0, device="cpu")

    assert sampler.window_frames == 50
    assert sampler.window_starts.tolist() == [0, 50, 100, 120, 170]
    assert sampler.window_ends.tolist() == [50, 100, 120, 170, 180]
    assert sampler.window_clip_ids.tolist() == [0, 0, 0, 1, 1]

    torch.manual_seed(7)
    window_ids, clip_ids, phases = sampler.sample(512)
    assert torch.equal(clip_ids, sampler.window_clip_ids[window_ids])
    starts = sampler.window_starts[window_ids]
    first_half_ends = starts + (sampler.window_lengths[window_ids] + 1) // 2
    assert torch.all(phases >= starts)
    assert torch.all(phases < first_half_ends)
    assert torch.all(phases < sampler.window_ends[window_ids])


def test_failure_raises_window_probability_while_uniform_component_remains():
    sampler = AdaptiveEloSampler([0], [150], fps=50.0, uniform_ratio=0.5, device="cpu")
    before = sampler.sampling_probabilities().clone()

    sampler.record_outcomes(torch.tensor([1, 2]), torch.tensor([1.0, 0.0]))
    assert sampler.apply_pending_feedback()
    after = sampler.sampling_probabilities()

    assert torch.allclose(before, torch.full_like(before, 1.0 / 3.0))
    assert sampler.ratings[1] > sampler.initial_rating
    assert sampler.ratings[2] < sampler.initial_rating
    assert after[1] > before[1]
    assert after[2] < before[2]
    assert torch.all(after > 0.0)
    assert torch.isclose(after.sum(), torch.tensor(1.0))


def test_invalid_sampler_configuration_is_rejected():
    for kwargs in (
        {"fps": 0.0},
        {"fps": 50.0, "uniform_ratio": 1.1},
        {"fps": 50.0, "window_s": 0.0},
    ):
        try:
            AdaptiveEloSampler([0], [50], **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid adaptive sampler configuration to fail: {kwargs}")
