import torch
from adaptive_sampling import AdaptiveTrackingSampler


def test_quarter_second_bins_preserve_clips_and_exclude_terminal_frames():
    sampler = AdaptiveTrackingSampler([0, 31], [31, 17], fps=50.0, device="cpu")

    assert sampler.bin_frames == 12
    assert sampler.bin_starts.tolist() == [0, 12, 24, 31, 43]
    assert sampler.bin_ends.tolist() == [12, 24, 30, 43, 47]
    assert sampler.bin_clip_ids.tolist() == [0, 0, 0, 1, 1]

    torch.manual_seed(7)
    bin_ids, clip_ids, phases = sampler.sample(512)
    assert torch.equal(clip_ids, sampler.bin_clip_ids[bin_ids])
    assert torch.all(phases >= sampler.bin_starts[bin_ids])
    assert torch.all(phases < sampler.bin_ends[bin_ids])
    assert not torch.any(phases == 30)
    assert not torch.any(phases == 47)


def test_initial_distribution_is_clip_balanced_and_phase_uniform():
    sampler = AdaptiveTrackingSampler([0, 31], [31, 17], fps=50.0, device="cpu")
    probabilities = sampler.sampling_probabilities()

    torch.testing.assert_close(probabilities[:3].sum(), torch.tensor(0.5))
    torch.testing.assert_close(probabilities[3:].sum(), torch.tensor(0.5))
    # Valid phase counts are [12, 12, 6] and [12, 4], respectively.
    torch.testing.assert_close(probabilities, torch.tensor([0.2, 0.2, 0.1, 0.375, 0.125]))


def test_full_body_error_and_failure_rate_raise_the_corresponding_bin_probability():
    sampler = AdaptiveTrackingSampler(
        [0],
        [31],
        fps=50.0,
        uniform_ratio=0.25,
        ema_alpha=1.0,
        tracking_error_weight=0.25,
        device="cpu",
    )
    sampler.record_tracking_errors(
        torch.tensor([0, 0, 0]), torch.tensor([1, 13, 25]), torch.tensor([0.0, 2.0, 4.0])
    )
    sampler.record_failures(
        torch.tensor([0, 0, 0]), torch.tensor([1, 13, 25]), torch.tensor([False, True, False])
    )

    assert sampler.apply_pending_feedback()
    torch.testing.assert_close(sampler.tracking_errors, torch.tensor([0.0, 2.0, 4.0]))
    # The failed bin has one normal visit plus its terminal failure step.
    torch.testing.assert_close(sampler.failure_rates, torch.tensor([0.0, 0.5, 0.0]))
    probabilities = sampler.sampling_probabilities()
    # Bins 1 and 2 have equal difficulty. Their mass therefore follows their
    # lengths, while both are prioritized above the easy first bin.
    torch.testing.assert_close(probabilities[1], 2.0 * probabilities[2])
    assert probabilities[1] > probabilities[0]


def test_tracking_error_is_averaged_per_visit_and_clipped():
    sampler = AdaptiveTrackingSampler(
        [0], [20], fps=50.0, ema_alpha=1.0, tracking_error_clip=5.0, device="cpu"
    )
    sampler.record_tracking_errors(
        torch.tensor([0, 0, 0]), torch.tensor([1, 2, 3]), torch.tensor([2.0, 4.0, float("inf")])
    )

    assert sampler.apply_pending_feedback()
    torch.testing.assert_close(sampler.tracking_errors[0], torch.tensor(11.0 / 3.0))
    assert sampler.pending_visit_counts.sum() == 0.0


def test_failure_rate_and_probabilities_are_invariant_to_repeated_samples():
    single = AdaptiveTrackingSampler([0], [31], fps=50.0, uniform_ratio=0.5, ema_alpha=1.0, device="cpu")
    repeated = AdaptiveTrackingSampler([0], [31], fps=50.0, uniform_ratio=0.5, ema_alpha=1.0, device="cpu")

    single.record_tracking_errors(torch.tensor([0]), torch.tensor([13]), torch.tensor([2.0]))
    single.record_failures(torch.tensor([0]), torch.tensor([13]), torch.tensor([True]))

    repeated.record_tracking_errors(
        torch.zeros(10, dtype=torch.long), torch.full((10,), 13), torch.full((10,), 2.0)
    )
    repeated.record_failures(
        torch.zeros(10, dtype=torch.long), torch.full((10,), 13), torch.ones(10, dtype=torch.bool)
    )

    assert single.apply_pending_feedback()
    assert repeated.apply_pending_feedback()
    torch.testing.assert_close(single.tracking_errors, repeated.tracking_errors)
    torch.testing.assert_close(single.failure_rates, repeated.failure_rates)
    torch.testing.assert_close(single.sampling_probabilities(), repeated.sampling_probabilities())


def test_unvisited_failure_rate_does_not_decay():
    sampler = AdaptiveTrackingSampler([0], [31], fps=50.0, ema_alpha=1.0, device="cpu")
    sampler.record_failures(torch.tensor([0]), torch.tensor([13]), torch.tensor([True]))
    assert sampler.apply_pending_feedback()
    before = sampler.failure_rates.clone()

    assert sampler.apply_pending_feedback()
    torch.testing.assert_close(sampler.failure_rates, before)


def test_phase_mapping_clamps_the_terminal_frame_to_its_clips_last_bin():
    sampler = AdaptiveTrackingSampler([0, 31], [31, 17], fps=50.0, device="cpu")
    bin_ids = sampler.phase_to_bin(torch.tensor([0, 1]), torch.tensor([30, 47]))
    assert bin_ids.tolist() == [2, 4]


def test_invalid_sampler_configuration_is_rejected():
    for kwargs in (
        {"fps": 0.0},
        {"fps": 50.0, "uniform_ratio": 1.1},
        {"fps": 50.0, "bin_duration_s": 0.0},
        {"fps": 50.0, "ema_alpha": 0.0},
        {"fps": 50.0, "tracking_error_weight": -1.0},
        {"fps": 50.0, "tracking_error_weight": float("nan")},
        {"fps": 50.0, "tracking_error_clip": float("inf")},
        {"fps": 50.0, "priority_epsilon": 0.0},
    ):
        try:
            AdaptiveTrackingSampler([0], [50], **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid adaptive sampler configuration to fail: {kwargs}")
