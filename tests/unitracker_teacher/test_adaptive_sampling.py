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


def test_initial_distribution_gives_each_clip_equal_probability_mass():
    sampler = AdaptiveTrackingSampler([0, 31], [31, 17], fps=50.0, device="cpu")
    probabilities = sampler.sampling_probabilities()

    torch.testing.assert_close(probabilities[:3].sum(), torch.tensor(0.5))
    torch.testing.assert_close(probabilities[3:].sum(), torch.tensor(0.5))
    torch.testing.assert_close(probabilities[:3], torch.full((3,), 1.0 / 6.0))
    torch.testing.assert_close(probabilities[3:], torch.full((2,), 0.25))


def test_full_body_error_and_failure_raise_the_corresponding_bin_probability():
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
    torch.testing.assert_close(sampler.failure_scores, torch.tensor([0.0, 1.0, 0.0]))
    # Difficulty [0, 1.5, 1.0], mixed with 25% uniform exploration.
    torch.testing.assert_close(
        sampler.sampling_probabilities(), torch.tensor([1.0 / 12.0, 8.0 / 15.0, 23.0 / 60.0])
    )


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
    ):
        try:
            AdaptiveTrackingSampler([0], [50], **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid adaptive sampler configuration to fail: {kwargs}")
