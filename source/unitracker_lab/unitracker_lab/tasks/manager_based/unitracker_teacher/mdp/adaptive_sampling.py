"""Simulator-independent adaptive sampling for multi-clip teacher motions."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch


class AdaptiveEloSampler:
    """Sample fixed-duration motion windows, prioritizing windows that fail."""

    def __init__(
        self,
        clip_starts: Sequence[int] | torch.Tensor,
        clip_lengths: Sequence[int] | torch.Tensor,
        *,
        fps: float,
        window_s: float = 1.0,
        uniform_ratio: float = 0.5,
        initial_rating: float = 100.0,
        rating_k: float = 32.0,
        sampling_temperature: float = 0.3,
        device: torch.device | str = "cpu",
    ):
        if fps <= 0.0 or window_s <= 0.0:
            raise ValueError("Adaptive sampling requires positive fps and window_s.")
        if not 0.0 <= uniform_ratio <= 1.0:
            raise ValueError("adaptive uniform_ratio must be in [0, 1].")
        if initial_rating <= 0.0 or rating_k <= 0.0 or sampling_temperature <= 0.0:
            raise ValueError("Adaptive ELO settings must be positive.")

        self.device = torch.device(device)
        self.initial_rating = float(initial_rating)
        self.rating_k = float(rating_k)
        self.sampling_temperature = float(sampling_temperature)
        self.uniform_ratio = float(uniform_ratio)
        self.window_frames = int(math.ceil(window_s * fps))

        starts = torch.as_tensor(clip_starts, dtype=torch.long, device=self.device).reshape(-1)
        lengths = torch.as_tensor(clip_lengths, dtype=torch.long, device=self.device).reshape(-1)
        if starts.numel() == 0 or starts.numel() != lengths.numel() or bool((lengths < 2).any()):
            raise ValueError("Every adaptive-sampling clip must contain at least two frames.")

        window_starts: list[int] = []
        window_ends: list[int] = []
        window_clip_ids: list[int] = []
        for clip_id, (clip_start, clip_length) in enumerate(zip(starts.tolist(), lengths.tolist(), strict=True)):
            clip_end = clip_start + clip_length
            for start in range(clip_start, clip_end, self.window_frames):
                window_starts.append(start)
                window_ends.append(min(start + self.window_frames, clip_end))
                window_clip_ids.append(clip_id)

        self.window_starts = torch.tensor(window_starts, dtype=torch.long, device=self.device)
        self.window_ends = torch.tensor(window_ends, dtype=torch.long, device=self.device)
        self.window_clip_ids = torch.tensor(window_clip_ids, dtype=torch.long, device=self.device)
        self.window_lengths = self.window_ends - self.window_starts
        self.ratings = torch.full((len(window_starts),), self.initial_rating, dtype=torch.float32, device=self.device)
        self.pending_failures = torch.zeros_like(self.ratings)
        self.pending_successes = torch.zeros_like(self.ratings)
        self.sync_count = 0

    @property
    def num_windows(self) -> int:
        return int(self.window_starts.numel())

    def _failure_probability(self, ratings: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.pow(10.0, (self.initial_rating - ratings) / 400.0))

    def adaptive_probabilities(self) -> torch.Tensor:
        failure_probability = self._failure_probability(self.ratings)
        weights = torch.exp(failure_probability / self.sampling_temperature)
        return weights / weights.sum()

    def sampling_probabilities(self) -> torch.Tensor:
        adaptive = self.adaptive_probabilities()
        uniform = torch.full_like(adaptive, 1.0 / self.num_windows)
        return (1.0 - self.uniform_ratio) * adaptive + self.uniform_ratio * uniform

    def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(window_ids, clip_ids, phase_indices)`` for RSI."""

        if count <= 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty, empty
        window_ids = torch.multinomial(self.sampling_probabilities(), count, replacement=True)
        starts = self.window_starts[window_ids]
        # Mirror UniTracker's RSI rule: choose from a window's first half.
        first_half_lengths = torch.clamp((self.window_lengths[window_ids] + 1) // 2, min=1)
        offsets = torch.floor(torch.rand(count, device=self.device) * first_half_lengths).to(dtype=torch.long)
        return window_ids, self.window_clip_ids[window_ids], starts + offsets

    def record_outcomes(self, window_ids: torch.Tensor, failures: torch.Tensor) -> None:
        window_ids = window_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        failures = failures.to(device=self.device, dtype=torch.float32).reshape(-1)
        if window_ids.numel() != failures.numel():
            raise ValueError("Adaptive ELO outcomes and window IDs must have equal length.")
        if window_ids.numel() == 0:
            return
        if bool(((window_ids < 0) | (window_ids >= self.num_windows)).any()):
            raise ValueError("Adaptive ELO received an invalid window ID.")
        failures = torch.clamp(failures, min=0.0, max=1.0)
        self.pending_failures.scatter_add_(0, window_ids, failures)
        self.pending_successes.scatter_add_(0, window_ids, 1.0 - failures)

    def _apply_feedback(self, failures: torch.Tensor, successes: torch.Tensor, scale: float) -> None:
        total = failures + successes
        prior = self._failure_probability(self.ratings)
        self.ratings.add_(self.rating_k * scale * (failures - total * prior))
        self.ratings.clamp_(self.initial_rating / 2.0, self.initial_rating * 2.0)

    def apply_pending_feedback(self) -> bool:
        """Update locally, or aggregate and broadcast ratings under torchrun."""

        if self._distributed_requested():
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                return False
            feedback = torch.stack((self.pending_failures, self.pending_successes))
            torch.distributed.all_reduce(feedback, op=torch.distributed.ReduceOp.SUM)
            world_size = torch.distributed.get_world_size()
            if torch.distributed.get_rank() == 0:
                self._apply_feedback(feedback[0], feedback[1], scale=1.0 / world_size)
            torch.distributed.broadcast(self.ratings, src=0)
        else:
            self._apply_feedback(self.pending_failures, self.pending_successes, scale=1.0)

        self.pending_failures.zero_()
        self.pending_successes.zero_()
        self.sync_count += 1
        return True

    @staticmethod
    def _distributed_requested() -> bool:
        return int(os.getenv("WORLD_SIZE", "1")) > 1
