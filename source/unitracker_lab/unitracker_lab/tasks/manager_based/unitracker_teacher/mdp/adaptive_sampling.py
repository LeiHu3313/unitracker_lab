"""Simulator-independent full-body-error sampling for teacher motions."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch


class AdaptiveTrackingSampler:
    """Sample short motion bins using failure and continuous tracking error."""

    def __init__(
        self,
        clip_starts: Sequence[int] | torch.Tensor,
        clip_lengths: Sequence[int] | torch.Tensor,
        *,
        fps: float,
        bin_duration_s: float = 0.25,
        uniform_ratio: float = 0.25,
        ema_alpha: float = 0.01,
        tracking_error_weight: float = 0.25,
        tracking_error_clip: float = 5.0,
        device: torch.device | str = "cpu",
    ) -> None:
        if not math.isfinite(fps) or not math.isfinite(bin_duration_s) or fps <= 0.0 or bin_duration_s <= 0.0:
            raise ValueError("Adaptive sampling requires positive fps and bin_duration_s.")
        if not math.isfinite(uniform_ratio) or not 0.0 <= uniform_ratio <= 1.0:
            raise ValueError("adaptive uniform_ratio must be in [0, 1].")
        if not math.isfinite(ema_alpha) or not 0.0 < ema_alpha <= 1.0:
            raise ValueError("adaptive ema_alpha must be in (0, 1].")
        if (
            not math.isfinite(tracking_error_weight)
            or not math.isfinite(tracking_error_clip)
            or tracking_error_weight < 0.0
            or tracking_error_clip <= 0.0
        ):
            raise ValueError("Adaptive tracking-error settings must be non-negative and finite.")

        self.device = torch.device(device)
        self.uniform_ratio = float(uniform_ratio)
        self.ema_alpha = float(ema_alpha)
        self.tracking_error_weight = float(tracking_error_weight)
        self.tracking_error_clip = float(tracking_error_clip)
        self.bin_frames = max(1, int(round(bin_duration_s * fps)))

        starts = torch.as_tensor(clip_starts, dtype=torch.long, device=self.device).reshape(-1)
        lengths = torch.as_tensor(clip_lengths, dtype=torch.long, device=self.device).reshape(-1)
        if starts.numel() == 0 or starts.numel() != lengths.numel() or bool((lengths < 2).any()):
            raise ValueError("Every adaptive-sampling clip must contain at least two frames.")
        self.clip_starts = starts

        bin_starts: list[int] = []
        bin_ends: list[int] = []
        bin_clip_ids: list[int] = []
        clip_bin_offsets = [0]
        for clip_id, (clip_start, clip_length) in enumerate(zip(starts.tolist(), lengths.tolist(), strict=True)):
            # The last clip frame is not a valid reset phase because the teacher
            # tracks k -> k+1.
            valid_end = clip_start + clip_length - 1
            for start in range(clip_start, valid_end, self.bin_frames):
                bin_starts.append(start)
                bin_ends.append(min(start + self.bin_frames, valid_end))
                bin_clip_ids.append(clip_id)
            clip_bin_offsets.append(len(bin_starts))

        self.bin_starts = torch.tensor(bin_starts, dtype=torch.long, device=self.device)
        self.bin_ends = torch.tensor(bin_ends, dtype=torch.long, device=self.device)
        self.bin_lengths = self.bin_ends - self.bin_starts
        self.bin_clip_ids = torch.tensor(bin_clip_ids, dtype=torch.long, device=self.device)
        self.clip_bin_offsets = torch.tensor(clip_bin_offsets, dtype=torch.long, device=self.device)

        self.failure_scores = torch.zeros(self.num_bins, dtype=torch.float32, device=self.device)
        self.tracking_errors = torch.zeros_like(self.failure_scores)
        self.pending_failure_counts = torch.zeros_like(self.failure_scores)
        self.pending_tracking_error_sums = torch.zeros_like(self.failure_scores)
        self.pending_visit_counts = torch.zeros_like(self.failure_scores)
        self.sync_count = 0

    @property
    def num_bins(self) -> int:
        return int(self.bin_starts.numel())

    def phase_to_bin(self, clip_ids: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """Map global reference phases to bins without crossing clip boundaries."""

        clip_ids = clip_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        phases = phases.to(device=self.device, dtype=torch.long).reshape(-1)
        if clip_ids.numel() != phases.numel():
            raise ValueError("clip_ids and phases must have equal length.")
        local_phases = torch.clamp(phases - self.clip_starts[clip_ids], min=0)
        bin_ids = self.clip_bin_offsets[clip_ids] + local_phases // self.bin_frames
        return torch.minimum(bin_ids, self.clip_bin_offsets[clip_ids + 1] - 1)

    def sampling_probabilities(self) -> torch.Tensor:
        """Return clip-balanced probabilities with a uniform exploration floor."""

        difficulty = (self.failure_scores + self.tracking_error_weight * self.tracking_errors).clamp_min(0.0)
        bins_per_clip = self.clip_bin_offsets[1:] - self.clip_bin_offsets[:-1]
        uniform_within_clip = 1.0 / bins_per_clip[self.bin_clip_ids]
        difficulty_per_clip = torch.zeros(
            len(bins_per_clip), dtype=difficulty.dtype, device=self.device
        ).scatter_add_(0, self.bin_clip_ids, difficulty)
        focused = torch.where(
            difficulty_per_clip[self.bin_clip_ids] > 0.0,
            difficulty / difficulty_per_clip[self.bin_clip_ids].clamp_min(1.0e-12),
            uniform_within_clip,
        )
        within_clip = self.uniform_ratio * uniform_within_clip + (1.0 - self.uniform_ratio) * focused
        return within_clip / len(bins_per_clip)

    def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(bin_ids, clip_ids, phase_indices)`` for RSI."""

        if count <= 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty, empty
        bin_ids = torch.multinomial(self.sampling_probabilities(), count, replacement=True)
        offsets = torch.floor(torch.rand(count, device=self.device) * self.bin_lengths[bin_ids]).to(torch.long)
        return bin_ids, self.bin_clip_ids[bin_ids], self.bin_starts[bin_ids] + offsets

    def record_tracking_errors(
        self, clip_ids: torch.Tensor, phases: torch.Tensor, tracking_errors: torch.Tensor
    ) -> None:
        """Accumulate one normalized full-body error per visited phase."""

        tracking_errors = tracking_errors.to(device=self.device, dtype=torch.float32).reshape(-1)
        bin_ids = self.phase_to_bin(clip_ids, phases)
        if bin_ids.numel() != tracking_errors.numel():
            raise ValueError("Adaptive tracking errors and phases must have equal length.")
        tracking_errors = torch.nan_to_num(
            tracking_errors, nan=self.tracking_error_clip, posinf=self.tracking_error_clip, neginf=0.0
        ).clamp_(0.0, self.tracking_error_clip)
        self.pending_tracking_error_sums.scatter_add_(0, bin_ids, tracking_errors)
        self.pending_visit_counts.scatter_add_(0, bin_ids, torch.ones_like(tracking_errors))

    def record_failures(self, clip_ids: torch.Tensor, phases: torch.Tensor, failures: torch.Tensor) -> None:
        """Assign actual terminations to the phase at which they occurred."""

        failures = failures.to(device=self.device, dtype=torch.bool).reshape(-1)
        clip_ids = clip_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        phases = phases.to(device=self.device, dtype=torch.long).reshape(-1)
        if clip_ids.numel() != phases.numel() or phases.numel() != failures.numel():
            raise ValueError("Adaptive failure inputs must have equal length.")
        failed = failures.nonzero().flatten()
        if failed.numel() == 0:
            return
        bin_ids = self.phase_to_bin(clip_ids[failed], phases[failed])
        self.pending_failure_counts.scatter_add_(0, bin_ids, torch.ones_like(bin_ids, dtype=torch.float32))

    def apply_pending_feedback(self) -> bool:
        """All-reduce rollout statistics once, then update per-bin EMA scores."""

        feedback = torch.stack(
            (self.pending_tracking_error_sums, self.pending_visit_counts, self.pending_failure_counts)
        )
        world_size = 1
        if self._distributed_requested():
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                return False
            torch.distributed.all_reduce(feedback, op=torch.distributed.ReduceOp.SUM)
            world_size = torch.distributed.get_world_size()

        error_sums, visit_counts, failure_counts = feedback
        visited = visit_counts > 0.0
        observed_error = error_sums / visit_counts.clamp_min(1.0)
        alpha = self.ema_alpha
        self.tracking_errors[visited] = (1.0 - alpha) * self.tracking_errors[visited] + alpha * observed_error[
            visited
        ]
        # Keep the failure term independent of DDP world size. Its relative
        # distribution still identifies the phases that actually terminate.
        self.failure_scores.mul_(1.0 - alpha).add_(failure_counts / world_size, alpha=alpha)

        self.pending_tracking_error_sums.zero_()
        self.pending_visit_counts.zero_()
        self.pending_failure_counts.zero_()
        self.sync_count += 1
        return True

    @staticmethod
    def _distributed_requested() -> bool:
        return int(os.getenv("WORLD_SIZE", "1")) > 1
