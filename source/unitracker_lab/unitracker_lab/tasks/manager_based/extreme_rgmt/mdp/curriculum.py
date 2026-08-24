"""Reward curriculum shared by Extreme-RGMT regularization terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def linear_curriculum_scale(iteration: int, start_iter: int, end_iter: int) -> float:
    if end_iter <= start_iter:
        raise ValueError("end_iter must be greater than start_iter.")
    return min(1.0, max(0.0, (float(iteration) - start_iter) / (end_iter - start_iter)))


def regularization_scale(env: ManagerBasedRLEnv, start_iter: int, end_iter: int, num_steps_per_iter: int) -> float:
    if num_steps_per_iter <= 0:
        raise ValueError("num_steps_per_iter must be positive.")
    iteration = int(env.common_step_counter // num_steps_per_iter)
    return linear_curriculum_scale(iteration, start_iter, end_iter)
