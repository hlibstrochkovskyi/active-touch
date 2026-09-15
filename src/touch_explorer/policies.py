"""Action selectors receive public observations and a model, never a world."""

from dataclasses import dataclass

import numpy as np

from .model import RadialGP, variance_reduction
from .types import MotionConfig, ProbeAction, PublicState, finite

POLICIES = ("random", "sweep", "gap", "variance", "cost")
TAU = 2 * np.pi


def angular_distance(a, b):
    return np.abs(np.arctan2(np.sin(np.asarray(a) - b), np.cos(np.asarray(a) - b)))


def estimated_times(angles, means, current_angle: float, motion: MotionConfig):
    radius = np.clip(means, motion.min_radius, motion.max_radius)
    return (
        motion.outer_radius * angular_distance(angles, current_angle) / motion.transit_speed
        + (motion.outer_radius - radius) * (1 / motion.inward_speed + 1 / motion.outward_speed)
        + motion.dwell
    )


@dataclass(frozen=True)
class Decision:
    action: ProbeAction
    score: float
    estimated_time: float
    reason: str


class Policy:
    def __init__(
        self,
        name: str,
        candidates,
        motion: MotionConfig,
        *,
        noise_variance: float,
        budget: int,
        integration_points: int = 128,
        seed: int = 0,
    ):
        if name not in POLICIES:
            raise ValueError(f"unknown policy: {name}")
        finite(noise_variance, "noise_variance")
        self.candidates = np.asarray(candidates, dtype=float).copy()
        if (
            self.candidates.ndim != 1
            or not len(self.candidates)
            or not np.isfinite(self.candidates).all()
        ):
            raise ValueError("candidates must be a nonempty finite vector")
        if budget < 1 or len(self.candidates) % budget or integration_points < 1:
            raise ValueError(
                "budget must divide candidate count; integration_points must be positive"
            )
        # Index-based sweep and initialization require an evenly spaced angular ring.
        expected = self.candidates[0] + TAU * np.arange(len(self.candidates)) / len(self.candidates)
        if not np.allclose(angular_distance(self.candidates, expected), 0, atol=1e-12):
            raise ValueError("candidates must be an evenly spaced ring in increasing angular order")
        self.name, self.motion = name, motion
        self.noise_variance, self.budget = noise_variance, budget
        self.integration_angles = (
            self.candidates[0] + TAU * (np.arange(integration_points) + 0.5) / integration_points
        )
        self.rng = np.random.default_rng(seed)

    def _unvisited(self, state):
        remaining = np.ones(len(self.candidates), dtype=bool)
        remaining[list(state.visited_indices)] = False
        return np.flatnonzero(remaining)

    def scores(self, state: PublicState, model: RadialGP) -> np.ndarray:
        if self.name == "variance":
            return model.predict(self.candidates)[1] ** 2
        if self.name == "cost":
            reduction = variance_reduction(
                model, self.candidates, self.integration_angles, self.noise_variance
            )
            means, _ = model.predict(self.candidates)
            return reduction / estimated_times(
                self.candidates, means, state.current_outer_angle, self.motion
            )
        remaining = self._unvisited(state)
        if not len(remaining):
            raise RuntimeError("all candidate angles have been visited")
        scores = np.full(len(self.candidates), -np.inf)
        if self.name == "random":
            scores[self.rng.choice(remaining)] = 1.0
        elif self.name == "sweep":
            grid = remaining[remaining % (len(self.candidates) // self.budget) == 0]
            if not len(grid):
                raise RuntimeError("uniform sweep is complete")
            clockwise = (state.current_outer_angle - self.candidates[grid]) % TAU
            scores[grid[np.argmin(clockwise)]] = 1.0
        else:
            angles = np.unique(np.mod([o.theta for o in state.observations], TAU))
            if not len(angles):
                scores[remaining] = 1.0
                return scores
            gaps = np.diff(np.r_[angles, angles[0] + TAU])
            for start, gap in zip(angles, gaps, strict=True):
                if not np.isclose(gap, gaps.max(), rtol=1e-10, atol=1e-12):
                    continue
                distances = angular_distance(self.candidates[remaining], start + gap / 2)
                nearest = remaining[np.isclose(distances, distances.min(), atol=1e-12, rtol=1e-10)]
                scores[nearest] = gap
        return scores

    def choose(self, state: PublicState, model: RadialGP) -> Decision:
        scores = self.scores(state, model)
        means, _ = model.predict(self.candidates)
        costs = estimated_times(self.candidates, means, state.current_outer_angle, self.motion)
        tied = np.flatnonzero(np.isclose(scores, np.max(scores), rtol=1e-10, atol=1e-14))
        low_cost = tied[np.isclose(costs[tied], costs[tied].min(), rtol=1e-10, atol=1e-12)]
        index = int(low_cost[0])
        return Decision(
            ProbeAction(index, float(self.candidates[index])),
            float(scores[index]),
            float(costs[index]),
            self.name,
        )
