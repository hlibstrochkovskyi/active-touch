"""Hidden radial geometry and exact kinematic contact events."""

from dataclasses import dataclass
from math import cos, hypot, isfinite, pi, sin

import numpy as np

from .types import ContactObservation, MotionConfig, ProbeAction, finite

TAU = 2 * pi


def signed_arc(start: float, end: float) -> float:
    """Shortest signed rotation; antipodal ties go clockwise."""
    return (end - start + pi) % TAU - pi


@dataclass(frozen=True)
class Shape:
    kind: str = "recess"
    base: float = 0.55
    a: float = 0.65
    b: float = 0.4
    depth: float = 0.18
    width: float = 0.08
    phase: float = 0.0

    def __post_init__(self):
        if self.kind not in {"circle", "ellipse", "rectangle", "fourier", "recess"}:
            raise ValueError(f"unknown shape: {self.kind}")
        for name in ("base", "a", "b", "width"):
            finite(getattr(self, name), name, strict=True)
        finite(self.depth, "depth")
        if not isfinite(self.phase):
            raise ValueError("phase must be finite")
        low, high = self.bounds
        if low < 0.25 or high > 0.8:
            raise ValueError("shape must lie within the public radius bounds [0.25, 0.8]")

    @property
    def bounds(self) -> tuple[float, float]:
        # Conservative analytic bounds; do not certify a shape using a sampled grid.
        if self.kind == "ellipse":
            return min(self.a, self.b), max(self.a, self.b)
        if self.kind == "rectangle":
            return min(self.a, self.b), hypot(self.a, self.b)
        if self.kind == "fourier":
            return self.base - 0.12, self.base + 0.12
        if self.kind == "recess":
            return self.base - self.depth, self.base
        return self.base, self.base

    def radius(self, theta):
        t = np.asarray(theta, dtype=float) - self.phase
        if self.kind == "ellipse":
            return self.a * self.b / np.hypot(self.b * np.cos(t), self.a * np.sin(t))
        if self.kind == "rectangle":
            c, s = np.abs(np.cos(t)), np.abs(np.sin(t))
            x = np.divide(self.a, c, out=np.full_like(t, np.inf), where=c > 1e-15)
            y = np.divide(self.b, s, out=np.full_like(t, np.inf), where=s > 1e-15)
            return np.minimum(x, y)
        if self.kind == "fourier":
            return self.base + 0.08 * np.cos(2 * t) + 0.04 * np.sin(3 * t)
        if self.kind == "recess":
            return self.base - self.depth * np.exp((np.cos(t) - 1) / self.width**2)
        return np.full_like(t, self.base)


@dataclass(frozen=True)
class PrivateEvent:
    """Ground truth for evaluation/replay only, never handed to a policy."""

    start_angle: float
    theta: float
    true_radius: float
    motion: MotionConfig

    @property
    def segment_durations(self) -> tuple[float, float, float, float]:
        travel = self.motion.outer_radius - self.true_radius
        return (
            self.motion.outer_radius
            * abs(signed_arc(self.start_angle, self.theta))
            / self.motion.transit_speed,
            travel / self.motion.inward_speed,
            self.motion.dwell,
            travel / self.motion.outward_speed,
        )

    @property
    def duration(self) -> float:
        return sum(self.segment_durations)

    @property
    def distance(self) -> float:
        return self.motion.outer_radius * abs(signed_arc(self.start_angle, self.theta)) + 2 * (
            self.motion.outer_radius - self.true_radius
        )

    def position(self, elapsed: float) -> tuple[float, float]:
        t = float(np.clip(elapsed, 0, self.duration))
        arc, inward, dwell, _ = self.segment_durations
        angle, radius = self.theta, self.motion.outer_radius
        if t < arc:
            angle = self.start_angle + signed_arc(self.start_angle, self.theta) * t / arc
        elif t < arc + inward:
            radius -= self.motion.inward_speed * (t - arc)
        elif t < arc + inward + dwell:
            radius = self.true_radius
        else:
            radius = min(
                self.motion.outer_radius,
                self.true_radius + self.motion.outward_speed * (t - arc - inward - dwell),
            )
        return radius * cos(angle), radius * sin(angle)


class ContactWorld:
    def __init__(
        self,
        shape: Shape,
        motion: MotionConfig,
        *,
        noise_std: float,
        seed: int,
        initial_angle: float = 0,
    ):
        finite(noise_std, "noise_std")
        if not isinstance(seed, int) or seed < 0 or not isfinite(initial_angle):
            raise ValueError("invalid world seed or initial angle")
        low, high = shape.bounds
        if low < motion.min_radius or high > motion.max_radius:
            raise ValueError("shape does not satisfy the configured workspace bounds")
        self._shape, self._motion = shape, motion
        self._noise_std, self._seed = noise_std, seed
        self._angle = initial_angle
        self._visits: dict[int, int] = {}

    def execute(self, action: ProbeAction) -> tuple[ContactObservation, PrivateEvent]:
        radius = float(self._shape.radius(action.theta))
        if not self._motion.min_radius <= radius <= self._motion.max_radius:
            raise ValueError("invalid contact: shape violates workspace bounds")
        repeat = self._visits.get(action.candidate_index, 0)
        rng = np.random.default_rng(
            np.random.SeedSequence([self._seed, action.candidate_index, repeat])
        )
        observation = ContactObservation(
            action.theta, radius + rng.normal(0, self._noise_std), self._noise_std**2
        )
        event = PrivateEvent(self._angle, action.theta, radius, self._motion)
        self._visits[action.candidate_index] = repeat + 1
        self._angle = action.theta
        return observation, event
