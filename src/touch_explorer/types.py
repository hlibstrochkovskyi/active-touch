"""The public sensor contract. Exact geometry and timing do not belong here."""

from dataclasses import dataclass
from math import isfinite


def finite(value: float, name: str, *, minimum: float = 0, strict: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isfinite(value)
        or value < minimum
        or (strict and value == minimum)
    ):
        raise ValueError(f"{name} must be finite and {'>' if strict else '>='} {minimum}")


@dataclass(frozen=True)
class ProbeAction:
    candidate_index: int
    theta: float

    def __post_init__(self):
        if isinstance(self.candidate_index, bool) or not isinstance(self.candidate_index, int):
            raise ValueError("candidate_index must be an integer")
        if self.candidate_index < 0 or not isfinite(self.theta):
            raise ValueError("invalid probe action")


@dataclass(frozen=True)
class ContactObservation:
    theta: float
    measured_radius: float
    noise_variance: float

    def __post_init__(self):
        if not isfinite(self.theta) or not isfinite(self.measured_radius):
            raise ValueError("contact coordinates must be finite")
        finite(self.noise_variance, "noise_variance")


@dataclass(frozen=True)
class PublicState:
    current_outer_angle: float
    observations: tuple[ContactObservation, ...]
    visited_indices: tuple[int, ...]


@dataclass(frozen=True)
class MotionConfig:
    outer_radius: float = 1.0
    min_radius: float = 0.25
    max_radius: float = 0.8
    transit_speed: float = 1.0
    inward_speed: float = 0.5
    outward_speed: float = 1.0
    dwell: float = 0.2

    def __post_init__(self):
        for name in (
            "outer_radius",
            "min_radius",
            "max_radius",
            "transit_speed",
            "inward_speed",
            "outward_speed",
        ):
            finite(getattr(self, name), name, strict=True)
        finite(self.dwell, "dwell")
        if not self.min_radius < self.max_radius < self.outer_radius:
            raise ValueError("require min_radius < max_radius < outer_radius")
