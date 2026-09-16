"""Observation-only stopping state; ground-truth errors never enter this module."""

from dataclasses import dataclass
from math import isfinite, radians

import numpy as np

from .types import finite


@dataclass(frozen=True)
class StoppingConfig:
    mode: str = "budget"
    half_width: float = 0.02
    min_touches: int = 24
    max_gap_degrees: float = 15.0
    audit_touches: int = 2
    innovation_limit: float = 2.5

    def __post_init__(self):
        if self.mode not in ("budget", "uncertainty", "guarded"):
            raise ValueError("stopping mode must be budget, uncertainty, or guarded")
        for name in ("min_touches", "audit_touches"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("half_width", "max_gap_degrees", "innovation_limit"):
            finite(getattr(self, name), name, strict=True)
        if self.max_gap_degrees > 360:
            raise ValueError("max_gap_degrees must be <= 360")


def largest_gap(angles) -> float:
    angles = np.unique(np.asarray(angles, dtype=float) % (2 * np.pi))
    if not len(angles):
        return 2 * np.pi
    return float(np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi])))


class StopController:
    def __init__(self, config: StoppingConfig):
        self.config = config
        self.audit_pending = False
        self.audits_passed = 0

    def observe(
        self,
        *,
        touches: int,
        max_half_width: float,
        max_gap: float,
        innovation: float,
        was_audit: bool,
    ) -> bool:
        if self.config.mode == "budget":
            return False
        narrow = isfinite(max_half_width) and max_half_width <= self.config.half_width
        if self.config.mode == "uncertainty":
            return narrow
        eligible = (
            narrow
            and touches >= self.config.min_touches
            and max_gap <= radians(self.config.max_gap_degrees) + 1e-12
        )
        if not eligible or (
            was_audit
            and (not isfinite(innovation) or abs(innovation) > self.config.innovation_limit)
        ):
            self.audits_passed = 0
            self.audit_pending = False
            return False
        if was_audit and self.audit_pending:
            self.audits_passed += 1
        done = self.audits_passed >= self.config.audit_touches
        self.audit_pending = not done
        return done
