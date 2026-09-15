"""Validated experiment settings, loaded from a small strict TOML file."""

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .model import ModelConfig
from .policies import POLICIES
from .types import MotionConfig, finite
from .world import Shape


@dataclass(frozen=True)
class ExperimentConfig:
    policy: str = "cost"
    touches: int = 32
    initial: int = 8
    candidates: int = 128
    integration_points: int = 128
    evaluation_points: int = 1024
    noise_std: float = 0.005
    seed: int = 7

    def __post_init__(self):
        for name in ("touches", "initial", "candidates", "integration_points", "evaluation_points"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            self.initial > self.touches
            or self.touches % self.initial
            or self.candidates % self.touches
        ):
            raise ValueError("initial must divide touches, and touches must divide candidates")
        if self.policy not in POLICIES:
            raise ValueError(f"unknown policy: {self.policy}")
        finite(self.noise_std, "noise_std")


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    shape: Shape = field(default_factory=Shape)
    model: ModelConfig = field(default_factory=ModelConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)

    def __post_init__(self):
        low, high = self.shape.bounds
        if low < self.motion.min_radius or high > self.motion.max_radius:
            raise ValueError("shape exceeds configured workspace bounds")


def config_from_dict(data: dict) -> Config:
    classes = {
        "experiment": ExperimentConfig,
        "shape": Shape,
        "model": ModelConfig,
        "motion": MotionConfig,
    }
    unknown = data.keys() - classes.keys()
    if unknown:
        raise ValueError(f"unknown config sections: {', '.join(sorted(unknown))}")
    resolved = {}
    for section, cls in classes.items():
        values = data.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a table")
        unknown = values.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown {section} settings: {', '.join(sorted(unknown))}")
        try:
            resolved[section] = cls(**values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {section}: {exc}") from exc
    return Config(**resolved)


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as handle:
        return config_from_dict(tomllib.load(handle))
