"""Reproducible episode orchestration with separate public and private logs."""

import csv
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .config import Config
from .metrics import reconstruction_metrics
from .model import RadialGP
from .policies import Decision, Policy, estimated_times
from .types import ContactObservation, ProbeAction, PublicState
from .world import TAU, ContactWorld, PrivateEvent


@dataclass
class Episode:
    config: Config
    observations: list[ContactObservation]
    events: list[PrivateEvent]
    decisions: list[Decision]
    metrics: list[dict]
    angles: np.ndarray
    truth: np.ndarray
    means: np.ndarray
    stds: np.ndarray


def run_episode(config: Config) -> Episode:
    # Small dense matrices lose time to oversized BLAS pools on many-core hosts.
    with threadpool_limits(limits=1, user_api="blas"):
        return _run_episode(config)


def _run_episode(config: Config) -> Episode:
    settings = config.experiment
    noise_seed, phase_seed, policy_seed = [
        int(np.random.SeedSequence([settings.seed, stream]).generate_state(1)[0])
        for stream in range(3)
    ]
    phase = np.random.default_rng(phase_seed).uniform(0, TAU)
    candidates = (phase + TAU * np.arange(settings.candidates) / settings.candidates) % TAU
    world = ContactWorld(
        config.shape,
        config.motion,
        noise_std=settings.noise_std,
        seed=noise_seed,
        initial_angle=float(candidates[0]),
    )
    model = RadialGP(config.model)
    policy = Policy(
        settings.policy,
        candidates,
        config.motion,
        noise_variance=settings.noise_std**2,
        budget=settings.touches,
        integration_points=settings.integration_points,
        seed=policy_seed,
    )
    angles = TAU * (np.arange(settings.evaluation_points) + 0.37) / settings.evaluation_points
    truth = config.shape.radius(angles)
    observations, events, decisions, visited = [], [], [], []
    means, stds, metrics = [], [], []
    current_angle = float(candidates[0])
    total_time, total_distance = 0.0, 0.0

    def record(fit_seconds=0.0, selection_seconds=0.0, innovation=None):
        mean, std = model.predict(angles)
        means.append(mean)
        stds.append(std)
        metrics.append(
            {
                "touches": len(observations),
                "motion_time": total_time,
                "distance": total_distance,
                **reconstruction_metrics(mean, std, truth),
                "fit_seconds": fit_seconds,
                "selection_seconds": selection_seconds,
                "innovation": innovation,
                "jitter": model.last_jitter,
            }
        )

    record()
    for step in range(settings.touches):
        state = PublicState(current_angle, tuple(observations), tuple(visited))
        start = perf_counter()
        if step < settings.initial:
            index = (-step * (settings.candidates // settings.initial)) % settings.candidates
            theta = float(candidates[index])
            mean, _ = model.predict([theta])
            cost = estimated_times([theta], mean, current_angle, config.motion)[0]
            decision = Decision(ProbeAction(index, theta), 0.0, float(cost), "initial")
        else:
            decision = policy.choose(state, model)
        selection_seconds = perf_counter() - start
        prior_mean, prior_std = model.predict([decision.action.theta])
        observation, event = world.execute(decision.action)
        innovation = float(
            (observation.measured_radius - prior_mean[0])
            / np.sqrt(prior_std[0] ** 2 + observation.noise_variance + model.last_jitter)
        )
        observations.append(observation)
        events.append(event)
        decisions.append(decision)
        visited.append(decision.action.candidate_index)
        current_angle = observation.theta
        start = perf_counter()
        model.fit(
            tuple(observations),
            optimize=(
                config.model.learn_hyperparameters
                and len(observations) >= 16
                and len(observations) % 8 == 0
            ),
        )
        fit_seconds = perf_counter() - start
        # Exact motion data are used only below, after the agent decision/update.
        total_time += event.duration
        total_distance += event.distance
        record(fit_seconds, selection_seconds, innovation)
    return Episode(
        config,
        observations,
        events,
        decisions,
        metrics,
        angles,
        truth,
        np.asarray(means),
        np.asarray(stds),
    )


def provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "revision": revision,
        "dirty": dirty,
        "blas_threads": 1,
        "packages": {
            name: version(name)
            for name in ("touch-explorer", "numpy", "scipy", "scikit-learn", "matplotlib")
        },
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty metrics table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_episode(result: Episode, folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "config.json", asdict(result.config))
    write_json(folder / "metadata.json", {**provenance(), "status": "budget_exhausted"})
    for filename, values in (
        ("observations.jsonl", result.observations),
        ("truth.jsonl", result.events),
        ("decisions.jsonl", result.decisions),
    ):
        with (folder / filename).open("w") as handle:
            for item in values:
                handle.write(json.dumps(asdict(item), allow_nan=False) + "\n")
    write_csv(folder / "metrics.csv", result.metrics)
    np.savez_compressed(
        folder / "snapshots.npz",
        angles=result.angles,
        truth=result.truth,
        means=result.means,
        stds=result.stds,
    )
