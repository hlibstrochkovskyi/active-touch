"""Frozen protocols and paired, shape-level analysis for the primary benchmark."""

import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib.figure import Figure

from .config import config_from_dict
from .policies import POLICIES
from .runner import provenance, run_episode, save_episode, write_json
from .types import finite
from .world import SensorConfig

PRIMARY_METRICS = ("final_rmse", "time_averaged_rmse")
SECONDARY_METRICS = (
    "final_max_error",
    "coverage_95",
    "interval_width",
    "motion_time",
    "distance",
    "repeat_fraction",
    "compute_p95_seconds",
)


def validate_protocol(protocol: dict) -> None:
    required = {
        "config",
        "cases",
        "noise_levels",
        "noise_seeds",
        "policies",
        "checkpoints",
        "time_horizon",
        "bootstrap_seed",
        "bootstrap_samples",
    }
    if set(protocol) != required:
        raise ValueError("protocol has missing or unknown fields")
    config = config_from_dict(protocol["config"])
    if (
        config.model.learn_hyperparameters
        or config.experiment.safeguard_every
        or config.stopping.mode != "budget"
        or config.sensor != SensorConfig()
    ):
        raise ValueError("primary study requires a fixed kernel, full budget, and nominal sensor")
    finite(protocol["time_horizon"], "time_horizon", strict=True)
    for name in ("noise_levels", "noise_seeds", "policies", "checkpoints"):
        values = protocol[name]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be a nonempty unique list")
    for value in protocol["noise_levels"]:
        finite(value, "noise level")
    for value in [*protocol["noise_seeds"], protocol["bootstrap_seed"]]:
        if type(value) is not int or value < 0:
            raise ValueError("seeds must be nonnegative integers")
    if type(protocol["bootstrap_samples"]) is not int or protocol["bootstrap_samples"] < 1000:
        raise ValueError("use at least 1000 bootstrap samples")
    if (
        set(protocol["policies"]) - set(POLICIES)
        or "gap" not in protocol["policies"]
        or len(protocol["policies"]) < 2
    ):
        raise ValueError("policies must include gap and another supported policy")
    if any(
        type(n) is not int or not 1 <= n <= config.experiment.touches
        for n in protocol["checkpoints"]
    ):
        raise ValueError("checkpoints must be within the touch budget")
    if not protocol["cases"]:
        raise ValueError("cases must not be empty")
    ids = []
    for case in protocol["cases"]:
        if set(case) != {"id", "shape_seed", "episode_seed", "shape"}:
            raise ValueError("invalid case fields")
        if not isinstance(case["id"], str) or not re.fullmatch(r"[a-z0-9-]+", case["id"]):
            raise ValueError("case id must use lowercase letters, digits and hyphens")
        ids.append(case["id"])
        for key in ("shape_seed", "episode_seed"):
            if type(case[key]) is not int or case[key] < 0:
                raise ValueError("case seeds must be nonnegative integers")
        config_from_dict({**protocol["config"], "shape": case["shape"]})
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case id")


def schedule(protocol: dict) -> list[tuple[dict, object]]:
    validate_protocol(protocol)
    jobs = []
    for case in protocol["cases"]:
        base = config_from_dict({**protocol["config"], "shape": case["shape"]})
        for level, noise_std in enumerate(protocol["noise_levels"]):
            for seed in protocol["noise_seeds"]:
                sensor_seed = int(
                    np.random.SeedSequence([case["episode_seed"], seed, 71]).generate_state(1)[0]
                )
                for policy in protocol["policies"]:
                    row = {
                        "case": case["id"],
                        "family": base.shape.kind,
                        "noise_std": noise_std,
                        "noise_seed": seed,
                        "policy": policy,
                        "run": f"{case['id']}-n{level}-s{seed}-{policy}",
                    }
                    config = replace(
                        base,
                        experiment=replace(
                            base.experiment,
                            seed=case["episode_seed"],
                            noise_seed=sensor_seed,
                            noise_std=noise_std,
                            policy=policy,
                        ),
                    )
                    jobs.append((row, config))
    return jobs


def time_average(times, errors, horizon: float) -> float:
    """Integrate completed-observation estimates, holding the final value after completion."""
    times, errors = np.asarray(times), np.asarray(errors)
    finite(horizon, "horizon", strict=True)
    if (
        times.ndim != 1
        or times.shape != errors.shape
        or not len(times)
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
        or not np.all(np.isfinite(times))
        or not np.all(np.isfinite(errors))
    ):
        raise ValueError("expected finite errors at increasing times starting at zero")
    edges = np.r_[times[times < horizon], horizon]
    return float(np.dot(np.diff(edges), errors[: len(edges) - 1]) / horizon)


def paired_interval(by_family: dict[str, list[float]], samples: int, seed: int) -> dict:
    """Percentile bootstrap of paired shape differences, stratified by family."""
    rng = np.random.default_rng(seed)
    bootstraps, means = [], []
    for family in sorted(by_family):
        values = np.asarray(by_family[family], dtype=float)
        if not len(values) or not np.all(np.isfinite(values)):
            raise ValueError("each family needs finite shape differences")
        means.append(float(values.mean()))
        indices = rng.integers(len(values), size=(samples, len(values)))
        bootstraps.append(values[indices].mean(axis=1))
    if not means:
        raise ValueError("no shape differences")
    return {
        "mean": float(np.mean(means)),
        "ci95": np.quantile(np.mean(bootstraps, axis=0), [0.025, 0.975]).tolist(),
        "shapes": sum(len(v) for v in by_family.values()),
    }


def analyze(protocol: dict, rows: list[dict]) -> dict:
    expected = {r["run"]: r for r, _ in schedule(protocol)}
    seen = {}
    for row in rows:
        if row["run"] in seen:
            raise ValueError("duplicate run in analysis")
        if row["run"] not in expected or any(row[k] != v for k, v in expected[row["run"]].items()):
            raise ValueError("run does not match frozen protocol")
        seen[row["run"]] = row
    failed = sum(row["status"] != "budget_exhausted" for row in rows)
    report = {
        "scheduled": len(expected),
        "recorded": len(rows),
        "failed": failed,
        "missing": len(expected) - len(rows),
        "complete": len(expected) == len(rows) and failed == 0,
        "paired": [],
        "means": [],
        "by_family": [],
        "curves": [],
        "interpretation": "policy minus gap; negative error differences favor policy; "
        "noise repeats averaged within shapes; equal family weights; "
        "95% percentile intervals resample shapes within families; "
        "marginal intervals, no multiple-comparison correction",
    }
    if not report["complete"]:
        return report  # Do not silently discard a failed or missing matched run.
    cases = {c["id"]: c["shape"]["kind"] for c in protocol["cases"]}
    families = sorted(set(cases.values()))
    metrics = [m for m in (*PRIMARY_METRICS, *SECONDARY_METRICS) if all(m in row for row in rows)]

    def shape_means(selected, policy, key):
        return {
            case: np.mean(
                [r[key] for r in selected if r["case"] == case and r["policy"] == policy], axis=0
            )
            for case in cases
        }

    def family_mean(values):
        return np.mean(
            [
                np.mean([v for c, v in values.items() if cases[c] == family], axis=0)
                for family in families
            ],
            axis=0,
        )

    for noise in [None, *protocol["noise_levels"]]:
        selected = [r for r in rows if noise is None or r["noise_std"] == noise]
        for metric in metrics:
            baseline = shape_means(selected, "gap", metric)
            for policy in protocol["policies"]:
                values = shape_means(selected, policy, metric)
                report["means"].append(
                    {
                        "noise_std": noise,
                        "policy": policy,
                        "metric": metric,
                        "mean": float(family_mean(values)),
                    }
                )
                if noise is not None:
                    for family in families:
                        report["by_family"].append(
                            {
                                "noise_std": noise,
                                "family": family,
                                "policy": policy,
                                "metric": metric,
                                "mean": float(
                                    np.mean([v for c, v in values.items() if cases[c] == family])
                                ),
                            }
                        )
                if policy == "gap" or metric not in PRIMARY_METRICS:
                    continue
                differences = {
                    family: [float(values[c] - baseline[c]) for c in cases if cases[c] == family]
                    for family in families
                }
                report["paired"].append(
                    {
                        "noise_std": noise,
                        "policy": policy,
                        "metric": metric,
                        **paired_interval(
                            differences, protocol["bootstrap_samples"], protocol["bootstrap_seed"]
                        ),
                    }
                )
    for policy in protocol["policies"]:
        curves = {"policy": policy}
        for key in ("touch_curve", "time_curve"):
            if all(key in r for r in rows):
                curves[key] = family_mean(shape_means(rows, policy, key)).tolist()
        report["curves"].append(curves)
    return report


def protocol_hash(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def analyze_folder(folder: Path) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text())
    protocol = manifest["protocol"]
    if protocol_hash(protocol) != manifest["protocol_sha256"]:
        raise ValueError("saved protocol checksum mismatch")
    report = analyze(protocol, json.loads((folder / "runs.json").read_text()))
    write_json(folder / "analysis.json", report)
    if report["complete"]:
        figure = Figure(figsize=(11, 4), layout="constrained")
        left, right = figure.subplots(1, 2)
        for curve in report["curves"]:
            left.step(
                np.arange(len(curve["touch_curve"])),
                curve["touch_curve"],
                where="post",
                label=curve["policy"],
            )
            right.step(
                np.linspace(0, protocol["time_horizon"], len(curve["time_curve"])),
                curve["time_curve"],
                where="post",
                label=curve["policy"],
            )
        left.set_xlabel("Completed touches")
        right.set_xlabel("Modeled motion time (s)")
        for axis in (left, right):
            axis.set_ylabel("Mean radial RMSE / R")
            axis.grid(alpha=0.2)
            axis.legend()
        figure.suptitle("Held-out fixed-kernel comparison · equal family and noise-level weights")
        figure.savefig(folder / "comparison.png", dpi=150)
        figure.savefig(folder / "comparison.svg")
    return report


def _execute_job(job):
    row, config, folder, protocol = job
    row = dict(row)
    start = perf_counter()
    try:
        result = run_episode(config)
        save_episode(result, folder / row["run"])
        final = result.metrics[-1]
        times = np.array([m["motion_time"] for m in result.metrics])
        errors = np.array([m["rmse"] for m in result.metrics])
        indices = (
            np.searchsorted(times, np.linspace(0, protocol["time_horizon"], 201), side="right") - 1
        )
        row.update(
            status=result.status,
            final_rmse=final["rmse"],
            final_max_error=final["max_radial_error"],
            coverage_95=final["coverage_95"],
            interval_width=final["mean_interval_width"],
            motion_time=final["motion_time"],
            distance=final["distance"],
            time_averaged_rmse=time_average(times, errors, protocol["time_horizon"]),
            checkpoints={str(n): float(errors[n]) for n in protocol["checkpoints"]},
            touch_curve=errors.tolist(),
            time_curve=errors[indices].tolist(),
            repeat_fraction=1
            - len({d.action.candidate_index for d in result.decisions}) / len(result.decisions),
            compute_p95_seconds=float(
                np.quantile(
                    [m["fit_seconds"] + m["selection_seconds"] for m in result.metrics[1:]], 0.95
                )
            ),
        )
    except Exception as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        failure_folder = folder / row["run"]
        failure_folder.mkdir(exist_ok=True)
        write_json(failure_folder / "failure.json", {"config": asdict(config), **row})
    row["wall_seconds"] = perf_counter() - start
    return row


def run_heldout(protocol: dict, folder: Path, workers: int = 1) -> int:
    jobs = schedule(protocol)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    folder.mkdir(parents=True, exist_ok=False)
    write_json(
        folder / "manifest.json",
        {
            **provenance(),
            "scope": "held-out fixed-kernel benchmark",
            "protocol": protocol,
            "protocol_sha256": protocol_hash(protocol),
            "workers": workers,
        },
    )
    jobs = [(row, config, folder, protocol) for row, config in jobs]
    rows = []

    def collect(results):
        for row in results:
            rows.append(row)
            write_json(folder / "runs.json", rows)
            print(
                f"{len(rows)}/{len(jobs)} {row['run']}: {row['status']} "
                f"({row['wall_seconds']:.2f}s)",
                flush=True,
            )

    if workers == 1:
        collect(map(_execute_job, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            collect(pool.map(_execute_job, jobs))
    report = analyze_folder(folder)
    return int(not report["complete"])
