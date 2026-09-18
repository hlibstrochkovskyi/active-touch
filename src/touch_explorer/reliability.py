"""Matched development experiments for fitting, coverage, and premature stopping."""

from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib.figure import Figure

from .config import Config
from .runner import Episode, provenance, run_episode, save_episode, write_json
from .world import SensorConfig

VARIANTS = (
    "fixed-budget",
    "learned-budget",
    "learned-coverage",
    "uncertainty-stop",
    "guarded-stop",
)
RMSE_TOLERANCE = 0.01
MAX_ERROR_TOLERANCE = 0.03


def variant_configs(config: Config) -> dict[str, Config]:
    fixed = replace(
        config,
        model=replace(config.model, learn_hyperparameters=False),
        experiment=replace(config.experiment, policy="cost", safeguard_every=0),
        stopping=replace(config.stopping, mode="budget"),
    )
    learned = replace(fixed, model=replace(config.model, learn_hyperparameters=True))
    coverage = replace(learned, experiment=replace(learned.experiment, safeguard_every=5))
    return dict(
        zip(
            VARIANTS,
            (
                fixed,
                learned,
                coverage,
                replace(coverage, stopping=replace(config.stopping, mode="uncertainty")),
                replace(coverage, stopping=replace(config.stopping, mode="guarded")),
            ),
            strict=True,
        )
    )


def mismatch_configs(config: Config) -> dict[str, Config]:
    """Predetermined one-fault-at-a-time conditions, sharing the agent's noise model."""
    methods = variant_configs(config)
    sensors = {
        "nominal": SensorConfig(),
        "bias": SensorConfig(bias=0.015),
        "noise": SensorConfig(noise_scale=3),
        "outliers": SensorConfig(outlier_probability=0.05, outlier_std=0.05),
    }
    return {
        f"{fault}-{method}": replace(base, sensor=sensor)
        for fault, sensor in sensors.items()
        for method, base in (
            ("budget", methods["learned-coverage"]),
            ("guarded", methods["guarded-stop"]),
        )
    }


def episode_outcome(
    result: Episode,
    *,
    rmse_tolerance: float = RMSE_TOLERANCE,
    max_error_tolerance: float = MAX_ERROR_TOLERANCE,
) -> dict:
    final = result.metrics[-1]
    acceptable = (
        final["rmse"] <= rmse_tolerance and final["max_radial_error"] <= max_error_tolerance
    )
    return {
        "status": result.status,
        "touches": len(result.observations),
        "false_stop": result.status == "confident" and not acceptable,
        "quality_met": acceptable,
        "final_rmse": final["rmse"],
        "final_max_error": final["max_radial_error"],
        "coverage_95": final["coverage_95"],
        "interval_width": final["mean_interval_width"],
        "motion_time": final["motion_time"],
        "audit_touches": sum(d.reason == "audit" for d in result.decisions),
        "coverage_touches": sum(d.reason == "coverage" for d in result.decisions),
        "optimizer_fallbacks": sum(d["status"] == "fallback" for d in result.model_diagnostics),
    }


def summarize(folder: Path, rows: list[dict], *, study: str = "reliability") -> dict:
    summaries = []
    for name in dict.fromkeys(r["variant"] for r in rows):
        attempted = [r for r in rows if r["variant"] == name]
        completed = [r for r in attempted if r["status"] != "failed"]
        stops = [r for r in completed if r["status"] == "confident"]
        false_stops = sum(r["false_stop"] for r in stops)

        def mean(key, runs=completed):
            return float(np.mean([r[key] for r in runs])) if runs else None

        summaries.append(
            {
                "variant": name,
                "attempted": len(attempted),
                "completed": len(completed),
                "failed": len(attempted) - len(completed),
                "stops": len(stops),
                "budget_exhausted": sum(r["status"] == "budget_exhausted" for r in completed),
                "false_stops": false_stops,
                "false_stop_rate": false_stops / len(stops) if stops else None,
                "mean_touches": mean("touches"),
                "mean_final_rmse": mean("final_rmse"),
                "mean_final_max_error": mean("final_max_error"),
                "mean_coverage_95": mean("coverage_95"),
                "mean_interval_width": mean("interval_width"),
                "mean_motion_time": mean("motion_time"),
                "quality_met": sum(r["quality_met"] for r in completed),
                "optimizer_fallbacks": sum(r["optimizer_fallbacks"] for r in completed),
            }
        )
    scope = "sensor mismatch" if study == "mismatch" else "reliability"
    report = {
        "scope": f"development {scope} study; aggregates over successful runs",
        "rmse_tolerance": RMSE_TOLERANCE,
        "max_error_tolerance": MAX_ERROR_TOLERANCE,
        "variants": summaries,
    }
    write_json(folder / f"{study}.json", report)
    figure = Figure(figsize=(12, 7), layout="constrained")
    axes = figure.subplots(2, 2)
    x = np.arange(len(summaries))
    labels = [s["variant"].replace("-", "\n", 1) for s in summaries]
    for ax, key, title in (
        (axes[0, 0], "mean_touches", "Mean completed touches"),
        (axes[1, 0], "mean_final_rmse", "Mean final radial RMSE"),
        (axes[1, 1], "mean_coverage_95", "Mean pointwise interval coverage"),
    ):
        ax.bar(x, [s[key] if s[key] is not None else np.nan for s in summaries], color="#2463a7")
        ax.set_title(title)
    ax = axes[0, 1]
    ax.bar(x, [s["false_stop_rate"] or 0 for s in summaries], color="#c65c22")
    for i, s in enumerate(summaries):
        label = f"{s['false_stops']}/{s['stops']}" if s["stops"] else "no stops"
        ax.text(i, (s["false_stop_rate"] or 0) + 0.035, label, ha="center", fontsize=9)
    ax.set(title="False stops / confidence stops", ylim=(0, 1.15))
    axes[1, 1].axhline(0.95, ls="--", color="gray")
    axes[1, 1].set_ylim(0, 1.05)
    for ax in axes.flat:
        ax.set_xticks(x, labels, fontsize=8)
        ax.grid(axis="y", alpha=0.15)
    figure.suptitle(f"Development {scope} study · different sensing budgets")
    figure.savefig(folder / f"{study}.png", dpi=150)
    figure.savefig(folder / f"{study}.svg")
    return report


def run_study(
    config: Config,
    folder: Path,
    seeds: int,
    shapes: list[str],
    shape_factory,
    shape_kinds: tuple[str, ...],
    *,
    study: str = "reliability",
) -> int:
    if seeds < 1 or not shapes or len(set(shapes)) != len(shapes) or set(shapes) - set(shape_kinds):
        raise ValueError("use positive --seeds and supported unique shapes")
    if study not in {"reliability", "mismatch"}:
        raise ValueError("unknown study")
    variants = mismatch_configs(config) if study == "mismatch" else variant_configs(config)
    scope = "sensor mismatch" if study == "mismatch" else "reliability"
    folder.mkdir(parents=True, exist_ok=False)
    write_json(
        folder / "manifest.json",
        {
            **provenance(),
            "scope": f"development {scope} study",
            "shapes": shapes,
            "seeds_per_shape": seeds,
            "variants": {k: asdict(v) for k, v in variants.items()},
            "rmse_tolerance": RMSE_TOLERANCE,
            "max_error_tolerance": MAX_ERROR_TOLERANCE,
        },
    )
    rows = []
    for kind in shapes:
        for repeat in range(seeds):
            seed = config.experiment.seed + shape_kinds.index(kind) * 1_000_003 + repeat
            shape = shape_factory(kind, seed)
            case = f"{kind}-{seed}"
            for name, base in variants.items():
                run_name = f"{case}-{name}"
                run_config = replace(
                    base, shape=shape, experiment=replace(base.experiment, seed=seed)
                )
                row = {"case": case, "shape": kind, "seed": seed, "run": run_name, "variant": name}
                start = perf_counter()
                try:
                    result = run_episode(run_config)
                    save_episode(result, folder / run_name)
                    row.update(episode_outcome(result))
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    failure_folder = folder / run_name
                    failure_folder.mkdir(exist_ok=True)
                    write_json(
                        failure_folder / "failure.json", {"config": asdict(run_config), **row}
                    )
                row["wall_seconds"] = perf_counter() - start
                rows.append(row)
                write_json(folder / "runs.json", rows)
                print(f"{run_name}: {row['status']} ({row['wall_seconds']:.2f}s)", flush=True)
    summarize(folder, rows, study=study)
    failures = sum(r["status"] == "failed" for r in rows)
    print(f"Saved {len(rows)} {study} runs; {failures} failed: {folder.resolve()}")
    return int(failures > 0)
