"""Static reports and a self-contained browser replay from saved snapshots."""

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .runner import write_json


def read_metrics(folder: Path) -> list[dict[str, float]]:
    with (folder / "metrics.csv").open() as handle:
        return [
            {key: float(value) if value else float("nan") for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def render_run(folder: str | Path) -> None:
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    observations = [
        json.loads(line) for line in (folder / "observations.jsonl").read_text().splitlines()
    ]
    events = [json.loads(line) for line in (folder / "truth.jsonl").read_text().splitlines()]
    decisions = [json.loads(line) for line in (folder / "decisions.jsonl").read_text().splitlines()]
    metrics = read_metrics(folder)
    with np.load(folder / "snapshots.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    angles, truth, means, stds = (arrays[key] for key in ("angles", "truth", "means", "stds"))
    figure = Figure(figsize=(11, 7), layout="constrained")
    axes = figure.subplots(2, 2)
    final, std = means[-1], stds[-1]
    closed = np.r_[np.arange(len(angles)), 0]
    axes[0, 0].plot(
        (truth * np.cos(angles))[closed],
        (truth * np.sin(angles))[closed],
        color="0.45",
        label="True outline",
    )
    axes[0, 0].plot(
        (final * np.cos(angles))[closed],
        (final * np.sin(angles))[closed],
        color="#2463a7",
        label="GP estimate",
    )
    observed_angle = np.array([o["theta"] for o in observations])
    observed_radius = np.array([o["measured_radius"] for o in observations])
    axes[0, 0].scatter(
        observed_radius * np.cos(observed_angle),
        observed_radius * np.sin(observed_angle),
        s=12,
        color="#c65c22",
        label="Contacts",
    )
    axes[0, 0].set(aspect="equal", title="Reconstructed outline", xlabel="x / R", ylabel="y / R")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].fill_between(
        angles,
        final - 1.96 * std,
        final + 1.96 * std,
        color="#2463a7",
        alpha=0.18,
        label="Pointwise 95% interval",
    )
    axes[0, 1].plot(angles, truth, color="0.45", label="Truth")
    axes[0, 1].plot(angles, final, color="#2463a7", label="Estimate")
    axes[0, 1].set(title="Radius and uncertainty", xlabel="Angle (radians)", ylabel="Radius / R")
    axes[0, 1].legend(fontsize=8)
    for axis, key, label in (
        (axes[1, 0], "touches", "Completed touches"),
        (axes[1, 1], "motion_time", "Modeled motion time (s)"),
    ):
        axis.step(
            [m[key] for m in metrics],
            [m["rmse"] for m in metrics],
            where="post",
            color="#2463a7",
            label="Radial RMSE",
        )
        axis.step(
            [m[key] for m in metrics],
            [m["max_radial_error"] for m in metrics],
            where="post",
            color="#c65c22",
            alpha=0.8,
            label="Sampled maximum error",
        )
        axis.set(xlabel=label, ylabel="Error / R")
        axis.legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.18)
    figure.suptitle(
        f"Touch Explorer · {config['shape']['kind']} · {config['experiment']['policy']}"
    )
    figure.savefig(folder / "summary.png", dpi=150)
    figure.savefig(folder / "summary.svg")
    # Replay uses saved posterior samples, not fitting or execution on the animation clock.
    take = np.arange(0, len(angles), max(1, len(angles) // 512))
    payload = {
        "status": json.loads((folder / "metadata.json").read_text())["status"],
        "config": config,
        "observations": observations,
        "events": events,
        "decisions": decisions,
        "angles": angles[take].round(7).tolist(),
        "truth": truth[take].round(7).tolist(),
        "means": means[:, take].round(7).tolist(),
        "stds": stds[:, take].round(7).tolist(),
        "metrics": [{k: v for k, v in row.items() if np.isfinite(v)} for row in metrics],
    }
    template = Path(__file__).with_name("replay.html").read_text()
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    (folder / "replay.html").write_text(template.replace("__RUN_DATA__", encoded))


def comparison(folder: Path, runs: list[dict]) -> None:
    """Paired pilot means, with stepwise estimates at a common physical-time horizon."""
    successful = [row for row in runs if row["status"] != "failed"]
    if not successful:
        return
    # Only compare complete matched cases; failures remain in runs.json.
    policy_names = sorted({row["policy"] for row in runs})
    cases = {row["case"] for row in runs}
    complete = {
        case
        for case in cases
        if sum(row["case"] == case for row in successful) == len(policy_names)
    }
    matched = [row for row in successful if row["case"] in complete]
    if not matched:
        return
    traces = {row["run"]: read_metrics(folder / row["run"]) for row in matched}
    touch_horizon = max(len(trace) for trace in traces.values())
    # Descriptive pilot horizon only, recorded explicitly; preregister a final-study horizon.
    horizon = min(trace[-1]["motion_time"] for trace in traces.values())
    time_grid = np.linspace(0, horizon, 300)
    figure = Figure(figsize=(11, 4), layout="constrained")
    left, right = figure.subplots(1, 2)
    summary = []
    for policy in policy_names:
        selected = [row for row in matched if row["policy"] == policy]
        touch_curves, time_curves, aucs, final_errors, computes = [], [], [], [], []
        for row in selected:
            trace = traces[row["run"]]
            times = np.array([m["motion_time"] for m in trace])
            errors = np.array([m["rmse"] for m in trace])
            touch_curves.append(np.pad(errors, (0, touch_horizon - len(errors)), mode="edge"))
            indices = np.searchsorted(times, time_grid, side="right") - 1
            time_curves.append(errors[indices])
            edges = np.r_[times[times < horizon], horizon]
            aucs.append(float(np.sum(np.diff(edges) * errors[: len(edges) - 1]) / horizon))
            final_errors.append(float(errors[-1]))
            computes.extend(m["fit_seconds"] + m["selection_seconds"] for m in trace[1:])
        left.step(
            np.arange(len(touch_curves[0])),
            np.mean(touch_curves, axis=0),
            where="post",
            label=policy,
        )
        right.step(time_grid, np.mean(time_curves, axis=0), where="post", label=policy)
        summary.append(
            {
                "policy": policy,
                "matched_cases": len(selected),
                "final_rmse_mean": float(np.mean(final_errors)),
                "time_averaged_rmse": float(np.mean(aucs)),
                "fit_and_select_p95_seconds": float(np.quantile(computes, 0.95)),
            }
        )
    for axis in (left, right):
        axis.grid(alpha=0.2)
        axis.set_ylabel("Mean radial RMSE / R")
        axis.legend()
    left.set_xlabel("Available touches (estimate held after stopping)")
    right.set_xlabel("Modeled motion time (s)")
    figure.suptitle(f"Pilot comparison · {len(complete)} matched cases")
    figure.savefig(folder / "comparison.png", dpi=150)
    figure.savefig(folder / "comparison.svg")
    write_json(
        folder / "summary.json",
        {
            "time_horizon_seconds": horizon,
            "excluded_incomplete_cases": sorted(cases - complete),
            "policies": summary,
        },
    )
