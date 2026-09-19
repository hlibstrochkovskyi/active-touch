"""Render the project article's exploration and cost figures from saved results."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def build(sequence: Path, analysis: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    steps = np.array([8, 16, 32, 128])
    observations = [
        json.loads(line) for line in (sequence / "observations.jsonl").read_text().splitlines()
    ]
    with np.load(sequence / "snapshots.npz") as saved:
        angles, truth = saved["angles"], saved["truth"]
        means, stds = saved["means"][steps], saved["stds"][steps]
    observed_angles = np.array([o["theta"] for o in observations])
    observed_radii = np.array([o["measured_radius"] for o in observations])
    np.savez_compressed(
        output / "exploration-data.npz",
        angles=angles,
        truth=truth,
        means=means,
        stds=stds,
        steps=steps,
        observed_angles=observed_angles,
        observed_radii=observed_radii,
        source=sequence.name,
    )
    figure = Figure(figsize=(11, 6), layout="constrained")
    axes = figure.subplots(2, 2)
    low = min(float(np.min(means - 1.96 * stds)), float(np.min(truth))) - 0.01
    high = max(float(np.max(means + 1.96 * stds)), float(np.max(truth))) + 0.01
    for axis, n, mean, std in zip(axes.flat, steps, means, stds, strict=True):
        axis.fill_between(
            angles,
            mean - 1.96 * std,
            mean + 1.96 * std,
            color="#2463a7",
            alpha=0.2,
            label="Pointwise 95% interval",
        )
        axis.plot(angles, truth, color="0.35", label="Truth")
        axis.plot(angles, mean, color="#2463a7", label="Estimate")
        axis.scatter(
            observed_angles[:n], observed_radii[:n], s=9, color="#c65c22", label="Readings"
        )
        axis.set(
            title=f"{n} completed touches",
            xlabel="Angle (rad)",
            ylabel="Radius / R",
            ylim=(low, high),
        )
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7, ncols=2)
    figure.suptitle(f"Exploration example · {sequence.name}")
    figure.savefig(output / "exploration.png", dpi=150)
    figure.savefig(output / "exploration.svg")

    report = json.loads(analysis.read_text())
    policies = [c["policy"] for c in report["curves"]]

    def values(key):
        return [
            next(
                r["mean"]
                for r in report["means"]
                if r["noise_std"] is None and r["policy"] == policy and r["metric"] == key
            )
            for policy in policies
        ]

    figure = Figure(figsize=(10, 4), layout="constrained")
    left, right = figure.subplots(1, 2)
    left.barh(policies, values("motion_time"), color="#2463a7")
    left.set_xlabel("Mean modeled seconds · all 128 touches")
    right.barh(policies, np.array(values("compute_p95_seconds")) * 1000, color="#c65c22")
    right.set_xlabel("Mean within-episode p95 fit + select (ms)")
    for axis in (left, right):
        axis.grid(axis="x", alpha=0.2)
        axis.invert_yaxis()
    figure.suptitle("Physical motion and computation · primary held-out study")
    figure.savefig(output / "cost-summary.png", dpi=150)
    figure.savefig(output / "cost-summary.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.sequence, args.analysis, args.output)
