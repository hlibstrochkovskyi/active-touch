"""Animate the four archived exploration checkpoints without rerunning inference."""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image


def build(source: Path, output: Path):
    with np.load(source) as saved:
        angles, truth = saved["angles"], saved["truth"]
        means, stds, steps = saved["means"], saved["stds"], saved["steps"]
        observed_angles, observed_radii = saved["observed_angles"], saved["observed_radii"]

    # Close spatial curves at the seam; the radial plot keeps its original grid.
    ring = np.r_[angles, angles[0] + 2 * np.pi]
    cosine, sine = np.cos(ring), np.sin(ring)
    frames = []
    low = min(float(np.min(means - 1.96 * stds)), float(np.min(truth))) - 0.02
    high = max(float(np.max(means + 1.96 * stds)), float(np.max(truth))) + 0.02
    for count, mean, std in zip(steps, means, stds, strict=True):
        figure = Figure(figsize=(10, 4.7), dpi=100, facecolor="white")
        canvas = FigureCanvasAgg(figure)
        spatial, radial = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.15]})
        figure.subplots_adjust(left=0.065, right=0.975, bottom=0.19, top=0.8, wspace=0.26)
        figure.suptitle(f"Active Touch  |  {count} / 128 contacts", x=0.065, ha="left", fontsize=17)
        figure.text(
            0.065,
            0.865,
            "Saved reconstruction checkpoints · cost policy · noisy recess case",
            fontsize=10,
            color="#475569",
        )

        closed_mean = np.r_[mean, mean[0]]
        closed_std = np.r_[std, std[0]]
        upper, lower = closed_mean + 1.96 * closed_std, closed_mean - 1.96 * closed_std
        spatial.fill(
            np.r_[upper * cosine, (lower * cosine)[::-1]],
            np.r_[upper * sine, (lower * sine)[::-1]],
            color="#2463a7",
            alpha=0.18,
        )
        closed_truth = np.r_[truth, truth[0]]
        spatial.plot(cosine, sine, color="#94a3b8", linestyle=":", linewidth=1)
        spatial.plot(closed_truth * cosine, closed_truth * sine, color="#374151", linewidth=1.5)
        spatial.plot(closed_mean * cosine, closed_mean * sine, color="#2463a7", linewidth=2)
        spatial.scatter(
            observed_radii[:count] * np.cos(observed_angles[:count]),
            observed_radii[:count] * np.sin(observed_angles[:count]),
            color="#c65c22",
            s=12,
            zorder=4,
        )
        spatial.plot(0, 0, "+", color="#64748b", markersize=7)
        spatial.set(
            aspect="equal",
            xlim=(-1.06, 1.06),
            ylim=(-1.06, 1.06),
            xlabel="x / outer radius",
            ylabel="y / outer radius",
        )

        radial.fill_between(
            angles,
            mean - 1.96 * std,
            mean + 1.96 * std,
            color="#2463a7",
            alpha=0.18,
            label="Pointwise 95% interval",
        )
        radial.plot(angles, truth, color="#374151", linewidth=1.5, label="True outline")
        radial.plot(angles, mean, color="#2463a7", linewidth=2, label="Estimated outline")
        radial.scatter(
            observed_angles[:count],
            observed_radii[:count],
            color="#c65c22",
            s=12,
            zorder=4,
            label="Noisy contacts",
        )
        radial.set(xlim=(0, 2 * np.pi), ylim=(low, high), xlabel="Angle (rad)", ylabel="Radius / R")
        for axis in (spatial, radial):
            axis.grid(alpha=0.15)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=8)
        handles, labels = radial.get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncols=4, frameon=False, fontsize=9)
        canvas.draw()
        frames.append(Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()).convert("RGB"))

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=[1800] * (len(frames) - 1) + [3500],
        loop=0,
        disposal=2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
