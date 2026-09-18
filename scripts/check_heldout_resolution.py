"""Re-evaluate saved fixed-kernel recess runs without choosing new probe actions."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from touch_explorer.config import config_from_dict
from touch_explorer.metrics import reconstruction_metrics
from touch_explorer.model import RadialGP
from touch_explorer.runner import write_json
from touch_explorer.types import ContactObservation


def check(folder: Path) -> dict:
    rows = json.loads((folder / "runs.json").read_text())
    diagnostics = []
    with threadpool_limits(limits=1, user_api="blas"):
        for row in rows:
            if row["family"] != "recess":
                continue
            if row["status"] != "budget_exhausted":
                raise ValueError("cannot check an incomplete recess episode")
            run = folder / row["run"]
            config = config_from_dict(json.loads((run / "config.json").read_text()))
            if config.model.learn_hyperparameters:
                raise ValueError("this check requires the frozen fixed-kernel study")
            observations = tuple(
                ContactObservation(**json.loads(line))
                for line in (run / "observations.jsonl").read_text().splitlines()
            )
            model = RadialGP(config.model)
            model.fit(observations)
            with np.load(run / "snapshots.npz") as saved:
                mean, std = model.predict(saved["angles"])
                np.testing.assert_allclose(mean, saved["means"][-1], atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(std, saved["stds"][-1], atol=1e-10, rtol=1e-10)
            points = 2 * config.experiment.evaluation_points
            angles = 2 * np.pi * (np.arange(points) + 0.37) / points
            mean, std = model.predict(angles)
            dense = reconstruction_metrics(mean, std, config.shape.radius(angles))
            diagnostics.append(
                {
                    "run": row["run"],
                    "points": points,
                    "original_rmse": row["final_rmse"],
                    "dense_rmse": dense["rmse"],
                    "original_max_error": row["final_max_error"],
                    "dense_max_error": dense["max_radial_error"],
                }
            )
    if not diagnostics:
        raise ValueError("no recess episodes to check")
    report = {
        "scope": "post-run evaluation-resolution diagnostic; primary metrics unchanged",
        "episodes": len(diagnostics),
        "max_abs_rmse_change": max(abs(r["dense_rmse"] - r["original_rmse"]) for r in diagnostics),
        "max_abs_max_error_change": max(
            abs(r["dense_max_error"] - r["original_max_error"]) for r in diagnostics
        ),
        "runs": diagnostics,
    }
    write_json(folder / "resolution.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    report = check(parser.parse_args().folder)
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))
