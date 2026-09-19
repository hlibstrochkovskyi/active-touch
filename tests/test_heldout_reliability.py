import json
from dataclasses import asdict

import numpy as np
import pytest

from touch_explorer.config import Config, ExperimentConfig
from touch_explorer.heldout import analyze, run_heldout, schedule, validate_protocol
from touch_explorer.reliability import VARIANTS
from touch_explorer.runner import run_episode
from touch_explorer.world import Shape


@pytest.fixture
def protocol():
    return {
        "study": "reliability",
        "config": asdict(
            Config(
                experiment=ExperimentConfig(
                    touches=8, initial=4, candidates=16, integration_points=16, evaluation_points=64
                )
            )
        ),
        "cases": [
            {
                "id": f"circle-{i}",
                "shape_seed": 9900 + i,
                "episode_seed": 8800 + i,
                "shape": asdict(Shape("circle", base=0.5 + i * 0.01)),
            }
            for i in range(2)
        ],
        "noise_levels": [0.005],
        "noise_seeds": [601, 602],
        "variants": list(VARIANTS),
        "checkpoints": [4, 8],
        "time_horizon": 50.0,
        "bootstrap_seed": 123,
        "bootstrap_samples": 1000,
        "rmse_tolerance": 0.01,
        "max_error_tolerance": 0.03,
    }


def test_schedule_keeps_methods_and_noise_matched(protocol):
    jobs = schedule(protocol)
    assert len(jobs) == 20
    selected = [(r, c) for r, c in jobs if r["case"] == "circle-0" and r["noise_seed"] == 601]
    assert len(selected) == 5
    assert {c.experiment.policy for _, c in selected} == {"cost"}
    readings = [run_episode(c).observations[:4] for _, c in selected]
    assert all(r == readings[0] for r in readings)
    settings = {r["variant"]: c for r, c in selected}
    assert not settings["fixed-budget"].model.learn_hyperparameters
    assert settings["learned-budget"].model.learn_hyperparameters
    assert settings["learned-coverage"].experiment.safeguard_every == 5
    assert settings["uncertainty-stop"].stopping.mode == "uncertainty"
    assert settings["guarded-stop"].stopping.mode == "guarded"


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(variants=["fixed-budget"]),
        lambda p: p.update(study="typo"),
        lambda p: p.update(rmse_tolerance=0),
        lambda p: p.update(max_error_tolerance=float("nan")),
        lambda p: p.pop("rmse_tolerance"),
        lambda p: p["config"]["model"].update(learn_hyperparameters=True),
    ],
)
def test_invalid_reliability_protocol(protocol, change):
    change(protocol)
    with pytest.raises(ValueError):
        validate_protocol(protocol)


def rows_for(protocol):
    rows = []
    for row, _ in schedule(protocol):
        stopping = row["variant"] in ("uncertainty-stop", "guarded-stop")
        wrong = row["variant"] == "uncertainty-stop"
        rows.append(
            {
                **row,
                "status": "confident" if stopping else "budget_exhausted",
                "touches": 4 if stopping else 8,
                "false_stop": wrong,
                "quality_met": not wrong,
                "final_rmse": 0.02 if wrong else 0.001,
                "final_max_error": 0.04 if wrong else 0.002,
                "motion_time": 10 if stopping else 20,
                "optimizer_fallbacks": 0,
                "coverage_95": 0.9,
                "interval_width": 0.02,
            }
        )
    return rows


def test_paired_false_stop_events_are_shape_level_and_conditional_rates_are_separate(protocol):
    report = analyze(protocol, rows_for(protocol))
    assert report["complete"]
    summaries = {r["variant"]: r for r in report["variants"]}
    assert summaries["fixed-budget"]["false_stop_rate"] is None
    assert summaries["uncertainty-stop"]["false_stop_rate"] == 1
    assert summaries["guarded-stop"]["false_stop_rate"] == 0
    contrast = next(r for r in report["paired"] if r["metric"] == "false_stop")
    assert contrast["variant"] == "guarded-stop"
    assert contrast["reference"] == "uncertainty-stop"
    assert contrast["mean"] == -1 and contrast["ci95"] == [-1, -1]
    assert contrast["shapes"] == 2  # Four runs are two shapes, not four independent samples.


def test_no_stop_is_not_success_and_missing_runs_withhold_inference(protocol):
    rows = rows_for(protocol)
    for r in rows:
        if r["variant"] == "guarded-stop":
            r.update(status="budget_exhausted", touches=8, false_stop=False, quality_met=False)
    report = analyze(protocol, rows)
    guarded = next(r for r in report["variants"] if r["variant"] == "guarded-stop")
    assert guarded["false_stop_rate"] is None
    assert guarded["budget_exhausted"] == 4 and guarded["quality_met"] == 0
    incomplete = analyze(protocol, rows[:-1])
    assert not incomplete["complete"] and incomplete["missing"] == 1
    assert incomplete["paired"] == []


def test_budget_variant_cannot_be_reported_as_confident(protocol):
    rows = rows_for(protocol)
    rows[0]["status"] = "confident"
    report = analyze(protocol, rows)
    assert not report["complete"] and report["failed"] == 1


def test_early_stops_hold_estimates_in_analysis_and_use_frozen_quality_limits(tmp_path, protocol):
    protocol["config"]["stopping"].update(
        half_width=1, min_touches=4, max_gap_degrees=360, innovation_limit=100
    )
    protocol["rmse_tolerance"] = 1e-12
    protocol["max_error_tolerance"] = 1e-12
    folder = tmp_path / "study"
    assert run_heldout(protocol, folder) == 0
    rows = json.loads((folder / "runs.json").read_text())
    early = [r for r in rows if r["status"] == "confident"]
    assert early and all(r["false_stop"] for r in early)
    for row in early:
        assert len(row["touch_curve"]) == 9
        np.testing.assert_allclose(row["touch_curve"][row["touches"] :], row["final_rmse"])
        assert row["checkpoints"]["8"] == row["final_rmse"]
    from touch_explorer.cli import main

    before = json.loads((folder / "analysis.json").read_text())
    assert "held-out" in json.loads((folder / "manifest.json").read_text())["scope"]
    assert main(["analyze", str(folder)]) == 0
    assert json.loads((folder / "analysis.json").read_text()) == before
