import json
from dataclasses import replace

from touch_explorer.config import Config, ExperimentConfig
from touch_explorer.reliability import episode_outcome, mismatch_configs, summarize
from touch_explorer.runner import run_episode
from touch_explorer.stopping import StoppingConfig
from touch_explorer.world import SensorConfig


def test_mismatch_matrix_changes_one_sensor_fault_at_a_time():
    config = Config(sensor=SensorConfig(bias=0.9))
    variants = mismatch_configs(config)
    expected = {
        "nominal": SensorConfig(),
        "bias": SensorConfig(bias=0.015),
        "noise": SensorConfig(noise_scale=3),
        "outliers": SensorConfig(outlier_probability=0.05, outlier_std=0.05),
    }
    assert len(variants) == 8
    reference = variants["nominal-budget"]
    for fault, sensor in expected.items():
        for method, mode in (("budget", "budget"), ("guarded", "guarded")):
            variant = variants[f"{fault}-{method}"]
            assert variant.sensor == sensor
            assert variant.stopping.mode == mode
            assert (
                replace(variant, sensor=reference.sensor, stopping=reference.stopping) == reference
            )
    assert reference.experiment.noise_std == config.experiment.noise_std
    assert reference.experiment.safeguard_every == 5
    assert reference.model.learn_hyperparameters


def test_false_stop_is_evaluated_after_the_episode():
    config = Config(
        experiment=ExperimentConfig(
            touches=8, initial=4, candidates=16, integration_points=16, evaluation_points=64
        ),
        stopping=StoppingConfig(mode="uncertainty", half_width=1),
    )
    result = run_episode(config)
    outcome = episode_outcome(result, rmse_tolerance=1e-10, max_error_tolerance=1e-10)
    assert outcome["false_stop"]
    assert outcome["status"] == "confident"
    exhausted = replace(result, status="budget_exhausted")
    assert not episode_outcome(exhausted, rmse_tolerance=1e-10, max_error_tolerance=1e-10)[
        "false_stop"
    ]


def test_no_stops_has_no_false_stop_rate(tmp_path):
    config = Config(
        experiment=ExperimentConfig(
            touches=4, initial=4, candidates=16, integration_points=16, evaluation_points=64
        )
    )
    result = run_episode(config)
    row = {"variant": "fixed-budget", "case": "a", **episode_outcome(result)}
    summarize(tmp_path, [row])
    summary = json.loads((tmp_path / "reliability.json").read_text())["variants"][0]
    assert summary["stops"] == 0
    assert summary["false_stop_rate"] is None
    assert summary["budget_exhausted"] == 1


def test_failed_runs_remain_counted(tmp_path):
    summarize(tmp_path, [{"variant": "guarded-stop", "case": "bad", "status": "failed"}])
    summary = json.loads((tmp_path / "reliability.json").read_text())["variants"][0]
    assert summary["attempted"] == summary["failed"] == 1
    assert summary["mean_touches"] is None
    assert summary["false_stop_rate"] is None
