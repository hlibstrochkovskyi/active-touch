import json
from dataclasses import asdict, replace

import pytest

from touch_explorer.config import Config, ExperimentConfig
from touch_explorer.heldout import (
    analyze,
    paired_interval,
    schedule,
    time_average,
    validate_protocol,
)
from touch_explorer.runner import run_episode
from touch_explorer.world import Shape


@pytest.fixture
def protocol():
    return {
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
                "shape_seed": 900 + i,
                "episode_seed": 800 + i,
                "shape": asdict(Shape("circle", base=0.5 + i * 0.01)),
            }
            for i in range(2)
        ],
        "noise_levels": [0.0, 0.005],
        "noise_seeds": [301, 302],
        "policies": ["gap", "cost"],
        "checkpoints": [4, 8],
        "time_horizon": 50.0,
        "bootstrap_seed": 123,
        "bootstrap_samples": 1000,
    }


def test_noise_seed_changes_readings_without_changing_candidate_or_policy_rng():
    config = Config(
        experiment=ExperimentConfig(
            policy="random",
            touches=8,
            initial=4,
            candidates=16,
            integration_points=16,
            evaluation_points=64,
            noise_seed=101,
        )
    )
    a = run_episode(config)
    b = run_episode(replace(config, experiment=replace(config.experiment, noise_seed=102)))
    assert [d.action for d in a.decisions] == [d.action for d in b.decisions]
    assert a.observations != b.observations
    assert a.events == b.events


def test_protocol_schedules_matched_independent_noise_repeats(protocol):
    jobs = schedule(protocol)
    assert len(jobs) == 16
    assert len({row["run"] for row, _ in jobs}) == 16
    selected = [(r, c) for r, c in jobs if r["case"] == "circle-0"]
    assert len({c.experiment.seed for _, c in selected}) == 1
    assert len({c.experiment.noise_seed for _, c in selected}) == 2
    assert len({c.shape for _, c in selected}) == 1
    same = [c for r, c in selected if r["noise_std"] == 0.005 and r["noise_seed"] == 301]
    assert run_episode(same[0]).observations[:4] == run_episode(same[1]).observations[:4]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(noise_levels=[0, 0]),
        lambda p: p.update(noise_seeds=[True]),
        lambda p: p.update(policies=["cost"]),
        lambda p: p.update(time_horizon=0),
        lambda p: p.update(checkpoints=[9]),
        lambda p: p["cases"].append(p["cases"][0]),
        lambda p: p["config"]["model"].update(learn_hyperparameters=True),
        lambda p: p["config"]["sensor"].update(bias=0.1),
        lambda p: p["config"]["stopping"].update(mode="guarded"),
    ],
)
def test_invalid_or_confounded_protocol_rejected(protocol, mutation):
    mutation(protocol)
    with pytest.raises(ValueError):
        validate_protocol(protocol)


def test_time_average_is_exact_and_holds_final_estimate():
    assert time_average([0, 2, 5], [4, 2, 1], 3) == pytest.approx(10 / 3)
    assert time_average([0, 2, 5], [4, 2, 1], 10) == pytest.approx(1.9)
    assert time_average([0, 2, 5], [4, 2, 1], 2) == 4


def test_bootstrap_weights_families_equally_and_keeps_shape_pairs():
    result = paired_interval({"circle": [1, 1], "ellipse": [3, 3, 3, 3]}, 1000, 123)
    assert result["mean"] == 2
    assert result["ci95"] == [2, 2]
    assert result["shapes"] == 6
    varied = {"circle": [-1, 1], "ellipse": [-2, 2]}
    assert paired_interval(varied, 1000, 123) == paired_interval(varied, 1000, 123)


def fake_rows(protocol):
    rows = []
    for row, _ in schedule(protocol):
        # Pairing cancels arbitrary case/seed effects, leaving the policy difference.
        value = 10 * int(row["case"].split("-")[-1]) + row["noise_seed"]
        offset = 2 if row["policy"] == "cost" else 0
        rows.append(
            {
                **row,
                "status": "budget_exhausted",
                "final_rmse": value + offset,
                "time_averaged_rmse": value + offset,
                "checkpoints": {"4": value + offset, "8": value + offset},
            }
        )
    return rows


def test_analysis_averages_noise_repeats_within_shapes_before_inference(protocol):
    report = analyze(protocol, fake_rows(protocol))
    assert report["complete"]
    assert len(report["paired"]) == 6  # pooled and two noise levels, two metrics
    for result in report["paired"]:
        assert result["mean"] == 2
        assert result["ci95"] == [2, 2]
        assert result["shapes"] == 2


def test_analysis_does_not_hide_missing_or_failed_runs(protocol):
    rows = fake_rows(protocol)
    rows[0] = {**rows[0], "status": "failed", "error": "numerical failure"}
    report = analyze(protocol, rows[:-1])
    assert not report["complete"]
    assert report["failed"] == 1 and report["missing"] == 1
    assert report["paired"] == []
    with pytest.raises(ValueError, match="duplicate"):
        analyze(protocol, rows + [rows[0]])


@pytest.mark.parametrize("workers", [1, 2])
def test_heldout_cli_and_analysis_replay(tmp_path, protocol, monkeypatch, workers):
    from touch_explorer import heldout
    from touch_explorer.cli import main

    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    folder = tmp_path / "study"
    assert (
        main(
            ["heldout", "--protocol", str(path), "--output", str(folder), "--workers", str(workers)]
        )
        == 0
    )
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["protocol"] == json.loads(path.read_text())
    report = json.loads((folder / "analysis.json").read_text())
    assert report["complete"] and report["scheduled"] == 16
    assert (folder / "comparison.png").stat().st_size > 1000

    def forbidden(*args, **kwargs):
        raise AssertionError("analysis must not execute episodes")

    monkeypatch.setattr(heldout, "run_episode", forbidden)
    assert main(["analyze", str(folder)]) == 0
    assert json.loads((folder / "analysis.json").read_text()) == report
    assert main(["heldout", "--protocol", str(path), "--output", str(folder)]) == 2


def test_execution_failure_is_saved_and_withholds_inference(tmp_path, protocol, monkeypatch):
    from touch_explorer import heldout

    def broken(config):
        raise ArithmeticError("deliberate failure")

    monkeypatch.setattr(heldout, "run_episode", broken)
    folder = tmp_path / "failed"
    assert heldout.run_heldout(protocol, folder) == 1
    rows = json.loads((folder / "runs.json").read_text())
    assert len(rows) == 16 and all(row["status"] == "failed" for row in rows)
    assert (folder / rows[0]["run"] / "failure.json").exists()
    report = json.loads((folder / "analysis.json").read_text())
    assert report["failed"] == 16 and not report["paired"]


@pytest.mark.parametrize("seed", [-1, True, 0.5])
def test_invalid_noise_seed(seed):
    with pytest.raises(ValueError, match="noise_seed"):
        ExperimentConfig(noise_seed=seed)
