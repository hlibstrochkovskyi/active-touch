import json

import pytest

from touch_explorer.cli import main
from touch_explorer.model import RadialGP
from touch_explorer.plotting import render_run


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "small.toml"
    path.write_text(
        "[experiment]\ntouches=8\ninitial=4\ncandidates=16\n"
        'integration_points=16\nevaluation_points=64\npolicy="cost"\n'
    )
    return path


def test_demo_and_replay_do_not_refit_saved_models(tmp_path, config_file, monkeypatch):
    output = tmp_path / "demo"
    assert main(["demo", "--config", str(config_file), "--output", str(output)]) == 0
    assert (output / "summary.png").stat().st_size > 1000
    html = (output / "replay.html").read_text()
    assert "<canvas" in html and "requestAnimationFrame" in html
    assert "https://" not in html and "<script src=" not in html

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("replay must not fit a GP")

    monkeypatch.setattr(RadialGP, "fit", forbidden_fit)
    render_run(output)
    assert main(["demo", "--config", str(config_file), "--output", str(output)]) == 2


def test_small_benchmark_records_every_scheduled_policy(tmp_path, config_file):
    output = tmp_path / "benchmark"
    assert (
        main(
            [
                "benchmark",
                "--config",
                str(config_file),
                "--output",
                str(output),
                "--seeds",
                "1",
                "--shapes",
                "circle",
            ]
        )
        == 0
    )
    rows = json.loads((output / "runs.json").read_text())
    assert len(rows) == 5
    assert all(row["status"] == "budget_exhausted" for row in rows)
    assert {row["policy"] for row in rows} == {"random", "sweep", "gap", "variance", "cost"}
    assert (output / "comparison.png").stat().st_size > 1000


def test_benchmark_failures_are_reported_and_return_nonzero(tmp_path, config_file, monkeypatch):
    from touch_explorer import cli

    def fail(config):
        raise ArithmeticError("deliberate numerical failure")

    monkeypatch.setattr(cli, "run_episode", fail)
    output = tmp_path / "failed"
    assert (
        main(
            [
                "benchmark",
                "--config",
                str(config_file),
                "--output",
                str(output),
                "--seeds",
                "1",
                "--shapes",
                "circle",
            ]
        )
        == 1
    )
    rows = json.loads((output / "runs.json").read_text())
    assert len(rows) == 5
    assert all(row["status"] == "failed" for row in rows)
    assert "deliberate numerical failure" in rows[0]["error"]


def test_reliability_command_writes_all_variants(tmp_path, config_file):
    output = tmp_path / "reliability"
    assert (
        main(
            [
                "reliability",
                "--config",
                str(config_file),
                "--output",
                str(output),
                "--seeds",
                "1",
                "--shapes",
                "circle",
            ]
        )
        == 0
    )
    rows = json.loads((output / "runs.json").read_text())
    assert len(rows) == 5
    assert {r["variant"] for r in rows} == {
        "fixed-budget",
        "learned-budget",
        "learned-coverage",
        "uncertainty-stop",
        "guarded-stop",
    }
    assert (output / "reliability.json").exists()
    assert (output / "reliability.png").stat().st_size > 1000


def test_variable_length_benchmark_and_demo_report_real_stop(tmp_path, config_file, capsys):
    with config_file.open("a") as handle:
        handle.write('\n[stopping]\nmode="uncertainty"\nhalf_width=1.0\n')
    output = tmp_path / "stopped"
    assert main(["demo", "--config", str(config_file), "--output", str(output)]) == 0
    assert "4 / 8 touches" in capsys.readouterr().out
    assert "confident" in (output / "replay.html").read_text()
    assert (
        main(
            [
                "benchmark",
                "--config",
                str(config_file),
                "--output",
                str(tmp_path / "bench"),
                "--seeds",
                "1",
                "--shapes",
                "circle",
            ]
        )
        == 0
    )
