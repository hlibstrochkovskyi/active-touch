import json

import pytest

from touch_explorer.plotting import comparison
from touch_explorer.runner import write_csv


def test_comparison_handles_different_stop_lengths_and_stepwise_error(tmp_path):
    runs = []
    for case, errors in (("a", [0.3, 0.2, 0.1]), ("b", [0.3, 0.25, 0.2, 0.15])):
        folder = tmp_path / case
        folder.mkdir()
        write_csv(
            folder / "metrics.csv",
            [
                {
                    "touches": n,
                    "motion_time": n,
                    "rmse": error,
                    "fit_seconds": 0.0,
                    "selection_seconds": 0.0,
                }
                for n, error in enumerate(errors)
            ],
        )
        runs.append({"case": case, "run": case, "policy": "cost", "status": "confident"})
    comparison(tmp_path, runs)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["time_horizon_seconds"] == 2
    assert summary["policies"][0]["matched_cases"] == 2
    assert summary["policies"][0]["time_averaged_rmse"] == pytest.approx(0.2625)
    assert summary["policies"][0]["final_rmse_mean"] == pytest.approx(0.125)
