import numpy as np
import pytest

from touch_explorer.metrics import reconstruction_metrics


def test_error_and_pointwise_coverage_are_distinct():
    metrics = reconstruction_metrics(
        np.array([0.5, 0.6, 0.8]), np.array([0.01, 0.1, 0.01]), np.array([0.5, 0.5, 0.5])
    )
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.1 / 3))
    assert metrics["max_radial_error"] == pytest.approx(0.3)
    assert metrics["coverage_95"] == pytest.approx(2 / 3)
    assert metrics["mean_interval_width"] == pytest.approx(3.92 * 0.04)


def test_exact_reconstruction_has_zero_error():
    values = np.array([0.4, 0.5, 0.6])
    metrics = reconstruction_metrics(values, np.zeros(3), values)
    assert metrics["rmse"] == metrics["max_radial_error"] == 0
    assert metrics["coverage_95"] == 1
