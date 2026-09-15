"""Truth-dependent evaluation; none of these values goes to a policy."""

import numpy as np


def reconstruction_metrics(mean, std, truth) -> dict[str, float]:
    error = np.asarray(mean) - truth
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_radial_error": float(np.max(np.abs(error))),
        "coverage_95": float(np.mean(np.abs(error) <= 1.96 * std)),
        "mean_interval_width": float(np.mean(2 * 1.96 * std)),
    }
