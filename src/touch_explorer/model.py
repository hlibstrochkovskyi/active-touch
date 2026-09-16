"""Periodic radius regression with explicit latent uncertainty."""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from .types import ContactObservation, finite


@dataclass(frozen=True)
class ModelConfig:
    mean: float = 0.525
    signal_std: float = 0.2
    length_scale: float = 0.4
    jitter: float = 1e-10
    learn_hyperparameters: bool = False
    signal_bounds: tuple[float, float] = (0.05, 0.4)
    length_bounds: tuple[float, float] = (0.08, 1.2)

    def __post_init__(self):
        for name in ("mean", "signal_std", "length_scale", "jitter"):
            finite(getattr(self, name), name, strict=True)
        if self.jitter > 1e-7:
            raise ValueError("jitter must stay <= 1e-7; model sensor noise separately")
        if type(self.learn_hyperparameters) is not bool:
            raise ValueError("learn_hyperparameters must be a boolean")
        for name in ("signal_bounds", "length_bounds"):
            bounds = getattr(self, name)
            if len(bounds) != 2:
                raise ValueError("hyperparameter bounds must be pairs")
            for bound in bounds:
                finite(bound, name, strict=True)
            object.__setattr__(self, name, tuple(bounds))
        if (
            len(self.signal_bounds) != 2
            or len(self.length_bounds) != 2
            or not 0 < self.signal_bounds[0] < self.signal_bounds[1]
            or not 0 < self.length_bounds[0] < self.length_bounds[1]
        ):
            raise ValueError("hyperparameter bounds must be increasing positive pairs")
        if self.learn_hyperparameters and not (
            self.signal_bounds[0] <= self.signal_std <= self.signal_bounds[1]
            and self.length_bounds[0] <= self.length_scale <= self.length_bounds[1]
        ):
            raise ValueError("initial learned parameters must lie within their bounds")


def optimize_kernel(objective, initial_theta, bounds):
    """Make optimizer failure explicit so a fit can fall back without losing new data."""
    result = minimize(objective, initial_theta, jac=True, bounds=bounds, method="L-BFGS-B")
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f"kernel optimizer failed: {result.message}")
    return result.x, result.fun


def circle_inputs(angles):
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 1 or not angles.size or not np.isfinite(angles).all():
        raise ValueError("angles must be a nonempty finite vector")
    return np.column_stack((np.cos(angles), np.sin(angles)))


class RadialGP:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self._signal_std = self.config.signal_std
        self._length_scale = self.config.length_scale
        self.fit_status = "prior"
        self.fit_message = None
        self.fit_warnings: list[str] = []
        self.last_jitter = self.config.jitter
        self._gp = self._new_gp(self.last_jitter)

    @property
    def hyperparameters(self) -> dict[str, float]:
        return {"signal_std": self._signal_std, "length_scale": self._length_scale}

    def _new_gp(self, alpha, *, optimize=False):
        kernel = ConstantKernel(
            self._signal_std**2,
            constant_value_bounds=tuple(v**2 for v in self.config.signal_bounds)
            if optimize
            else "fixed",
        ) * Matern(
            length_scale=self._length_scale,
            length_scale_bounds=self.config.length_bounds if optimize else "fixed",
            nu=1.5,
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimize_kernel if optimize else None,
            normalize_y=False,
        )

    def fit(self, observations: tuple[ContactObservation, ...], *, optimize: bool = False) -> None:
        self.fit_message = None
        self.fit_warnings = []
        if not observations:
            self._signal_std = self.config.signal_std
            self._length_scale = self.config.length_scale
            self.last_jitter = self.config.jitter
            self._gp = self._new_gp(self.last_jitter)
            self.fit_status = "prior"
            return
        x = circle_inputs([o.theta for o in observations])
        y = np.array([o.measured_radius for o in observations]) - self.config.mean
        noise = np.array([o.noise_variance for o in observations])

        def fit_with_jitter(learn):
            jitter = self.config.jitter
            while True:
                gp = self._new_gp(noise + jitter, optimize=learn)
                try:
                    with warnings.catch_warnings(record=True) as messages:
                        warnings.simplefilter("always", ConvergenceWarning)
                        gp.fit(x, y)
                    if not np.isfinite(gp.kernel_.theta).all():
                        raise FloatingPointError("nonfinite fitted kernel")
                except np.linalg.LinAlgError:
                    if jitter >= 1e-7:
                        raise
                    jitter = min(jitter * 10, 1e-7)
                else:
                    return gp, jitter, [str(w.message) for w in messages]

        try:
            gp, jitter, messages = fit_with_jitter(optimize)
            status = "optimized" if optimize else "fixed"
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            if not optimize:
                raise
            self.fit_message = f"{type(exc).__name__}: {exc}"
            gp, jitter, messages = fit_with_jitter(False)
            status = "fallback"
        self._gp, self.last_jitter = gp, jitter
        self._signal_std = float(np.sqrt(gp.kernel_.k1.constant_value))
        self._length_scale = float(gp.kernel_.k2.length_scale)
        self.fit_status, self.fit_warnings = status, messages

    def predict(self, angles) -> tuple[np.ndarray, np.ndarray]:
        mean, std = self._gp.predict(circle_inputs(angles), return_std=True)
        return np.atleast_1d(mean) + self.config.mean, np.atleast_1d(std)

    def posterior_covariance(self, angles) -> np.ndarray:
        _, covariance = self._gp.predict(circle_inputs(angles), return_cov=True)
        if not np.isfinite(covariance).all() or np.min(np.diag(covariance)) < -1e-10:
            raise ArithmeticError("invalid posterior covariance")
        return covariance


def variance_reduction(model: RadialGP, candidates, integration_angles, noise_variance: float):
    """Expected average latent-variance reduction for one additional observation.

    Hyperparameters stay fixed. This is not mutual information or a route optimum.
    """
    finite(noise_variance, "noise_variance")
    count = len(candidates)
    covariance = model.posterior_covariance(np.concatenate((candidates, integration_angles)))
    variances = np.maximum(np.diag(covariance)[:count], 0)
    cross = covariance[count:, :count]
    return np.mean(cross**2, axis=0) / (variances + noise_variance + model.last_jitter)
