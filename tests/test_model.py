import numpy as np
import pytest

from touch_explorer.model import ModelConfig, RadialGP, variance_reduction
from touch_explorer.types import ContactObservation


def observations(values=(0.45, 0.65, 0.5), noise=0.005):
    return tuple(
        ContactObservation(t, r, noise**2) for t, r in zip([0.0, 1.1, 3.2], values, strict=True)
    )


def independent_kernel(a, b, signal=0.2, length=0.4):
    distance = np.sqrt(2 - 2 * np.cos(np.asarray(a)[:, None] - np.asarray(b)[None, :]))
    z = np.sqrt(3) * distance / length
    return signal**2 * (1 + z) * np.exp(-z)


def test_gp_matches_independent_posterior():
    model = RadialGP()
    obs = observations()
    model.fit(obs)
    train = [o.theta for o in obs]
    query = [0.2, 1.4, 5.5]
    a = independent_kernel(train, train) + np.eye(3) * (0.005**2 + 1e-10)
    cross = independent_kernel(query, train)
    mean = 0.525 + cross @ np.linalg.solve(a, np.array([o.measured_radius for o in obs]) - 0.525)
    cov = independent_kernel(query, query) - cross @ np.linalg.solve(a, cross.T)
    actual, std = model.predict(query)
    np.testing.assert_allclose(actual, mean, atol=1e-12)
    np.testing.assert_allclose(model.posterior_covariance(query), cov, atol=1e-12)
    np.testing.assert_allclose(std**2, np.diag(cov), atol=1e-12)


def test_periodicity_and_value_independent_covariance():
    first, second = RadialGP(), RadialGP()
    first.fit(observations())
    second.fit(observations((0.7, 0.3, 0.4)))
    q = np.linspace(0, 2 * np.pi, 50)
    np.testing.assert_allclose(first.predict(q), first.predict(q + 2 * np.pi), atol=1e-12)
    np.testing.assert_allclose(first.posterior_covariance(q), second.posterior_covariance(q))
    assert not np.allclose(first.predict(q)[0], second.predict(q)[0])


@pytest.mark.parametrize("noise", [0.0, 0.005, 0.015])
@pytest.mark.parametrize("candidate", [0.0, 0.7])
def test_variance_reduction_matches_refit(noise, candidate):
    model = RadialGP()
    obs = observations(noise=noise)
    model.fit(obs)
    grid = np.linspace(0.1, 6.2, 21)
    before = model.predict(grid)[1] ** 2
    predicted = variance_reduction(model, [candidate], grid, noise**2)[0]
    model.fit((*obs, ContactObservation(candidate, 0.6, noise**2)))
    actual = np.mean(before - model.predict(grid)[1] ** 2)
    assert predicted == pytest.approx(actual, abs=1e-11)


def test_prior_and_duplicate_noiseless_contacts():
    model = RadialGP()
    assert model.predict([0])[0].shape == (1,)
    assert model.predict([0])[1].shape == (1,)
    mean, std = model.predict([0, 1])
    np.testing.assert_allclose(mean, 0.525)
    np.testing.assert_allclose(std, 0.2)
    obs = ContactObservation(0, 0.5, 0)
    model.fit((obs,) * 5)
    assert model.predict([0])[0][0] == pytest.approx(0.5, abs=1e-8)
    assert np.isfinite(model.predict([0])[1]).all()


def test_invalid_model_settings_and_queries():
    with pytest.raises(ValueError):
        ModelConfig(length_scale=0)
    with pytest.raises(ValueError):
        RadialGP().predict([float("nan")])


def test_optional_hyperparameter_fit_stays_within_declared_bounds():
    config = ModelConfig(learn_hyperparameters=True)
    model = RadialGP(config)
    model.fit(observations((0.35, 0.72, 0.42), noise=0.01), optimize=True)
    signal = float(np.sqrt(model._gp.kernel_.k1.constant_value))
    length = float(model._gp.kernel_.k2.length_scale)
    assert config.signal_bounds[0] <= signal <= config.signal_bounds[1]
    assert config.length_bounds[0] <= length <= config.length_bounds[1]
    assert np.isfinite(model.predict(np.linspace(0, 2 * np.pi, 20))[0]).all()


def test_learned_kernel_survives_unscheduled_fit_and_warm_starts(monkeypatch):
    from touch_explorer import model as module

    gp = RadialGP(ModelConfig(learn_hyperparameters=True))
    obs = observations((0.35, 0.72, 0.42), noise=0.01)
    gp.fit(obs, optimize=True)
    learned = gp.hyperparameters.copy()
    assert not np.isclose(learned["length_scale"], gp.config.length_scale)
    extended = (*obs, ContactObservation(2.0, 0.61, 0.0001))
    gp.fit(extended)
    assert gp.hyperparameters == learned

    optimizer = module.optimize_kernel
    starts = []

    def capture(objective, initial_theta, bounds):
        starts.append(initial_theta.copy())
        return optimizer(objective, initial_theta, bounds)

    monkeypatch.setattr(module, "optimize_kernel", capture)
    gp.fit(extended, optimize=True)
    np.testing.assert_allclose(
        starts[0], np.log([learned["signal_std"] ** 2, learned["length_scale"]])
    )


def test_optimizer_failure_refits_new_data_with_last_valid_kernel(monkeypatch):
    from touch_explorer import model as module

    gp = RadialGP(ModelConfig(learn_hyperparameters=True))
    obs = observations((0.35, 0.72, 0.42), noise=0.01)
    gp.fit(obs, optimize=True)
    learned = gp.hyperparameters.copy()

    def fail(*args, **kwargs):
        raise RuntimeError("test optimizer failure")

    monkeypatch.setattr(module, "optimize_kernel", fail)
    extended = (*obs, ContactObservation(5.0, 0.31, 0.0001))
    gp.fit(extended, optimize=True)
    assert gp.hyperparameters == learned
    assert gp.fit_status == "fallback"
    assert "test optimizer failure" in gp.fit_message
    reference = RadialGP(
        ModelConfig(signal_std=learned["signal_std"], length_scale=learned["length_scale"])
    )
    reference.fit(extended)
    np.testing.assert_allclose(gp.predict([0.2, 5.0]), reference.predict([0.2, 5.0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learn_hyperparameters": "true"},
        {"length_bounds": [0.1, float("inf")]},
        {"signal_bounds": [0.05, float("nan")]},
        {"learn_hyperparameters": True, "length_scale": 2.0},
    ],
)
def test_invalid_learning_settings(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)
