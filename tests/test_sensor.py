from dataclasses import asdict, replace

import numpy as np
import pytest

from touch_explorer.config import Config, ExperimentConfig, config_from_dict
from touch_explorer.runner import run_episode, save_episode
from touch_explorer.types import MotionConfig, ProbeAction
from touch_explorer.world import ContactWorld, SensorConfig, Shape


def world(sensor=None, noise_std=0.005):
    return ContactWorld(
        Shape("circle", base=0.5), MotionConfig(), noise_std=noise_std, seed=23, sensor=sensor
    )


def test_default_sensor_preserves_existing_random_draws():
    simulator = world()
    for index, repeat in [(3, 0), (5, 0), (3, 1)]:
        rng = np.random.default_rng(np.random.SeedSequence([23, index, repeat]))
        observation, _ = simulator.execute(ProbeAction(index, 0.2))
        assert observation.measured_radius == 0.5 + rng.normal(0, 0.005)
        assert observation.noise_variance == 0.005**2


def test_bias_and_noise_scale_change_only_the_sensor_reading():
    nominal = world()
    faulty = world(SensorConfig(bias=-0.02, noise_scale=3))
    for index in range(12):
        action = ProbeAction(index, index / 12)
        clean, clean_event = nominal.execute(action)
        changed, changed_event = faulty.execute(action)
        assert changed.measured_radius == pytest.approx(
            0.5 - 0.02 + 3 * (clean.measured_radius - 0.5)
        )
        assert changed.noise_variance == clean.noise_variance
        assert changed_event == clean_event
        assert set(asdict(changed)) == {"theta", "measured_radius", "noise_variance"}


def test_outliers_preserve_pairing_across_order_and_repeats():
    sensor = SensorConfig(outlier_probability=0.5, outlier_std=0.1)
    first, second = world(sensor), world(sensor)
    a, b = ProbeAction(3, 0.2), ProbeAction(7, 1.3)
    a0 = first.execute(a)[0]
    first.execute(b)
    a1 = first.execute(a)[0]
    second.execute(b)
    assert second.execute(a)[0] == a0
    assert second.execute(a)[0] == a1
    assert a0 != a1


def test_contamination_frequency_and_mixture_variance():
    # With zero nominal noise, nonzero residuals identify contamination exactly.
    simulator = world(SensorConfig(outlier_probability=0.1, outlier_std=0.1), noise_std=0)
    residuals = np.array(
        [simulator.execute(ProbeAction(i, 0))[0].measured_radius - 0.5 for i in range(6000)]
    )
    assert np.count_nonzero(residuals) / len(residuals) == pytest.approx(0.1, abs=0.015)
    assert abs(residuals.mean()) < 0.002
    assert residuals.var() == pytest.approx(0.1 * 0.1**2, rel=0.2)


def test_sensor_readings_are_not_clipped_to_workspace_bounds():
    observation, event = world(SensorConfig(bias=1)).execute(ProbeAction(0, 0))
    assert observation.measured_radius > 0.8
    assert event.true_radius == 0.5


@pytest.mark.parametrize(
    "settings",
    [
        {"bias": float("nan")},
        {"bias": True},
        {"noise_scale": -1},
        {"noise_scale": float("inf")},
        {"outlier_probability": -0.1},
        {"outlier_probability": 1.01},
        {"outlier_probability": True},
        {"outlier_std": -0.1},
        {"outlier_std": float("nan")},
    ],
)
def test_invalid_sensor_settings(settings):
    with pytest.raises(ValueError):
        SensorConfig(**settings)


def test_sensor_config_round_trip_and_strict_loading(tmp_path):
    import json

    config = config_from_dict({"sensor": {"bias": -0.01, "noise_scale": 3}})
    config = replace(
        config,
        experiment=ExperimentConfig(
            touches=4, initial=4, candidates=16, integration_points=16, evaluation_points=64
        ),
    )
    save_episode(run_episode(config), tmp_path / "run")
    saved = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config_from_dict(saved) == config
    assert config_from_dict({}).sensor == SensorConfig()
    with pytest.raises(ValueError, match="unknown sensor"):
        config_from_dict({"sensor": {"bais": 0.01}})


def test_sensor_parameters_do_not_leak_into_agent_decisions(monkeypatch):
    from touch_explorer import runner

    config = Config(
        experiment=ExperimentConfig(
            touches=16, initial=4, candidates=32, integration_points=32, evaluation_points=64
        )
    )
    original = runner.ContactWorld.execute

    def identical_readings(self, action):
        observation, event = original(self, action)
        return replace(observation, measured_radius=0.55), event

    monkeypatch.setattr(runner.ContactWorld, "execute", identical_readings)
    reference = run_episode(config)
    changed = run_episode(
        replace(
            config,
            sensor=SensorConfig(noise_scale=3, bias=0.1, outlier_probability=1, outlier_std=0.2),
        )
    )
    assert changed.observations == reference.observations
    assert changed.decisions == reference.decisions
    assert changed.status == reference.status
    np.testing.assert_array_equal(changed.means, reference.means)
    np.testing.assert_array_equal(changed.stds, reference.stds)


def test_runner_applies_sensor_bias():
    config = Config(
        experiment=ExperimentConfig(
            policy="sweep",
            touches=8,
            initial=4,
            candidates=16,
            integration_points=16,
            evaluation_points=64,
        )
    )
    reference = run_episode(config)
    changed = run_episode(replace(config, sensor=SensorConfig(bias=0.02)))
    np.testing.assert_allclose(
        [o.measured_radius for o in changed.observations],
        np.array([o.measured_radius for o in reference.observations]) + 0.02,
    )
    assert changed.events == reference.events
