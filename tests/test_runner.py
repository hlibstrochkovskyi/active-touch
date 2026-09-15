import json
from dataclasses import replace

import numpy as np
import pytest

from touch_explorer.config import Config, ExperimentConfig, load_config
from touch_explorer.runner import run_episode, save_episode
from touch_explorer.world import Shape


def small_config(policy="cost"):
    return Config(
        experiment=ExperimentConfig(
            policy=policy,
            touches=16,
            initial=4,
            candidates=32,
            integration_points=32,
            evaluation_points=128,
            seed=17,
        ),
        shape=Shape("ellipse"),
    )


def test_reproducible_runs_and_accounting():
    a, b = run_episode(small_config()), run_episode(small_config())
    assert a.observations == b.observations
    assert a.decisions == b.decisions
    np.testing.assert_allclose(a.means, b.means)
    assert len(a.observations) == 16
    assert a.means.shape == (17, 128)  # prior plus every completed action
    assert a.metrics[-1]["motion_time"] == pytest.approx(sum(e.duration for e in a.events))
    assert a.metrics[-1]["distance"] == pytest.approx(sum(e.distance for e in a.events))
    assert a.metrics[-1]["touches"] == 16
    assert a.metrics[-1]["rmse"] < 0.03


def test_policies_share_initial_measurements():
    a, b = run_episode(small_config("random")), run_episode(small_config("sweep"))
    assert a.observations[:4] == b.observations[:4]
    assert a.events[:4] == b.events[:4]


def test_public_history_drives_decisions_not_private_evaluation(monkeypatch):
    from touch_explorer import runner

    original = runner.ContactWorld.execute
    reference = run_episode(small_config())

    def changed_private_event(self, action):
        observation, event = original(self, action)
        return observation, replace(event, true_radius=0.3)

    monkeypatch.setattr(runner.ContactWorld, "execute", changed_private_event)
    changed = run_episode(small_config())
    assert reference.decisions == changed.decisions
    assert reference.observations == changed.observations
    assert reference.metrics[-1]["motion_time"] != changed.metrics[-1]["motion_time"]


def test_saved_run_separates_observations_and_truth(tmp_path):
    result = run_episode(small_config())
    folder = tmp_path / "run"
    save_episode(result, folder)
    public = [json.loads(line) for line in (folder / "observations.jsonl").read_text().splitlines()]
    assert set(public[0]) == {"theta", "measured_radius", "noise_variance"}
    assert len(public) == 16
    assert (folder / "truth.jsonl").is_file()
    assert (folder / "metrics.csv").is_file()
    with np.load(folder / "snapshots.npz") as data:
        np.testing.assert_allclose(data["means"], result.means)
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata["status"] == "budget_exhausted"
    assert metadata["packages"]["numpy"]
    with pytest.raises(FileExistsError):
        save_episode(result, folder)


def test_strict_config_loading(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[experiment]\npolicy="gap"\ntouches=16\n[shape]\nkind="circle"\n')
    assert load_config(path).experiment.policy == "gap"
    path.write_text("[experiment]\ntuches=16\n")
    with pytest.raises(ValueError, match="tuches"):
        load_config(path)
    path.write_text("[typo]\nx=1\n")
    with pytest.raises(ValueError, match="typo"):
        load_config(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"touches": 0},
        {"candidates": 31},
        {"initial": 3},
        {"seed": -1},
        {"touches": True},
        {"noise_std": -1},
        {"policy": "bad"},
    ],
)
def test_invalid_experiment_settings(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)
