import numpy as np
import pytest

from touch_explorer.model import RadialGP
from touch_explorer.policies import POLICIES, Policy, estimated_times
from touch_explorer.types import ContactObservation, MotionConfig, PublicState


def setup():
    candidates = np.arange(32) * 2 * np.pi / 32
    visited = (0, 8, 16, 24)
    obs = tuple(ContactObservation(candidates[i], 0.5, 0.005**2) for i in visited)
    model = RadialGP()
    model.fit(obs)
    return candidates, PublicState(candidates[24], obs, visited), model


def test_largest_gap_handles_seam_and_never_repeats():
    candidates, state, model = setup()
    policy = Policy("gap", candidates, MotionConfig(), noise_variance=0.005**2, budget=16)
    decision = policy.choose(state, model)
    assert decision.action.candidate_index in {4, 12, 20, 28}
    assert decision.action.candidate_index not in state.visited_indices


def test_sweep_moves_clockwise_on_budget_grid():
    candidates, state, model = setup()
    policy = Policy("sweep", candidates, MotionConfig(), noise_variance=0, budget=16)
    decision = policy.choose(state, model)
    assert decision.action.candidate_index == 22  # next clockwise from index 24


@pytest.mark.parametrize("name", POLICIES)
def test_policy_produces_finite_feasible_decision(name):
    candidates, state, model = setup()
    policy = Policy(name, candidates, MotionConfig(), noise_variance=0.005**2, budget=16, seed=3)
    decision = policy.choose(state, model)
    assert decision.action.theta in candidates
    assert np.isfinite(decision.score)
    assert decision.estimated_time > 0


def test_travel_cost_includes_both_radial_legs_and_dwell():
    result = estimated_times([0, np.pi / 2], [0.5, 0.5], 0, MotionConfig())
    np.testing.assert_allclose(result, [1.7, 1.7 + np.pi / 2])


def test_random_policy_is_seeded_and_does_not_repeat():
    candidates, state, model = setup()
    a = Policy("random", candidates, MotionConfig(), noise_variance=0, budget=16, seed=99)
    b = Policy("random", candidates, MotionConfig(), noise_variance=0, budget=16, seed=99)
    assert a.choose(state, model) == b.choose(state, model)
    assert a.choose(state, model).action.candidate_index not in state.visited_indices


def test_cost_score_really_uses_time():
    candidates, state, model = setup()
    a = Policy("cost", candidates, MotionConfig(), noise_variance=0.005**2, budget=16)
    decision = a.choose(state, model)
    scores = a.scores(state, model)
    assert scores[decision.action.candidate_index] == pytest.approx(max(scores))


def test_unknown_policy_rejected():
    candidates, _, _ = setup()
    with pytest.raises(ValueError):
        Policy("typo", candidates, MotionConfig(), noise_variance=0, budget=16)
