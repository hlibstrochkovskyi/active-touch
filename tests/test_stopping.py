from dataclasses import replace
from math import pi

import numpy as np
import pytest

from touch_explorer.stopping import StopController, StoppingConfig, largest_gap


def update(controller, *, n=24, width=0.01, gap=0.2, z=0.0, audit=False):
    return controller.observe(
        touches=n, max_half_width=width, max_gap=gap, innovation=z, was_audit=audit
    )


def test_budget_mode_never_claims_confidence():
    controller = StopController(StoppingConfig())
    assert not update(controller, width=0, gap=0)
    assert not controller.audit_pending


def test_uncertainty_mode_uses_width_only():
    controller = StopController(StoppingConfig(mode="uncertainty"))
    assert not update(controller, n=8, width=0.03)
    assert update(controller, n=8, width=0.019, gap=pi)


def test_guarded_stop_requires_two_new_audits():
    controller = StopController(StoppingConfig(mode="guarded"))
    assert not update(controller)
    assert controller.audit_pending
    assert not update(controller, n=25, audit=True)
    assert controller.audits_passed == 1
    assert update(controller, n=26, audit=True)


def test_failed_audit_and_failed_conditions_reset_audits():
    controller = StopController(StoppingConfig(mode="guarded"))
    update(controller)
    update(controller, n=25, audit=True)
    assert not update(controller, n=26, z=3.0, audit=True)
    assert not controller.audit_pending
    assert controller.audits_passed == 0
    update(controller, n=27)  # one normal exploration step before rearming
    assert controller.audit_pending
    assert not update(controller, n=28, width=0.04, audit=True)
    assert not controller.audit_pending


@pytest.mark.parametrize("kwargs", [{"n": 23}, {"gap": 0.5}, {"width": 0.03}])
def test_guarded_trigger_requires_all_conditions(kwargs):
    controller = StopController(StoppingConfig(mode="guarded"))
    assert not update(controller, **kwargs)
    assert not controller.audit_pending


def test_nonfinite_audit_cannot_pass():
    controller = StopController(StoppingConfig(mode="guarded"))
    update(controller)
    assert not update(controller, audit=True, z=float("nan"))
    assert not controller.audit_pending


def test_gap_handles_seam_and_duplicates():
    assert largest_gap([]) == pytest.approx(2 * pi)
    assert largest_gap([0, 2 * pi]) == pytest.approx(2 * pi)
    assert largest_gap(np.arange(8) * pi / 4) == pytest.approx(pi / 4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "typo"},
        {"audit_touches": 0},
        {"min_touches": True},
        {"half_width": float("nan")},
        {"max_gap_degrees": 361},
    ],
)
def test_invalid_stopping_settings(kwargs):
    with pytest.raises(ValueError):
        replace(StoppingConfig(), **kwargs)
