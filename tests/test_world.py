from dataclasses import asdict
from math import pi

import numpy as np
import pytest

from touch_explorer.types import MotionConfig, ProbeAction
from touch_explorer.world import ContactWorld, Shape, signed_arc


def test_analytic_shapes_and_periodicity():
    ellipse = Shape("ellipse", a=0.65, b=0.4)
    np.testing.assert_allclose(ellipse.radius([0, pi / 2, 2 * pi]), [0.65, 0.4, 0.65])
    rectangle = Shape("rectangle", a=0.5, b=0.3)
    np.testing.assert_allclose(rectangle.radius([0, pi / 2, pi / 4]), [0.5, 0.3, 0.3 * 2**0.5])
    np.testing.assert_allclose(Shape("circle", base=0.55).radius([0, 1, 4]), 0.55)


@pytest.mark.parametrize("kind", ["circle", "ellipse", "rectangle", "fourier", "recess"])
def test_generated_shapes_obey_public_bounds(kind):
    shape = Shape(kind)
    theta = np.linspace(0, 2 * pi, 4096, endpoint=False)
    radius = shape.radius(theta)
    assert np.all((radius >= 0.25) & (radius <= 0.8))
    np.testing.assert_allclose(radius, shape.radius(theta + 2 * pi), atol=1e-13)


def test_contact_noise_does_not_change_collision_or_cost():
    action = ProbeAction(4, pi / 2)
    exact = ContactWorld(Shape("circle", base=0.5), MotionConfig(), noise_std=0, seed=3)
    noisy = ContactWorld(Shape("circle", base=0.5), MotionConfig(), noise_std=0.02, seed=3)
    obs0, event0 = exact.execute(action)
    obs1, event1 = noisy.execute(action)
    assert obs0.measured_radius == 0.5
    assert obs1.measured_radius != 0.5
    assert event0 == event1
    assert event0.distance == pytest.approx(pi / 2 + 1)
    assert event0.duration == pytest.approx(pi / 2 + 1 + 0.5 + 0.2)
    assert set(asdict(obs1)) == {"theta", "measured_radius", "noise_variance"}


def test_path_stops_on_surface_and_returns_to_ring():
    world = ContactWorld(Shape("circle", base=0.5), MotionConfig(), noise_std=0, seed=1)
    _, event = world.execute(ProbeAction(0, pi / 2))
    transit_end = pi / 2
    approach_end = transit_end + 1
    np.testing.assert_allclose(event.position(0), [1, 0], atol=1e-14)
    np.testing.assert_allclose(event.position(transit_end), [0, 1], atol=1e-14)
    np.testing.assert_allclose(event.position(approach_end), [0, 0.5], atol=1e-14)
    np.testing.assert_allclose(event.position(approach_end + 0.1), [0, 0.5], atol=1e-14)
    np.testing.assert_allclose(event.position(event.duration), [0, 1], atol=1e-14)
    path = np.array([event.position(t) for t in np.linspace(0, event.duration, 301)])
    assert np.min(np.linalg.norm(path, axis=1)) >= 0.5 - 1e-12


def test_seam_and_antipodal_arc():
    assert signed_arc(2 * pi - 0.1, 0.1) == pytest.approx(0.2)
    assert signed_arc(0, pi) == pytest.approx(-pi)
    assert signed_arc(pi, 0) == pytest.approx(-pi)


def test_noise_is_coupled_by_candidate_and_repeat_not_policy_order():
    first = ContactWorld(Shape("ellipse"), MotionConfig(), noise_std=0.01, seed=15)
    second = ContactWorld(Shape("ellipse"), MotionConfig(), noise_std=0.01, seed=15)
    a, b = ProbeAction(2, 0.2), ProbeAction(3, 1.2)
    a0, _ = first.execute(a)
    first.execute(b)
    a1, _ = first.execute(a)
    second.execute(b)
    assert second.execute(a)[0] == a0
    assert second.execute(a)[0] == a1
    assert a0 != a1


def test_noise_statistics():
    world = ContactWorld(Shape("circle", base=0.5), MotionConfig(), noise_std=0.02, seed=23)
    residuals = np.array(
        [world.execute(ProbeAction(i, 0))[0].measured_radius - 0.5 for i in range(4000)]
    )
    assert abs(residuals.mean()) < 0.001
    assert residuals.std() == pytest.approx(0.02, rel=0.05)


@pytest.mark.parametrize(
    "kwargs", [{"a": -1}, {"kind": "unknown"}, {"depth": 1}, {"width": 0}, {"base": float("nan")}]
)
def test_invalid_shapes_rejected(kwargs):
    with pytest.raises(ValueError):
        Shape(**kwargs)


def test_invalid_motion_and_sensor_rejected():
    with pytest.raises(ValueError):
        MotionConfig(inward_speed=0)
    with pytest.raises(ValueError):
        ContactWorld(Shape(), MotionConfig(), noise_std=-0.1, seed=0)
    world = ContactWorld(Shape(), MotionConfig(), noise_std=0, seed=0)
    with pytest.raises(ValueError):
        world.execute(ProbeAction(0, float("nan")))
