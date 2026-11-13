import numpy as np
import pytest

from mani_skill.utils.geometry.geometry import (
    angle_between_vec,
    rotation_between_vec,
)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_angle_between_identical_vector_is_zero():
    a = _unit([1.0, 1.0, 1.0])
    # The naive dot product of a unit vector with itself can round above 1.0,
    # which used to make arccos return nan.
    assert angle_between_vec(a, a) == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_allclose(rotation_between_vec(a, a).apply(a), a, atol=1e-7)


def test_angle_between_opposite_vectors_is_pi():
    a = _unit([1.0, 1.0, 1.0])
    assert angle_between_vec(a, -a) == pytest.approx(np.pi, abs=1e-7)
    np.testing.assert_allclose(rotation_between_vec(a, -a).apply(a), -a, atol=1e-7)


def test_right_angle_is_unchanged():
    a = _unit([1.0, 0.0, 0.0])
    b = _unit([0.0, 1.0, 0.0])
    assert angle_between_vec(a, b) == pytest.approx(np.pi / 2, abs=1e-7)
    np.testing.assert_allclose(rotation_between_vec(a, b).apply(a), b, atol=1e-7)


@pytest.mark.parametrize(
    "a,b",
    [
        ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        ([1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]),
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.3, -0.2, 0.9], [-0.5, 0.7, 0.1]),
    ],
)
def test_angle_and_rotation_are_finite(a, b):
    a, b = _unit(a), _unit(b)
    assert np.isfinite(angle_between_vec(a, b))
    np.testing.assert_allclose(rotation_between_vec(a, b).apply(a), b, atol=1e-7)


@pytest.mark.parametrize(
    "a",
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.9, 0.4],
    ],
)
def test_antiparallel_rotation_uses_a_perpendicular_axis(a):
    a = _unit(a)
    R = rotation_between_vec(a, -a)
    np.testing.assert_allclose(R.apply(a), -a, atol=1e-7)
    # A proper rotation, not a reflection.
    assert np.isclose(np.linalg.det(R.as_matrix()), 1.0)


def test_zero_length_vector_raises():
    a = _unit([1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        angle_between_vec(np.zeros(3), a)
    with pytest.raises(ValueError):
        rotation_between_vec(a, np.zeros(3))
