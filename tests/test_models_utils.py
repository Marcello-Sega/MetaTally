import doctest

import numpy as np
import pytest

import metatally as mt
import metatally.models.utils as utils


class QuadraticModel:
    def energy(self, x):
        x = np.asarray(x, dtype=float)
        return float(np.sum(x**2))


def test_scan_coordinate_with_explicit_values():
    model = QuadraticModel()
    values, energies = mt.scan_coordinate(
        model,
        variable=1,
        background=[1.0, 0.0, 3.0],
        values=[-1.0, 0.0, 2.0],
        subtract_min=False,
    )

    np.testing.assert_allclose(values, np.array([-1.0, 0.0, 2.0]))
    np.testing.assert_allclose(energies, np.array([11.0, 10.0, 14.0]))


def test_scan_coordinate_subtracts_minimum():
    model = QuadraticModel()
    _, energies = mt.scan_coordinate(
        model,
        variable=0,
        background=[0.0, 2.0],
        values=[-1.0, 0.0, 1.0],
        subtract_min=True,
    )

    np.testing.assert_allclose(energies, np.array([1.0, 0.0, 1.0]))


def test_scan_coordinate_default_grid():
    model = QuadraticModel()
    values, energies = mt.scan_coordinate(
        model,
        variable=0,
        background=[0.0, 0.0],
        n_points=8,
        bounds=(-1.0, 1.0),
    )

    assert values.shape == (8,)
    assert energies.shape == (8,)
    assert values[0] == pytest.approx(-1.0)
    assert values[-1] == pytest.approx(0.75)
    assert np.min(energies) == pytest.approx(0.0)


def test_scan_coordinate_torsion_chain_minima():
    model = mt.TorsionChain(n_variables=1, radix=3, barrier=2.0)
    values, energies = mt.scan_coordinate(
        model,
        variable=0,
        background=model.initial_state(),
        values=mt.TorsionChain(n_variables=1, radix=3).state_space().assigner.centers,
    )

    assert values.shape == (3,)
    np.testing.assert_allclose(energies, np.zeros(3), atol=1e-12)


def test_scan_coordinate_rejects_invalid_inputs():
    model = QuadraticModel()

    with pytest.raises(ValueError, match="background"):
        mt.scan_coordinate(model, variable=0, background=[[0.0]])

    with pytest.raises(ValueError, match="variable"):
        mt.scan_coordinate(model, variable=2, background=[0.0, 1.0])

    with pytest.raises(ValueError, match="n_points"):
        mt.scan_coordinate(model, variable=0, background=[0.0], n_points=1)

    with pytest.raises(ValueError, match="bounds"):
        mt.scan_coordinate(model, variable=0, background=[0.0], bounds=(1.0, 0.0))

    with pytest.raises(ValueError, match="values must be a one-dimensional"):
        mt.scan_coordinate(model, variable=0, background=[0.0], values=[[0.0]])

    with pytest.raises(ValueError, match="values must contain"):
        mt.scan_coordinate(model, variable=0, background=[0.0], values=[])

    with pytest.raises(ValueError, match="values must be finite"):
        mt.scan_coordinate(model, variable=0, background=[0.0], values=[np.nan])


def test_doctests_for_models_utils_module():
    result = doctest.testmod(utils, verbose=False)
    assert result.failed == 0
