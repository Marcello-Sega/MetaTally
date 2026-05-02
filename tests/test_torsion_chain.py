import numpy as np
import pytest

import metatally as mt
from metatally.models.torsion_chain import torsion_centers


def test_torsion_centers_are_minima():
    for radix in range(2, 8):
        centers = torsion_centers(radix)
        np.testing.assert_allclose(np.cos(radix * centers), np.ones(radix), atol=1e-12)


def test_torsion_chain_energy_minima_without_state_potential():
    model = mt.TorsionChain(n_variables=3, radix=3, barrier=5.0)
    assert model.energy([0.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert model.energy([2 * np.pi / 3, -2 * np.pi / 3, 0.0]) == pytest.approx(0.0)


def test_torsion_chain_state_space():
    model = mt.TorsionChain(n_variables=3, radix=3)
    space = model.state_space()
    assert space.n_states == 27
    assert space.encode([0.1, 2.0, -2.0]) == 7


def test_torsion_chain_coupling_scalar_and_array():
    model = mt.TorsionChain(n_variables=4, radix=3, barrier=0.0, coupling=2.0)
    np.testing.assert_allclose(model.coupling_array, np.array([2.0, 2.0, 2.0]))
    assert model.energy(np.zeros(4)) == pytest.approx(-6.0)
    coupling = np.array([0.0, 0.5, 1.0])
    model2 = mt.TorsionChain(n_variables=4, radix=3, barrier=0.0, coupling=coupling)
    np.testing.assert_allclose(model2.coupling_array, coupling)
    assert model2.energy(np.zeros(4)) == pytest.approx(-np.sum(coupling))


def test_torsion_chain_state_potential():
    uxi = np.array([0.0, 1.0, 2.0])
    model = mt.TorsionChain(n_variables=1, radix=3, barrier=0.0, state_potential=uxi)
    space = model.state_space()
    for xi in range(3):
        phi = torsion_centers(3)[xi]
        assigned = int(space.encode([phi]))
        assert model.energy([phi]) == pytest.approx(uxi[assigned])
    np.testing.assert_allclose(model.state_free_energy_differences(reference=0), uxi - uxi[0])


def test_torsion_chain_rejects_invalid_state_potential():
    with pytest.raises(ValueError, match="state_potential"):
        mt.TorsionChain(n_variables=2, radix=3, state_potential=np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        mt.TorsionChain(n_variables=1, radix=3, state_potential=[0.0, np.nan, 1.0])


def test_torsion_chain_rejects_invalid_coupling():
    with pytest.raises(ValueError, match="coupling"):
        mt.TorsionChain(n_variables=4, coupling=[1.0, 2.0])
    with pytest.raises(ValueError, match="coupling"):
        mt.TorsionChain(n_variables=4, coupling=np.nan)


def test_scan_coordinate():
    model = mt.TorsionChain(n_variables=2, radix=3, barrier=2.0)
    values, energies = mt.scan_coordinate(model, variable=0, background=model.initial_state(), n_points=16)
    assert values.shape == energies.shape == (16,)
    assert np.min(energies) == pytest.approx(0.0)
