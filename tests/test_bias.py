import copy

import numpy as np
import pytest

import metatally as mt


def test_discrete_state_bias_updates_and_freezes():
    bias = mt.DiscreteStateBias(n_states=3, height=0.5)
    bias.update(1)
    assert bias.value(1) == 0.5
    bias.freeze()
    bias.update(1)
    assert bias.value(1) == 0.5
    bias.unfreeze()
    bias.update(1)
    assert bias.value(1) == 1.0


def test_independent_torsion_bias_updates():
    bias = mt.IndependentTorsionBias(n_variables=2, height=0.5, sigma=0.3, n_grid=32)
    assert bias.value(0, x=[0.0, 0.0]) == 0.0
    bias.update(0, x=[0.0, 0.0])
    assert bias.array.sum() > 0.0
    assert bias.value(1, x=[0.0, 0.0]) == pytest.approx(bias.value(0, x=[0.0, 0.0]))


def test_state_conditioned_torsion_bias_updates():
    bias = mt.StateConditionedTorsionBias(n_states=3, n_variables=1, height=0.5, sigma=0.3, n_grid=32)
    bias.update(0, x=[0.0])
    assert bias.value(0, x=[0.0]) > 0.0
    assert bias.value(1, x=[0.0]) == 0.0


def test_composite_bias_sums_and_updates():
    space = mt.TorsionChain(n_variables=1, radix=3).state_space()
    state = mt.DiscreteStateBias.from_space(space, height=2.0)
    torsion = mt.IndependentTorsionBias.from_space(space, height=1.0, sigma=0.3, n_grid=32)
    bias = mt.CompositeBias([state, torsion])
    assert bias.value(1, x=[0.0]) == 0.0
    bias.update(1, x=[0.0])
    assert bias.value(1, x=[0.0]) > 2.0
    before_state = state.array.copy()
    before_torsion = torsion.array.copy()
    bias.freeze()
    bias.update(1, x=[0.0])
    np.testing.assert_allclose(state.array, before_state)
    np.testing.assert_allclose(torsion.array, before_torsion)


def test_composite_bias_reset():
    space = mt.TorsionChain(n_variables=1, radix=3).state_space()
    bias = mt.CompositeBias([
        mt.DiscreteStateBias.from_space(space, height=1.0),
        mt.IndependentTorsionBias.from_space(space, height=1.0),
    ])
    bias.update(0, x=[0.0])
    assert bias.value(0, x=[0.0]) > 0.0
    bias.reset()
    assert bias.value(0, x=[0.0]) == 0.0


def test_frozen_copy_can_be_used_for_production():
    space = mt.TorsionChain(n_variables=1, radix=3).state_space()
    bias = mt.DiscreteStateBias.from_space(space, height=1.0)
    bias.update(0)
    frozen = copy.deepcopy(bias)
    frozen.freeze()
    frozen.update(0)
    assert frozen.value(0) == 1.0
    assert bias.value(0) == 1.0


def test_bad_bias_inputs():
    with pytest.raises(ValueError):
        mt.CompositeBias([])
    with pytest.raises(ValueError):
        mt.DiscreteStateBias(n_states=0)
    with pytest.raises(ValueError):
        mt.IndependentTorsionBias(n_variables=0)
    with pytest.raises(ValueError):
        mt.StateConditionedTorsionBias(n_states=0, n_variables=1)


def test_discrete_state_bias_values_setter_updates_live_array():
    bias = mt.DiscreteStateBias(n_states=3, height=0.0)
    bias.values = np.array([0.0, 1.0, 2.0])

    assert bias.value(0) == 0.0
    assert bias.value(1) == 1.0
    assert bias.value(2) == 2.0

    bias.values[:] = np.array([3.0, 4.0, 5.0])
    assert bias.value(1) == 4.0


def test_discrete_state_bias_freeze_keeps_values_active():
    bias = mt.DiscreteStateBias(n_states=3, height=1.0)
    bias.values = np.array([0.0, 1.0, 2.0])
    bias.freeze()

    assert bias.is_frozen
    assert bias.value(1) == 1.0
    bias.update(1)
    np.testing.assert_allclose(bias.values, np.array([0.0, 1.0, 2.0]))


def test_torsion_bias_values_setters_update_live_arrays():
    ind = mt.IndependentTorsionBias(n_variables=2, n_grid=4)
    ind.values = np.ones((2, 4))
    assert ind.value(0, x=[0.0, 0.0]) == 2.0

    sc = mt.StateConditionedTorsionBias(n_states=3, n_variables=2, n_grid=4)
    sc.values = np.ones((3, 2, 4))
    assert sc.value(1, x=[0.0, 0.0]) == 2.0


def test_well_tempered_discrete_state_bias_update_decreases_increment():
    bias = mt.WellTemperedDiscreteStateBias(
        n_states=3, height=1.0, bias_factor=10.0, beta=1.0
    )

    bias.update(1)
    first = bias.values[1]
    bias.update(1)
    second_increment = bias.values[1] - first

    assert first == pytest.approx(1.0)
    assert second_increment < first


def test_well_tempered_discrete_state_bias_freeze_keeps_value_active():
    bias = mt.WellTemperedDiscreteStateBias(n_states=3, height=1.0)
    bias.values = [0.0, 1.0, 2.0]
    bias.freeze()

    assert bias.value(2) == pytest.approx(2.0)
    bias.update(2)
    assert bias.value(2) == pytest.approx(2.0)


def test_well_tempered_free_energy_differences_from_bias():
    bias = mt.WellTemperedDiscreteStateBias(
        n_states=3, height=1.0, bias_factor=5.0, beta=1.0
    )
    bias.values = [0.0, 1.0, 2.0]

    dF = bias.free_energy_differences_from_bias(reference=0)

    np.testing.assert_allclose(dF, np.array([0.0, -1.25, -2.5]))
