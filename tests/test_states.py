import doctest

import numpy as np
import pytest

import metatally as mt
import metatally.core.states as states
from metatally.core.encoding import MixedRadixEncoder


def test_discrete_state_assigner_properties():
    assigner = mt.DiscreteStateAssigner([2, 3, 4])

    assert assigner.radices == (2, 3, 4)
    assert assigner.n_variables == 3


def test_discrete_state_assigner_assign_one():
    assigner = mt.DiscreteStateAssigner([2, 3, 4])

    np.testing.assert_array_equal(assigner.assign([1, 2, 3]), np.array([1, 2, 3]))


def test_discrete_state_assigner_assign_many():
    assigner = mt.DiscreteStateAssigner([2, 3, 4])

    x = [[0, 0, 0], [1, 2, 3]]
    np.testing.assert_array_equal(assigner.assign(x), np.array(x))


def test_discrete_state_assigner_rejects_invalid_radices():
    with pytest.raises(ValueError, match="All radices must be >= 2"):
        mt.DiscreteStateAssigner([2, 1])


def test_discrete_state_assigner_rejects_wrong_shape():
    assigner = mt.DiscreteStateAssigner([2, 2, 2])

    with pytest.raises(ValueError, match="Expected the last digit axis"):
        assigner.assign([1, 0])


def test_discrete_state_assigner_rejects_out_of_range_digits():
    assigner = mt.DiscreteStateAssigner([2, 2, 2])

    with pytest.raises(ValueError, match="Digits out of range"):
        assigner.assign([1, 0, 2])


def test_threshold_state_assigner_properties():
    assigner = mt.ThresholdStateAssigner([0.5], n_variables=3)

    assert assigner.thresholds == (0.5,)
    assert assigner.n_variables == 3
    assert assigner.radices == (2, 2, 2)


def test_threshold_state_assigner_ternary_properties():
    assigner = mt.ThresholdStateAssigner([-1.0, 1.0], n_variables=2)

    assert assigner.thresholds == (-1.0, 1.0)
    assert assigner.n_variables == 2
    assert assigner.radices == (3, 3)


def test_threshold_state_assigner_assign_one_binary():
    assigner = mt.ThresholdStateAssigner([0.5], n_variables=3)

    np.testing.assert_array_equal(
        assigner.assign([0.1, 0.7, 0.2]),
        np.array([0, 1, 0]),
    )


def test_threshold_state_assigner_assign_many_binary():
    assigner = mt.ThresholdStateAssigner([0.5], n_variables=3)

    x = [[0.1, 0.7, 0.2], [0.8, 0.9, 0.0]]
    expected = [[0, 1, 0], [1, 1, 0]]

    np.testing.assert_array_equal(assigner.assign(x), np.array(expected))


def test_threshold_state_assigner_assign_ternary():
    assigner = mt.ThresholdStateAssigner([-1.0, 1.0], n_variables=4)

    x = [-2.0, -0.5, 0.5, 2.0]
    expected = [0, 1, 1, 2]

    np.testing.assert_array_equal(assigner.assign(x), np.array(expected))


def test_threshold_state_assigner_rejects_invalid_n_variables():
    with pytest.raises(ValueError, match="n_variables must be >= 1"):
        mt.ThresholdStateAssigner([0.5], n_variables=0)


def test_threshold_state_assigner_rejects_empty_thresholds():
    with pytest.raises(ValueError, match="At least one threshold"):
        mt.ThresholdStateAssigner([], n_variables=3)


def test_threshold_state_assigner_rejects_unsorted_thresholds():
    with pytest.raises(ValueError, match="strictly increasing"):
        mt.ThresholdStateAssigner([1.0, 0.0], n_variables=3)

    with pytest.raises(ValueError, match="strictly increasing"):
        mt.ThresholdStateAssigner([1.0, 1.0], n_variables=3)


def test_threshold_state_assigner_rejects_scalar_input():
    assigner = mt.ThresholdStateAssigner([0.5], n_variables=3)

    with pytest.raises(ValueError, match="Input must have at least one dimension"):
        assigner.assign(0.1)


def test_threshold_state_assigner_rejects_wrong_shape():
    assigner = mt.ThresholdStateAssigner([0.5], n_variables=3)

    with pytest.raises(ValueError, match="Expected the last axis"):
        assigner.assign([0.1, 0.7])


def test_torsion_state_assigner_properties():
    centers = [-2 * np.pi / 3, 0.0, 2 * np.pi / 3]
    assigner = mt.TorsionStateAssigner(centers=centers, n_variables=4)

    assert assigner.n_variables == 4
    assert assigner.radices == (3, 3, 3, 3)
    np.testing.assert_allclose(assigner.centers, tuple(centers))


def test_torsion_state_assigner_wrap_angle():
    wrapped = mt.TorsionStateAssigner.wrap_angle(
        np.array([-3 * np.pi, -np.pi, 0.0, np.pi, 3 * np.pi])
    )

    np.testing.assert_allclose(
        wrapped,
        np.array([-np.pi, -np.pi, 0.0, -np.pi, -np.pi]),
    )


def test_torsion_state_assigner_assign_one():
    centers = [-2 * np.pi / 3, 0.0, 2 * np.pi / 3]
    assigner = mt.TorsionStateAssigner(centers=centers, n_variables=3)

    np.testing.assert_array_equal(
        assigner.assign([0.1, 2.0, -2.0]),
        np.array([1, 2, 0]),
    )


def test_torsion_state_assigner_assign_many():
    centers = [-2 * np.pi / 3, 0.0, 2 * np.pi / 3]
    assigner = mt.TorsionStateAssigner(centers=centers, n_variables=3)

    x = [[0.1, 2.0, -2.0], [0.0, 0.1, -0.1]]
    expected = [[1, 2, 0], [1, 1, 1]]

    np.testing.assert_array_equal(assigner.assign(x), np.array(expected))


def test_torsion_state_assigner_periodic_assignment():
    centers = [-np.pi + 0.1, 0.0, np.pi - 0.1]
    assigner = mt.TorsionStateAssigner(centers=centers, n_variables=2)

    # These angles are close to the positive-pi-side centre once periodicity
    # is handled.
    np.testing.assert_array_equal(
        assigner.assign([np.pi - 0.05, -np.pi + 0.05]),
        np.array([2, 0]),
    )


def test_torsion_state_assigner_rejects_invalid_n_variables():
    with pytest.raises(ValueError, match="n_variables must be >= 1"):
        mt.TorsionStateAssigner([0.0, 1.0], n_variables=0)


def test_torsion_state_assigner_rejects_too_few_centers():
    with pytest.raises(ValueError, match="At least two torsional"):
        mt.TorsionStateAssigner([0.0], n_variables=3)


def test_torsion_state_assigner_rejects_scalar_input():
    assigner = mt.TorsionStateAssigner([0.0, 1.0], n_variables=3)

    with pytest.raises(ValueError, match="Input must have at least one dimension"):
        assigner.assign(0.1)


def test_torsion_state_assigner_rejects_wrong_shape():
    assigner = mt.TorsionStateAssigner([0.0, 1.0], n_variables=3)

    with pytest.raises(ValueError, match="Expected the last axis"):
        assigner.assign([0.1, 0.2])


def test_state_space_from_radix_properties():
    space = mt.StateSpace.from_radix(n_variables=3, radix=2)

    assert space.radices == (2, 2, 2)
    assert space.n_variables == 3
    assert space.n_states == 8


def test_state_space_from_radices_properties():
    space = mt.StateSpace.from_radices([2, 3, 4])

    assert space.radices == (2, 3, 4)
    assert space.n_variables == 3
    assert space.n_states == 24


def test_state_space_discrete_assign_encode_decode_one():
    space = mt.StateSpace.from_radix(n_variables=3, radix=2)

    np.testing.assert_array_equal(space.assign([1, 0, 1]), np.array([1, 0, 1]))
    assert space.encode([1, 0, 1]) == 5
    np.testing.assert_array_equal(space.decode(5), np.array([1, 0, 1]))


def test_state_space_discrete_assign_encode_decode_many():
    space = mt.StateSpace.from_radix(n_variables=3, radix=2)

    x = [[0, 0, 0], [1, 0, 1], [1, 1, 1]]

    np.testing.assert_array_equal(space.assign(x), np.array(x))
    np.testing.assert_array_equal(space.encode(x), np.array([0, 5, 7]))
    np.testing.assert_array_equal(space.decode([0, 5, 7]), np.array(x))


def test_state_space_with_threshold_assigner():
    assigner = mt.ThresholdStateAssigner([0.5], n_variables=3)
    space = mt.StateSpace(assigner)

    assert space.radices == (2, 2, 2)
    assert space.encode([0.1, 0.7, 0.2]) == 2

    np.testing.assert_array_equal(
        space.assign([[0.1, 0.7, 0.2], [0.8, 0.9, 0.0]]),
        np.array([[0, 1, 0], [1, 1, 0]]),
    )

    np.testing.assert_array_equal(
        space.encode([[0.1, 0.7, 0.2], [0.8, 0.9, 0.0]]),
        np.array([2, 3]),
    )


def test_state_space_with_torsion_assigner():
    centers = [-2 * np.pi / 3, 0.0, 2 * np.pi / 3]
    assigner = mt.TorsionStateAssigner(centers=centers, n_variables=3)
    space = mt.StateSpace(assigner)

    assert space.radices == (3, 3, 3)
    np.testing.assert_array_equal(
        space.assign([0.1, 2.0, -2.0]),
        np.array([1, 2, 0]),
    )
    assert space.encode([0.1, 2.0, -2.0]) == 7


def test_state_space_rejects_mismatched_encoder():
    assigner = mt.DiscreteStateAssigner([2, 2, 2])
    enc = MixedRadixEncoder([3, 3, 3])

    with pytest.raises(ValueError, match="Encoder radices must match"):
        mt.StateSpace(assigner=assigner, encoder=enc)


def test_state_space_all_digits():
    space = mt.StateSpace.from_radix(n_variables=2, radix=2)

    np.testing.assert_array_equal(
        space.all_digits(),
        np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
    )


def test_runtime_protocol():
    assigner = mt.DiscreteStateAssigner([2, 2])

    assert isinstance(assigner, states.StateAssigner)


def test_doctests_for_states_module():
    result = doctest.testmod(states, verbose=False)
    assert result.failed == 0

