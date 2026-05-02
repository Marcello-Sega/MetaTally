import numpy as np
import pytest

import metatally as mt


def test_single_variable_gaussian_move_changes_one_coordinate():
    move = mt.SingleVariableGaussianMove(step_size=0.5)
    rng = np.random.default_rng(1)
    x = np.zeros(5)
    prop = move.propose(x, rng)

    assert prop.shape == (5,)
    assert np.count_nonzero(prop != x) == 1


def test_single_variable_periodic_gaussian_move_wraps():
    move = mt.SingleVariablePeriodicGaussianMove(step_size=10.0)
    rng = np.random.default_rng(1)
    x = np.full(4, np.pi - 1e-3)
    prop = move.propose(x, rng)

    assert prop.shape == (4,)
    assert np.all(prop >= -np.pi)
    assert np.all(prop < np.pi)


def test_moves_reject_bad_step_size():
    with pytest.raises(ValueError, match="step_size"):
        mt.SingleVariableGaussianMove(step_size=0.0)
    with pytest.raises(ValueError, match="step_size"):
        mt.SingleVariablePeriodicGaussianMove(step_size=0.0)
