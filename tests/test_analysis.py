import numpy as np
import pytest

import metatally as mt


def test_free_energy_differences_zero_probabilities_are_nan():
    dF = mt.free_energy_differences([0.5, 0.0, 0.5], reference=0)
    assert dF[0] == 0.0
    assert np.isnan(dF[1])
    assert dF[2] == pytest.approx(0.0)


def test_reweighted_state_probabilities_fixed_bias_synthetic():
    states = np.array([0, 1, 2])
    bias = np.array([0.0, 1.0, 2.0])
    # Biased probabilities for equal underlying weights under U+V.
    p_biased = np.exp(-bias)
    p_biased /= p_biased.sum()
    counts = np.round(10000 * p_biased).astype(int)
    traj = np.repeat(states, counts)
    bias_values = bias[traj]
    probs = mt.reweighted_state_probabilities(traj, bias_values, n_states=3)
    np.testing.assert_allclose(probs, np.ones(3) / 3.0, atol=5e-4)


def test_reweighting_diagnostics():
    diag = mt.reweighting_diagnostics([0.0, 1.0, 2.0])
    assert diag.log_weight_range == pytest.approx(2.0)
    assert diag.effective_sample_size > 0.0
