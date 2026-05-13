"""Analysis helpers for MetaTally trajectories."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]
IntegerArray = NDArray[np.integer]


def state_probabilities(
    states: ArrayLike | None = None,
    *,
    visits: ArrayLike | None = None,
    n_states: int | None = None,
    weights: ArrayLike | None = None,
) -> FloatArray:
    """
    Estimate discrete state probabilities from states or visit counts.

    Parameters
    ----------
    states
        Encoded state trajectory. Required unless ``visits`` is provided.
    visits
        Precomputed visit counts for each state. If supplied, ``states`` and
        ``weights`` must be omitted.
    n_states
        Total number of states. Required when ``states`` is supplied and some
        states may be unvisited.
    weights
        Optional statistical weights for each trajectory frame.

    Returns
    -------
    FloatArray
        Normalised probabilities with shape ``(n_states,)``.

    Examples
    --------
    >>> import metatally as mt
    >>> p = mt.state_probabilities(states=[0, 1, 1, 2], n_states=4)
    >>> p.tolist()
    [0.25, 0.5, 0.25, 0.0]
    """
    if visits is not None:
        if states is not None or weights is not None:
            raise ValueError("If visits is provided, states and weights must be omitted.")
        counts = np.asarray(visits, dtype=float)
        if counts.ndim != 1:
            raise ValueError("visits must be one-dimensional.")
        if np.any(counts < 0.0):
            raise ValueError("visits must be non-negative.")
    else:
        if states is None:
            raise ValueError("Either states or visits must be provided.")
        idx = np.asarray(states)
        if idx.ndim == 0:
            idx = idx.reshape(1)
        if not np.issubdtype(idx.dtype, np.integer):
            if np.all(np.equal(idx, np.asarray(idx, dtype=np.int64))):
                idx = idx.astype(np.int64)
            else:
                raise ValueError("states must contain integer state indices.")
        else:
            idx = idx.astype(np.int64, copy=False)
        if idx.size == 0:
            raise ValueError("states must not be empty.")
        if np.any(idx < 0):
            raise ValueError("states must be non-negative.")
        if n_states is None:
            n_states = int(np.max(idx)) + 1
        n_states = int(n_states)
        if n_states < 1:
            raise ValueError("n_states must be >= 1.")
        if np.any(idx >= n_states):
            raise ValueError(f"states out of range for n_states={n_states}.")
        if weights is None:
            counts = np.bincount(idx.ravel(), minlength=n_states).astype(float)
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != idx.shape:
                raise ValueError("weights must have the same shape as states.")
            if np.any(w < 0.0) or not np.all(np.isfinite(w)):
                raise ValueError("weights must be finite and non-negative.")
            counts = np.bincount(idx.ravel(), weights=w.ravel(), minlength=n_states).astype(float)

    total = float(np.sum(counts))
    if total <= 0.0:
        raise ValueError("At least one state count/weight must be positive.")
    return counts / total


def free_energy_differences(
    probabilities: ArrayLike,
    *,
    beta: float = 1.0,
    reference: int | None = None,
) -> FloatArray:
    """
    Compute state free-energy differences relative to a reference state.

    The returned array contains

    ``DeltaF[i] = F[i] - F[reference]``

    with

    ``F[i] - F[j] = -1/beta * log(p[i] / p[j])``.

    Parameters
    ----------
    probabilities
        State probabilities with shape ``(n_states,)``.
    beta
        Inverse thermal energy, ``1/(k_B T)``. The returned free energies are
        in the corresponding energy units.
    reference
        Encoded state label used as zero of free energy. The default is None 
        and selects the state with minimum free energy. Otherwise, pass an
        integer to select a reference state.

    Returns
    -------
    FloatArray
        Free-energy differences relative to ``reference``.

    Examples
    --------
    >>> import numpy as np
    >>> import metatally as mt
    >>> p = np.array([0.5, 0.25, 0.25])
    >>> mt.free_energy_differences(p, reference=0).round(6).tolist()
    [-0.0, 0.693147, 0.693147]
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1:
        raise ValueError("probabilities must be one-dimensional.")
    if p.size == 0:
        raise ValueError("probabilities must not be empty.")
    if np.any(p < 0.0) or not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite and non-negative.")
    beta = float(beta)
    if beta <= 0.0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and > 0.")
    if reference is None: 
        reference = np.argmax(probabilities)
    reference = int(reference)
    if reference < 0 or reference >= p.size:
        raise ValueError(f"reference out of range. Expected 0 <= reference < {p.size}.")
    if p[reference] <= 0.0:
        raise ValueError("reference probability must be positive.")

    out = np.full(p.shape, np.inf, dtype=float)
    mask = p > 0.0
    out[mask] = -(1.0 / beta) * np.log(p[mask] / p[reference])
    return out


def free_energy_differences_from_visits(
    visits: ArrayLike,
    *,
    beta: float = 1.0,
    reference: int | None = None,
    pseudocount: float = 0.0,
) -> FloatArray:
    """
    Compute free-energy differences from state visit counts.

    Parameters
    ----------
    visits
        Visit counts for each encoded state.
    beta
        Inverse thermal energy, ``1/(k_B T)``.
    reference
        State used as zero of free energy. The default is ``0``.
    pseudocount
        Optional non-negative pseudocount added to every state before
        normalisation. Use this only when a finite estimate for unvisited states
        is explicitly desired.

    Returns
    -------
    FloatArray
        Free-energy differences relative to ``reference``.

    Examples
    --------
    >>> import metatally as mt
    >>> mt.free_energy_differences_from_visits([2, 1, 1], reference=0).round(6).tolist()
    [-0.0, 0.693147, 0.693147]
    """
    counts = np.asarray(visits, dtype=float)
    if counts.ndim != 1:
        raise ValueError("visits must be one-dimensional.")
    pseudocount = float(pseudocount)
    if pseudocount < 0.0 or not np.isfinite(pseudocount):
        raise ValueError("pseudocount must be finite and >= 0.")
    probabilities = state_probabilities(visits=counts + pseudocount)
    return free_energy_differences(probabilities, beta=beta, reference=reference)


def reweight_from_bias(bias_values: ArrayLike, *, beta: float = 1.0) -> FloatArray:
    """
    Return stable Boltzmann reweighting factors for a biased trajectory.

    If a configuration was sampled under a bias ``V_bias``, its unbiased weight
    is proportional to ``exp(beta * V_bias)``. The returned weights are shifted
    by a constant for numerical stability; this does not affect normalised
    probabilities or free-energy differences.
    """
    bias = np.asarray(bias_values, dtype=float)
    if bias.ndim == 0:
        bias = bias.reshape(1)
    if not np.all(np.isfinite(bias)):
        raise ValueError("bias_values must be finite.")
    beta = float(beta)
    if beta <= 0.0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and > 0.")
    shifted = beta * (bias - np.max(bias))
    return np.exp(shifted)


def reweighted_state_probabilities(
    states: ArrayLike,
    bias_values: ArrayLike,
    *,
    beta: float = 1.0,
    n_states: int | None = None,
) -> FloatArray:
    """
    Estimate unbiased state probabilities from biased samples.

    Parameters
    ----------
    states
        Encoded state trajectory.
    bias_values
        Total bias value acting on each sampled configuration, recorded before
        the next bias deposition/update.
    beta
        Inverse thermal energy.
    n_states
        Total number of encoded states.
    """
    states_arr = np.asarray(states)
    bias_arr = np.asarray(bias_values, dtype=float)
    if states_arr.shape != bias_arr.shape:
        raise ValueError("states and bias_values must have the same shape.")
    weights = reweight_from_bias(bias_arr, beta=beta)
    return state_probabilities(states_arr, n_states=n_states, weights=weights)


def reweighted_free_energy_differences(
    states: ArrayLike,
    bias_values: ArrayLike,
    *,
    beta: float = 1.0,
    n_states: int | None = None,
    reference: int | None = None,
) -> FloatArray:
    """
    Compute free-energy differences from biased samples by reweighting.

    This returns ``F[i] - F[reference]`` for all encoded states. The bias values
    must be the total bias applied to each recorded configuration.
    """
    probabilities = reweighted_state_probabilities(
        states,
        bias_values,
        beta=beta,
        n_states=n_states,
    )
    return free_energy_differences(probabilities, beta=beta, reference=reference)
