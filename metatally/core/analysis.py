"""Analysis utilities for state probabilities and free energies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]


def state_probabilities(visits: ArrayLike) -> FloatArray:
    """Convert state visit counts to probabilities."""
    visits_arr = np.asarray(visits, dtype=float)
    if visits_arr.ndim != 1:
        raise ValueError("visits must be a one-dimensional array.")
    if np.any(visits_arr < 0.0):
        raise ValueError("visits must be non-negative.")
    total = np.sum(visits_arr)
    if total <= 0.0:
        return np.full(visits_arr.shape, np.nan, dtype=float)
    return visits_arr / total


def free_energy_differences(
    probabilities: ArrayLike,
    *,
    reference: int = 0,
    beta: float = 1.0,
) -> FloatArray:
    """Compute ``F_i - F_reference`` from state probabilities.

    States with zero probability are returned as ``NaN``. If the reference
    state has zero probability, all differences are ``NaN``.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1:
        raise ValueError("probabilities must be a one-dimensional array.")
    if beta <= 0.0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and > 0.")
    reference = int(reference)
    if reference < 0 or reference >= p.size:
        raise ValueError("reference index out of range.")
    if np.any(p < 0.0):
        raise ValueError("probabilities must be non-negative.")
    out = np.full(p.shape, np.nan, dtype=float)
    p_ref = p[reference]
    if not np.isfinite(p_ref) or p_ref <= 0.0:
        return out
    mask = np.isfinite(p) & (p > 0.0)
    out[mask] = -(1.0 / beta) * np.log(p[mask] / p_ref)
    out[reference] = 0.0
    return out


def free_energy_differences_from_visits(
    visits: ArrayLike,
    *,
    reference: int = 0,
    beta: float = 1.0,
) -> FloatArray:
    """Compute free-energy differences from unbiased visit counts."""
    return free_energy_differences(
        state_probabilities(visits),
        reference=reference,
        beta=beta,
    )


def _logsumexp(values: FloatArray) -> float:
    """Small local log-sum-exp helper."""
    if values.size == 0:
        return -np.inf
    vmax = np.max(values)
    if not np.isfinite(vmax):
        return float(vmax)
    return float(vmax + np.log(np.sum(np.exp(values - vmax))))


def reweighted_state_probabilities(
    states: ArrayLike,
    bias_values: ArrayLike,
    *,
    beta: float = 1.0,
    n_states: int | None = None,
) -> FloatArray:
    """Estimate state probabilities from a fixed-bias trajectory.

    The weights are ``exp(beta * bias_values)``. For strongly time-dependent
    adaptive biases this is only a diagnostic; use frozen-bias production for
    quantitative estimates.
    """
    states_arr = np.asarray(states, dtype=np.int64)
    bias_arr = np.asarray(bias_values, dtype=float)
    if states_arr.ndim != 1:
        raise ValueError("states must be a one-dimensional array.")
    if bias_arr.shape != states_arr.shape:
        raise ValueError("bias_values must have the same shape as states.")
    if beta <= 0.0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and > 0.")
    if states_arr.size == 0:
        if n_states is None:
            raise ValueError("n_states is required for an empty trajectory.")
        return np.full(int(n_states), np.nan, dtype=float)
    if np.any(states_arr < 0):
        raise ValueError("states must be non-negative.")
    if n_states is None:
        n_states = int(states_arr.max()) + 1
    n_states = int(n_states)
    if n_states < 1:
        raise ValueError("n_states must be >= 1.")
    if np.any(states_arr >= n_states):
        raise ValueError("states contains values >= n_states.")

    logw = beta * bias_arr
    log_counts = np.full(n_states, -np.inf, dtype=float)
    for state in range(n_states):
        log_counts[state] = _logsumexp(logw[states_arr == state])
    log_total = _logsumexp(logw)
    if not np.isfinite(log_total):
        return np.full(n_states, np.nan, dtype=float)
    probs = np.exp(log_counts - log_total)
    probs[~np.isfinite(probs)] = 0.0
    return probs


def reweighted_free_energy_differences(
    states: ArrayLike,
    bias_values: ArrayLike,
    *,
    reference: int = 0,
    beta: float = 1.0,
    n_states: int | None = None,
) -> FloatArray:
    """Compute free-energy differences from a fixed-bias trajectory."""
    probs = reweighted_state_probabilities(
        states,
        bias_values,
        beta=beta,
        n_states=n_states,
    )
    return free_energy_differences(probs, reference=reference, beta=beta)


@dataclass(frozen=True)
class ReweightingDiagnostics:
    """Diagnostics for exponential reweighting."""

    log_weight_min: float
    log_weight_max: float
    log_weight_range: float
    effective_sample_size: float
    effective_sample_fraction: float


def reweighting_diagnostics(bias_values: ArrayLike, *, beta: float = 1.0) -> ReweightingDiagnostics:
    """Return log-weight range and effective sample size diagnostics."""
    bias_arr = np.asarray(bias_values, dtype=float)
    if bias_arr.ndim != 1:
        raise ValueError("bias_values must be a one-dimensional array.")
    if beta <= 0.0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and > 0.")
    if bias_arr.size == 0:
        return ReweightingDiagnostics(np.nan, np.nan, np.nan, 0.0, 0.0)
    logw = beta * bias_arr
    shifted = logw - np.max(logw)
    w = np.exp(shifted)
    ess = float(np.sum(w) ** 2 / np.sum(w**2))
    return ReweightingDiagnostics(
        log_weight_min=float(np.min(logw)),
        log_weight_max=float(np.max(logw)),
        log_weight_range=float(np.max(logw) - np.min(logw)),
        effective_sample_size=ess,
        effective_sample_fraction=ess / float(bias_arr.size),
    )
