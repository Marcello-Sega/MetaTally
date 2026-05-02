"""Proposal moves for Monte Carlo samplers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]


def wrap_periodic(x: ArrayLike) -> FloatArray:
    """Wrap angles to ``[-pi, pi)``."""
    return ((np.asarray(x, dtype=float) + np.pi) % (2.0 * np.pi)) - np.pi


@runtime_checkable
class Move(Protocol):
    """Protocol for proposal moves."""

    def propose(self, x: ArrayLike, rng: np.random.Generator) -> FloatArray:
        """Return a proposed configuration."""
        ...


@dataclass(frozen=True)
class SingleVariableGaussianMove:
    """Gaussian move applied to one randomly chosen coordinate."""

    step_size: float = 0.5

    def __post_init__(self) -> None:
        """Validate the proposal width."""
        if self.step_size <= 0.0 or not np.isfinite(self.step_size):
            raise ValueError("step_size must be finite and > 0.")

    def propose(self, x: ArrayLike, rng: np.random.Generator) -> FloatArray:
        """Return a proposal with one coordinate displaced."""
        out = np.asarray(x, dtype=float).copy()
        if out.ndim != 1:
            raise ValueError("x must be a one-dimensional configuration.")
        i = int(rng.integers(out.size))
        out[i] += rng.normal(scale=self.step_size)
        return out


@dataclass(frozen=True)
class SingleVariablePeriodicGaussianMove:
    """Gaussian move on one randomly chosen periodic coordinate."""

    step_size: float = 0.5

    def __post_init__(self) -> None:
        """Validate the proposal width."""
        if self.step_size <= 0.0 or not np.isfinite(self.step_size):
            raise ValueError("step_size must be finite and > 0.")

    def propose(self, x: ArrayLike, rng: np.random.Generator) -> FloatArray:
        """Return a wrapped proposal with one coordinate displaced."""
        out = np.asarray(x, dtype=float).copy()
        if out.ndim != 1:
            raise ValueError("x must be a one-dimensional configuration.")
        i = int(rng.integers(out.size))
        out[i] += rng.normal(scale=self.step_size)
        return wrap_periodic(out)
