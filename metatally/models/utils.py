"""Model utilities."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]


class EnergyModel(Protocol):
    """Protocol for objects exposing an energy function."""

    def energy(self, x: ArrayLike) -> float:
        """Return the energy of configuration ``x``."""
        ...


def scan_coordinate(
    model: EnergyModel,
    variable: int,
    background: ArrayLike,
    values: ArrayLike | None = None,
    *,
    n_points: int = 512,
    bounds: tuple[float, float] = (-np.pi, np.pi),
    subtract_min: bool = True,
) -> tuple[FloatArray, FloatArray]:
    """Scan a model potential along one coordinate."""
    background_arr = np.asarray(background, dtype=float)
    if background_arr.ndim != 1:
        raise ValueError("background must be a one-dimensional configuration.")
    variable = int(variable)
    if variable < 0 or variable >= background_arr.size:
        raise ValueError(f"variable must satisfy 0 <= variable < {background_arr.size}.")
    if values is None:
        if n_points < 2:
            raise ValueError("n_points must be >= 2.")
        if len(bounds) != 2:
            raise ValueError("bounds must contain exactly two values.")
        low, high = float(bounds[0]), float(bounds[1])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("bounds must be finite and increasing.")
        values_arr = np.linspace(low, high, n_points, endpoint=False, dtype=float)
    else:
        values_arr = np.asarray(values, dtype=float)
        if values_arr.ndim != 1:
            raise ValueError("values must be a one-dimensional array.")
        if values_arr.size == 0:
            raise ValueError("values must contain at least one point.")
        if not np.all(np.isfinite(values_arr)):
            raise ValueError("values must be finite.")
    energies = np.empty(values_arr.size, dtype=float)
    for i, value in enumerate(values_arr):
        x = background_arr.copy()
        x[variable] = value
        energies[i] = float(model.energy(x))
    if subtract_min:
        energies = energies - np.min(energies)
    return values_arr, energies
