"""Bias objects for MetaTally samplers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]
IntegerArray = NDArray[np.integer]


def _as_integer_indices(xi: int | ArrayLike, *, n_states: int) -> int | IntegerArray:
    """Validate encoded state indices."""
    arr = np.asarray(xi)
    if not np.issubdtype(arr.dtype, np.integer):
        if np.all(np.equal(arr, np.asarray(arr, dtype=np.int64))):
            arr = arr.astype(np.int64)
        else:
            raise ValueError("xi must contain integer state indices.")
    else:
        arr = arr.astype(np.int64, copy=False)

    if np.any(arr < 0) or np.any(arr >= n_states):
        raise ValueError(f"xi out of range. Expected 0 <= xi < {n_states}.")

    if arr.ndim == 0:
        return int(arr)
    return arr


def _validate_values(values: ArrayLike, shape: tuple[int, ...]) -> FloatArray:
    """Return a finite float array with the requested shape."""
    arr = np.asarray(values, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"values must have shape {shape}, got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite.")
    return arr


@runtime_checkable
class Bias(Protocol):
    """Protocol for bias objects used by samplers."""

    def value(self, xi: int | ArrayLike, x: ArrayLike | None = None) -> float | FloatArray:
        """Return the bias value for one or more states/configurations."""
        ...

    def update(self, xi: int | ArrayLike, x: ArrayLike | None = None) -> None:
        """Update the bias after visiting one or more states/configurations."""
        ...

    def reset(self) -> None:
        """Reset the bias to its initial state."""
        ...

    def freeze(self) -> None:
        """Disable future bias updates while keeping current bias values."""
        ...

    def unfreeze(self) -> None:
        """Re-enable future bias updates."""
        ...

    @property
    def is_frozen(self) -> bool:
        """Whether updates are disabled."""
        ...


@dataclass
class NoBias:
    """Null bias with the same interface as active bias objects."""

    _frozen: bool = field(default=False, init=False, repr=False)

    @property
    def is_frozen(self) -> bool:
        """Whether the null bias is marked frozen."""
        return self._frozen

    def freeze(self) -> None:
        """Mark the null bias as frozen."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Mark the null bias as unfrozen."""
        self._frozen = False

    def value(self, xi: int | ArrayLike, x: ArrayLike | None = None) -> float | FloatArray:
        """Return zero for one or more states."""
        arr = np.asarray(xi)
        if arr.ndim == 0:
            return 0.0
        return np.zeros_like(arr, dtype=float)

    def update(self, xi: int | ArrayLike, x: ArrayLike | None = None) -> None:
        """Do nothing."""
        return None

    def reset(self) -> None:
        """Do nothing."""
        return None


@dataclass
class CompositeBias:
    """Sum several bias objects and update them together.

    Examples
    --------
    >>> import metatally as mt
    >>> space = mt.TorsionChain(n_variables=2).state_space()
    >>> bias = mt.CompositeBias([
    ...     mt.DiscreteStateBias.from_space(space, height=0.1),
    ...     mt.IndependentTorsionBias.from_space(space, height=0.01),
    ... ])
    >>> bias.value(0, x=[0.0, 0.0])
    0.0
    """

    biases: Sequence[Bias]
    _frozen: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate bias list."""
        biases = tuple(self.biases)
        if len(biases) == 0:
            raise ValueError("CompositeBias requires at least one component bias.")
        object.__setattr__(self, "biases", biases)

    @property
    def is_frozen(self) -> bool:
        """Whether the composite bias is frozen."""
        return self._frozen

    def freeze(self) -> None:
        """Freeze the composite and all component biases."""
        self._frozen = True
        for bias in self.biases:
            bias.freeze()

    def unfreeze(self) -> None:
        """Unfreeze the composite and all component biases."""
        self._frozen = False
        for bias in self.biases:
            bias.unfreeze()

    def value(self, xi: int | ArrayLike, x: ArrayLike | None = None) -> float | FloatArray:
        """Return the sum of all component bias values."""
        out = None
        for bias in self.biases:
            val = bias.value(xi, x=x)
            out = val if out is None else out + val
        return out

    def update(self, xi: int | ArrayLike, x: ArrayLike | None = None) -> None:
        """Update all component biases unless the composite is frozen."""
        if self._frozen:
            return None
        for bias in self.biases:
            bias.update(xi, x=x)
        return None

    def reset(self) -> None:
        """Reset all component biases."""
        for bias in self.biases:
            bias.reset()


class DiscreteStateBias:
    """History-dependent delta bias over encoded global states.

    ``values`` is the live mutable bias array. Both assignment and in-place
    modification update the array used by ``value()`` and ``update()``.
    """

    def __init__(
        self,
        n_states: int,
        height: float = 1.0,
        values: ArrayLike | None = None,
    ) -> None:
        """Initialise a discrete state bias."""
        n_states = int(n_states)
        if n_states < 1:
            raise ValueError("n_states must be >= 1.")
        height = float(height)
        if not np.isfinite(height):
            raise ValueError("height must be finite.")
        self.n_states = n_states
        self.height = height
        self._values = np.zeros(n_states, dtype=float)
        self._frozen = False
        if values is not None:
            self.values = values

    @classmethod
    def from_space(
        cls,
        space: object,
        *,
        height: float = 1.0,
        values: ArrayLike | None = None,
    ) -> "DiscreteStateBias":
        """Create a bias using ``space.n_states``."""
        return cls(n_states=int(space.n_states), height=height, values=values)

    @property
    def is_frozen(self) -> bool:
        """Whether updates are disabled."""
        return self._frozen

    def freeze(self) -> None:
        """Disable future updates while keeping the current bias active."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Re-enable updates."""
        self._frozen = False

    @property
    def values(self) -> FloatArray:
        """Mutable bias array of shape ``(n_states,)``."""
        return self._values

    @values.setter
    def values(self, new_values: ArrayLike) -> None:
        arr = _validate_values(new_values, (self.n_states,))
        self._values[...] = arr

    @property
    def array(self) -> FloatArray:
        """Alias for ``values``."""
        return self._values

    @array.setter
    def array(self, new_values: ArrayLike) -> None:
        self.values = new_values

    def value(self, xi: int | ArrayLike, x: ArrayLike | None = None) -> float | FloatArray:
        """Return the bias value for one or more encoded states."""
        idx = _as_integer_indices(xi, n_states=self.n_states)
        out = self._values[idx]
        if np.asarray(out).ndim == 0:
            return float(out)
        return out

    def update(self, xi: int | ArrayLike, x: ArrayLike | None = None) -> None:
        """Increment the bias at one or more states unless frozen."""
        if self._frozen:
            return None
        idx = _as_integer_indices(xi, n_states=self.n_states)
        np.add.at(self._values, idx, self.height)
        return None

    def reset(self) -> None:
        """Set all bias values to zero."""
        self._values.fill(0.0)


class WellTemperedDiscreteStateBias(DiscreteStateBias):
    """Well-tempered discrete state-counting bias.

    On visiting state ``xi``, the deposited increment is reduced as the
    accumulated bias grows::

        height * exp(-V[xi] / ((bias_factor - 1) / beta))

    This is the discrete-state analogue of well-tempered metadynamics.
    """

    def __init__(
        self,
        n_states: int,
        height: float = 1.0,
        bias_factor: float = 10.0,
        beta: float = 1.0,
        values: ArrayLike | None = None,
    ) -> None:
        """Initialise a well-tempered discrete state bias."""
        super().__init__(n_states=n_states, height=height, values=values)
        bias_factor = float(bias_factor)
        beta = float(beta)
        if bias_factor <= 1.0 or not np.isfinite(bias_factor):
            raise ValueError("bias_factor must be finite and > 1.")
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError("beta must be finite and > 0.")
        self.bias_factor = bias_factor
        self.beta = beta

    @classmethod
    def from_space(
        cls,
        space: object,
        *,
        height: float = 1.0,
        bias_factor: float = 10.0,
        beta: float = 1.0,
        values: ArrayLike | None = None,
    ) -> "WellTemperedDiscreteStateBias":
        """Create a well-tempered bias using ``space.n_states``."""
        return cls(
            n_states=int(space.n_states),
            height=height,
            bias_factor=bias_factor,
            beta=beta,
            values=values,
        )

    @property
    def bias_temperature(self) -> float:
        """Return ``(bias_factor - 1) / beta`` in energy units."""
        return (self.bias_factor - 1.0) / self.beta

    def update(self, xi: int | ArrayLike, x: ArrayLike | None = None) -> None:
        """Deposit well-tempered increments unless frozen."""
        if self.is_frozen:
            return None
        idx = _as_integer_indices(xi, n_states=self.n_states)
        flat = np.ravel(np.asarray(idx, dtype=np.int64))
        for single_idx in flat:
            increment = self.height * np.exp(
                -self._values[int(single_idx)] / self.bias_temperature
            )
            self._values[int(single_idx)] += increment
        return None

    def free_energy_differences_from_bias(self, reference: int = 0) -> FloatArray:
        """Estimate ``F_i - F_reference`` from the current WT bias.

        Uses the asymptotic well-tempered relation
        ``F_i - F_ref ~= -gamma/(gamma-1) * (V_i - V_ref)``.
        """
        reference = int(reference)
        if reference < 0 or reference >= self.n_states:
            raise ValueError("reference index out of range.")
        scale = -self.bias_factor / (self.bias_factor - 1.0)
        return scale * (self.values - self.values[reference])


class IndependentTorsionBias:
    """Independent Gaussian torsional biases, not conditioned on ``xi``."""

    def __init__(
        self,
        n_variables: int,
        height: float = 1.0,
        sigma: float = 0.25,
        n_grid: int = 128,
        values: ArrayLike | None = None,
    ) -> None:
        """Initialise independent torsional grid biases."""
        n_variables = int(n_variables)
        n_grid = int(n_grid)
        height = float(height)
        sigma = float(sigma)
        if n_variables < 1:
            raise ValueError("n_variables must be >= 1.")
        if n_grid < 2:
            raise ValueError("n_grid must be >= 2.")
        if not np.isfinite(height):
            raise ValueError("height must be finite.")
        if sigma <= 0.0 or not np.isfinite(sigma):
            raise ValueError("sigma must be finite and > 0.")
        self.n_variables = n_variables
        self.height = height
        self.sigma = sigma
        self.n_grid = n_grid
        self._values = np.zeros((n_variables, n_grid), dtype=float)
        self._grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False, dtype=float)
        self._frozen = False
        if values is not None:
            self.values = values

    @classmethod
    def from_space(
        cls,
        space: object,
        *,
        height: float = 1.0,
        sigma: float = 0.25,
        n_grid: int = 128,
        values: ArrayLike | None = None,
    ) -> "IndependentTorsionBias":
        """Create an independent torsional bias using ``space.n_variables``."""
        return cls(
            n_variables=int(space.n_variables),
            height=height,
            sigma=sigma,
            n_grid=n_grid,
            values=values,
        )

    @property
    def is_frozen(self) -> bool:
        """Whether updates are disabled."""
        return self._frozen

    def freeze(self) -> None:
        """Disable future updates while keeping the current bias active."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Re-enable updates."""
        self._frozen = False

    @property
    def values(self) -> FloatArray:
        """Mutable bias grid of shape ``(n_variables, n_grid)``."""
        return self._values

    @values.setter
    def values(self, new_values: ArrayLike) -> None:
        arr = _validate_values(new_values, (self.n_variables, self.n_grid))
        self._values[...] = arr

    @property
    def array(self) -> FloatArray:
        """Alias for ``values``."""
        return self._values

    @array.setter
    def array(self, new_values: ArrayLike) -> None:
        self.values = new_values

    @property
    def grid(self) -> FloatArray:
        """Torsion grid points in radians."""
        return self._grid

    @staticmethod
    def wrap_angle(angle: ArrayLike) -> FloatArray:
        """Wrap angles to ``[-pi, pi)``."""
        return ((np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi)) - np.pi

    def _validate_x(self, x: ArrayLike | None) -> FloatArray:
        """Validate one torsional configuration."""
        if x is None:
            raise ValueError("x must be provided for IndependentTorsionBias.")
        arr = np.asarray(x, dtype=float)
        if arr.shape != (self.n_variables,):
            raise ValueError(f"Expected x with shape ({self.n_variables},), got {arr.shape}.")
        return self.wrap_angle(arr)

    def _grid_indices(self, x: FloatArray) -> IntegerArray:
        """Return nearest lower grid indices for one torsional configuration."""
        frac = (x + np.pi) / (2.0 * np.pi)
        return np.floor(frac * self.n_grid).astype(np.int64) % self.n_grid

    def value(self, xi: int | ArrayLike, x: ArrayLike | None = None) -> float | FloatArray:
        """Return the independent torsional bias for one configuration."""
        phi = self._validate_x(x)
        grid_idx = self._grid_indices(phi)
        return float(np.sum(self._values[np.arange(self.n_variables), grid_idx]))

    def update(self, xi: int | ArrayLike, x: ArrayLike | None = None) -> None:
        """Deposit periodic Gaussians for all torsions unless frozen."""
        if self._frozen:
            return None
        phi = self._validate_x(x)
        for n, angle in enumerate(phi):
            d = self.wrap_angle(self._grid - angle)
            self._values[n, :] += self.height * np.exp(-0.5 * (d / self.sigma) ** 2)
        return None

    def reset(self) -> None:
        """Set all bias values to zero."""
        self._values.fill(0.0)


class StateConditionedTorsionBias:
    """Gaussian torsional bias conditioned on the global state ``xi``."""

    def __init__(
        self,
        n_states: int,
        n_variables: int,
        height: float = 1.0,
        sigma: float = 0.25,
        n_grid: int = 128,
        values: ArrayLike | None = None,
    ) -> None:
        """Initialise state-conditioned torsional grid biases."""
        n_states = int(n_states)
        n_variables = int(n_variables)
        n_grid = int(n_grid)
        height = float(height)
        sigma = float(sigma)
        if n_states < 1:
            raise ValueError("n_states must be >= 1.")
        if n_variables < 1:
            raise ValueError("n_variables must be >= 1.")
        if n_grid < 2:
            raise ValueError("n_grid must be >= 2.")
        if not np.isfinite(height):
            raise ValueError("height must be finite.")
        if sigma <= 0.0 or not np.isfinite(sigma):
            raise ValueError("sigma must be finite and > 0.")
        self.n_states = n_states
        self.n_variables = n_variables
        self.height = height
        self.sigma = sigma
        self.n_grid = n_grid
        self._values = np.zeros((n_states, n_variables, n_grid), dtype=float)
        self._grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False, dtype=float)
        self._frozen = False
        if values is not None:
            self.values = values

    @classmethod
    def from_space(
        cls,
        space: object,
        *,
        height: float = 1.0,
        sigma: float = 0.25,
        n_grid: int = 128,
        values: ArrayLike | None = None,
    ) -> "StateConditionedTorsionBias":
        """Create a torsional Gaussian bias using state-space dimensions."""
        return cls(
            n_states=int(space.n_states),
            n_variables=int(space.n_variables),
            height=height,
            sigma=sigma,
            n_grid=n_grid,
            values=values,
        )

    @property
    def is_frozen(self) -> bool:
        """Whether updates are disabled."""
        return self._frozen

    def freeze(self) -> None:
        """Disable future updates while keeping the current bias active."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Re-enable updates."""
        self._frozen = False

    @property
    def values(self) -> FloatArray:
        """Mutable bias grid with shape ``(n_states, n_variables, n_grid)``."""
        return self._values

    @values.setter
    def values(self, new_values: ArrayLike) -> None:
        arr = _validate_values(new_values, (self.n_states, self.n_variables, self.n_grid))
        self._values[...] = arr

    @property
    def array(self) -> FloatArray:
        """Alias for ``values``."""
        return self._values

    @array.setter
    def array(self, new_values: ArrayLike) -> None:
        self.values = new_values

    @property
    def grid(self) -> FloatArray:
        """Torsion grid points in radians."""
        return self._grid

    @staticmethod
    def wrap_angle(angle: ArrayLike) -> FloatArray:
        """Wrap angles to ``[-pi, pi)``."""
        return ((np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi)) - np.pi

    def _validate_x(self, x: ArrayLike | None) -> FloatArray:
        """Validate one torsional configuration."""
        if x is None:
            raise ValueError("x must be provided for StateConditionedTorsionBias.")
        arr = np.asarray(x, dtype=float)
        if arr.shape != (self.n_variables,):
            raise ValueError(f"Expected x with shape ({self.n_variables},), got {arr.shape}.")
        return self.wrap_angle(arr)

    def _grid_indices(self, x: FloatArray) -> IntegerArray:
        """Return nearest lower grid indices for one torsional configuration."""
        frac = (x + np.pi) / (2.0 * np.pi)
        return np.floor(frac * self.n_grid).astype(np.int64) % self.n_grid

    def value(self, xi: int | ArrayLike, x: ArrayLike | None = None) -> float | FloatArray:
        """Return the state-conditioned torsional bias for one configuration."""
        idx = _as_integer_indices(xi, n_states=self.n_states)
        if not isinstance(idx, int):
            raise ValueError("StateConditionedTorsionBias currently expects scalar xi.")
        phi = self._validate_x(x)
        grid_idx = self._grid_indices(phi)
        return float(np.sum(self._values[idx, np.arange(self.n_variables), grid_idx]))

    def update(self, xi: int | ArrayLike, x: ArrayLike | None = None) -> None:
        """Deposit periodic Gaussians for all torsions unless frozen."""
        if self._frozen:
            return None
        idx = _as_integer_indices(xi, n_states=self.n_states)
        if not isinstance(idx, int):
            for single_xi, single_x in zip(np.ravel(idx), np.asarray(x), strict=True):
                self.update(int(single_xi), single_x)
            return None
        phi = self._validate_x(x)
        for n, angle in enumerate(phi):
            d = self.wrap_angle(self._grid - angle)
            self._values[idx, n, :] += self.height * np.exp(-0.5 * (d / self.sigma) ** 2)
        return None

    def reset(self) -> None:
        """Set all bias values to zero."""
        self._values.fill(0.0)
