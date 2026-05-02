"""Simple torsion-chain model used in examples and tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from metatally.core.encoding import encoder
from metatally.core.states import StateSpace, TorsionStateAssigner
from metatally.sampling.moves import SingleVariablePeriodicGaussianMove, wrap_periodic

FloatArray = NDArray[np.floating]


def torsion_centers(radix: int) -> FloatArray:
    """Return torsion-state centres at minima of ``1 - cos(radix * phi)``."""
    radix = int(radix)
    if radix < 2:
        raise ValueError("radix must be >= 2.")
    centers = 2.0 * np.pi * np.arange(radix, dtype=float) / radix
    centers = wrap_periodic(centers)
    return np.sort(centers)


def _coupling_array(coupling: float | ArrayLike, n_variables: int) -> FloatArray:
    """Validate and return nearest-neighbour coupling constants."""
    expected_shape = (max(int(n_variables) - 1, 0),)
    arr = np.asarray(coupling, dtype=float)
    if arr.ndim == 0:
        value = float(arr)
        if not np.isfinite(value):
            raise ValueError("coupling must contain finite values.")
        return np.full(expected_shape, value, dtype=float)
    if arr.shape != expected_shape:
        raise ValueError(
            "coupling must be a scalar or an array with shape "
            f"{expected_shape}, got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("coupling must contain finite values.")
    return arr.astype(float, copy=True)


def _state_potential_array(
    state_potential: ArrayLike | None,
    n_variables: int,
    radix: int,
) -> FloatArray:
    """Validate and return discrete state-potential values."""
    n_states = int(radix) ** int(n_variables)
    if state_potential is None:
        return np.zeros(n_states, dtype=float)
    arr = np.asarray(state_potential, dtype=float)
    if arr.shape != (n_states,):
        raise ValueError(
            f"state_potential must have shape ({n_states},), got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("state_potential must contain finite values.")
    return arr.copy()


@dataclass(frozen=True)
class TorsionChain:
    """Periodic torsion chain with optional nearest-neighbour and state terms.

    The local energy is ``barrier * sum_i (1 - cos(radix * phi_i))``.
    The nearest-neighbour coupling is ``-sum_i J_i cos(phi[i+1] - phi[i])``.
    A discrete potential ``U_xi[xi]`` can also be added to test recovery of
    known state free energies.
    """

    n_variables: int = 3
    radix: int = 3
    barrier: float = 5.0
    coupling: float | ArrayLike = 0.0
    state_potential: ArrayLike | None = None
    _coupling: FloatArray = field(init=False, repr=False)
    _state_potential: FloatArray = field(init=False, repr=False)
    _state_space: StateSpace = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate model parameters."""
        n_variables = int(self.n_variables)
        radix = int(self.radix)
        barrier = float(self.barrier)
        if n_variables < 1:
            raise ValueError("n_variables must be >= 1.")
        if radix < 2:
            raise ValueError("radix must be >= 2.")
        if not np.isfinite(barrier) or barrier < 0.0:
            raise ValueError("barrier must be finite and >= 0.")
        coupling_arr = _coupling_array(self.coupling, n_variables)
        state_pot = _state_potential_array(self.state_potential, n_variables, radix)
        enc = encoder(n_variables=n_variables, radix=radix)
        assigner = TorsionStateAssigner(centers=torsion_centers(radix), n_variables=n_variables)
        space = StateSpace(assigner=assigner, encoder=enc)

        object.__setattr__(self, "n_variables", n_variables)
        object.__setattr__(self, "radix", radix)
        object.__setattr__(self, "barrier", barrier)
        object.__setattr__(self, "coupling", coupling_arr)
        object.__setattr__(self, "state_potential", state_pot)
        object.__setattr__(self, "_coupling", coupling_arr)
        object.__setattr__(self, "_state_potential", state_pot)
        object.__setattr__(self, "_state_space", space)

    @property
    def coupling_array(self) -> FloatArray:
        """Nearest-neighbour coupling constants with shape ``(n_variables - 1,)``."""
        return self._coupling.copy()

    @property
    def state_potential_array(self) -> FloatArray:
        """Discrete potential values ``U_xi`` with shape ``(n_states,)``."""
        return self._state_potential.copy()

    def state_free_energy_differences(self, reference: int = 0) -> FloatArray:
        """Known ``U_xi - U_reference`` values for symmetric uncoupled wells."""
        reference = int(reference)
        if reference < 0 or reference >= self._state_potential.size:
            raise ValueError("reference index out of range.")
        return self._state_potential - self._state_potential[reference]

    def energy(self, x: ArrayLike) -> float:
        """Return the potential energy for one torsional configuration."""
        phi = np.asarray(x, dtype=float)
        if phi.shape != (self.n_variables,):
            raise ValueError(f"Expected x with shape ({self.n_variables},), got {phi.shape}.")
        local = self.barrier * np.sum(1.0 - np.cos(self.radix * phi))
        coupled = 0.0
        if self.n_variables > 1:
            dphi = phi[1:] - phi[:-1]
            coupled = -float(np.sum(self._coupling * np.cos(dphi)))
        xi = int(self._state_space.encode(phi))
        return float(local + coupled + self._state_potential[xi])

    def initial_state(self, state: int = 1) -> FloatArray:
        """Return an initial configuration at one torsional minimum."""
        centers = torsion_centers(self.radix)
        idx = int(state) % self.radix
        return np.full(self.n_variables, centers[idx], dtype=float)

    def state_space(self) -> StateSpace:
        """Return the torsional ``StateSpace`` for this model."""
        return self._state_space

    def default_move(self, step_size: float = 0.5) -> SingleVariablePeriodicGaussianMove:
        """Return the default local periodic Gaussian move."""
        return SingleVariablePeriodicGaussianMove(step_size=step_size)

    @staticmethod
    def wrap(x: ArrayLike) -> FloatArray:
        """Wrap torsion angles to ``[-pi, pi)``."""
        return wrap_periodic(x)
