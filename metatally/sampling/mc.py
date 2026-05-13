"""Stateful Metropolis Monte Carlo sampler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from metatally.core.bias import Bias, NoBias
from metatally.core.states import StateSpace
from metatally.sampling.moves import Move
try: 
    from tqdm.auto import tqdm
    progress_bar_available = True
except(ModuleNotFoundError,ImportError): 
    progress_bar_available = False

FloatArray = NDArray[np.floating]
IntegerArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]


@runtime_checkable
class EnergyModel(Protocol):
    """Protocol for models that provide an energy function."""

    def energy(self, x: ArrayLike) -> float:
        """Return the model energy for a configuration."""
        ...

    def initial_state(self) -> FloatArray:
        """Return a default initial configuration."""
        ...

    def default_move(self, step_size: float = 0.5) -> Move:
        """Return the default proposal move for the model."""
        ...


@dataclass(frozen=True)
class SamplingResult:
    """Snapshot of a sampler trajectory and accumulated statistics."""

    states: IntegerArray
    energies: FloatArray
    accepted: BoolArray
    visits: IntegerArray
    bias_values: FloatArray | None = None

    def __post_init__(self) -> None:
        """Fill missing bias values for backwards-compatible construction."""
        if self.bias_values is None:
            object.__setattr__(
                self,
                "bias_values",
                np.zeros_like(self.energies, dtype=float),
            )

    @property
    def n_steps(self) -> int:
        """Number of recorded MC steps."""
        return int(self.states.size)

    @property
    def steps(self) -> IntegerArray:
        """One-based MC step numbers."""
        return np.arange(1, self.n_steps + 1, dtype=np.int64)

    @property
    def n_visited(self) -> int:
        """Number of unique global states visited."""
        return int(np.count_nonzero(self.visits))

    @property
    def coverage(self) -> float:
        """Fraction of global states visited at least once."""
        if self.visits.size == 0:
            return 0.0
        return self.n_visited / int(self.visits.size)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of accepted MC moves."""
        if self.accepted.size == 0:
            return 0.0
        return float(np.mean(self.accepted))

    @property
    def n_unique_by_step(self) -> IntegerArray:
        """Cumulative number of unique states visited at each step."""
        seen: set[int] = set()
        out = np.empty(self.states.size, dtype=np.int64)
        for i, xi in enumerate(self.states):
            seen.add(int(xi))
            out[i] = len(seen)
        return out

    def cumulative_unique(self) -> IntegerArray:
        """Alias for ``n_unique_by_step``."""
        return self.n_unique_by_step


@dataclass
class MetropolisSampler:
    """Stateful Metropolis sampler with optional history-dependent bias."""

    model: EnergyModel
    state_space: StateSpace
    initial: ArrayLike | None = None
    bias: Bias | None = None
    beta: float = 1.0
    move: Move | None = None
    step_size: float = 0.5
    seed: int | None = None
    rng: np.random.Generator = field(init=False, repr=False)
    x: FloatArray = field(init=False)
    energy: float = field(init=False)
    xi: int = field(init=False)
    _states: list[int] = field(init=False, repr=False)
    _energies: list[float] = field(init=False, repr=False)
    _accepted: list[bool] = field(init=False, repr=False)
    _bias_values: list[float] = field(init=False, repr=False)
    visits: IntegerArray = field(init=False)

    def __post_init__(self) -> None:
        """Initialise RNG, current state, move, bias, and statistics."""
        beta = float(self.beta)
        if beta <= 0.0 or not np.isfinite(beta):
            raise ValueError("beta must be finite and > 0.")
        self.beta = beta
        self.rng = np.random.default_rng(self.seed)
        if self.bias is None:
            self.bias = NoBias()
        if self.move is None:
            self.move = self.model.default_move(step_size=self.step_size)
        self.reset(initial=self.initial, reset_bias=False)

    @property
    def n_steps(self) -> int:
        """Number of completed MC steps."""
        return len(self._states)

    @property
    def states(self) -> IntegerArray:
        """Encoded global-state trajectory."""
        return np.asarray(self._states, dtype=np.int64)

    @property
    def energies(self) -> FloatArray:
        """Physical energy trajectory after each MC step."""
        return np.asarray(self._energies, dtype=float)

    @property
    def accepted(self) -> BoolArray:
        """Boolean acceptance trajectory."""
        return np.asarray(self._accepted, dtype=bool)

    @property
    def bias_values(self) -> FloatArray:
        """Total bias value recorded before each bias update."""
        return np.asarray(self._bias_values, dtype=float)

    @property
    def steps(self) -> IntegerArray:
        """One-based MC step numbers."""
        return np.arange(1, self.n_steps + 1, dtype=np.int64)

    @property
    def n_visited(self) -> int:
        """Number of unique global states visited."""
        return int(np.count_nonzero(self.visits))

    @property
    def coverage(self) -> float:
        """Fraction of global states visited at least once."""
        return self.n_visited / int(self.state_space.n_states)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of accepted MC moves."""
        if self.n_steps == 0:
            return 0.0
        return float(np.mean(self.accepted))

    @property
    def n_unique_by_step(self) -> IntegerArray:
        """Cumulative number of unique states visited at each MC step."""
        return self.snapshot().n_unique_by_step

    def cumulative_unique(self) -> IntegerArray:
        """Alias for ``n_unique_by_step``."""
        return self.n_unique_by_step

    def _effective_energy(self, x: ArrayLike, energy: float, xi: int) -> float:
        """Return physical plus bias energy for one state."""
        return float(energy + self.bias.value(xi, x=x))

    def step(self) -> bool:
        """Perform one Metropolis step and update live statistics."""
        x_prop = self.move.propose(self.x, self.rng)
        energy_prop = float(self.model.energy(x_prop))
        xi_prop = int(self.state_space.encode(x_prop))

        current_eff = self._effective_energy(self.x, self.energy, self.xi)
        prop_eff = self._effective_energy(x_prop, energy_prop, xi_prop)
        delta = prop_eff - current_eff
        accept = bool(delta <= 0.0 or self.rng.random() < np.exp(-self.beta * delta))

        if accept:
            self.x = np.asarray(x_prop, dtype=float).copy()
            self.energy = energy_prop
            self.xi = xi_prop

        current_bias = float(self.bias.value(self.xi, x=self.x))
        self.bias.update(self.xi, x=self.x)
        self._record(accept, current_bias)
        return accept

    def _record(self, accepted: bool, bias_value: float) -> None:
        """Append the current state to trajectory statistics."""
        self._states.append(int(self.xi))
        self._energies.append(float(self.energy))
        self._accepted.append(bool(accepted))
        self._bias_values.append(float(bias_value))
        self.visits[int(self.xi)] += 1

    def run(self, n_steps: int, progress: bool) -> "MetropolisSampler":
        """Run ``n_steps`` additional MC steps and return ``self``."""
        visited,old=0,0
        n_steps = int(n_steps)
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")

        if progress and progress_bar_available: 
            iterable = tqdm(range(n_steps))
        else: 
            iterable = range(nsteps)

        for _ in iterable: 
            self.step()
            if progress:
                visited = self.n_visited
                if visited > old: 
                    old = visited
                    fraction = 100*visited/self.state_space.n_states
                    iterable.set_postfix_str(f"visited: {visited} / {self.state_space.n_states} = {fraction:3.3f}%")
        return self

    def reset(
        self,
        initial: ArrayLike | None = None,
        *,
        reset_bias: bool = False,
    ) -> None:
        """Reset current state and accumulated sampler statistics."""
        if initial is None:
            initial = self.model.initial_state()
        self.x = np.asarray(initial, dtype=float).copy()
        self.energy = float(self.model.energy(self.x))
        self.xi = int(self.state_space.encode(self.x))
        self._states = []
        self._energies = []
        self._accepted = []
        self._bias_values = []
        self.visits = np.zeros(int(self.state_space.n_states), dtype=np.int64)
        if reset_bias:
            self.bias.reset()

    def snapshot(self) -> SamplingResult:
        """Return an immutable copy of current trajectory statistics."""
        return SamplingResult(
            states=self.states.copy(),
            energies=self.energies.copy(),
            accepted=self.accepted.copy(),
            visits=self.visits.copy(),
            bias_values=self.bias_values.copy(),
        )
