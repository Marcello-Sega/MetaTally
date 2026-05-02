"""
State assignment utilities for MetaTally.

This module converts configurations into local discrete state labels, and then
into global MetaTally state indices using an encoder.

The public interface follows the same NumPy-like convention as
``metatally.core.encoding``:

- assigners return local-state digits with the local-variable axis last;
- ``StateSpace.encode(x)`` assigns local states and encodes them;
- ``StateSpace.decode(indices)`` decodes global indices back to local digits.

Examples
--------
>>> import metatally as mt
>>> space = mt.StateSpace.from_radix(n_variables=3, radix=2)
>>> space.encode([1, 0, 1])
5
>>> space.decode(5).tolist()
[1, 0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from metatally.core.encoding import IntegerArray, MixedRadixEncoder, encoder


FloatArray = NDArray[np.floating]


@runtime_checkable
class StateAssigner(Protocol):
    """
    Protocol for objects that assign configurations to local discrete states.

    An assigner converts a configuration ``x`` into local-state digits. The last
    axis of the returned array must have length ``n_variables``.
    """

    @property
    def radices(self) -> tuple[int, ...]:
        """Number of allowed states for each local variable."""
        ...

    @property
    def n_variables(self) -> int:
        """Number of local variables assigned by this object."""
        ...

    def assign(self, x: ArrayLike) -> IntegerArray:
        """
        Assign configurations to local-state digits.

        Parameters
        ----------
        x
            Configuration or array of configurations.

        Returns
        -------
        IntegerArray
            Local-state digits. The last axis is the local-variable axis.
        """
        ...


@dataclass(frozen=True)
class DiscreteStateAssigner:
    """
    State assigner for configurations that are already discrete.

    This assigner does not transform continuous coordinates into states. It
    only validates that the input is already a vector of local state labels.

    Parameters
    ----------
    radices
        Number of allowed states for each local variable. For example,
        ``radices=[2, 3, 4]`` means:

        - variable 0 can take states ``0`` or ``1``;
        - variable 1 can take states ``0``, ``1``, or ``2``;
        - variable 2 can take states ``0``, ``1``, ``2``, or ``3``.

    Examples
    --------
    A binary spin-like system with three local variables can be represented as
    three digits, each equal to ``0`` or ``1``:

    >>> import metatally as mt
    >>> assigner = mt.DiscreteStateAssigner([2, 2, 2])
    >>> assigner.assign([1, 0, 1]).tolist()
    [1, 0, 1]

    The assigner accepts arrays of configurations, treating the last axis as
    the local-variable axis:

    >>> assigner.assign([[0, 0, 0], [1, 0, 1]]).tolist()
    [[0, 0, 0], [1, 0, 1]]

    Invalid local states are rejected with a ``ValueError``.
    """

    radices: Sequence[int]

    def __post_init__(self) -> None:
        """Validate and canonicalise the radix sequence."""
        enc = MixedRadixEncoder(self.radices)
        object.__setattr__(self, "radices", enc.radices)

    @property
    def n_variables(self) -> int:
        """Number of local discrete variables."""
        return len(self.radices)

    def assign(self, x: ArrayLike) -> IntegerArray:
        """
        Return validated local-state digits.

        Parameters
        ----------
        x
            Array-like object whose last axis contains local-state digits.

        Returns
        -------
        IntegerArray
            Validated integer local-state digits.
        """
        enc = MixedRadixEncoder(self.radices)
        return enc.validate_digits(x)


@dataclass(frozen=True)
class ThresholdStateAssigner:
    """
    Assign scalar variables to states using thresholds.

    The same threshold set is applied independently to each local variable.
    With one threshold, the assigner is binary. With two thresholds, it is
    ternary, and so on.

    Parameters
    ----------
    thresholds
        Monotonically increasing threshold values. The number of states is
        ``len(thresholds) + 1``.
    n_variables
        Number of scalar variables to assign.

    Examples
    --------
    >>> import metatally as mt
    >>> assigner = mt.ThresholdStateAssigner([0.5], n_variables=3)
    >>> assigner.assign([0.1, 0.7, 0.2]).tolist()
    [0, 1, 0]

    The last axis is treated as the local-variable axis:

    >>> assigner.assign([[0.1, 0.7, 0.2], [0.8, 0.9, 0.0]]).tolist()
    [[0, 1, 0], [1, 1, 0]]
    """

    thresholds: Sequence[float]
    n_variables: int

    def __post_init__(self) -> None:
        """Validate thresholds and number of variables."""
        n_variables = int(self.n_variables)
        if n_variables < 1:
            raise ValueError("n_variables must be >= 1.")

        thresholds = tuple(float(t) for t in self.thresholds)
        if len(thresholds) == 0:
            raise ValueError("At least one threshold is required.")

        if any(b <= a for a, b in zip(thresholds[:-1], thresholds[1:])):
            raise ValueError("Thresholds must be strictly increasing.")

        object.__setattr__(self, "n_variables", n_variables)
        object.__setattr__(self, "thresholds", thresholds)

    @property
    def radices(self) -> tuple[int, ...]:
        """Number of states for each thresholded variable."""
        return (len(self.thresholds) + 1,) * self.n_variables

    def assign(self, x: ArrayLike) -> IntegerArray:
        """
        Assign scalar values to threshold bins.

        Parameters
        ----------
        x
            Array-like object whose last axis has length ``n_variables``.

        Returns
        -------
        IntegerArray
            Integer state labels with the same shape as ``x``.
        """
        arr = np.asarray(x, dtype=float)

        if arr.ndim == 0:
            raise ValueError(
                "Input must have at least one dimension with the last axis "
                f"of length {self.n_variables}."
            )

        if arr.shape[-1] != self.n_variables:
            raise ValueError(
                "Expected the last axis to have length "
                f"{self.n_variables}, got shape {arr.shape}."
            )

        return np.digitize(arr, self.thresholds).astype(np.int64)


@dataclass(frozen=True)
class TorsionStateAssigner:
    """
    Assign periodic angular variables to nearest torsional state centres.

    The same set of angular centres is applied independently to each torsion.
    Angles and centres are in radians. Periodic distances are used, so centres
    close to ``-pi`` and ``pi`` are treated as neighbouring points.

    Parameters
    ----------
    centers
        Angular state centres in radians.
    n_variables
        Number of angular variables to assign.

    Examples
    --------
    >>> import numpy as np
    >>> import metatally as mt
    >>> assigner = mt.TorsionStateAssigner(
    ...     centers=[-2*np.pi/3, 0.0, 2*np.pi/3],
    ...     n_variables=3,
    ... )
    >>> assigner.assign([0.1, 2.0, -2.0]).tolist()
    [1, 2, 0]
    """

    centers: Sequence[float]
    n_variables: int

    def __post_init__(self) -> None:
        """Validate angular centres and number of variables."""
        n_variables = int(self.n_variables)
        if n_variables < 1:
            raise ValueError("n_variables must be >= 1.")

        centers = tuple(float(c) for c in self.centers)
        if len(centers) < 2:
            raise ValueError("At least two torsional state centres are required.")

        object.__setattr__(self, "n_variables", n_variables)
        object.__setattr__(self, "centers", centers)

    @property
    def radices(self) -> tuple[int, ...]:
        """Number of torsional states for each angular variable."""
        return (len(self.centers),) * self.n_variables

    @staticmethod
    def wrap_angle(angle: ArrayLike) -> FloatArray:
        """
        Wrap angles to the interval ``[-pi, pi)``.

        Parameters
        ----------
        angle
            Angle or array of angles in radians.

        Returns
        -------
        FloatArray
            Wrapped angle array.
        """
        return ((np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi)) - np.pi

    def assign(self, x: ArrayLike) -> IntegerArray:
        """
        Assign each angle to the nearest periodic state centre.

        Parameters
        ----------
        x
            Array-like object whose last axis has length ``n_variables``.

        Returns
        -------
        IntegerArray
            Integer torsional state labels.
        """
        angles = np.asarray(x, dtype=float)

        if angles.ndim == 0:
            raise ValueError(
                "Input must have at least one dimension with the last axis "
                f"of length {self.n_variables}."
            )

        if angles.shape[-1] != self.n_variables:
            raise ValueError(
                "Expected the last axis to have length "
                f"{self.n_variables}, got shape {angles.shape}."
            )

        centers = np.asarray(self.centers, dtype=float)

        # Shape: (..., n_variables, n_centers)
        diff = self.wrap_angle(angles[..., :, None] - centers)
        return np.argmin(np.abs(diff), axis=-1).astype(np.int64)


@dataclass(frozen=True)
class StateSpace:
    """
    Combine a state assigner with a global state encoder.

    ``StateSpace`` is the main user-facing object for going from a configuration
    to a MetaTally global state index.

    Parameters
    ----------
    assigner
        Object that converts configurations to local-state digits.
    encoder
        Optional encoder. If omitted, it is built from ``assigner.radices``.

    Examples
    --------
    >>> import metatally as mt
    >>> space = mt.StateSpace.from_radix(n_variables=3, radix=2)
    >>> space.assign([1, 0, 1]).tolist()
    [1, 0, 1]
    >>> space.encode([1, 0, 1])
    5
    >>> space.decode(5).tolist()
    [1, 0, 1]
    """

    assigner: StateAssigner
    encoder: MixedRadixEncoder | None = None

    def __post_init__(self) -> None:
        """Create or validate the encoder associated with the assigner."""
        if self.encoder is None:
            enc = MixedRadixEncoder(self.assigner.radices)
            object.__setattr__(self, "encoder", enc)
            return

        if tuple(self.encoder.radices) != tuple(self.assigner.radices):
            raise ValueError(
                "Encoder radices must match assigner radices. "
                f"Got encoder radices={self.encoder.radices}, "
                f"assigner radices={self.assigner.radices}."
            )

    @classmethod
    def from_radix(cls, n_variables: int, radix: int) -> StateSpace:
        """
        Create a state space for already-discrete fixed-radix variables.

        Parameters
        ----------
        n_variables
            Number of local discrete variables.
        radix
            Number of states for each variable.

        Returns
        -------
        StateSpace
            State space using a ``DiscreteStateAssigner``.

        Examples
        --------
        >>> import metatally as mt
        >>> space = mt.StateSpace.from_radix(n_variables=3, radix=2)
        >>> space.encode([1, 0, 1])
        5
        """
        enc = encoder(n_variables=n_variables, radix=radix)
        return cls(assigner=DiscreteStateAssigner(enc.radices), encoder=enc)

    @classmethod
    def from_radices(cls, radices: Sequence[int]) -> StateSpace:
        """
        Create a state space for already-discrete mixed-radix variables.

        Parameters
        ----------
        radices
            Number of states for each local variable.

        Returns
        -------
        StateSpace
            State space using a ``DiscreteStateAssigner``.
        """
        enc = MixedRadixEncoder(radices)
        return cls(assigner=DiscreteStateAssigner(enc.radices), encoder=enc)

    @property
    def radices(self) -> tuple[int, ...]:
        """Number of allowed states for each local variable."""
        return self.encoder.radices

    @property
    def n_variables(self) -> int:
        """Number of local variables in the state space."""
        return self.encoder.n_variables

    @property
    def n_states(self) -> int:
        """Total number of global states in the joint catalogue."""
        return self.encoder.n_states

    def assign(self, x: ArrayLike) -> IntegerArray:
        """
        Assign configurations to local-state digits.

        Parameters
        ----------
        x
            Configuration or array of configurations.

        Returns
        -------
        IntegerArray
            Local-state digits with the local-variable axis last.
        """
        return self.assigner.assign(x)

    def encode(self, x: ArrayLike) -> int | IntegerArray:
        """
        Assign configurations and encode them as global state indices.

        Parameters
        ----------
        x
            Configuration or array of configurations.

        Returns
        -------
        int or IntegerArray
            Encoded global state index or array of indices.

        Examples
        --------
        >>> import metatally as mt
        >>> space = mt.StateSpace.from_radix(n_variables=3, radix=2)
        >>> space.encode([1, 0, 1])
        5
        >>> space.encode([[0, 0, 0], [1, 0, 1]]).tolist()
        [0, 5]
        """
        return self.encoder.encode(self.assign(x))

    def decode(self, indices: int | ArrayLike) -> IntegerArray:
        """
        Decode global state indices into local-state digits.

        Parameters
        ----------
        indices
            Scalar state index or array-like object of state indices.

        Returns
        -------
        IntegerArray
            Decoded local-state digits.
        """
        return self.encoder.decode(indices)

    def all_digits(self) -> IntegerArray:
        """
        Return the full catalogue of local-state vectors.

        Returns
        -------
        IntegerArray
            Array with shape ``(n_states, n_variables)``.
        """
        return self.encoder.all_digits()

