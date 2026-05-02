"""
Encoding utilities for MetaTally.

The central object is a reversible map between local discrete state labels,

    digits = [a_0, a_1, ..., a_{N-1}],

with

    0 <= a_i < radix_i,

and a single global state index,

    S = a_0 + radix_0*a_1 + radix_0*radix_1*a_2 + ...

The public interface is NumPy-like:

- ``encode(digits)`` treats the last axis as the local-state digit axis.
- ``decode(indices)`` appends the local-state digit axis as the last axis.

Examples
--------
>>> import metatally as mt
>>> enc = mt.encoder(n_variables=3, radix=2)

Encode one state:

>>> enc.encode([1, 0, 1])
5

Encode many states:

>>> enc.encode([[0, 0, 0], [1, 0, 1], [1, 1, 1]]).tolist()
[0, 5, 7]

Decode one state:

>>> enc.decode(5).tolist()
[1, 0, 1]

Decode many states:

>>> enc.decode([0, 5, 7]).tolist()
[[0, 0, 0], [1, 0, 1], [1, 1, 1]]
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import prod
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


IntegerArray = NDArray[np.integer]


@dataclass(frozen=True)
class MixedRadixEncoder:
    """
    Reversible encoder for mixed-radix discrete state vectors.

    Parameters
    ----------
    radices
        Number of allowed states for each local variable. For example,
        ``[2, 2, 2]`` for three binary variables, ``[3, 3, 3]`` for three
        ternary variables, or ``[2, 3, 4]`` for a mixed system.

    Notes
    -----
    The encoding convention is little-endian:

        S = a_0 + r_0 a_1 + r_0 r_1 a_2 + ...

    This means digit 0 is the least significant digit.
    """

    radices: Sequence[int]

    def __post_init__(self) -> None:
        """Validate and canonicalise the radix sequence after initialisation."""
        radices = tuple(int(r) for r in self.radices)

        if len(radices) == 0:
            raise ValueError("At least one radix is required.")

        if any(r < 2 for r in radices):
            raise ValueError(
                "All radices must be >= 2. "
                f"Received radices={radices}."
            )

        object.__setattr__(self, "radices", radices)

    @property
    def n_variables(self) -> int:
        """Number of local discrete variables represented by the encoder."""
        return len(self.radices)

    @cached_property
    def n_states(self) -> int:
        """
        Total number of global states in the joint catalogue.

        This is the product of all local radices.
        """
        return prod(self.radices)

    @cached_property
    def multipliers(self) -> tuple[int, ...]:
        """
        Positional multipliers used in the encoding.

        For radices ``[r0, r1, r2]``, this returns ``(1, r0, r0*r1)``.
        """
        out: list[int] = [1]
        for radix in self.radices[:-1]:
            out.append(out[-1] * radix)
        return tuple(out)

    def _as_integer_array(self, values: ArrayLike, *, name: str) -> IntegerArray:
        """
        Convert array-like input to an integer array when safe.

        Floating-point values are accepted only if they are exactly integer-like,
        for example ``1.0`` but not ``1.2``.
        """
        arr = np.asarray(values)

        if np.issubdtype(arr.dtype, np.integer):
            return arr.astype(np.int64, copy=False)

        if np.all(np.equal(arr, np.asarray(arr, dtype=np.int64))):
            return arr.astype(np.int64)

        raise ValueError(f"{name} must contain integer values.")

    def validate_digits(self, digits: ArrayLike) -> IntegerArray:
        """
        Validate local-state digits.

        Parameters
        ----------
        digits
            Array-like object whose last axis contains the local-state digits.
            Valid shapes are ``(n_variables,)`` for one state or
            ``(..., n_variables)`` for many states.

        Returns
        -------
        IntegerArray
            Validated integer array with the same shape as the input.

        Raises
        ------
        ValueError
            If the last dimension is not ``n_variables``, if the digits are not
            integer-like, or if any digit is outside its allowed range.
        """
        arr = self._as_integer_array(digits, name="digits")

        if arr.ndim == 0:
            raise ValueError(
                "Digits must have at least one dimension with the last axis "
                f"of length {self.n_variables}."
            )

        if arr.shape[-1] != self.n_variables:
            raise ValueError(
                "Expected the last digit axis to have length "
                f"{self.n_variables}, got shape {arr.shape}."
            )

        radices = np.asarray(self.radices, dtype=np.int64)

        if np.any(arr < 0) or np.any(arr >= radices):
            raise ValueError(
                "Digits out of range. Expected 0 <= digit_i < radix_i. "
                f"Received radices={self.radices}."
            )

        return arr

    def validate_indices(self, indices: int | ArrayLike) -> int | IntegerArray:
        """
        Validate encoded global state indices.

        Parameters
        ----------
        indices
            Scalar index or array-like object of indices.

        Returns
        -------
        int or IntegerArray
            Validated scalar integer or integer array.

        Raises
        ------
        ValueError
            If any index is not integer-like or lies outside ``[0, n_states)``.
        """
        arr = self._as_integer_array(indices, name="indices")

        if np.any(arr < 0) or np.any(arr >= self.n_states):
            raise ValueError(
                f"State index out of range. Expected 0 <= index < {self.n_states}."
            )

        if arr.ndim == 0:
            return int(arr)

        return arr

    def encode(self, digits: ArrayLike) -> int | IntegerArray:
        """
        Encode local-state digits into global state indices.

        The last axis of ``digits`` is interpreted as the local-state digit
        axis. Therefore:

        - shape ``(n_variables,)`` returns one integer;
        - shape ``(..., n_variables)`` returns an array with shape ``...``.

        Parameters
        ----------
        digits
            Local-state vector or array of local-state vectors.

        Returns
        -------
        int or IntegerArray
            Encoded global state index or array of indices.

        Examples
        --------
        >>> import metatally as mt
        >>> enc = mt.encoder(n_variables=3, radix=3)

        Encode one state:

        >>> enc.encode([2, 0, 1])
        11

        Encode several states:

        >>> enc.encode([[0, 0, 0], [2, 0, 1], [2, 2, 2]]).tolist()
        [0, 11, 26]
        """
        arr = self.validate_digits(digits)
        multipliers = np.asarray(self.multipliers, dtype=np.int64)
        encoded = np.sum(arr * multipliers, axis=-1, dtype=np.int64)

        if encoded.ndim == 0:
            return int(encoded)

        return encoded

    def decode(self, indices: int | ArrayLike) -> IntegerArray:
        """
        Decode global state indices into local-state digits.

        The local-state digit axis is appended as the last axis. Therefore:

        - a scalar index returns shape ``(n_variables,)``;
        - input shape ``...`` returns shape ``(..., n_variables)``.

        Parameters
        ----------
        indices
            Scalar state index or array-like object of state indices.

        Returns
        -------
        IntegerArray
            Decoded local-state digits.

        Examples
        --------
        >>> import metatally as mt
        >>> enc = mt.encoder(n_variables=3, radix=3)

        Decode one state:

        >>> enc.decode(11).tolist()
        [2, 0, 1]

        Decode several states:

        >>> enc.decode([0, 11, 26]).tolist()
        [[0, 0, 0], [2, 0, 1], [2, 2, 2]]
        """
        validated = self.validate_indices(indices)
        idx = np.asarray(validated, dtype=np.int64)

        out_shape = idx.shape + (self.n_variables,)
        digits = np.empty(out_shape, dtype=np.int64)

        work = idx.copy()
        for i, radix in enumerate(self.radices):
            digits[..., i] = work % radix
            work //= radix

        return digits

    def iter_digits(self) -> Iterable[IntegerArray]:
        """
        Iterate over all local-state vectors in encoded order.

        Yields
        ------
        IntegerArray
            One local-state vector at a time.

        Warning
        -------
        This scales as ``n_states`` and should only be used for small systems.
        """
        for index in range(self.n_states):
            yield self.decode(index)

    def all_digits(self) -> IntegerArray:
        """
        Return the full catalogue of local-state vectors.

        Returns
        -------
        IntegerArray
            Array with shape ``(n_states, n_variables)``. Row ``i`` contains
            the decoded local-state vector for global state ``i``.

        Examples
        --------
        >>> import metatally as mt
        >>> enc = mt.binary_encoder(2)
        >>> enc.all_digits().tolist()
        [[0, 0], [1, 0], [0, 1], [1, 1]]

        Warning
        -------
        This allocates an array of shape ``(n_states, n_variables)``. Use only
        for small systems.
        """
        return self.decode(np.arange(self.n_states, dtype=np.int64))

    def hamming_distance(self, index_a: int, index_b: int) -> int:
        """
        Count how many local variables differ between two encoded states.

        Parameters
        ----------
        index_a, index_b
            Encoded global state indices.

        Returns
        -------
        int
            Number of differing local digits.
        """
        a = self.decode(index_a)
        b = self.decode(index_b)
        return int(np.count_nonzero(a != b))

    def cyclic_digit_distance(self, index_a: int, index_b: int) -> int:
        """
        Compute the sum of cyclic distances between decoded digits.

        For ternary torsional states, this treats states ``0`` and ``2`` as
        one step apart. This can be useful for rotamer graphs, but it is not a
        general physical distance.

        Parameters
        ----------
        index_a, index_b
            Encoded global state indices.

        Returns
        -------
        int
            Sum of local cyclic digit distances.
        """
        a = self.decode(index_a)
        b = self.decode(index_b)
        radices = np.asarray(self.radices, dtype=np.int64)

        diff = np.abs(a - b)
        cyclic = np.minimum(diff, radices - diff)

        return int(np.sum(cyclic))

    def are_neighbours(
        self,
        index_a: int,
        index_b: int,
        *,
        cyclic: bool = False,
    ) -> bool:
        """
        Check whether two global states differ by one local move.

        Parameters
        ----------
        index_a, index_b
            Encoded global state indices.
        cyclic
            If ``True``, local digit states are treated as cyclic. For example,
            in a ternary torsional variable, states ``0`` and ``2`` are then
            considered neighbours. If ``False``, neighbour states must differ
            by exactly one in exactly one digit.

        Returns
        -------
        bool
            ``True`` if the two states are local neighbours.
        """
        a = self.decode(index_a)
        b = self.decode(index_b)

        changed = np.flatnonzero(a != b)
        if changed.size != 1:
            return False

        i = int(changed[0])
        diff = abs(int(a[i]) - int(b[i]))

        if cyclic:
            return min(diff, self.radices[i] - diff) == 1

        return diff == 1


def encoder(n_variables: int, radix: int) -> MixedRadixEncoder:
    """
    Create a fixed-radix encoder.

    Parameters
    ----------
    n_variables
        Number of local discrete variables.
    radix
        Number of allowed states for each variable.

    Returns
    -------
    MixedRadixEncoder
        Encoder with ``radices=[radix] * n_variables``.

    Examples
    --------
    Suppose we have three ternary metastable variables, for example torsional
    basins with states ``0``, ``1``, and ``2``.

    >>> import metatally as mt
    >>> enc = mt.encoder(n_variables=3, radix=3)

    Encode one state:

    >>> enc.encode([2, 0, 1])
    11

    Encode many states:

    >>> enc.encode([[0, 0, 0], [2, 0, 1], [2, 2, 2]]).tolist()
    [0, 11, 26]

    Decode many states:

    >>> enc.decode([0, 11, 26]).tolist()
    [[0, 0, 0], [2, 0, 1], [2, 2, 2]]

    A minimal MetaTally-style coverage counter can then track which global
    states have been visited:

    >>> trajectory = [
    ...     [0, 0, 0],
    ...     [1, 0, 0],
    ...     [2, 0, 0],
    ...     [2, 1, 0],
    ...     [2, 0, 1],
    ...     [0, 0, 0],
    ... ]
    >>> visited = set(enc.encode(trajectory).tolist())
    >>> sorted(visited)
    [0, 1, 2, 5, 11]

    >>> len(visited) / enc.n_states
    0.18518518518518517

    This measures coverage of the joint state catalogue, not just whether each
    local variable has individually visited its possible states.
    """
    if n_variables < 1:
        raise ValueError("n_variables must be >= 1.")

    if radix < 2:
        raise ValueError("radix must be >= 2.")

    return MixedRadixEncoder([int(radix)] * int(n_variables))


def binary_encoder(n_variables: int) -> MixedRadixEncoder:
    """
    Create a fixed-radix encoder with ``radix=2``.

    Example: ``import metatally as mt; enc = mt.binary_encoder(3)``.
    """
    return encoder(n_variables=n_variables, radix=2)


def ternary_encoder(n_variables: int) -> MixedRadixEncoder:
    """
    Create a fixed-radix encoder with ``radix=3``.

    Example: ``import metatally as mt; enc = mt.ternary_encoder(3)``.
    """
    return encoder(n_variables=n_variables, radix=3)

