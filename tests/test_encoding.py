import numpy as np
import pytest

import metatally as mt
from metatally.core.encoding import MixedRadixEncoder


def test_fixed_radix_encoder_properties():
    enc = mt.encoder(n_variables=4, radix=3)

    assert enc.radices == (3, 3, 3, 3)
    assert enc.n_variables == 4
    assert enc.n_states == 81
    assert enc.multipliers == (1, 3, 9, 27)


def test_binary_encoder_properties():
    enc = mt.binary_encoder(5)

    assert enc.radices == (2, 2, 2, 2, 2)
    assert enc.n_variables == 5
    assert enc.n_states == 32
    assert enc.multipliers == (1, 2, 4, 8, 16)


def test_ternary_encoder_properties():
    enc = mt.ternary_encoder(3)

    assert enc.radices == (3, 3, 3)
    assert enc.n_variables == 3
    assert enc.n_states == 27
    assert enc.multipliers == (1, 3, 9)


def test_mixed_radix_encoder_properties():
    enc = MixedRadixEncoder([2, 3, 4])

    assert enc.radices == (2, 3, 4)
    assert enc.n_variables == 3
    assert enc.n_states == 24
    assert enc.multipliers == (1, 2, 6)


def test_invalid_radices():
    with pytest.raises(ValueError, match="At least one radix"):
        MixedRadixEncoder([])

    with pytest.raises(ValueError, match="All radices must be >= 2"):
        MixedRadixEncoder([2, 1, 3])

    with pytest.raises(ValueError, match="All radices must be >= 2"):
        MixedRadixEncoder([0])


def test_invalid_fixed_radix_encoder_arguments():
    with pytest.raises(ValueError, match="n_variables must be >= 1"):
        mt.encoder(n_variables=0, radix=2)

    with pytest.raises(ValueError, match="radix must be >= 2"):
        mt.encoder(n_variables=3, radix=1)

    with pytest.raises(ValueError, match="n_variables must be >= 1"):
        mt.binary_encoder(0)

    with pytest.raises(ValueError, match="n_variables must be >= 1"):
        mt.ternary_encoder(0)


def test_encode_one_binary_state():
    enc = mt.binary_encoder(3)

    assert enc.encode([0, 0, 0]) == 0
    assert enc.encode([1, 0, 0]) == 1
    assert enc.encode([0, 1, 0]) == 2
    assert enc.encode([1, 0, 1]) == 5
    assert enc.encode([1, 1, 1]) == 7


def test_encode_one_ternary_state():
    enc = mt.ternary_encoder(3)

    assert enc.encode([0, 0, 0]) == 0
    assert enc.encode([1, 0, 0]) == 1
    assert enc.encode([2, 0, 1]) == 11
    assert enc.encode([2, 2, 2]) == 26


def test_encode_one_mixed_radix_state():
    enc = MixedRadixEncoder([2, 3, 4])

    assert enc.encode([0, 0, 0]) == 0
    assert enc.encode([1, 0, 0]) == 1
    assert enc.encode([0, 1, 0]) == 2
    assert enc.encode([0, 0, 1]) == 6
    assert enc.encode([1, 2, 3]) == 23


def test_decode_one_binary_state():
    enc = mt.binary_encoder(3)

    np.testing.assert_array_equal(enc.decode(0), np.array([0, 0, 0]))
    np.testing.assert_array_equal(enc.decode(1), np.array([1, 0, 0]))
    np.testing.assert_array_equal(enc.decode(2), np.array([0, 1, 0]))
    np.testing.assert_array_equal(enc.decode(5), np.array([1, 0, 1]))
    np.testing.assert_array_equal(enc.decode(7), np.array([1, 1, 1]))


def test_decode_one_ternary_state():
    enc = mt.ternary_encoder(3)

    np.testing.assert_array_equal(enc.decode(0), np.array([0, 0, 0]))
    np.testing.assert_array_equal(enc.decode(1), np.array([1, 0, 0]))
    np.testing.assert_array_equal(enc.decode(11), np.array([2, 0, 1]))
    np.testing.assert_array_equal(enc.decode(26), np.array([2, 2, 2]))


def test_decode_one_mixed_radix_state():
    enc = MixedRadixEncoder([2, 3, 4])

    np.testing.assert_array_equal(enc.decode(0), np.array([0, 0, 0]))
    np.testing.assert_array_equal(enc.decode(1), np.array([1, 0, 0]))
    np.testing.assert_array_equal(enc.decode(2), np.array([0, 1, 0]))
    np.testing.assert_array_equal(enc.decode(6), np.array([0, 0, 1]))
    np.testing.assert_array_equal(enc.decode(23), np.array([1, 2, 3]))


def test_roundtrip_binary_all_states():
    enc = mt.binary_encoder(6)

    for index in range(enc.n_states):
        digits = enc.decode(index)
        assert enc.encode(digits) == index


def test_roundtrip_ternary_all_states():
    enc = mt.ternary_encoder(5)

    for index in range(enc.n_states):
        digits = enc.decode(index)
        assert enc.encode(digits) == index


def test_roundtrip_mixed_radix_all_states():
    enc = MixedRadixEncoder([2, 3, 4, 5])

    for index in range(enc.n_states):
        digits = enc.decode(index)
        assert enc.encode(digits) == index


def test_encode_many_list_of_lists():
    enc = mt.binary_encoder(3)

    digits = [
        [0, 0, 0],
        [1, 0, 1],
        [1, 1, 1],
    ]

    indices = enc.encode(digits)

    assert isinstance(indices, np.ndarray)
    np.testing.assert_array_equal(indices, np.array([0, 5, 7]))


def test_encode_many_nd_array():
    enc = mt.binary_encoder(3)

    digits = np.array(
        [
            [[0, 0, 0], [1, 0, 1]],
            [[1, 1, 1], [0, 1, 0]],
        ]
    )

    indices = enc.encode(digits)

    assert indices.shape == (2, 2)
    np.testing.assert_array_equal(indices, np.array([[0, 5], [7, 2]]))


def test_decode_many_list():
    enc = mt.binary_encoder(3)

    digits = enc.decode([0, 5, 7])

    assert isinstance(digits, np.ndarray)
    assert digits.shape == (3, 3)
    np.testing.assert_array_equal(
        digits,
        np.array(
            [
                [0, 0, 0],
                [1, 0, 1],
                [1, 1, 1],
            ]
        ),
    )


def test_decode_many_nd_array():
    enc = mt.binary_encoder(3)

    indices = np.array([[0, 5], [7, 2]])
    digits = enc.decode(indices)

    assert digits.shape == (2, 2, 3)
    np.testing.assert_array_equal(
        digits,
        np.array(
            [
                [[0, 0, 0], [1, 0, 1]],
                [[1, 1, 1], [0, 1, 0]],
            ]
        ),
    )


def test_encode_decode_array_roundtrip():
    enc = MixedRadixEncoder([2, 3, 4])

    digits = np.array(
        [
            [[0, 0, 0], [1, 0, 0]],
            [[0, 1, 2], [1, 2, 3]],
        ]
    )

    indices = enc.encode(digits)
    recovered = enc.decode(indices)

    np.testing.assert_array_equal(recovered, digits)


def test_decode_encode_array_roundtrip():
    enc = MixedRadixEncoder([2, 3, 4])

    indices = np.array([[0, 1, 2], [6, 12, 23]])
    recovered = enc.encode(enc.decode(indices))

    np.testing.assert_array_equal(recovered, indices)


def test_encode_accepts_integer_like_float_values():
    enc = mt.binary_encoder(3)

    assert enc.encode([1.0, 0.0, 1.0]) == 5

    np.testing.assert_array_equal(
        enc.encode([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
        np.array([0, 5]),
    )


def test_decode_accepts_integer_like_float_values():
    enc = mt.binary_encoder(3)

    np.testing.assert_array_equal(enc.decode(5.0), np.array([1, 0, 1]))
    np.testing.assert_array_equal(
        enc.decode([0.0, 5.0]),
        np.array([[0, 0, 0], [1, 0, 1]]),
    )


def test_encode_rejects_non_integer_digits():
    enc = mt.binary_encoder(3)

    with pytest.raises(ValueError, match="digits must contain integer values"):
        enc.encode([1.2, 0.0, 1.0])


def test_decode_rejects_non_integer_indices():
    enc = mt.binary_encoder(3)

    with pytest.raises(ValueError, match="indices must contain integer values"):
        enc.decode([1.2, 2.0])


def test_encode_rejects_scalar_digits():
    enc = mt.binary_encoder(3)

    with pytest.raises(ValueError, match="Digits must have at least one dimension"):
        enc.encode(1)


def test_encode_rejects_wrong_last_axis():
    enc = mt.binary_encoder(3)

    with pytest.raises(ValueError, match="Expected the last digit axis"):
        enc.encode([1, 0])

    with pytest.raises(ValueError, match="Expected the last digit axis"):
        enc.encode(np.zeros((2, 2), dtype=int))


def test_encode_rejects_out_of_range_digits():
    enc = mt.binary_encoder(3)

    with pytest.raises(ValueError, match="Digits out of range"):
        enc.encode([2, 0, 1])

    with pytest.raises(ValueError, match="Digits out of range"):
        enc.encode([-1, 0, 1])

    with pytest.raises(ValueError, match="Digits out of range"):
        enc.encode([[0, 0, 0], [1, 2, 0]])


def test_decode_rejects_out_of_range_indices():
    enc = mt.binary_encoder(3)

    with pytest.raises(ValueError, match="State index out of range"):
        enc.decode(-1)

    with pytest.raises(ValueError, match="State index out of range"):
        enc.decode(8)

    with pytest.raises(ValueError, match="State index out of range"):
        enc.decode([0, 8])


def test_all_digits_binary():
    enc = mt.binary_encoder(2)

    np.testing.assert_array_equal(
        enc.all_digits(),
        np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
            ]
        ),
    )


def test_all_digits_mixed_radix():
    enc = MixedRadixEncoder([2, 3])

    np.testing.assert_array_equal(
        enc.all_digits(),
        np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 2],
                [1, 2],
            ]
        ),
    )


def test_iter_digits_matches_all_digits():
    enc = MixedRadixEncoder([2, 3, 2])

    from_iter = np.array(list(enc.iter_digits()))
    np.testing.assert_array_equal(from_iter, enc.all_digits())


def test_hamming_distance():
    enc = mt.ternary_encoder(3)

    a = enc.encode([0, 0, 0])
    b = enc.encode([2, 0, 1])

    assert enc.hamming_distance(a, b) == 2
    assert enc.hamming_distance(a, a) == 0


def test_cyclic_digit_distance():
    enc = mt.ternary_encoder(3)

    a = enc.encode([0, 0, 0])
    b = enc.encode([2, 0, 1])

    # 0 -> 2 is one cyclic step in radix 3; 0 -> 1 is one step.
    assert enc.cyclic_digit_distance(a, b) == 2


def test_are_neighbours_noncyclic():
    enc = mt.ternary_encoder(2)

    assert enc.are_neighbours(enc.encode([0, 0]), enc.encode([1, 0]))
    assert not enc.are_neighbours(enc.encode([0, 0]), enc.encode([2, 0]))
    assert not enc.are_neighbours(enc.encode([0, 0]), enc.encode([1, 1]))
    assert not enc.are_neighbours(enc.encode([0, 0]), enc.encode([0, 0]))


def test_are_neighbours_cyclic():
    enc = mt.ternary_encoder(2)

    assert enc.are_neighbours(enc.encode([0, 0]), enc.encode([2, 0]), cyclic=True)
    assert enc.are_neighbours(enc.encode([0, 0]), enc.encode([1, 0]), cyclic=True)
    assert not enc.are_neighbours(enc.encode([0, 0]), enc.encode([2, 1]), cyclic=True)
    assert not enc.are_neighbours(enc.encode([0, 0]), enc.encode([0, 0]), cyclic=True)


def test_doctests_for_encoding_module():
    import doctest
    import metatally.core.encoding as encoding

    result = doctest.testmod(encoding, verbose=False)
    assert result.failed == 0
