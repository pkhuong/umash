"""
Test suite for the medium (9-16 bytes) fingerprinting case.
"""

from hypothesis import given
import hypothesis.strategies as st
from umash import C, FFI
from umash_reference import umash, UmashKey


U64S = st.integers(min_value=0, max_value=2**64 - 1)


FIELD = 2**61 - 1


@given(
    seed=U64S,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=st.lists(
        U64S,
        min_size=C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT,
        max_size=C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT,
    ),
    data=st.binary(min_size=9, max_size=16),
)
def test_umash_fp_medium(seed, multipliers, key, data):
    """Compare umash_fp_medium with two calls to the reference."""
    expected = [
        umash(UmashKey(poly=multipliers[0], oh=key), seed, data, secondary=False),
        umash(UmashKey(poly=multipliers[1], oh=key), seed, data, secondary=True),
    ]

    n_bytes = len(data)
    block = FFI.new("char[]", n_bytes)
    FFI.memmove(block, data, n_bytes)
    poly = FFI.new("uint64_t[2][2]")
    for i in range(2):
        poly[i][0] = (multipliers[i] ** 2) % FIELD
        poly[i][1] = multipliers[i]
    params = FFI.new("struct umash_params[1]")
    for i, param in enumerate(key):
        params[0].oh[i] = param

    actual = C.umash_fp_medium(poly, params[0].oh, seed, block, n_bytes)
    assert [actual.hash[0], actual.hash[1]] == expected


@given(
    seed=U64S,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=st.lists(
        U64S,
        min_size=C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT,
        max_size=C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT,
    ),
    data=st.binary(min_size=9, max_size=16),
)
def test_umash_fp_medium_matches_fprint(seed, multipliers, key, data):
    """umash_fp_medium must agree with umash_fprint for medium inputs."""
    n_bytes = len(data)
    block = FFI.new("char[]", n_bytes)
    FFI.memmove(block, data, n_bytes)
    params = FFI.new("struct umash_params[1]")
    for i, multiplier in enumerate(multipliers):
        params[0].poly[i][0] = (multiplier**2) % FIELD
        params[0].poly[i][1] = multiplier
    for i, param in enumerate(key):
        params[0].oh[i] = param

    fp = C.umash_fprint(params, seed, block, n_bytes)
    direct = C.umash_fp_medium(params[0].poly, params[0].oh, seed, block, n_bytes)
    assert [fp.hash[0], fp.hash[1]] == [direct.hash[0], direct.hash[1]]
