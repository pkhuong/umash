"""
Test suite for the UMASH parameter preparation function.
"""

import random
from hypothesis import example, given
import hypothesis.strategies as st
from umash import C, FFI
from umash_reference import umash, UmashKey


U64S = st.integers(min_value=0, max_value=2**64 - 1)
SEEDS = U64S | st.sampled_from([0, 1, 0xFF, 2**32 - 1, 2**32, 2**63, 2**64 - 1])


FIELD = 2**61 - 1


OH_COUNT = C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT


def assert_idempotent(params):
    """Asserts that calling `umash_params_prepare` on something that was
    successfully prepared is idempotent.
    """
    size = FFI.sizeof("struct umash_params")
    copy = FFI.new("struct umash_params[1]")
    FFI.memmove(copy, params, size)
    assert C.umash_params_prepare(copy) == True
    # The copy should still be the same as params.
    assert size % 8 == 0
    for i in range(size // 8):
        assert FFI.cast("uint64_t *", params)[i] == FFI.cast("uint64_t *", copy)[i]


# Multiplier values at interesting boundaries for the mod-2^61-1
# reduction in umash_params_prepare.
MULTIPLIER_BOUNDARIES = [
    0,  # zero — rejected
    1,  # smallest valid
    2,
    FIELD - 1,  # 2^61 - 2: largest valid
    FIELD,  # 2^61 - 1: equals modulo, rejected
    FIELD + 1,  # 2^61: masks to 0, rejected
    FIELD + 2,  # 2^61 + 1: masks to 1, valid
    2 * FIELD,  # 2^62 - 2: masks to FIELD - 1, valid
    2 * FIELD + 1,  # 2^62 - 1: masks to FIELD, rejected
    2**63 - 1,  # all lower 63 bits: masks to FIELD, rejected
    2**63,  # only bit 63: masks to 0, rejected
    2**63 + 1,  # bit 63 + 1: masks to 1, valid
    2**64 - 9,  # UINT64_MAX - 8: masks to FIELD - 8, valid
    2**64 - 2,  # UINT64_MAX - 1: masks to FIELD - 1, valid
    2**64 - 1,  # UINT64_MAX: masks to FIELD, rejected
]


# Explicit boundary examples: each multiplier at the field modulus boundary.
@example(multipliers=[0, FIELD], random=random.Random(1))
@example(multipliers=[0, 0], random=random.Random(1))
@example(multipliers=[FIELD, FIELD], random=random.Random(1))
@example(multipliers=[1, 1], random=random.Random(1))
@example(multipliers=[FIELD - 1, FIELD - 1], random=random.Random(1))
# 2^61 masks to 0; 2^61 + 1 masks to 1 (valid).
@example(multipliers=[FIELD + 1, FIELD + 2], random=random.Random(1))
# High-bit patterns that reduce to boundary values.
@example(multipliers=[2**64 - 1, 2**64 - 1], random=random.Random(1))
@example(multipliers=[2**64 - 2, 2**64 - 2], random=random.Random(1))
@example(multipliers=[2**63 - 1, 2**63], random=random.Random(1))
@example(multipliers=[2 * FIELD + 1, 2 * FIELD], random=random.Random(1))
# One valid, one at each rejection boundary.
@example(multipliers=[1, FIELD], random=random.Random(1))
@example(multipliers=[FIELD - 1, FIELD + 1], random=random.Random(1))
@example(multipliers=[2, 2**64 - 1], random=random.Random(1))
@given(
    multipliers=st.lists(
        st.integers(min_value=1, max_value=FIELD - 1)
        | U64S
        | st.sampled_from(MULTIPLIER_BOUNDARIES),
        min_size=2,
        max_size=2,
    ),
    random=st.randoms(note_method_calls=True, use_true_random=True),
)
def test_public_multiplier_reduction(multipliers, random):
    """Make sure multipliers are correctly reduced and rejected."""
    params = FFI.new("struct umash_params[1]")
    params[0].poly[0][0] = random.getrandbits(64)
    params[0].poly[0][1] = multipliers[0]
    params[0].poly[1][0] = random.getrandbits(64)
    params[0].poly[1][1] = multipliers[1]

    for i in range(OH_COUNT):
        params[0].oh[i] = i

    assert C.umash_params_prepare(params) == True
    assert_idempotent(params)
    for i in range(2):
        # If we passed in something clearly usable, it should be kept.
        if 0 < multipliers[i] < FIELD:
            assert params[0].poly[i][1] == multipliers[i]
        # The multipliers must be valid.
        assert 0 < params[0].poly[i][1] < FIELD
        assert params[0].poly[i][0] == (params[0].poly[i][1] ** 2) % FIELD

    # The OH params are valid.
    for i in range(C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT):
        assert params[0].oh[i] == i


@given(value=U64S | st.sampled_from([0, 1, 2**64 - 1]))
def test_public_all_identical_oh_fails(value):
    """umash_params_prepare must fail when every OH slot holds the same
    value: fixing all 34 duplicates requires far more entropy than the
    two-entry backup buffer can provide."""
    params = FFI.new("struct umash_params[1]")
    # Use valid multipliers so any failure is from the OH check.
    params[0].poly[0][0] = 0
    params[0].poly[0][1] = 1
    params[0].poly[1][0] = 0
    params[0].poly[1][1] = 2

    for i in range(OH_COUNT):
        params[0].oh[i] = value

    assert C.umash_params_prepare(params) == False


@example(oh=[0] * OH_COUNT, random=random.Random(1))
@given(
    oh=st.lists(
        st.integers(min_value=0, max_value=100), min_size=OH_COUNT, max_size=OH_COUNT
    ),
    random=st.randoms(note_method_calls=True, use_true_random=True),
)
def test_public_bad_oh(oh, random):
    """When the OH values repeat, we should replace them if we can."""
    repeated_values = len(oh) - len(set(oh))

    params = FFI.new("struct umash_params[1]")
    for i in range(2):
        params[0].poly[i][0] = random.getrandbits(64)
        params[0].poly[i][1] = random.getrandbits(64)

    for i, value in enumerate(oh):
        params[0].oh[i] = value

    result = C.umash_params_prepare(params)
    if repeated_values > 2:
        assert result == False
    if not result:
        return

    assert_idempotent(params)
    # On success, the OH parameters should be unique
    actual_oh = [params[0].oh[i] for i in range(OH_COUNT)]
    assert len(actual_oh) == len(set(actual_oh))


@given(
    random=st.randoms(note_method_calls=True, use_true_random=True),
    seed=SEEDS,
    data=st.binary(),
)
def test_public_smoke_matches(random, seed, data):
    """Prepare a params struct, and make sure the UMASH function matches
    our reference."""
    params = FFI.new("struct umash_params[1]")
    size = FFI.sizeof("struct umash_params")
    assert size % 8 == 0
    for i in range(size // 8):
        FFI.cast("uint64_t *", params)[i] = random.getrandbits(64)

    # Pseudorandom input should always succeed.
    assert C.umash_params_prepare(params) == True
    assert_idempotent(params)
    expected0 = umash(
        UmashKey(params[0].poly[0][1], [params[0].oh[i] for i in range(OH_COUNT)]),
        seed,
        data,
        secondary=False,
    )
    assert C.umash_full(params, seed, 0, data, len(data)) == expected0

    expected1 = umash(
        UmashKey(
            params[0].poly[1][1],
            [params[0].oh[i] for i in range(OH_COUNT)],
        ),
        seed,
        data,
        secondary=True,
    )
    assert C.umash_full(params, seed, 1, data, len(data)) == expected1


@given(
    bits=U64S,
    key=st.none() | st.binary(min_size=32, max_size=32),
)
def test_params_derive_valid(bits, key):
    """umash_params_derive must always produce a valid params struct:

    - poly[i][1] (the multiplier f) is in (0, 2**61 - 1)
    - poly[i][0] == (f ** 2) % (2**61 - 1)
    - all OH values are unique
    - the result is idempotent under umash_params_prepare
    """
    params = FFI.new("struct umash_params[1]")
    if key is None:
        C.umash_params_derive(params, bits, FFI.NULL)
    else:
        buf = FFI.new("char[]", len(key))
        FFI.memmove(buf, key, len(key))
        C.umash_params_derive(params, bits, buf)

    # Each polynomial multiplier must be a valid non-zero element of F.
    for i in range(2):
        f = params[0].poly[i][1]
        assert 0 < f < FIELD, f"poly[{i}][1] = {f} is not in (0, 2**61-1)"
        assert params[0].poly[i][0] == (f**2) % FIELD, f"poly[{i}][0] != f**2 mod FIELD"

    # All OH parameters must be unique.
    actual_oh = [params[0].oh[i] for i in range(OH_COUNT)]
    assert len(actual_oh) == len(set(actual_oh)), "OH parameters contain duplicates"

    # A valid params struct must be idempotent under prepare.
    assert_idempotent(params)
