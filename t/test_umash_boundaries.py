"""Test suite that targets important size boundaries in the UMASH
dispatch logic.

Hypothesis naturally tends toward small inputs and may not reliably
generate data that lands exactly at the thresholds where UMASH switches
between code paths.  This file complements the existing fuzz-style
tests by using Hypothesis to vary keys/seeds/content while pinning
data lengths to sizes that exercise every dispatch transition:

 - 0-8 bytes   -> umash_short  (with an internal split at 4 bytes)
 - 9-16 bytes  -> umash_medium
 - 17+ bytes   -> umash_long
 - 256 bytes   -> OH block boundary (BLOCK_SIZE)
 - 257+ bytes  -> multi-block long path
 - 1024, 2048  -> deeper multi-block, +/- {0,256,512} blocks
 - 4096, 8192, 65536 -> large multi-block, +/- {0,1,15,16,17,255,256,257}

The tests compare the C implementation against the Python reference
for batch hashing, batch fingerprinting, and the incremental API.
"""

from hypothesis import given, note, settings
import hypothesis.strategies as st
from umash import C, FFI
from umash_reference import umash, UmashKey


U64S = st.integers(min_value=0, max_value=2**64 - 1)
SEEDS = U64S | st.sampled_from([0, 1, 0xFF, 2**32 - 1, 2**32, 2**63, 2**64 - 1])
FIELD = 2**61 - 1

# Sizes that straddle every dispatch boundary.
BOUNDARY_SIZES = [
    0,
    1,
    3,
    4,
    5,
    7,
    8,  # short path; internal split at 4
    9,
    15,
    16,  # medium path
    17,
    31,
    32,  # long path, single partial block
    # Around one OH block (256 bytes = 16 chunks of 16 bytes).
    # PH handles the first 15 chunks (240 bytes), ENH the last chunk.
    239,
    240,
    241,
    255,
    256,
    257,
    # Two full blocks and surroundings.
    511,
    512,
    513,
    # Deeper multi-block: sizes around {1024, 2048} +/- {0, 256, 512},
    # plus off-by-{1,2} around each block-aligned boundary.  These
    # exercise the polynomial hash accumulation across 3-10 OH blocks
    # and the block_sink_update bulk path in the incremental API.
    # 512 +/- is already above.
    766, 767, 768, 769, 770,
    1022, 1023, 1024, 1025, 1026,
    1278, 1279, 1280, 1281, 1282,
    1534, 1535, 1536, 1537, 1538,
    1790, 1791, 1792, 1793, 1794,
    2046, 2047, 2048, 2049, 2050,
    2302, 2303, 2304, 2305, 2306,
    2558, 2559, 2560, 2561, 2562,
    # Large multi-block: sizes around {4096, 8192, 65536} with offsets
    # that hit block boundaries (256), INCREMENTAL_GRANULARITY boundaries
    # (16), and off-by-one around each.  These stress the polynomial
    # accumulation at depth and the umash_multiple_blocks fast path.
    3839, 3840, 3841, 4079, 4080, 4081,
    4095, 4096, 4097, 4111, 4112, 4113,
    4351, 4352, 4353,
    7935, 7936, 7937, 8175, 8176, 8177,
    8191, 8192, 8193, 8207, 8208, 8209,
    8447, 8448, 8449,
    65279, 65280, 65281, 65519, 65520, 65521,
    65535, 65536, 65537, 65551, 65552, 65553,
    65791, 65792, 65793,
]


def oh_key():
    return st.lists(
        U64S,
        min_size=C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT,
        max_size=C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT,
    )


def make_params(multipliers, key):
    """Build a C umash_params struct from Python values."""
    params = FFI.new("struct umash_params[1]")
    for i, multiplier in enumerate(multipliers):
        params[0].poly[i][0] = (multiplier**2) % FIELD
        params[0].poly[i][1] = multiplier
    for i, param in enumerate(key):
        params[0].oh[i] = param
    return params


def make_block(data):
    """Copy *data* into a heap-allocated CFFI buffer (helps ASan)."""
    n = len(data)
    block = FFI.new("char[]", n)
    FFI.memmove(block, data, n)
    return block


def incremental_hash(params, seed, which, data):
    """Compute a hash using the incremental API in one shot."""
    state = FFI.new("struct umash_state[1]")
    C.umash_init(state, params, seed, which)
    sink = FFI.addressof(state[0].sink)
    block = make_block(data)
    C.umash_sink_update(sink, block, len(data))
    return C.umash_digest(state)


def incremental_fprint(params, seed, data):
    """Compute a fingerprint using the incremental API in one shot."""
    state = FFI.new("struct umash_fp_state[1]")
    C.umash_fp_init(state, params, seed)
    sink = FFI.addressof(state[0].sink)
    block = make_block(data)
    C.umash_sink_update(sink, block, len(data))
    return C.umash_fp_digest(state)


# -- Batch hash at boundaries ----------------------------------------


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    # Random content for every boundary length.
    random=st.randoms(use_true_random=True),
)
def test_umash_full_at_boundaries(seed, multipliers, key, random):
    """umash_full must match the reference at every dispatch boundary."""
    params = make_params(multipliers, key)

    for n_bytes in BOUNDARY_SIZES:
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))
        block = make_block(data)

        for which in (0, 1):
            expected = umash(
                UmashKey(poly=multipliers[which], oh=key),
                seed,
                data,
                secondary=(which == 1),
            )
            actual = C.umash_full(params, seed, which, block, n_bytes)
            assert (
                actual == expected
            ), f"umash_full mismatch: which={which} len={n_bytes}"

            incr = incremental_hash(params, seed, which, data)
            assert (
                incr == expected
            ), f"incremental hash mismatch: which={which} len={n_bytes}"


# -- Batch fingerprint at boundaries ---------------------------------


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    random=st.randoms(use_true_random=True),
)
def test_umash_fprint_at_boundaries(seed, multipliers, key, random):
    """umash_fprint must match the reference at every dispatch boundary."""
    params = make_params(multipliers, key)

    for n_bytes in BOUNDARY_SIZES:
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))
        block = make_block(data)

        expected = [
            umash(
                UmashKey(poly=multipliers[i], oh=key),
                seed,
                data,
                secondary=(i == 1),
            )
            for i in range(2)
        ]
        actual = C.umash_fprint(params, seed, block, n_bytes)
        assert [
            actual.hash[0],
            actual.hash[1],
        ] == expected, f"umash_fprint mismatch: len={n_bytes}"

        incr = incremental_fprint(params, seed, data)
        assert [
            incr.hash[0],
            incr.hash[1],
        ] == expected, f"incremental fprint mismatch: len={n_bytes}"


# -- Incremental hash at boundaries ----------------------------------


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    random=st.randoms(use_true_random=True),
)
def test_incremental_hash_at_boundaries(seed, multipliers, key, random):
    """The incremental API must agree with umash_full at boundary sizes."""
    params = make_params(multipliers, key)

    for n_bytes in BOUNDARY_SIZES:
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))
        block = make_block(data)

        for which in (0, 1):
            expected = C.umash_full(params, seed, which, block, n_bytes)
            actual = incremental_hash(params, seed, which, data)
            assert (
                actual == expected
            ), f"incremental hash mismatch: which={which} len={n_bytes}"


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    random=st.randoms(use_true_random=True),
)
def test_incremental_fprint_at_boundaries(seed, multipliers, key, random):
    """The incremental fingerprint API must agree with umash_fprint."""
    params = make_params(multipliers, key)

    for n_bytes in BOUNDARY_SIZES:
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))
        block = make_block(data)

        expected = C.umash_fprint(params, seed, block, n_bytes)
        actual = incremental_fprint(params, seed, data)
        assert [actual.hash[0], actual.hash[1]] == [
            expected.hash[0],
            expected.hash[1],
        ], f"incremental fprint mismatch: len={n_bytes}"


# -- Incremental with chunked feeding at boundaries ------------------


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    random=st.randoms(use_true_random=True),
)
def test_incremental_chunked_at_boundaries(seed, multipliers, key, random):
    """Feed data in 16-byte chunks (INCREMENTAL_GRANULARITY) via the
    incremental API and compare with batch results.

    This exercises the buffer-flush and OH iteration logic at every
    boundary size.
    """
    params = make_params(multipliers, key)
    CHUNK = 16  # INCREMENTAL_GRANULARITY

    for n_bytes in BOUNDARY_SIZES:
        if n_bytes == 0:
            continue
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))

        state = FFI.new("struct umash_state[1]")
        C.umash_init(state, params, seed, 0)
        sink = FFI.addressof(state[0].sink)

        # Feed in CHUNK-sized pieces, with a possibly short final piece.
        offset = 0
        while offset < n_bytes:
            end = min(offset + CHUNK, n_bytes)
            piece = data[offset:end]
            buf = make_block(piece)
            C.umash_sink_update(sink, buf, len(piece))
            offset = end

        block = make_block(data)
        expected = C.umash_full(params, seed, 0, block, n_bytes)
        actual = C.umash_digest(state)
        assert actual == expected, f"chunked incremental mismatch: len={n_bytes}"


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    random=st.randoms(use_true_random=True),
)
def test_incremental_byte_at_a_time(seed, multipliers, key, random):
    """Feed data one byte at a time around the medium/long boundary.

    This is the most adversarial pattern for the incremental state
    machine: every call to umash_sink_update only advances bufsz by 1.
    """
    params = make_params(multipliers, key)

    for n_bytes in (15, 16, 17, 32, 255, 256, 257):
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))

        state = FFI.new("struct umash_fp_state[1]")
        C.umash_fp_init(state, params, seed)
        sink = FFI.addressof(state[0].sink)

        for i in range(n_bytes):
            byte = FFI.new("char[]", 1)
            byte[0] = data[i : i + 1]
            C.umash_sink_update(sink, byte, 1)

        block = make_block(data)
        expected = C.umash_fprint(params, seed, block, n_bytes)
        actual = C.umash_fp_digest(state)
        assert [actual.hash[0], actual.hash[1]] == [
            expected.hash[0],
            expected.hash[1],
        ], f"byte-at-a-time fprint mismatch: len={n_bytes}"
