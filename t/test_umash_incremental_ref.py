"""Test suite for the incremental hashing API compared against the
Python reference implementation, targeting structural boundaries in
the incremental state machine.

The existing test_umash_incremental.py uses Hypothesis stateful
testing to fuzz arbitrary update sequences, but doesn't control the
total accumulated size or the split points.  The existing
test_umash_boundaries.py tests at dispatch boundaries but compares
incremental results against the batch C implementation, not the
Python reference.

This file fills the gap: it uses Hypothesis to vary keys/seeds/content
while explicitly controlling how data is split across umash_sink_update
calls, and compares the final digest against the Python reference
umash() function.

Structural boundaries in the incremental state machine:

 - 16 bytes: INCREMENTAL_GRANULARITY, first buffer flush
 - oh_iter increments by 2 per 16-byte chunk; PH while oh_iter < 30
 - 240 bytes into a block: oh_iter hits 30, PH->ENH transition
 - 256 bytes: full OH block, sink_update_poly + reset
 - >256 bytes with oh_iter==0: block_sink_update bulk path
 - Multiple blocks: polynomial hash accumulates across blocks
"""

from hypothesis import given, settings
import hypothesis.strategies as st
from umash import C, FFI
from umash_reference import umash, UmashKey


U64S = st.integers(min_value=0, max_value=2**64 - 1)
SEEDS = U64S | st.sampled_from([0, 1, 0xFF, 2**32 - 1, 2**32, 2**63, 2**64 - 1])
FIELD = 2**61 - 1

INCREMENTAL_GRANULARITY = 16
BLOCK_SIZE = 256


def _uniform_splits(total, chunk):
    """Split points for feeding *total* bytes in uniform *chunk*-byte pieces."""
    return list(range(chunk, total, chunk))


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
    """Copy *data* into a heap-allocated CFFI buffer."""
    n = len(data)
    block = FFI.new("char[]", n)
    FFI.memmove(block, data, n)
    return block


def feed_incremental(sink, data, split_points):
    """Feed *data* to *sink* in pieces defined by *split_points*.

    split_points is a sorted list of byte offsets; the data is sliced
    at those points and each piece is passed to umash_sink_update.
    """
    offsets = sorted(set([0] + list(split_points) + [len(data)]))
    for start, end in zip(offsets, offsets[1:]):
        piece = data[start:end]
        if len(piece) == 0:
            continue
        buf = make_block(piece)
        C.umash_sink_update(sink, buf, len(piece))


def reference_hash(multipliers, key, seed, data, which):
    return umash(
        UmashKey(poly=multipliers[which], oh=key),
        seed,
        data,
        secondary=(which == 1),
    )


def reference_fprint(multipliers, key, seed, data):
    return [
        umash(
            UmashKey(poly=multipliers[i], oh=key),
            seed,
            data,
            secondary=(i == 1),
        )
        for i in range(2)
    ]


# -- Feed exactly at INCREMENTAL_GRANULARITY boundaries ---------------


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    random=st.randoms(use_true_random=True),
)
def test_incremental_granularity_aligned_hash(seed, multipliers, key, random):
    """Feed data in exact 16-byte pieces and compare hash against the
    Python reference at sizes that hit every OH iteration step."""
    params = make_params(multipliers, key)

    # One chunk per PH iteration, plus one for ENH, plus multi-block.
    # oh_iter = 0,2,4,...,28 are PH (chunks 1-15), oh_iter=30 is ENH (chunk 16).
    for n_chunks in [1, 2, 8, 15, 16, 17, 32, 33]:
        n_bytes = n_chunks * INCREMENTAL_GRANULARITY
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))

        for which in (0, 1):
            state = FFI.new("struct umash_state[1]")
            C.umash_init(state, params, seed, which)
            sink = FFI.addressof(state[0].sink)

            for i in range(n_chunks):
                piece = data[
                    i * INCREMENTAL_GRANULARITY : (i + 1) * INCREMENTAL_GRANULARITY
                ]
                buf = make_block(piece)
                C.umash_sink_update(sink, buf, INCREMENTAL_GRANULARITY)

            expected = reference_hash(multipliers, key, seed, data, which)
            actual = C.umash_digest(state)
            assert actual == expected, (
                f"granularity-aligned hash mismatch: "
                f"which={which} n_chunks={n_chunks}"
            )


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    random=st.randoms(use_true_random=True),
)
def test_incremental_granularity_aligned_fprint(seed, multipliers, key, random):
    """Feed data in exact 16-byte pieces and compare fingerprint against
    the Python reference."""
    params = make_params(multipliers, key)

    for n_chunks in [1, 2, 8, 15, 16, 17, 32, 33]:
        n_bytes = n_chunks * INCREMENTAL_GRANULARITY
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))

        state = FFI.new("struct umash_fp_state[1]")
        C.umash_fp_init(state, params, seed)
        sink = FFI.addressof(state[0].sink)

        for i in range(n_chunks):
            piece = data[
                i * INCREMENTAL_GRANULARITY : (i + 1) * INCREMENTAL_GRANULARITY
            ]
            buf = make_block(piece)
            C.umash_sink_update(sink, buf, INCREMENTAL_GRANULARITY)

        expected = reference_fprint(multipliers, key, seed, data)
        actual = C.umash_fp_digest(state)
        assert [
            actual.hash[0],
            actual.hash[1],
        ] == expected, f"granularity-aligned fprint mismatch: n_chunks={n_chunks}"


# -- Feed at OH block boundaries (256 bytes) --------------------------


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    random=st.randoms(use_true_random=True),
)
def test_incremental_block_aligned_hash(seed, multipliers, key, random):
    """Feed data in whole 256-byte blocks (triggering block_sink_update),
    with a variable-length tail, and compare against the Python reference."""
    params = make_params(multipliers, key)

    for n_blocks, tail in [
        (1, 0),
        (1, 1),
        (1, 15),
        (1, 16),
        (1, 17),
        (2, 0),
        (2, 128),
        (2, 255),
        (3, 0),
        (3, 1),
    ]:
        n_bytes = n_blocks * BLOCK_SIZE + tail
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))

        for which in (0, 1):
            state = FFI.new("struct umash_state[1]")
            C.umash_init(state, params, seed, which)
            sink = FFI.addressof(state[0].sink)

            # Feed full blocks, then the tail.
            offset = 0
            for _ in range(n_blocks):
                buf = make_block(data[offset : offset + BLOCK_SIZE])
                C.umash_sink_update(sink, buf, BLOCK_SIZE)
                offset += BLOCK_SIZE
            if tail > 0:
                buf = make_block(data[offset:])
                C.umash_sink_update(sink, buf, tail)

            expected = reference_hash(multipliers, key, seed, data, which)
            actual = C.umash_digest(state)
            assert actual == expected, (
                f"block-aligned hash mismatch: "
                f"which={which} n_blocks={n_blocks} tail={tail}"
            )


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    random=st.randoms(use_true_random=True),
)
def test_incremental_block_aligned_fprint(seed, multipliers, key, random):
    """Feed data in whole 256-byte blocks with a tail, and compare
    fingerprint against the Python reference."""
    params = make_params(multipliers, key)

    for n_blocks, tail in [
        (1, 0),
        (1, 1),
        (1, 16),
        (1, 17),
        (2, 0),
        (2, 128),
        (3, 0),
        (3, 1),
    ]:
        n_bytes = n_blocks * BLOCK_SIZE + tail
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))

        state = FFI.new("struct umash_fp_state[1]")
        C.umash_fp_init(state, params, seed)
        sink = FFI.addressof(state[0].sink)

        offset = 0
        for _ in range(n_blocks):
            buf = make_block(data[offset : offset + BLOCK_SIZE])
            C.umash_sink_update(sink, buf, BLOCK_SIZE)
            offset += BLOCK_SIZE
        if tail > 0:
            buf = make_block(data[offset:])
            C.umash_sink_update(sink, buf, tail)

        expected = reference_fprint(multipliers, key, seed, data)
        actual = C.umash_fp_digest(state)
        assert [actual.hash[0], actual.hash[1]] == expected, (
            f"block-aligned fprint mismatch: " f"n_blocks={n_blocks} tail={tail}"
        )


# -- Split patterns that straddle buffer/block boundaries -------------


# Split patterns: (total_size, [split_points]).
# Each split point is where an umash_sink_update call boundary falls.
SPLIT_PATTERNS = [
    # Buffer boundary: 15+1 = 16, the first 16 bytes trigger large_umash.
    (16, [15]),
    # 1 byte short of filling the buffer, then the rest.
    (32, [15]),
    # Exactly one buffer fill, then a second.
    (32, [16]),
    # Split in the middle of the second chunk.
    (32, [24]),
    # One byte at a time for the first buffer, then bulk.
    (48, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    # 15 PH chunks then the ENH chunk separately.
    (256, [240]),
    # Split right at the PH->ENH transition.
    (256, [239]),
    (256, [241]),
    # Split mid-block at various OH iteration points.
    (256, [16, 128, 240]),
    # Block boundary then a little more.
    (257, [256]),
    # One full block as one update, second block byte by byte start.
    (272, [256]),
    # Cross-block split: straddle the 256-byte boundary.
    (512, [255]),
    (512, [256]),
    (512, [257]),
    # Two full blocks then a partial block.
    (600, [256, 512]),
    # Large bulk: trigger block_sink_update path.
    (768, [256, 512]),
    # All at once (single update call, tests block_sink_update).
    (768, []),
    # --- Larger sizes up to 1536 ---
    # Four blocks, block-aligned.
    (1024, [256, 512, 768]),
    # Four blocks, straddle boundaries by +-1.
    (1024, [255, 512, 769]),
    # Four blocks with a tail.
    (1025, [256, 512, 768]),
    # Five blocks.
    (1280, [256, 512, 768, 1024]),
    # Five blocks, cross-boundary straddles.
    (1280, [257, 511, 769, 1023]),
    # Six blocks, block-aligned.
    (1536, [256, 512, 768, 1024, 1280]),
    # Six blocks, straddle every boundary.
    (1536, [255, 513, 767, 1025, 1279]),
    # 1536 all at once (single update, block_sink_update bulk path).
    (1536, []),
    # 1536 with a mid-block split in the first and last blocks.
    (1536, [128, 256, 512, 768, 1024, 1408]),
    # --- Odd chunk sizes: 7, 13, 15, 17, 31, 33 bytes ---
    # 7-byte chunks.
    (256, _uniform_splits(256, 7)),
    (512, _uniform_splits(512, 7)),
    (1024, _uniform_splits(1024, 7)),
    # 13-byte chunks.
    (256, _uniform_splits(256, 13)),
    (512, _uniform_splits(512, 13)),
    (1536, _uniform_splits(1536, 13)),
    # 15-byte chunks (just under INCREMENTAL_GRANULARITY).
    (256, _uniform_splits(256, 15)),
    (512, _uniform_splits(512, 15)),
    (1024, _uniform_splits(1024, 15)),
    # 17-byte chunks (just over INCREMENTAL_GRANULARITY).
    (256, _uniform_splits(256, 17)),
    (512, _uniform_splits(512, 17)),
    (1536, _uniform_splits(1536, 17)),
    # 31-byte chunks (just under 2*INCREMENTAL_GRANULARITY).
    (512, _uniform_splits(512, 31)),
    (1024, _uniform_splits(1024, 31)),
    (1536, _uniform_splits(1536, 31)),
    # 33-byte chunks (just over 2*INCREMENTAL_GRANULARITY).
    (512, _uniform_splits(512, 33)),
    (1024, _uniform_splits(1024, 33)),
    (1536, _uniform_splits(1536, 33)),
]


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    random=st.randoms(use_true_random=True),
)
def test_incremental_split_patterns_hash(seed, multipliers, key, random):
    """Test specific split patterns that straddle internal state machine
    boundaries, comparing hash against the Python reference."""
    params = make_params(multipliers, key)

    for n_bytes, splits in SPLIT_PATTERNS:
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))

        for which in (0, 1):
            state = FFI.new("struct umash_state[1]")
            C.umash_init(state, params, seed, which)
            sink = FFI.addressof(state[0].sink)

            feed_incremental(sink, data, splits)

            expected = reference_hash(multipliers, key, seed, data, which)
            actual = C.umash_digest(state)
            assert actual == expected, (
                f"split-pattern hash mismatch: "
                f"which={which} len={n_bytes} splits={splits}"
            )


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    random=st.randoms(use_true_random=True),
)
def test_incremental_split_patterns_fprint(seed, multipliers, key, random):
    """Test specific split patterns comparing fingerprint against the
    Python reference."""
    params = make_params(multipliers, key)

    for n_bytes, splits in SPLIT_PATTERNS:
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))

        state = FFI.new("struct umash_fp_state[1]")
        C.umash_fp_init(state, params, seed)
        sink = FFI.addressof(state[0].sink)

        feed_incremental(sink, data, splits)

        expected = reference_fprint(multipliers, key, seed, data)
        actual = C.umash_fp_digest(state)
        assert [actual.hash[0], actual.hash[1]] == expected, (
            f"split-pattern fprint mismatch: " f"len={n_bytes} splits={splits}"
        )


# -- Byte-at-a-time through structural boundaries --------------------


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    random=st.randoms(use_true_random=True),
)
def test_incremental_byte_at_a_time_hash_vs_ref(seed, multipliers, key, random):
    """Feed data one byte at a time and compare hash against the Python
    reference.  This is the worst case for the incremental state
    machine: bufsz advances by 1 each call, exercising every buffer
    fill and drain transition."""
    params = make_params(multipliers, key)

    # Sizes that cross each structural boundary.
    for n_bytes in [8, 9, 15, 16, 17, 240, 255, 256, 257, 512, 513]:
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))

        for which in (0, 1):
            state = FFI.new("struct umash_state[1]")
            C.umash_init(state, params, seed, which)
            sink = FFI.addressof(state[0].sink)

            for i in range(n_bytes):
                byte = FFI.new("char[]", 1)
                byte[0] = data[i : i + 1]
                C.umash_sink_update(sink, byte, 1)

            expected = reference_hash(multipliers, key, seed, data, which)
            actual = C.umash_digest(state)
            assert actual == expected, (
                f"byte-at-a-time hash mismatch: " f"which={which} len={n_bytes}"
            )


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    random=st.randoms(use_true_random=True),
)
def test_incremental_byte_at_a_time_fprint_vs_ref(seed, multipliers, key, random):
    """Feed data one byte at a time and compare fingerprint against the
    Python reference."""
    params = make_params(multipliers, key)

    for n_bytes in [8, 9, 15, 16, 17, 240, 255, 256, 257, 512, 513]:
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))

        state = FFI.new("struct umash_fp_state[1]")
        C.umash_fp_init(state, params, seed)
        sink = FFI.addressof(state[0].sink)

        for i in range(n_bytes):
            byte = FFI.new("char[]", 1)
            byte[0] = data[i : i + 1]
            C.umash_sink_update(sink, byte, 1)

        expected = reference_fprint(multipliers, key, seed, data)
        actual = C.umash_fp_digest(state)
        assert [
            actual.hash[0],
            actual.hash[1],
        ] == expected, f"byte-at-a-time fprint mismatch: len={n_bytes}"


# -- Hypothesis-chosen split points at fixed sizes --------------------


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    random=st.randoms(use_true_random=True),
    splits=st.lists(st.integers(min_value=1, max_value=255), min_size=0, max_size=8),
)
def test_incremental_random_splits_one_block_hash(
    seed, multipliers, key, random, splits
):
    """For a single-OH-block input (256 bytes), let Hypothesis choose
    where to split the data across umash_sink_update calls, and compare
    hash against the Python reference."""
    params = make_params(multipliers, key)
    n_bytes = 256
    data = bytes(random.getrandbits(8) for _ in range(n_bytes))

    for which in (0, 1):
        state = FFI.new("struct umash_state[1]")
        C.umash_init(state, params, seed, which)
        sink = FFI.addressof(state[0].sink)

        feed_incremental(sink, data, splits)

        expected = reference_hash(multipliers, key, seed, data, which)
        actual = C.umash_digest(state)
        assert actual == expected, (
            f"random-splits hash mismatch: "
            f"which={which} splits={sorted(set(splits))}"
        )


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    random=st.randoms(use_true_random=True),
    splits=st.lists(st.integers(min_value=1, max_value=511), min_size=0, max_size=8),
)
def test_incremental_random_splits_two_blocks_fprint(
    seed, multipliers, key, random, splits
):
    """For a two-block input (512 bytes), let Hypothesis choose split
    points and compare fingerprint against the Python reference."""
    params = make_params(multipliers, key)
    n_bytes = 512
    data = bytes(random.getrandbits(8) for _ in range(n_bytes))

    state = FFI.new("struct umash_fp_state[1]")
    C.umash_fp_init(state, params, seed)
    sink = FFI.addressof(state[0].sink)

    feed_incremental(sink, data, splits)

    expected = reference_fprint(multipliers, key, seed, data)
    actual = C.umash_fp_digest(state)
    assert [
        actual.hash[0],
        actual.hash[1],
    ] == expected, f"random-splits fprint mismatch: splits={sorted(set(splits))}"


# -- Snapshot consistency: digest must be repeatable -------------------


@settings(deadline=None)
@given(
    seed=SEEDS,
    multipliers=st.lists(
        st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2
    ),
    key=oh_key(),
    random=st.randoms(use_true_random=True),
)
def test_incremental_digest_repeatable(seed, multipliers, key, random):
    """Calling umash_digest or umash_fp_digest multiple times on the
    same state must return the same value, and that value must match
    the Python reference.

    digest_flush works on a copy of the sink, so the original state
    should not be modified."""
    params = make_params(multipliers, key)

    for n_bytes in [0, 8, 16, 17, 256, 257, 512]:
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))

        # Hash
        state = FFI.new("struct umash_state[1]")
        C.umash_init(state, params, seed, 0)
        sink = FFI.addressof(state[0].sink)
        if n_bytes > 0:
            buf = make_block(data)
            C.umash_sink_update(sink, buf, n_bytes)

        expected = reference_hash(multipliers, key, seed, data, 0)
        first = C.umash_digest(state)
        second = C.umash_digest(state)
        assert (
            first == second == expected
        ), f"digest not repeatable or wrong: len={n_bytes}"

        # Fingerprint
        fp_state = FFI.new("struct umash_fp_state[1]")
        C.umash_fp_init(fp_state, params, seed)
        fp_sink = FFI.addressof(fp_state[0].sink)
        if n_bytes > 0:
            buf = make_block(data)
            C.umash_sink_update(fp_sink, buf, n_bytes)

        fp_expected = reference_fprint(multipliers, key, seed, data)
        fp_first = C.umash_fp_digest(fp_state)
        fp_second = C.umash_fp_digest(fp_state)
        assert (
            [fp_first.hash[0], fp_first.hash[1]]
            == [fp_second.hash[0], fp_second.hash[1]]
            == fp_expected
        ), f"fp_digest not repeatable or wrong: len={n_bytes}"
