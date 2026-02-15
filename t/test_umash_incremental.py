"""Test suite for the incremental hashing and fingerprinting
interfaces.  Compares their results with the batch implementation and
the reference implementation.
"""

from hypothesis import given, note, settings
from hypothesis.stateful import (
    initialize,
    invariant,
    precondition,
    rule,
    RuleBasedStateMachine,
)
import hypothesis.strategies as st
from umash import C, FFI
from umash_reference import umash, UmashKey

U64S = st.integers(min_value=0, max_value=2**64 - 1)
SEEDS = U64S | st.sampled_from([
    0,
    1,
    0xFF,
    2**32 - 1,
    2**32,
    2**63 - 1,   # INT64_MAX: all bits set except MSB
    2**63,       # INT64_MIN as unsigned: only MSB set
    2**63 + 1,   # just past the sign-bit boundary
    2**64 - 9,   # UINT64_MAX - 8: near-max, stresses seed + param overflow
    2**64 - 2,   # UINT64_MAX - 1: one below max
    2**64 - 1,   # UINT64_MAX
])


FIELD = 2**61 - 1


def umash_params():
    """Generates a UMASH parameter tuple."""

    def make_params(multipliers, oh):
        params = FFI.new("struct umash_params[1]")
        for i, multiplier in enumerate(multipliers):
            params[0].poly[i][0] = (multiplier**2) % FIELD
            params[0].poly[i][1] = multiplier
        for i, param in enumerate(oh):
            params[0].oh[i] = param

        return (multipliers, oh, params)

    return st.builds(
        make_params,
        st.lists(st.integers(min_value=0, max_value=FIELD - 1), min_size=2, max_size=2),
        st.lists(
            U64S,
            min_size=C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT,
            max_size=C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT,
        ),
    )


class IncrementalUpdater(RuleBasedStateMachine):
    """Calls self.update with various byte buffers.

    Child classes are expected to:

    - define an @initialize rule to set the UMASH state
    - define reference_value() to compute the reference hash of self.acc
    - define batch_value() to compute the batch hash of self.acc
    - define digest_value() to extract the current incremental digest
    - update(buf, n) to feed buf[0 ... n - 1] to the incremental state
    """

    def __init__(self):
        super().__init__()
        self.multipliers = None
        self.oh = None
        self.params = None
        self.state = None
        self.acc = b""

    @invariant()
    def compare_values(self):
        if self.state is None:
            return

        reference = self.reference_value()
        batch = self.batch_value()
        actual = self.digest_value()

        note((len(self.acc), self.acc))
        assert reference == batch == actual, {
            "ref": reference,
            "batch": batch,
            "actual": actual,
        }

    def _update(self, buf):
        self.acc += buf
        n = len(buf)
        # Copy to the heap to help ASan
        copy = FFI.new("char[]", n)
        FFI.memmove(copy, buf, n)
        self.update(copy, n)

    @precondition(lambda self: self.state)
    @rule(buf=st.binary())
    def update_short(self, buf):
        note("update_short: %s" % len(buf))
        self._update(buf)

    @precondition(lambda self: self.state)
    @rule()
    def update_empty(self):
        """Explicitly send a 0-byte update."""
        self._update(b"")

    @precondition(lambda self: self.state)
    @rule(
        num=st.integers(min_value=1, max_value=2048),
        byte=st.binary(min_size=1, max_size=1),
    )
    def update_repeat(self, num, byte):
        buf = byte * num
        self._update(buf)

    @precondition(lambda self: self.state)
    @rule(
        n_blocks=st.integers(min_value=4, max_value=8),
        byte=st.binary(min_size=1, max_size=1),
    )
    def update_long_repeat(self, n_blocks, byte):
        num = n_blocks * 256
        buf = byte * num
        self._update(buf)

    @precondition(lambda self: self.state)
    @rule(
        length=st.integers(min_value=1, max_value=2048),
        random=st.randoms(use_true_random=True),
    )
    def update_long(self, length, random):
        buf = bytes((random.getrandbits(8) for _ in range(length)))
        note("update_long: %s" % buf)
        self._update(buf)


class IncrementalHasher(IncrementalUpdater):
    def __init__(self):
        super().__init__()
        self.seed = None
        self.which = None

    @initialize(
        params=umash_params(), seed=SEEDS, which=st.integers(min_value=0, max_value=1)
    )
    def create_state(self, params, seed, which):
        self.multipliers, self.oh, self.params = params
        self.state = FFI.new("struct umash_state[1]")
        self.sink = FFI.addressof(self.state[0].sink)
        C.umash_init(self.state, self.params, seed, which)
        self.seed = seed
        self.which = which

    def update(self, buf, n):
        C.umash_sink_update(self.sink, buf, n)

    def reference_value(self):
        return umash(
            UmashKey(
                poly=self.multipliers[self.which],
                oh=self.oh,
            ),
            self.seed,
            self.acc,
            secondary=(self.which == 1),
        )

    def batch_value(self):
        return C.umash_full(self.params, self.seed, self.which, self.acc, len(self.acc))

    def digest_value(self):
        return C.umash_digest(self.state)


test_public_incremental_hasher = IncrementalHasher.TestCase


class IncrementalFprinter(IncrementalUpdater):
    def __init__(self):
        super().__init__()
        self.seed = None

    @initialize(params=umash_params(), seed=SEEDS)
    def create_state(self, params, seed):
        self.multipliers, self.oh, self.params = params
        self.state = FFI.new("struct umash_fp_state[1]")
        self.sink = FFI.addressof(self.state[0].sink)
        C.umash_fp_init(self.state, self.params, seed)
        self.seed = seed

    def update(self, buf, n):
        C.umash_sink_update(self.sink, buf, n)

    def reference_value(self):
        return [
            umash(
                UmashKey(
                    poly=self.multipliers[which],
                    oh=self.oh,
                ),
                self.seed,
                self.acc,
                secondary=(which == 1),
            )
            for which in range(2)
        ]

    def batch_value(self):
        result = C.umash_fprint(self.params, self.seed, self.acc, len(self.acc))
        return [result.hash[0], result.hash[1]]

    def digest_value(self):
        result = C.umash_fp_digest(self.state)
        return [result.hash[0], result.hash[1]]


test_public_incremental_fprinter = IncrementalFprinter.TestCase


# -- Targeted 0-byte update tests ----------------------------------------

# Sizes that hit every dispatch path (short, medium, long, multi-block)
# and interesting buffer-fill points (INCREMENTAL_GRANULARITY = 16).
EMPTY_UPDATE_SIZES = [0, 1, 8, 9, 15, 16, 17, 240, 255, 256, 257, 512]


def _sink_empty_update(sink):
    """Call umash_sink_update with a 0-byte payload."""
    buf = FFI.new("char[]", 1)
    C.umash_sink_update(sink, buf, 0)


def _sink_data_update(sink, data):
    """Call umash_sink_update with a copy of *data*."""
    n = len(data)
    copy = FFI.new("char[]", n)
    FFI.memmove(copy, data, n)
    C.umash_sink_update(sink, copy, n)


@settings(deadline=None)
@given(
    params=umash_params(),
    seed=SEEDS,
    which=st.integers(min_value=0, max_value=1),
    random=st.randoms(use_true_random=True),
)
def test_public_empty_updates_hash(params, seed, which, random):
    """0-byte updates at the beginning and end of the input must not
    affect the incremental hash.

    The 16-byte case is especially interesting: 16 bytes exactly fill
    the internal buffer (bufsz == INCREMENTAL_GRANULARITY), so a
    subsequent 0-byte update enters the second branch of
    umash_sink_update (remaining == 0) rather than the fast path.
    """
    multipliers, oh, c_params = params

    for n_bytes in EMPTY_UPDATE_SIZES:
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))
        expected = C.umash_full(c_params, seed, which, data, n_bytes)

        state = FFI.new("struct umash_state[1]")
        C.umash_init(state, c_params, seed, which)
        sink = FFI.addressof(state[0].sink)

        # Multiple 0-byte updates at the very beginning.
        _sink_empty_update(sink)
        _sink_empty_update(sink)

        if n_bytes > 0:
            _sink_data_update(sink, data)

        # Multiple 0-byte updates at the very end.
        _sink_empty_update(sink)
        _sink_empty_update(sink)

        actual = C.umash_digest(state)
        assert (
            actual == expected
        ), f"empty-updates hash mismatch: which={which} len={n_bytes}"


@settings(deadline=None)
@given(
    params=umash_params(),
    seed=SEEDS,
    random=st.randoms(use_true_random=True),
)
def test_public_empty_updates_fprint(params, seed, random):
    """0-byte updates at the beginning and end of the input must not
    affect the incremental fingerprint."""
    multipliers, oh, c_params = params

    for n_bytes in EMPTY_UPDATE_SIZES:
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))
        expected = C.umash_fprint(c_params, seed, data, n_bytes)

        state = FFI.new("struct umash_fp_state[1]")
        C.umash_fp_init(state, c_params, seed)
        sink = FFI.addressof(state[0].sink)

        _sink_empty_update(sink)
        _sink_empty_update(sink)

        if n_bytes > 0:
            _sink_data_update(sink, data)

        _sink_empty_update(sink)
        _sink_empty_update(sink)

        actual = C.umash_fp_digest(state)
        assert [actual.hash[0], actual.hash[1]] == [
            expected.hash[0],
            expected.hash[1],
        ], f"empty-updates fprint mismatch: len={n_bytes}"


@settings(deadline=None)
@given(
    params=umash_params(),
    seed=SEEDS,
    which=st.integers(min_value=0, max_value=1),
    random=st.randoms(use_true_random=True),
)
def test_public_empty_updates_between_chunks_hash(params, seed, which, random):
    """0-byte updates interleaved between data chunks must not change
    the hash.

    When a chunk exactly fills the internal 16-byte buffer, a following
    0-byte update enters the branch where remaining == 0 and
    n_bytes == 0, setting large_umash but otherwise acting as a no-op.
    """
    multipliers, oh, c_params = params

    for chunk_size, n_chunks in [
        (1, 20),
        (8, 4),
        (15, 3),
        (16, 3),
        (17, 3),
        (256, 3),
    ]:
        n_bytes = chunk_size * n_chunks
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))
        expected = C.umash_full(c_params, seed, which, data, n_bytes)

        state = FFI.new("struct umash_state[1]")
        C.umash_init(state, c_params, seed, which)
        sink = FFI.addressof(state[0].sink)

        # Leading empty update.
        _sink_empty_update(sink)

        for i in range(n_chunks):
            chunk = data[i * chunk_size : (i + 1) * chunk_size]
            _sink_data_update(sink, chunk)
            # Empty update after every chunk.
            _sink_empty_update(sink)

        actual = C.umash_digest(state)
        assert actual == expected, (
            f"interleaved empty-updates hash mismatch: "
            f"which={which} chunk_size={chunk_size} n_chunks={n_chunks}"
        )


# -- Short-input incremental vs Python reference -------------------------

# Every size from 0 through 272 (one full OH block + one
# INCREMENTAL_GRANULARITY chunk).  Short enough that the Python
# reference is fast, thorough enough to cover every dispatch path
# and off-by-one within.
SHORT_INPUT_SIZES = range(273)


@settings(deadline=None)
@given(
    params=umash_params(),
    seed=SEEDS,
    random=st.randoms(use_true_random=True),
)
def test_public_incremental_short_hash_vs_ref(params, seed, random):
    """Incremental hash for both which=0 and which=1 must match the
    Python reference for every input size up to 272 bytes."""
    multipliers, oh, c_params = params

    for n_bytes in SHORT_INPUT_SIZES:
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))

        for which in (0, 1):
            state = FFI.new("struct umash_state[1]")
            C.umash_init(state, c_params, seed, which)
            if n_bytes > 0:
                _sink_data_update(FFI.addressof(state[0].sink), data)

            expected = umash(
                UmashKey(poly=multipliers[which], oh=oh),
                seed,
                data,
                secondary=(which == 1),
            )
            actual = C.umash_digest(state)
            assert actual == expected, (
                f"incremental hash vs ref: which={which} len={n_bytes}"
            )


@settings(deadline=None)
@given(
    params=umash_params(),
    seed=SEEDS,
    random=st.randoms(use_true_random=True),
)
def test_public_incremental_short_fprint_vs_ref(params, seed, random):
    """Incremental fingerprint (both hash[0] and hash[1]) must match
    the Python reference for every input size up to 272 bytes."""
    multipliers, oh, c_params = params

    for n_bytes in SHORT_INPUT_SIZES:
        data = bytes(random.getrandbits(8) for _ in range(n_bytes))

        state = FFI.new("struct umash_fp_state[1]")
        C.umash_fp_init(state, c_params, seed)
        if n_bytes > 0:
            _sink_data_update(FFI.addressof(state[0].sink), data)

        expected = [
            umash(
                UmashKey(poly=multipliers[i], oh=oh),
                seed,
                data,
                secondary=(i == 1),
            )
            for i in range(2)
        ]
        actual = C.umash_fp_digest(state)
        assert [actual.hash[0], actual.hash[1]] == expected, (
            f"incremental fprint vs ref: len={n_bytes}"
        )
