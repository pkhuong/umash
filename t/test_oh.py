"""
Test suite for OH block compression.
"""

from hypothesis import given
import hypothesis.strategies as st
from umash import C, FFI
from umash_reference import chunk_bytes, blockify_chunks, oh_compress_one_block


U64S = st.integers(min_value=0, max_value=2**64 - 1)

BLOCK_SIZE = 8 * C.UMASH_OH_PARAM_COUNT


def split_block(data):
    """Splits the last (potentially partial) block of 16-byte chunks in data."""
    last_block = []
    for block, _ in blockify_chunks(chunk_bytes(data)):
        last_block = block
    return last_block


@given(
    tag=U64S,
    key=st.lists(
        U64S, min_size=C.UMASH_OH_PARAM_COUNT, max_size=C.UMASH_OH_PARAM_COUNT
    ),
    data=st.binary(min_size=BLOCK_SIZE, max_size=BLOCK_SIZE),
)
def test_oh_full_block(tag, key, data):
    """Compare OH compression for full blocks."""
    # The C-side implicit accepts the high half of the tag, and leaves
    # the low half zeroed out.
    expected = oh_compress_one_block(key, split_block(data), tag << 64)

    # Copy to exactly-sized malloced buffers to help ASan.
    block = FFI.new("char[]", BLOCK_SIZE)
    FFI.memmove(block, data, BLOCK_SIZE)
    params = FFI.new("struct umash_params[1]")
    for i, param in enumerate(key):
        params[0].oh[i] = param

    actual = C.oh_varblock(params[0].oh, tag, block, BLOCK_SIZE)
    assert expected == actual.bits[0] + (actual.bits[1] << 64), (
        actual.bits[0],
        actual.bits[1],
    )


@given(
    tag=U64S,
    key=st.lists(
        U64S,
        min_size=C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT,
        max_size=C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT,
    ),
    data=st.binary(min_size=BLOCK_SIZE, max_size=BLOCK_SIZE),
)
def test_oh_full_block_fprint(tag, key, data):
    """Compare combined OH compression for full blocks."""
    expected = [
        oh_compress_one_block(key, split_block(data), tag << 64),
        oh_compress_one_block(key, split_block(data), tag << 64, secondary=True),
    ]

    # Copy to exactly-sized malloced buffers to help ASan.
    block = FFI.new("char[]", BLOCK_SIZE)
    FFI.memmove(block, data, BLOCK_SIZE)
    params = FFI.new("struct umash_params[1]")
    for i, param in enumerate(key):
        params[0].oh[i] = param

    actual = FFI.new("struct umash_oh[2]")
    C.oh_varblock_fprint(actual, params[0].oh, tag, block, BLOCK_SIZE)
    assert expected == [
        (actual[0].bits[0] + (actual[0].bits[1] << 64)),
        (actual[1].bits[0] + (actual[1].bits[1] << 64)),
    ], (
        actual[0].bits[0],
        actual[0].bits[1],
        actual[1].bits[0],
        actual[1].bits[1],
    )

    solo = C.oh_varblock(params[0].oh, tag, block, BLOCK_SIZE)
    assert (actual[0].bits[0], actual[0].bits[1]) == (solo.bits[0], solo.bits[1])


@given(
    tag=U64S,
    key=st.lists(
        U64S, min_size=C.UMASH_OH_PARAM_COUNT, max_size=C.UMASH_OH_PARAM_COUNT
    ),
    data=st.binary(min_size=16, max_size=BLOCK_SIZE),
)
def test_oh_tail_large(tag, key, data):
    """Compare OH compression for the last block, when it has enough data
    to fully contain the last 16-byte chunk."""
    expected = oh_compress_one_block(key, split_block(data), tag << 64)

    n_bytes = len(data)
    # Copy to exactly-sized malloced buffers to help ASan.
    block = FFI.new("char[]", n_bytes)
    FFI.memmove(block, data, n_bytes)
    params = FFI.new("struct umash_params[1]")
    for i, param in enumerate(key):
        params[0].oh[i] = param

    actual = C.oh_varblock(params[0].oh, tag, block, n_bytes)
    assert expected == actual.bits[0] + (actual.bits[1] << 64), (
        actual.bits[0],
        actual.bits[1],
    )


@given(
    tag=U64S,
    key=st.lists(
        U64S,
        min_size=C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT,
        max_size=C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT,
    ),
    data=st.binary(min_size=16, max_size=BLOCK_SIZE),
)
def test_oh_tail_large_fprint(tag, key, data):
    """Compare combined OH compression for the last block, when it has
    enough data to fully contain the last 16-byte chunk.
    """
    expected = [
        oh_compress_one_block(key, split_block(data), tag << 64),
        oh_compress_one_block(key, split_block(data), tag << 64, secondary=True),
    ]

    n_bytes = len(data)
    # Copy to exactly-sized malloced buffers to help ASan.
    block = FFI.new("char[]", n_bytes)
    FFI.memmove(block, data, n_bytes)
    params = FFI.new("struct umash_params[1]")
    for i, param in enumerate(key):
        params[0].oh[i] = param

    actual = FFI.new("struct umash_oh[2]")
    C.oh_varblock_fprint(actual, params[0].oh, tag, block, n_bytes)
    assert expected == [
        (actual[0].bits[0] + (actual[0].bits[1] << 64)),
        (actual[1].bits[0] + (actual[1].bits[1] << 64)),
    ], (
        actual[0].bits[0],
        actual[0].bits[1],
        actual[1].bits[0],
        actual[1].bits[1],
    )

    solo = C.oh_varblock(params[0].oh, tag, block, n_bytes)
    assert (actual[0].bits[0], actual[0].bits[1]) == (solo.bits[0], solo.bits[1])


@given(
    tag=U64S,
    key=st.lists(
        U64S, min_size=C.UMASH_OH_PARAM_COUNT, max_size=C.UMASH_OH_PARAM_COUNT
    ),
    prefix=st.binary(min_size=16, max_size=16),
    data=st.binary(min_size=1, max_size=16),
)
def test_oh_tail_short(tag, key, prefix, data):
    """Compare OH compression for the last block, when we must steal some
    data from the previous chunk."""
    expected = oh_compress_one_block(
        key, split_block(prefix * (C.UMASH_OH_PARAM_COUNT // 2) + data), tag << 64
    )

    offset = len(prefix)
    n_bytes = len(data)
    # Copy to exactly-sized malloced buffers to help ASan.
    block = FFI.new("char[]", offset + n_bytes)
    FFI.memmove(block, prefix + data, offset + n_bytes)
    params = FFI.new("struct umash_params[1]")
    for i, param in enumerate(key):
        params[0].oh[i] = param

    actual = C.oh_varblock(params[0].oh, tag, block + offset, n_bytes)
    assert expected == actual.bits[0] + (actual.bits[1] << 64), (
        actual.bits[0],
        actual.bits[1],
    )


@given(
    tag=U64S,
    key=st.lists(
        U64S,
        min_size=C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT,
        max_size=C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT,
    ),
    prefix=st.binary(min_size=16, max_size=16),
    data=st.binary(min_size=1, max_size=16),
)
def test_oh_tail_short_fprint(tag, key, prefix, data):
    """Compare fprint OH compression for the last block, when we must
    steal some data from the previous chunk."""
    expected = [
        oh_compress_one_block(
            key, split_block(prefix * (C.UMASH_OH_PARAM_COUNT // 2) + data), tag << 64
        ),
        oh_compress_one_block(
            key,
            split_block(prefix * (C.UMASH_OH_PARAM_COUNT // 2) + data),
            tag << 64,
            secondary=True,
        ),
    ]

    offset = len(prefix)
    n_bytes = len(data)
    # Copy to exactly-sized malloced buffers to help ASan.
    block = FFI.new("char[]", offset + n_bytes)
    FFI.memmove(block, prefix + data, offset + n_bytes)
    params = FFI.new("struct umash_params[1]")
    for i, param in enumerate(key):
        params[0].oh[i] = param

    actual = FFI.new("struct umash_oh[2]")
    C.oh_varblock_fprint(actual, params[0].oh, tag, block + offset, n_bytes)
    assert expected == [
        (actual[0].bits[0] + (actual[0].bits[1] << 64)),
        (actual[1].bits[0] + (actual[1].bits[1] << 64)),
    ], (
        actual[0].bits[0],
        actual[0].bits[1],
        actual[1].bits[0],
        actual[1].bits[1],
    )

    solo = C.oh_varblock(params[0].oh, tag, block + offset, n_bytes)
    assert (actual[0].bits[0], actual[0].bits[1]) == (solo.bits[0], solo.bits[1])


@given(
    tag=st.sampled_from([0, 2**64 - 1]),
    key_val=st.sampled_from([0, 2**64 - 1]) | U64S,
    byte_val=st.sampled_from([0x00, 0xFF]) | st.integers(min_value=0, max_value=255),
    n_bytes=st.integers(min_value=16, max_value=BLOCK_SIZE),
)
def test_oh_adversarial(tag, key_val, byte_val, n_bytes):
    """Compare OH compression with the reference for adversarial inputs:
    uniform keys, uniform data bytes, and extreme tag values."""
    key = [key_val] * C.UMASH_OH_PARAM_COUNT
    data = bytes([byte_val]) * n_bytes

    expected = oh_compress_one_block(key, split_block(data), tag << 64)

    block = FFI.new("char[]", n_bytes)
    FFI.memmove(block, data, n_bytes)
    params = FFI.new("struct umash_params[1]")
    for i, param in enumerate(key):
        params[0].oh[i] = param

    actual = C.oh_varblock(params[0].oh, tag, block, n_bytes)
    assert expected == actual.bits[0] + (actual.bits[1] << 64), (
        actual.bits[0],
        actual.bits[1],
    )


@given(
    tag=st.sampled_from([0, 2**64 - 1]),
    key_val=st.sampled_from([0, 2**64 - 1]) | U64S,
    byte_val=st.sampled_from([0x00, 0xFF]) | st.integers(min_value=0, max_value=255),
    n_bytes=st.integers(min_value=16, max_value=BLOCK_SIZE),
)
def test_oh_adversarial_fprint(tag, key_val, byte_val, n_bytes):
    """Compare combined OH compression with the reference for adversarial
    inputs: uniform keys, uniform data bytes, and extreme tag values."""
    key = [key_val] * (C.UMASH_OH_PARAM_COUNT + C.UMASH_OH_TWISTING_COUNT)
    data = bytes([byte_val]) * n_bytes

    expected = [
        oh_compress_one_block(key, split_block(data), tag << 64),
        oh_compress_one_block(key, split_block(data), tag << 64, secondary=True),
    ]

    block = FFI.new("char[]", n_bytes)
    FFI.memmove(block, data, n_bytes)
    params = FFI.new("struct umash_params[1]")
    for i, param in enumerate(key):
        params[0].oh[i] = param

    actual = FFI.new("struct umash_oh[2]")
    C.oh_varblock_fprint(actual, params[0].oh, tag, block, n_bytes)
    assert expected == [
        (actual[0].bits[0] + (actual[0].bits[1] << 64)),
        (actual[1].bits[0] + (actual[1].bits[1] << 64)),
    ], (
        actual[0].bits[0],
        actual[0].bits[1],
        actual[1].bits[0],
        actual[1].bits[1],
    )

    solo = C.oh_varblock(params[0].oh, tag, block, n_bytes)
    assert (actual[0].bits[0], actual[0].bits[1]) == (solo.bits[0], solo.bits[1])
