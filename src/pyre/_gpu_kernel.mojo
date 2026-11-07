"""GPU traversal over packed depth-first nodes, split by row chunk and tree.

Each block owns `chunk` consecutive rows (a power of two up to 32). Thread
`tid` evaluates row `tid % chunk` on trees `tid / chunk`, `+ groves`, ... and
accumulates privately. Per-row partials meet in dynamic shared memory and are
summed by one thread per row when there are few, otherwise by a warp per row.
Float32 rows are read from global memory: staging them in shared memory
measured slower, since it shrinks the L1 that caches nodes. Float64 rows are
staged as order-preserving keys when small, which removes the per-visit key
conversion.

Node encoding: see `_layout.mojo`. The GPU packs keyed float64 thresholds
and normalizes LE splits to LT wherever possible.
"""
from std.gpu import block_idx, block_dim, thread_idx, lane_id, WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.memory import external_memory, AddressSpace
from std.gpu.primitives import warp
from std.math import isnan, exp
from std.memory import bitcast

from ._layout import (
    OP_LT,
    OP_LE,
    OP_EQ,
    DEFAULT_DISTANT,
    COMPACT_OFFSET_BITS,
    COMPACT_FEATURE_BITS,
    WIDE_FEATURE_BITS,
)


@always_inline
def node_value[
    dtype: DType, words: Int
](node: SIMD[DType.uint32, words]) -> Scalar[dtype]:
    comptime if dtype == DType.float32:
        return bitcast[dtype, 1](node[0])
    else:
        comptime assert words == 4
        return bitcast[dtype, 1](SIMD[DType.uint32, 2](node[0], node[1]))


@always_inline
def warp_sum[dtype: DType](value: Scalar[dtype]) -> Scalar[dtype]:
    comptime if dtype == DType.float64:
        # Float64 shuffles are unavailable; exchange the two 32-bit halves.
        var total = value
        comptime for i in range(5):
            var halves = bitcast[DType.uint32, 2](total)
            var offset = UInt32(16 >> i)
            var other = SIMD[DType.uint32, 2](
                warp.shuffle_xor(halves[0], offset),
                warp.shuffle_xor(halves[1], offset),
            )
            total += bitcast[dtype, 1](other)
        return total
    else:
        return warp.sum(value)


@always_inline
def input_key(value: Float64) -> Int64:
    """Order-preserving key of an input; NaN maps to the Int64.MIN sentinel.

    Non-NaN keys are never Int64.MIN, and a NaN threshold key is Int64.MIN, so
    `key < threshold` is false for it as in IEEE. Callers test the sentinel.
    """
    var bits = bitcast[DType.int64, 1](value)
    var magnitude = bits & Int64.MAX
    if magnitude > Int64(0x7FF0000000000000):
        return Int64.MIN
    return -magnitude if bits < 0 else magnitude


@always_inline
def evaluate_tree[
    dtype: DType, words: Int, general: Bool, staged: Bool
](
    nodes: ImmPointer[UInt32, ImmutAnyOrigin],
    root: Int32,
    row: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    keys: Pointer[Int64, _, address_space=AddressSpace.SHARED],
) -> Scalar[dtype]:
    """Walk one tree for one row. Staged float64 rows read precomputed keys.

    Node indices stay 32-bit: the forest is limited to Int32 node indexing.
    """
    comptime compact = words == 2
    var index = UInt32(root)
    var node = nodes.unsafe_load[width=words, alignment=4 * words](
        Int(index * UInt32(words))
    )
    while True:
        var meta: UInt32
        var offset: UInt32
        var feature: UInt32
        comptime if compact:
            meta = node[1]
            offset = meta & ((1 << COMPACT_OFFSET_BITS) - 1)
            feature = (meta >> COMPACT_OFFSET_BITS) & (
                (1 << COMPACT_FEATURE_BITS) - 1
            )
        else:
            meta = node[2]
            offset = node[3]
            feature = meta & ((1 << WIDE_FEATURE_BITS) - 1)
        if offset == 0:
            break
        var condition: Bool
        comptime if dtype == DType.float64:
            # FP64 compare throughput is low on consumer GPUs; compare
            # order-preserving integer keys instead (see `float64_key`).
            var key: Int64
            comptime if staged:
                key = keys[unsafe_offset=Int(feature)]
            else:
                key = input_key(Float64(row[unsafe_offset=Int(feature)]))
            var missing = key == Int64.MIN
            var threshold = bitcast[DType.int64, 1](
                SIMD[DType.uint32, 2](node[0], node[1])
            )
            comptime if general:
                var op = (meta >> 29) & 3
                condition = (
                    (op == OP_LT and key < threshold)
                    or (op == OP_LE and key <= threshold)
                    or (op == OP_EQ and key == threshold)
                )
            else:
                condition = key < threshold
            if missing:
                condition = (meta & DEFAULT_DISTANT) != 0
        else:
            var value = row[unsafe_offset=Int(feature)]
            var threshold = node_value[dtype, words](node)
            comptime if general:
                var op = (meta >> 29) & 3
                condition = (
                    (op == OP_LT and value < threshold)
                    or (op == OP_LE and value <= threshold)
                    or (op == OP_EQ and value == threshold)
                )
            else:
                condition = value < threshold
            condition = condition or (
                (meta & DEFAULT_DISTANT) != 0 and isnan(value)
            )
        index += offset if condition else 1
        node = nodes.unsafe_load[width=words, alignment=4 * words](
            Int(index * UInt32(words))
        )
    return node_value[dtype, words](node)


def predict_kernel[
    dtype: DType,
    output_dtype: DType,
    mode: Int,
    words: Int,
    general: Bool,
    staged: Bool,
](
    nodes: ImmPointer[UInt32, ImmutAnyOrigin],
    roots: ImmPointer[Int32, ImmutAnyOrigin],
    x: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    y: MutPointer[Scalar[output_dtype], MutAnyOrigin],
    rows: Int32,
    trees: Int32,
    features: Int32,
    chunk_shift: Int32,
    average: Int32,
    bias: Float64,
    alpha: Float32,
    threshold: Scalar[dtype],
):
    """Dynamic shared memory: one partial per thread, then staged row keys.

    Staging converts each float64 input of the block's rows to a key once,
    instead of once per visited node; the host enables it for small rows.
    """
    comptime assert dtype.is_floating_point()
    comptime assert not staged or dtype == DType.float64
    var tid = thread_idx.x
    var threads = block_dim.x
    var shift = Int(chunk_shift)
    var chunk = 1 << shift
    var width = Int(features)
    var first = block_idx.x * chunk
    var active = min(chunk, Int(rows) - first)
    var partial = external_memory[
        Scalar[dtype], address_space=AddressSpace.SHARED, alignment=16
    ]()
    var keys = partial.unsafe_offset(threads).unsafe_bitcast[Int64]()
    comptime if staged:
        var base = x.unsafe_offset(first * width)
        for i in range(tid, active * width, threads):
            keys[unsafe_offset=i] = input_key(Float64(base[unsafe_offset=i]))
        barrier()
    var r = tid & (chunk - 1)
    var groves = threads >> shift
    var total = Scalar[dtype](0)
    if r < active:
        var row = x.unsafe_offset((first + r) * width)
        var row_keys = keys.unsafe_offset(r * width)
        for tree in range(tid >> shift, Int(trees), groves):
            total += evaluate_tree[dtype, words, general, staged](
                nodes, roots[unsafe_offset=tree], row, row_keys
            )
    partial[unsafe_offset=tid] = total
    barrier()
    if groves <= 16:
        # Narrow reductions: one thread per row beats a warp shuffle tree.
        if tid < active:
            var sum = partial[unsafe_offset=tid]
            for g in range(1, groves):
                sum += partial[unsafe_offset=g * chunk + tid]
            finish[mode](
                y, first + tid, sum, trees, average, bias, alpha, threshold
            )
    else:
        var lane = lane_id()
        for rr in range(tid // WARP_SIZE, active, threads // WARP_SIZE):
            var sum = Scalar[dtype](0)
            for g in range(Int(lane), groves, WARP_SIZE):
                sum += partial[unsafe_offset=g * chunk + rr]
            sum = warp_sum(sum)
            if lane == 0:
                finish[mode](
                    y, first + rr, sum, trees, average, bias, alpha, threshold
                )


@always_inline
def finish[
    dtype: DType, output_dtype: DType, //, mode: Int
](
    y: MutPointer[Scalar[output_dtype], MutAnyOrigin],
    row: Int,
    total: Scalar[dtype],
    trees: Int32,
    average: Int32,
    bias: Float64,
    alpha: Float32,
    threshold: Scalar[dtype],
):
    """Average, add the base score, apply the output transform and store."""
    comptime assert dtype.is_floating_point()
    var sum = total
    if average:
        sum /= Scalar[dtype](Int(trees))
    var value = Scalar[dtype](Float64(sum) + bias)
    comptime if mode != 0:
        var z = Scalar[dtype](alpha) * value
        var e = exp(-abs(z))
        value = 1 / (1 + e) if z >= 0 else e / (1 + e)
    comptime if mode == 1:
        y[unsafe_offset=2 * row] = Scalar[output_dtype](1 - value)
        y[unsafe_offset=2 * row + 1] = Scalar[output_dtype](value)
    elif mode == 2:
        y[unsafe_offset=row] = Scalar[output_dtype](value > threshold)
    else:
        y[unsafe_offset=row] = Scalar[output_dtype](value)
