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
from std.bit import log2_floor

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
        comptime for i in range(log2_floor(WARP_SIZE)):
            var halves = bitcast[DType.uint32, 2](total)
            var offset = UInt32((WARP_SIZE // 2) >> i)
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


def stage_type[dtype: DType]() -> DType:
    """Staged element type: float64 inputs are staged as comparison keys."""
    return DType.int64 if dtype == DType.float64 else dtype


@always_inline
def split[
    dtype: DType, words: Int, general: Bool, staged: Bool
](
    node: SIMD[DType.uint32, words],
    row: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    stage: Pointer[
        Scalar[stage_type[dtype]()], _, address_space=AddressSpace.SHARED
    ],
    stride: Int,
) -> UInt32:
    """Index step from `node` for one row: 0 at a leaf, else 1 or the offset.

    Unstaged rows read `row[feature]` from global memory; staged rows read
    `stage[feature * stride]`, a transposed copy (float64 as keys).
    """
    comptime compact = words == 2
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
        return 0
    var condition: Bool
    comptime if dtype == DType.float64:
        # FP64 compare throughput is low on consumer GPUs; compare
        # order-preserving integer keys instead (see `float64_key`).
        var key: Int64
        comptime if staged:
            key = rebind[Int64](stage[unsafe_offset=Int(feature) * stride])
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
        var value: Scalar[dtype]
        comptime if staged:
            value = rebind[Scalar[dtype]](
                stage[unsafe_offset=Int(feature) * stride]
            )
        else:
            value = row[unsafe_offset=Int(feature)]
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
    return offset if condition else 1


@always_inline
def evaluate[
    dtype: DType,
    words: Int,
    general: Bool,
    staged: Bool,
    rows: Int,
    trees: Int,
](
    nodes: ImmPointer[UInt32, ImmutAnyOrigin],
    roots: SIMD[DType.uint32, trees],
    x: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    stage: Pointer[
        Scalar[stage_type[dtype]()], _, address_space=AddressSpace.SHARED
    ],
    row_index: SIMD[DType.int32, rows],
    width: Int,
    stride: Int,
) -> SIMD[dtype, rows]:
    """Walk `rows x trees` independent chains in lockstep; sum leaves per row.

    Interleaved walks keep several node loads in flight per thread. A chain
    at its leaf steps by zero, reloading its cached leaf, until all arrive.
    Node indices stay 32-bit: the forest is limited to Int32 node indexing.
    """
    comptime chains = rows * trees
    var index = SIMD[DType.uint32, chains]()
    comptime for c in range(chains):
        index[c] = roots[c % trees]
    while True:
        var step = SIMD[DType.uint32, chains]()
        var leaf = SIMD[dtype, rows]()
        comptime for c in range(chains):
            comptime r = c // trees
            var node = nodes.unsafe_load[width=words, alignment=4 * words](
                Int(index[c] * UInt32(words))
            )
            var r_at = Int(row_index[r])
            step[c] = split[dtype, words, general, staged](
                node,
                x.unsafe_offset(r_at * width),
                stage.unsafe_offset(r_at),
                stride,
            )
            leaf[r] += node_value[dtype, words](node)
        if step.reduce_or() == 0:
            return leaf
        index += step


def predict_kernel[
    dtype: DType,
    output_dtype: DType,
    mode: Int,
    words: Int,
    general: Bool,
    staged: Bool,
    rows_per_thread: Int,
    trees_per_thread: Int,
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
    """Dynamic shared memory: `threads * rows_per_thread` partials, then
    staged row keys.

    The block's `chunk` rows split into `lanes = chunk / rows_per_thread`; a
    thread walks rows `lane + k * lanes` over `trees_per_thread` consecutive
    trees at a time, striding by groves. Staging converts each float64 input
    of the block's rows to a key once, instead of once per visited node.
    """
    comptime assert dtype.is_floating_point()
    comptime R = rows_per_thread
    comptime Q = trees_per_thread
    var tid = thread_idx.x
    var threads = block_dim.x
    var chunk = 1 << Int(chunk_shift)
    var lanes = chunk // R
    var width = Int(features)
    var first = block_idx.x * chunk
    var active = min(chunk, Int(rows) - first)
    var partial = external_memory[
        Scalar[dtype], address_space=AddressSpace.SHARED, alignment=16
    ]()
    comptime S = stage_type[dtype]()
    # Partials end 8-byte aligned: `threads * R` is even past one warp.
    var stage = partial.unsafe_offset(threads * R).unsafe_bitcast[Scalar[S]]()
    var block_x = x.unsafe_offset(first * width)
    # Odd stride: lanes on distinct rows and fill on distinct features both
    # land on distinct banks.
    var stride = chunk + 1
    comptime if staged:
        for i in range(tid, active * width, threads):
            var r = i // width
            var f = i - r * width
            var v = block_x[unsafe_offset=i]
            comptime if dtype == DType.float64:
                stage[unsafe_offset=f * stride + r] = rebind[Scalar[S]](
                    input_key(Float64(v))
                )
            else:
                stage[unsafe_offset=f * stride + r] = rebind[Scalar[S]](v)
        barrier()
    var lane = tid % lanes
    var groves = threads // lanes
    var grove = tid // lanes
    # Rows past the batch end repeat the last row; their sums are not stored.
    var row_index = SIMD[DType.int32, R]()
    comptime for k in range(R):
        row_index[k] = Int32(min(lane + k * lanes, active - 1))
    var total = SIMD[dtype, R]()
    if lane < active:
        var count = Int(trees)
        var tree = grove * Q
        while tree + Q <= count:
            var root = SIMD[DType.uint32, Q]()
            comptime for q in range(Q):
                root[q] = UInt32(roots[unsafe_offset=tree + q])
            total += evaluate[dtype, words, general, staged, R, Q](
                nodes, root, block_x, stage, row_index, width, stride
            )
            tree += groves * Q
        comptime if Q > 1:
            # Fewer than Q trees remain for this grove: walk them singly.
            while tree < count:
                var root = SIMD[DType.uint32, 1](
                    UInt32(roots[unsafe_offset=tree])
                )
                total += evaluate[dtype, words, general, staged, R, 1](
                    nodes, root, block_x, stage, row_index, width, stride
                )
                tree += 1
    comptime for k in range(R):
        partial[unsafe_offset=grove * chunk + lane + k * lanes] = total[k]
    barrier()
    if groves <= 16:
        # Narrow reductions: one thread per row beats a warp shuffle tree.
        for r in range(tid, active, threads):
            var sum = partial[unsafe_offset=r]
            for g in range(1, groves):
                sum += partial[unsafe_offset=g * chunk + r]
            finish[mode](
                y, first + r, sum, trees, average, bias, alpha, threshold
            )
    else:
        var warp_lane = lane_id()
        for r in range(tid // WARP_SIZE, active, threads // WARP_SIZE):
            var sum = Scalar[dtype](0)
            for g in range(Int(warp_lane), groves, WARP_SIZE):
                sum += partial[unsafe_offset=g * chunk + r]
            sum = warp_sum(sum)
            if warp_lane == 0:
                finish[mode](
                    y, first + r, sum, trees, average, bias, alpha, threshold
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
