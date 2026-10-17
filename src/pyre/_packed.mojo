"""CPU general traversal over packed depth-first nodes, owned by NuMojo.

Each node is one 64-bit word (compact) or two (wide), so a traversal step is
one node load per lane instead of separate child, feature, threshold, operator
and default loads. On ISAs without vector gathers (NEON) those loads are scalar;
on AVX2/AVX-512 they are fewer, wider gathers.

Layout: every split is normalized so that a true condition selects the distant
child at `node + offset`, a false condition the adjacent child at `node + 1`,
and a leaf has offset zero. GT/GE exchange children and become LE/LT, which is
exact for non-NaN inputs. Unlike the GPU, LE is never rewritten as LT on
`nextafter(t, +inf)`: worker tasks flush subnormals, which would make
`x <= 0` compare as `x < 0`. NaN inputs compare false and take the adjacent
child unless the default-distant bit is set; NaN thresholds keep their
original children, so every comparison is false, as in IEEE.

Compact (float32, trees below 2^16 nodes, at most 2^13 features):
  bits 0-31 value (threshold or leaf), 32-47 offset, 48-60 feature,
  61-62 operator, 63 default-distant.
Wide (otherwise): word 0 holds the value bits (float32 zero-extended);
  word 1 bits 0-28 feature, 29-30 operator, 31 default-distant, 32-63 offset.
"""

from std.math import isnan
from std.memory import bitcast
from balsa import Model, NodeType, Operator
from numojo import NDArray
from numojo.core.type_aliases import Shape

comptime _OP_LT = 0
comptime _OP_LE = 1
comptime _OP_EQ = 2
# Operator specializations: every split normalizes to LT, to LE, or mixed.
comptime _MIXED = 3
comptime _COMPACT_OFFSET_BITS = 16
comptime _COMPACT_FEATURE_BITS = 13
comptime _WIDE_FEATURE_BITS = 29


struct _Packed[dtype: DType](Movable):
    var nodes: NDArray[DType.uint64]
    var roots: NDArray[DType.int32]
    var wide: Bool
    # _OP_LT or _OP_LE when every split normalizes to it, otherwise _MIXED.
    var op: Int

    def __init__(out self, model: Model[Self.dtype]) raises:
        """Pack after validation; Balsa guarantees reachable, in-range nodes."""
        var count = 0
        var widest = 0
        for tree in model.trees:
            count += Int(tree.num_nodes)
            widest = max(widest, Int(tree.num_nodes))
        if count > Int(Int32.MAX) // 2:
            raise Error("Pyre CPU forest exceeds Int32 node indexing")
        var features = Int(model.num_feature)
        if features > (1 << _WIDE_FEATURE_BITS):
            raise Error("Pyre CPU forest exceeds feature indexing")
        self.wide = not (
            Self.dtype == DType.float32
            and widest <= (1 << _COMPACT_OFFSET_BITS)
            and features <= (1 << _COMPACT_FEATURE_BITS)
        )
        var words = 2 if self.wide else 1
        self.nodes = NDArray[DType.uint64](Shape(count * words))
        self.roots = NDArray[DType.int32](Shape(len(model.trees)))
        var nodes = self.nodes.unsafe_ptr()
        var uniform = -1
        var base = 0
        var pending = List[Int32]()
        var parents = List[Int]()
        for t in range(len(model.trees)):
            ref tree = model.trees[t]
            self.roots.itemset(t, Int32(base))
            var position = base
            pending.append(0)
            parents.append(-1)
            while len(pending):
                var n = Int(pending.pop())
                var parent = parents.pop()
                if parent >= 0:
                    var offset = UInt64(position - parent)
                    if self.wide:
                        nodes[unsafe_offset=2 * parent + 1] |= offset << 32
                    else:
                        nodes[unsafe_offset=parent] |= offset << 32
                var value: Scalar[Self.dtype]
                var meta = UInt64(0)
                if tree.node_type[n] == NodeType.LEAF:
                    value = tree.leaf_value[n]
                else:
                    value = tree.threshold[n]
                    var op = tree.cmp[n]
                    var left_distant = True
                    var code: Int
                    if isnan(value):
                        # Exchanging children would invert an always-false split.
                        code = _OP_LT
                    elif op == Operator.LT:
                        code = _OP_LT
                    elif op == Operator.LE:
                        code = _OP_LE
                    elif op == Operator.EQ:
                        code = _OP_EQ
                    elif op == Operator.GT:
                        code = _OP_LE
                        left_distant = False
                    elif op == Operator.GE:
                        code = _OP_LT
                        left_distant = False
                    else:
                        raise Error("Pyre does not support this split operator")
                    if uniform == -1:
                        uniform = code
                    elif uniform != code:
                        uniform = _MIXED
                    var distant = tree.cleft[
                        n
                    ] if left_distant else tree.cright[n]
                    var adjacent = tree.cright[
                        n
                    ] if left_distant else tree.cleft[n]
                    var flags = UInt64(code) << 29
                    if (tree.default_left[n] != 0) == left_distant:
                        flags |= UInt64(1) << 31
                    var feature = UInt64(tree.split_index[n])
                    if self.wide:
                        meta = flags | feature
                    else:
                        meta = (flags | (feature << 16)) << 32
                    # Adjacent child is popped next and stored at position + 1.
                    pending.append(distant)
                    parents.append(position)
                    pending.append(adjacent)
                    parents.append(-1)
                var bits: UInt64
                comptime if Self.dtype == DType.float32:
                    bits = UInt64(bitcast[DType.uint32, 1](value))
                else:
                    bits = bitcast[DType.uint64, 1](value)
                if self.wide:
                    nodes[unsafe_offset=2 * position] = bits
                    nodes[unsafe_offset=2 * position + 1] = meta
                else:
                    nodes[unsafe_offset=position] = bits | meta
                position += 1
            if position - base != Int(tree.num_nodes):
                raise Error("Pyre CPU packing found unreachable nodes")
            base = position
        # A forest of root leaves has no splits; any kernel evaluates it.
        self.op = _OP_LT if uniform == -1 else uniform

    def tree_range[
        width: Int, wide: Bool, op: Int
    ](
        self,
        x: NDArray[Self.dtype],
        num_features: Int,
        row: Int,
        count: Int,
        begin: Int,
        end: Int,
        var total: SIMD[Self.dtype, width],
    ) -> SIMD[Self.dtype, width]:
        """Continue the incoming total in original tree order."""
        comptime assert Self.dtype.is_floating_point()
        comptime words = 2 if wide else 1
        comptime offset_mask = (
            0xFFFFFFFF if wide else (1 << _COMPACT_OFFSET_BITS) - 1
        )
        comptime feature_shift = 0 if wide else 16
        comptime feature_bits = (
            _WIDE_FEATURE_BITS if wide else _COMPACT_FEATURE_BITS
        )
        comptime feature_mask = (1 << feature_bits) - 1
        var input = x.unsafe_ptr()
        var nodes = self.nodes.unsafe_ptr()
        var roots = self.roots.unsafe_ptr()
        comptime if width == 1:
            # A scalar fallback avoids masked-gather overhead on one-row calls.
            var offset_row = row * num_features
            for tree_id in range(begin, end):
                var index = Int(roots[unsafe_offset=tree_id])
                while True:
                    var info = nodes[unsafe_offset=words * index + words - 1]
                    var meta = UInt32(info if wide else info >> 32)
                    var offset = Int(UInt32(info >> 32)) & offset_mask
                    if offset == 0:
                        break
                    var value = input[
                        unsafe_offset=offset_row
                        + (Int(meta >> UInt32(feature_shift)) & feature_mask)
                    ]
                    var threshold = _value[Self.dtype](
                        nodes[unsafe_offset=words * index]
                    )
                    var condition: Bool
                    comptime if op == _OP_LT:
                        condition = value < threshold
                    elif op == _OP_LE:
                        condition = value <= threshold
                    else:
                        var code = (meta >> 29) & 3
                        condition = (
                            (code == _OP_LT and value < threshold)
                            or (code == _OP_LE and value <= threshold)
                            or (code == _OP_EQ and value == threshold)
                        )
                    if isnan(value):
                        condition = (meta >> 31) != 0
                    index += offset if condition else 1
                total += _value[Self.dtype](nodes[unsafe_offset=words * index])
        else:
            var lanes = SIMD[DType.int64, width](0)
            comptime for lane in range(width):
                lanes[lane] = Int64(lane)
            var valid = lanes.lt(Int64(count))
            var row_offsets = (lanes + Int64(row)) * Int64(num_features)
            for tree_id in range(begin, end):
                var ids = SIMD[DType.int32, width](roots[unsafe_offset=tree_id])
                var active = valid
                var value_word: SIMD[DType.uint64, width]
                while True:
                    # Finished lanes stay at their leaf; tail lanes at root.
                    var info: SIMD[DType.uint64, width]
                    comptime if wide:
                        value_word = nodes.unsafe_gather(ids * 2)
                        info = nodes.unsafe_gather(ids * 2 + 1)
                    else:
                        value_word = nodes.unsafe_gather(ids)
                        info = value_word
                    var meta = (info if wide else info >> 32).cast[
                        DType.uint32
                    ]()
                    # Both layouts keep the offset in the high word.
                    var offset = (info >> 32).cast[DType.uint32]() & UInt32(
                        offset_mask
                    )
                    active &= offset.ne(0)
                    if not active.reduce_or():
                        break
                    var features = (
                        (meta >> UInt32(feature_shift)) & UInt32(feature_mask)
                    ).cast[DType.int64]()
                    # Masking is essential: leaf words and absent rows are not
                    # valid input addresses, even when their result is unused.
                    var values = input.unsafe_gather(
                        row_offsets + features, active
                    )
                    var thresholds = _values[Self.dtype, width](value_word)
                    var condition: SIMD[DType.bool, width]
                    comptime if op == _OP_LT:
                        condition = values.lt(thresholds)
                    elif op == _OP_LE:
                        condition = values.le(thresholds)
                    else:
                        var code = (meta >> 29) & 3
                        condition = (
                            (code.eq(_OP_LT) & values.lt(thresholds))
                            | (code.eq(_OP_LE) & values.le(thresholds))
                            | (code.eq(_OP_EQ) & values.eq(thresholds))
                        )
                    condition = isnan(values).select(
                        (meta >> 31).ne(0), condition
                    )
                    ids = active.select(
                        ids
                        + condition.select(
                            offset.cast[DType.int32](),
                            SIMD[DType.int32, width](1),
                        ),
                        ids,
                    )
                total += valid.select(
                    _values[Self.dtype, width](value_word),
                    SIMD[Self.dtype, width](0),
                )
        return total


@always_inline
def _value[dtype: DType](word: UInt64) -> Scalar[dtype]:
    comptime if dtype == DType.float32:
        return bitcast[dtype, 1](UInt32(word))
    else:
        return bitcast[dtype, 1](word)


@always_inline
def _values[
    dtype: DType, width: Int
](words: SIMD[DType.uint64, width]) -> SIMD[dtype, width]:
    comptime if dtype == DType.float32:
        return bitcast[dtype, width](words.cast[DType.uint32]())
    else:
        return bitcast[dtype, width](words)
