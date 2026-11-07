"""Packed depth-first node layout shared by the CPU and GPU evaluators.

Every split is normalized so that a true condition selects the distant child at
`node + offset` and a false condition the adjacent child at `node + 1`; a leaf
has offset zero and stores its value in the threshold slot. The condition is
`x < t`, `x <= t` or `x == t`; NaN compares false and goes to the adjacent child
unless the default-distant bit is set.

Compact nodes are two 32-bit words, `[threshold, meta]`, with 16 offset bits
and 13 feature bits in `meta`. Wide nodes are four words,
`[threshold (one or two words), meta, offset]`, with 29 feature bits in `meta`.
Both encodings place the operator in bits 29-30 and the default-distant flag in
bit 31 of `meta`. Keyed float64 thresholds hold `float64_key` integers instead
of IEEE values, for GPU integer comparison; leaves always hold values.
"""
from balsa import Model, NodeType, Operator
from std.math import inf, nextafter
from std.memory import bitcast

comptime OP_LT = UInt32(0)
comptime OP_LE = UInt32(1)
comptime OP_EQ = UInt32(2)
comptime DEFAULT_DISTANT = UInt32(1) << 31
comptime COMPACT_OFFSET_BITS = 16
comptime COMPACT_FEATURE_BITS = 13
comptime WIDE_FEATURE_BITS = 29


def float64_key(value: Float64) -> Int64:
    """Order-preserving integer key: sign-magnitude to two's complement.

    For non-NaN values, keys compare exactly as IEEE values do, including
    -0 == +0. A NaN threshold maps to Int64.MIN, which no input key reaches,
    so every comparison against it is false, as in IEEE.
    """
    var bits = bitcast[DType.int64, 1](value)
    var magnitude = bits & Int64.MAX
    if magnitude > Int64(0x7FF0000000000000):
        return Int64.MIN
    return -magnitude if bits < 0 else magnitude


def node_words[dtype: DType](model: Model[dtype]) raises -> Int:
    """Two words when float32 offsets and features fit, otherwise four."""
    var widest = 0
    var count = 0
    for tree in model.trees:
        if count > 2147483647 - Int(tree.num_nodes):
            raise Error("Pyre forest exceeds Int32 node indexing")
        count += Int(tree.num_nodes)
        widest = max(widest, Int(tree.num_nodes))
    var features = Int(model.num_feature)
    if features > (1 << WIDE_FEATURE_BITS):
        raise Error("Pyre forest exceeds feature indexing")
    var compact = (
        dtype == DType.float32
        and widest < (1 << COMPACT_OFFSET_BITS)
        and features <= (1 << COMPACT_FEATURE_BITS)
    )
    # Evaluators index 32-bit words of four-word nodes with 32-bit arithmetic.
    if not compact and count >= (1 << 30):
        raise Error("Pyre forest exceeds wide-node indexing")
    # A distant offset is at most the size of the tree it lives in.
    return 2 if compact else 4


def node_count[dtype: DType](model: Model[dtype]) -> Int:
    var count = 0
    for tree in model.trees:
        count += Int(tree.num_nodes)
    return count


def pack[
    dtype: DType, *, keyed: Bool, strict: Bool
](
    model: Model[dtype],
    nodes: MutPointer[UInt32, _],
    roots: MutPointer[Int32, _],
    words: Int,
) raises -> UInt32:
    """Write depth-first normalized nodes; return a bitmask of used operators.

    Bit `1 << code` is set for each operator code present on a split. A true
    condition selects the distant child, stored after the adjacent child's
    whole subtree; GT/GE become LE/LT with the children exchanged. Unless
    `strict`, LE becomes LT on the next representable threshold where one
    exists. Strict packing keeps LE, so no subnormal threshold is introduced
    for evaluators that may run with denormals flushed to zero.
    """
    var used = UInt32(0)
    var base = 0
    var pending = List[Int32]()
    var parents = List[Int]()
    for t in range(len(model.trees)):
        ref tree = model.trees[t]
        roots[unsafe_offset=t] = Int32(base)
        var position = base
        pending.append(0)
        parents.append(-1)
        while len(pending):
            var n = Int(pending.pop())
            var parent = parents.pop()
            if parent >= 0:
                var offset = UInt32(position - parent)
                if words == 2:
                    nodes[unsafe_offset=2 * parent + 1] |= offset
                else:
                    nodes[unsafe_offset=4 * parent + 3] = offset
            var value: Scalar[dtype]
            var meta = UInt32(0)
            if tree.node_type[n] == NodeType.LEAF:
                value = tree.leaf_value[n]
            else:
                value = tree.threshold[n]
                var op = tree.cmp[n]
                var left_distant = True
                var code: UInt32
                if value != value:
                    # Every comparison with NaN is false, so the split always
                    # takes the right child; exchanging children would invert it.
                    code = OP_LT
                elif op == Operator.LT:
                    code = OP_LT
                elif op == Operator.LE:
                    code = OP_LE
                elif op == Operator.EQ:
                    code = OP_EQ
                elif op == Operator.GT:
                    code = OP_LE
                    left_distant = False
                elif op == Operator.GE:
                    code = OP_LT
                    left_distant = False
                else:
                    raise Error("Pyre does not support this split operator")
                comptime if not strict:
                    # For every non-NaN x, x <= t exactly when
                    # x < nextafter(t, inf), except t = +inf.
                    if code == OP_LE and value != inf[dtype]():
                        value = nextafter(value, inf[dtype]())
                        code = OP_LT
                used |= UInt32(1) << code
                var left = tree.cleft[n]
                var right = tree.cright[n]
                var distant = left if left_distant else right
                var adjacent = right if left_distant else left
                meta = code << 29
                if (tree.default_left[n] != 0) == left_distant:
                    meta |= DEFAULT_DISTANT
                var feature = UInt32(tree.split_index[n])
                if words == 2:
                    meta |= feature << COMPACT_OFFSET_BITS
                else:
                    meta |= feature
                # Adjacent child is popped next and stored at position + 1.
                pending.append(distant)
                parents.append(position)
                pending.append(adjacent)
                parents.append(-1)
            var at = words * position
            comptime if dtype == DType.float32:
                nodes[unsafe_offset=at] = bitcast[DType.uint32, 1](value)
                if words == 4:
                    nodes[unsafe_offset=at + 1] = 0
            else:
                var bits = bitcast[DType.uint64, 1](value)
                comptime if keyed:
                    if tree.node_type[n] != NodeType.LEAF:
                        bits = bitcast[DType.uint64, 1](
                            float64_key(Float64(value))
                        )
                var halves = bitcast[DType.uint32, 2](bits)
                nodes[unsafe_offset=at] = halves[0]
                nodes[unsafe_offset=at + 1] = halves[1]
            if words == 2:
                nodes[unsafe_offset=at + 1] = meta
            else:
                nodes[unsafe_offset=at + 2] = meta
                nodes[unsafe_offset=at + 3] = 0
            position += 1
        if position - base != Int(tree.num_nodes):
            raise Error("Pyre packing found unreachable nodes")
        base = position
    return used


def path_length[dtype: DType](model: Model[dtype]) -> Float64:
    """Sum over trees of the mean root-to-leaf node count, a per-row cost.

    Every tree contributes at least one evaluated node (its leaf).
    """
    var total = Float64(0)
    var stack = List[Int]()
    var depths = List[Int]()
    for tree in model.trees:
        var leaves = 0
        var sum = 0
        stack.append(0)
        depths.append(1)
        while len(stack):
            var n = stack.pop()
            var d = depths.pop()
            if tree.node_type[n] == NodeType.LEAF:
                leaves += 1
                sum += d
            else:
                stack.append(Int(tree.cleft[n]))
                depths.append(d + 1)
                stack.append(Int(tree.cright[n]))
                depths.append(d + 1)
        total += Float64(sum) / Float64(max(1, leaves))
    return total
