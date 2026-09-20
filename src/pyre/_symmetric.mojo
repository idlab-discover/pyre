"""CPU-only symmetric LE trees; packed storage and tile bins owned by NuMojo."""

from std.math import isfinite, isnan
from balsa import Model, NodeType, Operator
from numojo import NDArray
from numojo.core.type_aliases import Shape


struct _Symmetric[dtype: DType](Movable):
    var num_features: Int
    var enabled: Bool
    var binned: Bool
    var features: NDArray[DType.int32]
    var thresholds: NDArray[Self.dtype]
    var defaults: NDArray[DType.bool]
    var leaves: NDArray[Self.dtype]
    var splits: NDArray[DType.int64]
    var leaf_offsets: NDArray[DType.int64]
    var borders: NDArray[Self.dtype]
    var border_offsets: NDArray[DType.int64]
    var split_bins: NDArray[DType.uint8]

    def __init__(out self, model: Model[Self.dtype]) raises:
        self.num_features = Int(model.num_feature)
        self.enabled = False
        self.binned = False
        self.features = NDArray[DType.int32](Shape(0))
        self.thresholds = NDArray[Self.dtype](Shape(0))
        self.defaults = NDArray[DType.bool](Shape(0))
        self.leaves = NDArray[Self.dtype](Shape(0))
        self.splits = NDArray[DType.int64](Shape(0))
        self.leaf_offsets = NDArray[DType.int64](Shape(0))
        self.borders = NDArray[Self.dtype](Shape(0))
        self.border_offsets = NDArray[DType.int64](Shape(0))
        self.split_bins = NDArray[DType.uint8](Shape(0))
        var features = List[Int32]()
        var thresholds = List[Scalar[Self.dtype]]()
        var defaults = List[Bool]()
        var leaves = List[Scalar[Self.dtype]]()
        var splits: List[Int64] = [0]
        var offsets: List[Int64] = [0]
        # Structural validation has already run. Recover topology independently
        # of serialized node numbering; bit d means the right edge at depth d.
        for tree in model.trees:
            var frontier: List[Int] = [0]
            var depth = 0
            while tree.node_type[frontier[0]] != NodeType.LEAF:
                if depth >= 16:
                    return
                var first = frontier[0]
                if tree.cmp[first] != Operator.LE or not isfinite(
                    tree.threshold[first]
                ):
                    return
                var next = List[Int](length=2 * len(frontier), fill=0)
                for i in range(len(frontier)):
                    var node = frontier[i]
                    if (
                        tree.node_type[node] == NodeType.LEAF
                        or tree.cmp[node] != Operator.LE
                        or tree.split_index[node] != tree.split_index[first]
                        or tree.threshold[node] != tree.threshold[first]
                        or tree.default_left[node] != tree.default_left[first]
                    ):
                        return
                    next[i] = Int(tree.cleft[node])
                    next[i + len(frontier)] = Int(tree.cright[node])
                features.append(tree.split_index[first])
                thresholds.append(tree.threshold[first])
                defaults.append(tree.default_left[first] != 0)
                frontier = next^
                depth += 1
            for node in frontier:
                if tree.node_type[node] != NodeType.LEAF:
                    return
                leaves.append(tree.leaf_value[node])
            splits.append(Int64(len(features)))
            offsets.append(Int64(len(leaves)))
        self.features = NDArray[DType.int32](Shape(len(features)))
        self.thresholds = NDArray[Self.dtype](Shape(len(features)))
        self.defaults = NDArray[DType.bool](Shape(len(features)))
        self.leaves = NDArray[Self.dtype](Shape(len(leaves)))
        self.splits = NDArray[DType.int64](Shape(len(splits)))
        self.leaf_offsets = NDArray[DType.int64](Shape(len(offsets)))
        for i in range(len(features)):
            self.features.itemset(i, features[i])
            self.thresholds.itemset(i, thresholds[i])
            self.defaults.itemset(i, defaults[i])
        for i in range(len(leaves)):
            self.leaves.itemset(i, leaves[i])
        for i in range(len(splits)):
            self.splits.itemset(i, splits[i])
            self.leaf_offsets.itemset(i, offsets[i])
        self.enabled = True
        # Byte 255 is reserved for NaNs, so each feature may have 254 borders.
        # More borders retain the direct compact evaluator without narrowing.
        var by_feature = List[List[Scalar[Self.dtype]]]()
        for _ in range(Int(model.num_feature)):
            by_feature.append(List[Scalar[Self.dtype]]())
        for s in range(len(features)):
            ref values = by_feature[Int(features[s])]
            var value = thresholds[s]
            var pos = 0
            while pos < len(values) and values[pos] < value:
                pos += 1
            if pos < len(values) and values[pos] == value:
                continue
            if len(values) == 254:
                return
            values.append(value)
            for j in range(len(values) - 1, pos, -1):
                values[j] = values[j - 1]
            values[pos] = value
        var count = 0
        for values in by_feature:
            count += len(values)
        self.borders = NDArray[Self.dtype](Shape(count))
        self.border_offsets = NDArray[DType.int64](Shape(len(by_feature) + 1))
        self.split_bins = NDArray[DType.uint8](Shape(len(features)))
        var offset = 0
        for f in range(len(by_feature)):
            self.border_offsets.itemset(f, Int64(offset))
            for value in by_feature[f]:
                self.borders.itemset(offset, value)
                offset += 1
        self.border_offsets.itemset(len(by_feature), Int64(offset))
        for s in range(len(features)):
            ref values = by_feature[Int(features[s])]
            for j in range(len(values)):
                if values[j] == thresholds[s]:
                    self.split_bins.itemset(s, UInt8(j))
                    break
        # Comparison reuse must amortize preparation. Batch cutoff is applied
        # by Forest. Policy is deliberately conservative pending wider sweeps.
        self.binned = count > 0 and count < len(features)

    def prepare[
        bin_origin: MutOrigin, //, width: Int
    ](
        self,
        x: NDArray[Self.dtype],
        bins: Pointer[UInt8, bin_origin],
        tile: Int,
        size: Int,
        stride: Int,
    ):
        var lanes = SIMD[DType.int64, width](0)
        comptime for lane in range(width):
            lanes[lane] = Int64(lane)
        for f in range(self.num_features):
            var begin = Int(self.border_offsets.unsafe_ptr()[unsafe_offset=f])
            var end = Int(self.border_offsets.unsafe_ptr()[unsafe_offset=f + 1])
            if begin == end:
                continue
            for row in range(0, size, width):
                var valid = lanes.lt(Int64(min(width, size - row)))
                var values = x.unsafe_ptr().unsafe_gather(
                    (lanes + Int64(tile + row)) * Int64(self.num_features)
                    + Int64(f),
                    valid,
                )
                var ordinal = SIMD[DType.uint8, width](0)
                for j in range(begin, end):
                    ordinal += values.gt(
                        self.borders.unsafe_ptr()[unsafe_offset=j]
                    ).cast[DType.uint8]()
                ordinal = isnan(values).select(
                    SIMD[DType.uint8, width](255), ordinal
                )
                bins.unsafe_offset(f * stride + row).unsafe_store(ordinal)

    def tree_range[
        bin_origin: Origin, //, width: Int, quantized: Bool
    ](
        self,
        x: NDArray[Self.dtype],
        bins: Pointer[UInt8, bin_origin],
        row: Int,
        local_row: Int,
        count: Int,
        stride: Int,
        begin: Int,
        end: Int,
        var total: SIMD[Self.dtype, width],
    ) -> SIMD[Self.dtype, width]:
        var lanes = SIMD[DType.int64, width](0)
        comptime for lane in range(width):
            lanes[lane] = Int64(lane)
        var valid = lanes.lt(Int64(count))
        for t in range(begin, end):
            var idx = SIMD[DType.int32, width](0)
            var start = Int(self.splits.unsafe_ptr()[unsafe_offset=t])
            for s in range(
                start, Int(self.splits.unsafe_ptr()[unsafe_offset=t + 1])
            ):
                var f = Int(self.features.unsafe_ptr()[unsafe_offset=s])
                var right: SIMD[DType.bool, width]
                comptime if quantized:
                    var value = bins.unsafe_offset(
                        f * stride + local_row
                    ).unsafe_load[width=width]()
                    right = value.eq(255).select(
                        SIMD[DType.bool, width](
                            fill=not self.defaults.unsafe_ptr()[unsafe_offset=s]
                        ),
                        value.gt(self.split_bins.unsafe_ptr()[unsafe_offset=s]),
                    )
                else:
                    var value = x.unsafe_ptr().unsafe_gather(
                        (lanes + Int64(row)) * Int64(self.num_features)
                        + Int64(f),
                        valid,
                    )
                    right = isnan(value).select(
                        SIMD[DType.bool, width](
                            fill=not self.defaults.unsafe_ptr()[unsafe_offset=s]
                        ),
                        value.gt(self.thresholds.unsafe_ptr()[unsafe_offset=s]),
                    )
                idx |= right.cast[DType.int32]() << Int32(s - start)
            total += (
                self.leaves.unsafe_ptr()
                .unsafe_offset(
                    Int(self.leaf_offsets.unsafe_ptr()[unsafe_offset=t])
                )
                .unsafe_gather(idx, valid)
            )
        return total
