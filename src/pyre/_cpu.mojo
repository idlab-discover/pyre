"""Private CPU preparation and execution; Forest is the public interface."""

from std.math import ceildiv, exp, isnan
from std.sys import simd_width_of, size_of
from std.sys.info import num_physical_cores
from std.runtime import initialize_runtime
from max.runtime.asyncrt import parallelism_level
from ._executor import _Executor
from ._validation import validate_inference
from ._symmetric import _Symmetric
from ._layout import (
    OP_LT,
    OP_LE,
    DEFAULT_DISTANT,
    COMPACT_OFFSET_BITS,
    COMPACT_FEATURE_BITS,
    WIDE_FEATURE_BITS,
    node_count,
    node_words,
    pack,
    path_length,
)
from balsa import Model, NodeType, TaskType
from std.memory import bitcast
from numojo import NDArray
from numojo.core.type_aliases import Shape


# Internal locality policy; SIMD width and worker partitioning remain independent.
# Private execution strategies, also used by differential benchmark controls.
comptime _AUTO = -1
comptime _GENERAL = 0
comptime _DIRECT = 1
comptime _BINNED = 2
comptime _CONSTANT = 3
comptime _ROW_TILE = 256
comptime _TREE_GROUP = 4
# Gather traversal is latency-bound: 32-row blocks walking two trees at once
# keep 64 lanes in flight; one-row calls walk up to 32 trees per vector.
comptime _WIDE = 32
comptime _INTERLEAVE = 2
comptime _SINGLE = 32
# A task should carry several microseconds of work: waking a pooled worker and
# joining it costs a few microseconds on typical operating systems.
comptime _GRAIN_NS = 8000.0
# Operator specializations of the packed generic evaluator.
comptime _OPS_LT = 0
comptime _OPS_LE = 1
comptime _OPS_ANY = 2


struct _CPUPlan[dtype: DType = DType.float32](Movable):
    """A packed forest with SIMD CPU inference. Input precision must match.

    CPU inference vectorizes independent rows, preserving tree summation order.
    Supports scalar regression/identity and binary classification/sigmoid.
    """

    var _strategy: Int
    var _constant: Bool
    var _constant_total: Scalar[Self.dtype]
    # Inference facts; Balsa's per-field node storage is released after packing.
    var _features: Int
    var _trees: Int
    var _classifier: Bool
    var _average: Bool
    var _bias: Float64
    var _alpha: Float32
    var _symmetric: _Symmetric[Self.dtype]
    # Depth-first packed nodes (see _layout.mojo) for the generic evaluator.
    var _nodes: NDArray[DType.uint32]
    var _roots: NDArray[DType.int32]
    var _words: Int
    var _ops: Int
    # Mean evaluated nodes per row, for the task-count estimate.
    var _path: Float64
    var _cores: Int
    var _executor: _Executor

    def __init__(
        out self, var model: Model[Self.dtype], *, strategy: Int = _AUTO
    ) raises:
        """Validate support, pack nodes and recover symmetric trees.

        The model is consumed: only packed storage and scalar facts remain.
        Pass model.copy() to retain an editable source. Balsa remains
        responsible for structure.
        """
        _ = validate_inference(model)
        self._symmetric = _Symmetric(model)
        self._words = node_words(model)
        self._path = path_length(model)
        self._cores = max(1, num_physical_cores())
        self._nodes = NDArray[DType.uint32](
            Shape(max(1, node_count(model) * self._words))
        )
        self._roots = NDArray[DType.int32](Shape(len(model.trees)))
        # Strict packing keeps LE: workers flush denormals, so a nextafter
        # threshold above zero could compare as zero.
        var used = pack[keyed=False, strict=True](
            model,
            self._nodes.unsafe_ptr(),
            self._roots.unsafe_ptr(),
            self._words,
        )
        if used & ~(UInt32(1) << OP_LT) == 0:
            self._ops = _OPS_LT
        elif used & ~(UInt32(1) << OP_LE) == 0:
            self._ops = _OPS_LE
        else:
            self._ops = _OPS_ANY
        self._strategy = strategy
        self._constant = True
        self._constant_total = 0
        for tree in model.trees:
            self._constant &= tree.node_type[0] == NodeType.LEAF
        if self._constant:
            # Preserve the exact incoming-zero and ordered-sum semantics.
            for tree in model.trees:
                self._constant_total += tree.leaf_value[0]
        self._features = Int(model.num_feature)
        self._trees = len(model.trees)
        self._classifier = model.task_type == TaskType.BINARY_CLF
        self._average = model.average_tree_output != 0
        self._bias = model.base_scores[0]
        self._alpha = model.sigmoid_alpha
        self._executor = _Executor()
        self._use_strategy(strategy)

    def _use_strategy(mut self, strategy: Int) raises:
        """Private differential-testing control; preserve storage and facts.

        Production construction uses AUTO. Validation prevents a diagnostic
        override from selecting a kernel that cannot evaluate the model.
        """
        if strategy < _AUTO or strategy > _CONSTANT:
            raise Error("Unknown internal CPU strategy")
        if (
            strategy == _DIRECT
            and not self._symmetric.enabled
            or strategy == _BINNED
            and not self._symmetric.binned
            or strategy == _CONSTANT
            and not self._constant
        ):
            raise Error("Requested CPU strategy cannot evaluate this model")
        self._strategy = strategy

    def kernel(self, rows: Int) -> Int:
        """Choose from construction-time facts; never inspect nodes per call.

        Internal diagnostic seam: callers of Forest need not know strategies.
        Forced strategies are validated at construction for differential tests.
        """
        if self._strategy != _AUTO:
            return self._strategy
        if self._constant:
            return _CONSTANT
        if not self._symmetric.enabled:
            return _GENERAL
        if self._symmetric.binned and rows >= 256:
            return _BINNED
        if rows == 1:
            # One row walks many trees per vector on packed nodes, overlapping
            # their load chains; symmetric levels would fill one lane.
            return _GENERAL
        return _DIRECT

    def num_features(self) -> Int:
        return self._features

    def num_trees(self) -> Int:
        return self._trees

    def is_classifier(self) -> Bool:
        return self._classifier

    def _check(self, x: NDArray[Self.dtype]) raises:
        if x.ndim != 2:
            raise Error("Pyre input must be rank two [rows, features]")
        if x.shape[1] != self.num_features():
            raise Error("Pyre input feature count does not match the forest")
        if (
            not x.flags.C_CONTIGUOUS
            or (x.shape[0] > 1 and x.strides[0] != self.num_features())
            or (self.num_features() > 1 and x.strides[1] != 1)
        ):
            raise Error("Pyre MVP requires C-contiguous input")

    @always_inline
    def _condition[
        ops: Int, width: Int
    ](
        self,
        value: SIMD[Self.dtype, width],
        threshold: SIMD[Self.dtype, width],
        meta: SIMD[DType.uint32, width],
    ) -> SIMD[DType.bool, width]:
        """True selects the distant child; NaN takes it only by default."""
        var condition: SIMD[DType.bool, width]
        comptime if ops == _OPS_LT:
            condition = value.lt(threshold)
        elif ops == _OPS_LE:
            condition = value.le(threshold)
        else:
            var op = (meta >> 29) & 3
            condition = (
                (op.eq(OP_LT) & value.lt(threshold))
                | (op.eq(OP_LE) & value.le(threshold))
                | (op.gt(OP_LE) & value.eq(threshold))
            )
        return condition | (isnan(value) & (meta & DEFAULT_DISTANT).ne(0))

    def _tree_range[
        width: Int, words: Int, ops: Int
    ](
        self,
        x: NDArray[Self.dtype],
        row: Int,
        count: Int,
        begin: Int,
        end: Int,
        var total: SIMD[Self.dtype, width],
    ) -> SIMD[Self.dtype, width]:
        """Continue `total` over trees [begin, end) in original tree order.

        Independent group sums would change floating-point association.
        """
        comptime assert Self.dtype.is_floating_point()
        comptime assert words == 4 or Self.dtype == DType.float32
        var roots = self._roots.unsafe_ptr()
        var features = self.num_features()
        # 64-bit block base; in-block lane offsets fit Int32 (checked per call).
        var input = x.unsafe_ptr().unsafe_offset(row * features)
        comptime if width == 1:
            # One row: lanes walk consecutive trees, so their dependent load
            # chains overlap. Leaves are still added in original tree order.
            var tree = begin
            # Halve the lane count for remainders: 32, 16, 8, 4, then scalar.
            comptime for level in range(4):
                comptime trees = _SINGLE >> level
                var offsets = SIMD[DType.int32, trees](0)
                while tree + trees <= end:
                    var leaves = self._walk[trees, words, ops](
                        roots.unsafe_offset(tree).unsafe_load[width=trees](),
                        offsets,
                        input,
                    )
                    comptime for k in range(trees):
                        total += leaves[k]
                    tree += trees
            while tree < end:
                total += self._walk[1, words, ops](
                    SIMD[DType.int32, 1](roots[unsafe_offset=tree]),
                    SIMD[DType.int32, 1](0),
                    input,
                )[0]
                tree += 1
        else:
            var lanes = SIMD[DType.int32, width](0)
            comptime for lane in range(width):
                lanes[lane] = Int32(lane)
            # Tail lanes repeat the block's first row, so every lane reads a
            # valid input row and no gather needs a mask. Their sums are unused.
            lanes = lanes.lt(Int32(count)).select(lanes, 0)
            var row_offsets = lanes * Int32(features)
            var tree = begin
            comptime if _INTERLEAVE > 1:
                # Walk several trees at once: their dependent gather chains
                # overlap. Leaves are still added in original tree order.
                comptime wide = width * _INTERLEAVE
                var wide_offsets = SIMD[DType.int32, wide](0)
                comptime for i in range(wide):
                    wide_offsets[i] = row_offsets[i % width]
                while tree + _INTERLEAVE <= end:
                    var ids = SIMD[DType.int32, wide](0)
                    comptime for i in range(wide):
                        ids[i] = roots[unsafe_offset=tree + i // width]
                    var leaves = self._walk[wide, words, ops](
                        ids, wide_offsets, input
                    )
                    comptime for k in range(_INTERLEAVE):
                        total += leaves.slice[width, offset=k * width]()
                    tree += _INTERLEAVE
            while tree < end:
                total += self._walk[width, words, ops](
                    SIMD[DType.int32, width](roots[unsafe_offset=tree]),
                    row_offsets,
                    input,
                )
                tree += 1
        return total

    @always_inline
    def _walk[
        width: Int, words: Int, ops: Int
    ](
        self,
        var ids: SIMD[DType.int32, width],
        row_offsets: SIMD[DType.int32, width],
        input: ImmPointer[Scalar[Self.dtype], _],
    ) -> SIMD[Self.dtype, width]:
        """Advance every lane from its root to a leaf; return leaf values.

        Finished lanes step by zero, so their gathers stay on the leaf.
        """
        comptime compact = words == 2
        comptime offset_mask = UInt32((1 << COMPACT_OFFSET_BITS) - 1)
        comptime feature_mask = UInt32(
            (1 << COMPACT_FEATURE_BITS) - 1
        ) if compact else UInt32((1 << WIDE_FEATURE_BITS) - 1)
        comptime feature_shift = COMPACT_OFFSET_BITS if compact else 0
        var pairs = self._nodes.unsafe_ptr().unsafe_bitcast[UInt64]()
        while True:
            var meta: SIMD[DType.uint32, width]
            var offset: SIMD[DType.int32, width]
            var low: SIMD[DType.uint64, width]
            comptime if compact:
                # One 64-bit gather yields both threshold and meta.
                low = pairs.unsafe_gather(ids)
                meta = (low >> 32).cast[DType.uint32]()
                offset = (meta & offset_mask).cast[DType.int32]()
            else:
                var high = pairs.unsafe_gather(ids * 2 + 1)
                meta = high.cast[DType.uint32]()
                offset = (high >> 32).cast[DType.int32]()
                low = pairs.unsafe_gather(ids * 2)
            var threshold: SIMD[Self.dtype, width]
            comptime if Self.dtype == DType.float32:
                threshold = bitcast[Self.dtype, width](low.cast[DType.uint32]())
            else:
                threshold = bitcast[Self.dtype, width](low)
            var active = offset.ne(0)
            if not active.reduce_or():
                # Every lane is on a leaf, whose value is in the threshold slot.
                return threshold
            var feature = ((meta >> UInt32(feature_shift)) & feature_mask).cast[
                DType.int32
            ]()
            var value = input.unsafe_gather(row_offsets + feature)
            var condition = self._condition[ops, width](value, threshold, meta)
            ids += condition.select(offset, active.cast[DType.int32]())

    def _finish[
        width: Int
    ](self, var total: SIMD[Self.dtype, width]) -> SIMD[Self.dtype, width]:
        if self._average:
            total /= Scalar[Self.dtype](self.num_trees())
        # Treelite stores base scores as Float64, even for float32 forests.
        # Add in that precision before narrowing to the prediction dtype.
        return (total.cast[DType.float64]() + self._bias).cast[Self.dtype]()

    def _predict_default[
        output_dtype: DType, mode: Int
    ](
        self,
        x: NDArray[Self.dtype],
        num_threads: Int,
        threshold: Scalar[Self.dtype] = 0.5,
    ) raises -> NDArray[output_dtype]:
        # Gather traversal is bound by memory-level parallelism rather than
        # register width: _WIDE rows times _INTERLEAVE trees keeps about 64
        # independent lanes in flight. Small batches keep native vectors to
        # avoid oversized tail gathers.
        if x.ndim == 2 and x.shape[0] >= _WIDE:
            # Do not reduce the available task count below an explicit limit.
            if num_threads <= 1 or ceildiv(x.shape[0], _WIDE) >= num_threads:
                return self._predict[_WIDE, output_dtype, mode](
                    x, num_threads, threshold
                )
        return self._predict[simd_width_of[Self.dtype](), output_dtype, mode](
            x, num_threads, threshold
        )

    def _predict[
        width: Int, output_dtype: DType, mode: Int
    ](
        self,
        x: NDArray[Self.dtype],
        num_threads: Int,
        threshold: Scalar[Self.dtype] = 0.5,
    ) raises -> NDArray[output_dtype]:
        var route = self.kernel(x.shape[0] if x.ndim == 2 else 0)
        if route == _CONSTANT:
            return self._predict_impl[
                width, output_dtype, mode, 4, _OPS_ANY, _CONSTANT
            ](x, num_threads, threshold)
        if route == _BINNED:
            return self._predict_impl[
                width, output_dtype, mode, 4, _OPS_ANY, _BINNED
            ](x, num_threads, threshold)
        if route == _DIRECT:
            return self._predict_impl[
                width, output_dtype, mode, 4, _OPS_ANY, _DIRECT
            ](x, num_threads, threshold)
        # Inspect the immutable model once at construction, dispatch once per
        # prediction. Homogeneous kernels have no operator decode.
        comptime if Self.dtype == DType.float32:
            if self._words == 2:
                return self._generic[width, output_dtype, mode, 2](
                    x, num_threads, threshold
                )
        return self._generic[width, output_dtype, mode, 4](
            x, num_threads, threshold
        )

    def _generic[
        width: Int, output_dtype: DType, mode: Int, words: Int
    ](
        self,
        x: NDArray[Self.dtype],
        num_threads: Int,
        threshold: Scalar[Self.dtype],
    ) raises -> NDArray[output_dtype]:
        if self._ops == _OPS_LT:
            return self._predict_impl[
                width, output_dtype, mode, words, _OPS_LT
            ](x, num_threads, threshold)
        if self._ops == _OPS_LE:
            return self._predict_impl[
                width, output_dtype, mode, words, _OPS_LE
            ](x, num_threads, threshold)
        return self._predict_impl[width, output_dtype, mode, words, _OPS_ANY](
            x, num_threads, threshold
        )

    def _predict_impl[
        width: Int,
        output_dtype: DType,
        mode: Int,
        words: Int,
        ops: Int,
        compact: Int = 0,
    ](
        self,
        x: NDArray[Self.dtype],
        num_threads: Int,
        threshold: Scalar[Self.dtype] = 0.5,
    ) raises -> NDArray[output_dtype]:
        comptime assert (
            width > 0 and (width & (width - 1)) == 0
        ), "SIMD width must be a positive power of two"
        self._check(x)
        # Gathers address a block's rows with Int32 lane offsets.
        if width * self.num_features() > Int(Int32.MAX):
            raise Error("Pyre SIMD width times feature count exceeds Int32")
        var rows = x.shape[0]
        var result: NDArray[output_dtype]
        comptime if mode == 1:
            result = NDArray[output_dtype](Shape(rows, 2))
        else:
            result = NDArray[output_dtype](Shape(rows))
        var output = result.unsafe_ptr()
        var blocks = (rows + width - 1) // width
        var workers: Int
        comptime if compact == _CONSTANT:
            if num_threads < 0:
                raise Error(
                    "num_threads must be nonnegative (0 selects automatic)"
                )
            # A constant forest only stores; tasks cost more than they save.
            workers = 1
        else:
            workers = self._workers[compact](rows, blocks, num_threads)

        # Each task owns whole SIMD blocks. Model/input captures are borrowed;
        # The executor joins before result storage or either borrow can expire.
        def work(
            task: Int,
        ) {self, x, output, rows, blocks, workers, threshold}:
            var begin = blocks * task // workers
            var end = blocks * (task + 1) // workers
            # Tile inside the existing partition, never partition by tile count.
            # Each concurrent call owns its scratch until the executor joins.
            comptime tile_blocks = max(1, _ROW_TILE // width)
            # Only quantized tasks allocate: each bins its current tile.
            var bins = List[UInt8](
                unsafe_uninit_length=self.num_features()
                * tile_blocks
                * width if compact
                == _BINNED else 0
            )
            var bin_output = bins.unsafe_ptr()
            var scratch = Array[SIMD[Self.dtype, width], tile_blocks](
                uninitialized=True
            )
            for tile in range(begin, end, tile_blocks):
                var size = min(tile_blocks, end - tile)
                comptime if compact == 2:
                    self._symmetric.prepare[width](
                        x,
                        bin_output,
                        tile * width,
                        min(size * width, rows - tile * width),
                        tile_blocks * width,
                    )
                for i in range(size):
                    comptime if compact == _CONSTANT:
                        scratch[i] = SIMD[Self.dtype, width](
                            self._constant_total
                        )
                    else:
                        scratch[i] = SIMD[Self.dtype, width](0)
                comptime if compact != _CONSTANT:
                    # Groups keep a few trees cache-resident across a tile's
                    # blocks; a single block has no reuse to protect.
                    var group_size = (
                        _TREE_GROUP if size > 1 else self.num_trees()
                    )
                    for group in range(0, self.num_trees(), group_size):
                        var group_end = min(
                            group + group_size, self.num_trees()
                        )
                        for i in range(size):
                            var row = (tile + i) * width
                            var count = min(width, rows - row)
                            comptime if compact != 0:
                                scratch[i] = self._symmetric.tree_range[
                                    width, compact == 2
                                ](
                                    x,
                                    bin_output,
                                    row,
                                    i * width,
                                    count,
                                    tile_blocks * width,
                                    group,
                                    group_end,
                                    scratch[i],
                                )
                            else:
                                if count == 1:
                                    scratch[i][0] = self._tree_range[
                                        1, words, ops
                                    ](
                                        x,
                                        row,
                                        1,
                                        group,
                                        group_end,
                                        scratch[i][0],
                                    )
                                else:
                                    scratch[i] = self._tree_range[
                                        width, words, ops
                                    ](
                                        x,
                                        row,
                                        count,
                                        group,
                                        group_end,
                                        scratch[i],
                                    )
                for i in range(size):
                    var row = (tile + i) * width
                    var count = min(width, rows - row)
                    var value = self._finish(scratch[i])
                    comptime if mode != 0:
                        # Stable vector sigmoid; intermediate magnitudes stay <= 1.
                        comptime assert Self.dtype.is_floating_point()
                        var z = Scalar[Self.dtype](self._alpha) * value
                        var magnitude = abs(z)
                        var e = exp(-magnitude)
                        # Some CPU exp lowerings clamp extreme arguments to a
                        # tiny nonzero value. Enforce rounded underflow without
                        # depending on the caller's flush-to-zero mode: otherwise
                        # a negative extreme can become class 1 at threshold 0.
                        # First representable magnitudes above log(2) * 150
                        # (f32) / 1075 (f64), half the smallest subnormal.
                        comptime zero_cutoff = (
                            103.97208404541015625 if Self.dtype
                            == DType.float32 else 745.1332191019412
                        )
                        e = magnitude.ge(
                            Scalar[Self.dtype](zero_cutoff)
                        ).select(SIMD[Self.dtype, width](0), e)
                        value = z.ge(0).select(1 / (1 + e), e / (1 + e))
                    comptime if mode == 1:
                        comptime for lane in range(width):
                            if lane < count:
                                output[unsafe_offset=2 * (row + lane)] = Scalar[
                                    output_dtype
                                ](1 - value[lane])
                                output[
                                    unsafe_offset=2 * (row + lane) + 1
                                ] = Scalar[output_dtype](value[lane])
                    else:
                        var stored: SIMD[output_dtype, width]
                        comptime if mode == 2:
                            stored = value.gt(threshold).cast[output_dtype]()
                        else:
                            stored = value.cast[output_dtype]()
                        if count == width:
                            output.unsafe_offset(row).unsafe_store(stored)
                        else:
                            comptime for lane in range(width):
                                if lane < count:
                                    output[unsafe_offset=row + lane] = stored[
                                        lane
                                    ]

        if workers == 1:
            work(0)
        else:
            self._executor.run(work, workers)
        return result^

    def predict_margin(
        self,
        x: NDArray[Self.dtype],
        *,
        num_threads: Int = 0,
    ) raises -> NDArray[Self.dtype]:
        """Raw margins [rows]. num_threads=0 auto, 1 serial, >1 a task limit."""
        return self._predict_default[Self.dtype, 0](x, num_threads)

    def predict_margin[
        width: Int
    ](
        self,
        x: NDArray[Self.dtype],
        *,
        num_threads: Int = 0,
    ) raises -> NDArray[
        Self.dtype
    ]:
        """Raw margins [rows]. num_threads=0 auto, 1 serial, >1 a task limit."""
        return self._predict[width, Self.dtype, 0](x, num_threads)

    def _workers[
        compact: Int
    ](self, rows: Int, blocks: Int, num_threads: Int) raises -> Int:
        """Tasks for a call: bounded by blocks, the limit and estimated work.

        num_threads=0 caps at physical cores, since SMT siblings contend for the
        same gather and load units. Explicit limits also drop tasks that would
        carry less than one grain of work, which only adds scheduling latency.
        """
        if num_threads < 0:
            raise Error("num_threads must be nonnegative (0 selects automatic)")
        if num_threads == 1 or blocks < 2:
            return 1
        # Order-of-magnitude per-node cost: quantized symmetric levels are a
        # byte compare; packed traversal scales with input element size.
        comptime visit_ns = 0.4 if compact == _BINNED else 0.25 * Float64(
            size_of[Scalar[Self.dtype]]()
        )
        var tasks = Int(Float64(rows) * self._path * visit_ns / _GRAIN_NS)
        if tasks < 2:
            return 1
        initialize_runtime()
        var limit = self._cores
        if num_threads > 0:
            limit = min(num_threads, parallelism_level(None))
        return max(1, min(blocks, limit, tasks))

    def predict(
        self, x: NDArray[Self.dtype], *, num_threads: Int = 0
    ) raises -> NDArray[Self.dtype]:
        """Regression values. Classifiers must use predict_class/proba."""
        if self.is_classifier():
            raise Error(
                "Use predict_class or predict_proba for binary classification"
            )
        return self.predict_margin(x, num_threads=num_threads)

    def predict[
        width: Int
    ](self, x: NDArray[Self.dtype], *, num_threads: Int = 0) raises -> NDArray[
        Self.dtype
    ]:
        """Regression values. Classifiers must use predict_class/proba."""
        if self.is_classifier():
            raise Error(
                "Use predict_class or predict_proba for binary classification"
            )
        return self.predict_margin[width](x, num_threads=num_threads)

    def predict_proba(
        self, x: NDArray[Self.dtype], *, num_threads: Int = 0
    ) raises -> NDArray[Self.dtype]:
        """Binary class probabilities in columns [negative, positive]."""
        comptime assert Self.dtype.is_floating_point()
        if not self.is_classifier():
            raise Error("predict_proba requires a binary classifier")
        return self._predict_default[Self.dtype, 1](x, num_threads)

    def predict_proba[
        width: Int
    ](self, x: NDArray[Self.dtype], *, num_threads: Int = 0) raises -> NDArray[
        Self.dtype
    ]:
        """Binary class probabilities in columns [negative, positive]."""
        comptime assert Self.dtype.is_floating_point()
        if not self.is_classifier():
            raise Error("predict_proba requires a binary classifier")
        return self._predict[width, Self.dtype, 1](x, num_threads)

    def predict_class(
        self,
        x: NDArray[Self.dtype],
        threshold: Scalar[Self.dtype] = 0.5,
        *,
        num_threads: Int = 0,
    ) raises -> NDArray[DType.int64]:
        """Binary labels; probability must be strictly above threshold."""
        if not (threshold >= 0 and threshold <= 1):
            raise Error("Classification threshold must lie in [0, 1]")
        if not self.is_classifier():
            raise Error("predict_class requires a binary classifier")
        return self._predict_default[DType.int64, 2](x, num_threads, threshold)

    def predict_class[
        width: Int
    ](
        self,
        x: NDArray[Self.dtype],
        threshold: Scalar[Self.dtype] = 0.5,
        *,
        num_threads: Int = 0,
    ) raises -> NDArray[DType.int64]:
        """Binary labels; probability must be strictly above threshold."""
        if not (threshold >= 0 and threshold <= 1):
            raise Error("Classification threshold must lie in [0, 1]")
        if not self.is_classifier():
            raise Error("predict_class requires a binary classifier")
        return self._predict[width, DType.int64, 2](x, num_threads, threshold)
