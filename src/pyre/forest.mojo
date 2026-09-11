"""CPU tree-ensemble inference over NuMojo arrays, loaded through Balsa."""

from std.math import exp, isnan
from std.sys import simd_width_of
from std.runtime import initialize_runtime
from std.runtime.asyncrt import parallelism_level
from ._executor import _Executor
from balsa import Model, NodeType, Operator, TaskType, validate
from numojo import NDArray
from numojo.core.type_aliases import Shape


struct Forest[dtype: DType = DType.float32](Movable):
    """An owned Balsa model with SIMD CPU inference. Input precision must match.

    CPU inference vectorizes independent rows, preserving tree summation order.
    Supports scalar regression/identity and binary classification/sigmoid.
    """

    var _model: Model[Self.dtype]
    var _lt_only: Bool
    var _executor: _Executor

    def __init__(out self, var model: Model[Self.dtype]) raises:
        """Validate inference support, then take ownership without repacking.

        Pass model^ to transfer an existing model, or model.copy() to retain an
        independently mutable source. Balsa remains responsible for structure.
        """
        validate(model)
        if model.num_tree == 0:
            raise Error("Pyre requires at least one tree")
        if (
            model.num_target != 1
            or model.num_class[0] != 1
            or model.leaf_vector_shape[0] != 1
            or model.leaf_vector_shape[1] != 1
        ):
            raise Error("Pyre supports single-output scalar leaves only")
        var transform = model.postprocessor_name()
        if not (
            (model.task_type == TaskType.REGRESSOR and transform == "identity")
            or (
                model.task_type == TaskType.BINARY_CLF
                and transform == "sigmoid"
            )
        ):
            raise Error(
                "Pyre supports regression/identity and binary/sigmoid only"
            )
        if len(model.extensions) != 0:
            raise Error("Pyre cannot interpret model extensions")
        for tree_id in range(len(model.trees)):
            if model.target_id[tree_id] != 0 or model.class_id[tree_id] != 0:
                raise Error(
                    "Pyre requires scalar tree target/class IDs of zero"
                )
        var lt_only = True
        for tree in model.trees:
            if len(tree.tree_extensions) != 0 or len(tree.node_extensions) != 0:
                raise Error("Pyre cannot interpret tree/node extensions")
            for node in range(Int(tree.num_nodes)):
                if tree.node_type[node] == NodeType.CATEGORICAL:
                    raise Error("Pyre does not support categorical splits")
                if tree.node_type[node] != NodeType.LEAF:
                    lt_only &= tree.cmp[node] == Operator.LT
                if len(tree.leaf_values(node)) != 0:
                    raise Error("Pyre does not support vector leaves")
        self._lt_only = lt_only
        self._model = model^
        self._executor = _Executor()

    def num_features(self) -> Int:
        return Int(self._model.num_feature)

    def num_trees(self) -> Int:
        return len(self._model.trees)

    def is_classifier(self) -> Bool:
        return self._model.task_type == TaskType.BINARY_CLF

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

    def _block[
        width: Int, lt_only: Bool
    ](self, x: NDArray[Self.dtype], row: Int, count: Int) -> SIMD[
        Self.dtype, width
    ]:
        comptime assert Self.dtype.is_floating_point()
        var total = SIMD[Self.dtype, width](0)
        var input = x.unsafe_ptr()
        comptime if width == 1:
            # A scalar fallback avoids masked-gather overhead on one-row calls.
            for tree in self._model.trees:
                var node = 0
                while tree.node_type[node] != NodeType.LEAF:
                    var value = input[
                        unsafe_offset=row * self.num_features()
                        + Int(tree.split_index[node])
                    ]
                    var threshold = tree.threshold[node]
                    var left: Bool
                    comptime if lt_only:
                        left = value < threshold
                    else:
                        var op = tree.cmp[node]
                        left = (
                            (op == Operator.EQ and value == threshold)
                            or (op == Operator.LT and value < threshold)
                            or (op == Operator.LE and value <= threshold)
                            or (op == Operator.GT and value > threshold)
                            or (op == Operator.GE and value >= threshold)
                        )
                    if isnan(value):
                        left = tree.default_left[node] != 0
                    node = Int(tree.cleft[node] if left else tree.cright[node])
                total += tree.leaf_value[node]
        else:
            var lanes = SIMD[DType.int64, width](0)
            comptime for lane in range(width):
                lanes[lane] = Int64(lane)
            var valid = lanes.lt(Int64(count))
            var row_offsets = (lanes + Int64(row)) * Int64(self.num_features())
            for tree in self._model.trees:
                var ids = SIMD[DType.int32, width](0)
                var active = valid
                while True:
                    # Validation guarantees every reached node is in bounds.
                    # Finished lanes stay at their leaf; tail lanes stay at root.
                    # Balsa validates leaf children as -1, internal children >= 0.
                    var left = tree.cleft.unsafe_ptr().unsafe_gather(ids)
                    active &= left.ge(0)
                    if not active.reduce_or():
                        break
                    var features = (
                        tree.split_index.unsafe_ptr()
                        .unsafe_gather(ids)
                        .cast[DType.int64]()
                    )
                    # Masking is essential: leaf feature indices and absent rows
                    # are not valid input addresses, even when their result is unused.
                    var values = input.unsafe_gather(
                        row_offsets + features, active
                    )
                    var thresholds = tree.threshold.unsafe_ptr().unsafe_gather(
                        ids
                    )
                    var defaults = (
                        tree.default_left.unsafe_ptr().unsafe_gather(ids).ne(0)
                    )
                    var right = tree.cright.unsafe_ptr().unsafe_gather(ids)
                    var condition: SIMD[DType.bool, width]
                    comptime if lt_only:
                        condition = values.lt(thresholds)
                    else:
                        var ops = tree.cmp.unsafe_ptr().unsafe_gather(ids)
                        condition = (
                            (ops.eq(Operator.EQ) & values.eq(thresholds))
                            | (ops.eq(Operator.LT) & values.lt(thresholds))
                            | (ops.eq(Operator.LE) & values.le(thresholds))
                            | (ops.eq(Operator.GT) & values.gt(thresholds))
                            | (ops.eq(Operator.GE) & values.ge(thresholds))
                        )
                    condition = isnan(values).select(defaults, condition)
                    ids = active.select(condition.select(left, right), ids)
                total += tree.leaf_value.unsafe_ptr().unsafe_gather(ids, valid)
        if self._model.average_tree_output != 0:
            total /= Scalar[Self.dtype](self.num_trees())
        # Treelite stores base scores as Float64, even for float32 forests.
        # Add in that precision before narrowing to the prediction dtype.
        return (total.cast[DType.float64]() + self._model.base_scores[0]).cast[
            Self.dtype
        ]()

    def _predict[
        width: Int, output_dtype: DType, mode: Int
    ](
        self,
        x: NDArray[Self.dtype],
        num_threads: Int,
        threshold: Scalar[Self.dtype] = 0.5,
    ) raises -> NDArray[output_dtype]:
        # Inspect the immutable model once at construction, dispatch once per
        # prediction. The LT kernel has no operator loads or operator dispatch.
        if self._lt_only:
            return self._predict_impl[width, output_dtype, mode, True](
                x, num_threads, threshold
            )
        return self._predict_impl[width, output_dtype, mode, False](
            x, num_threads, threshold
        )

    def _predict_impl[
        width: Int, output_dtype: DType, mode: Int, lt_only: Bool
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
        var rows = x.shape[0]
        var result: NDArray[output_dtype]
        comptime if mode == 1:
            result = NDArray[output_dtype](Shape(rows, 2))
        else:
            result = NDArray[output_dtype](Shape(rows))
        var output = result.unsafe_ptr()
        var blocks = (rows + width - 1) // width
        var workers = self._workers(rows, blocks, num_threads)

        # Each task owns whole SIMD blocks. Model/input captures are borrowed;
        # The executor joins before result storage or either borrow can expire.
        def work(task: Int) {self, x, output, rows, blocks, workers, threshold}:
            for block in range(
                blocks * task // workers, blocks * (task + 1) // workers
            ):
                var row = block * width
                var count = min(width, rows - row)
                var value = SIMD[Self.dtype, width](0)
                if count == 1:
                    value[0] = self._block[1, lt_only](x, row, 1)
                else:
                    value = self._block[width, lt_only](x, row, count)
                comptime if mode != 0:
                    # Stable vector sigmoid; intermediate magnitudes stay <= 1.
                    comptime assert Self.dtype.is_floating_point()
                    var z = (
                        Scalar[Self.dtype](self._model.sigmoid_alpha) * value
                    )
                    var e = exp(-abs(z))
                    value = z.ge(0).select(1 / (1 + e), e / (1 + e))
                comptime if mode == 1:
                    comptime for lane in range(width):
                        if lane < count:
                            output[unsafe_offset=2 * (row + lane)] = Scalar[
                                output_dtype
                            ](1 - value[lane])
                            output[unsafe_offset=2 * (row + lane) + 1] = Scalar[
                                output_dtype
                            ](value[lane])
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
                                output[unsafe_offset=row + lane] = stored[lane]

        if workers == 1:
            work(0)
        else:
            self._executor.run(work, workers)
        return result^

    def predict_margin[
        width: Int = simd_width_of[Self.dtype]()
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

    def _workers(self, rows: Int, blocks: Int, num_threads: Int) raises -> Int:
        if num_threads < 0:
            raise Error("num_threads must be nonnegative (0 selects automatic)")
        if num_threads == 1 or blocks < 2:
            return 1
        if num_threads == 0 and (rows < 256 or rows * self.num_trees() < 32768):
            return 1
        initialize_runtime()
        var limit = min(8, parallelism_level()) if num_threads == 0 else min(
            num_threads, parallelism_level()
        )
        return max(1, min(blocks, limit))

    def predict[
        width: Int = simd_width_of[Self.dtype]()
    ](self, x: NDArray[Self.dtype], *, num_threads: Int = 0) raises -> NDArray[
        Self.dtype
    ]:
        """Regression values. Classifiers must use predict_class/proba."""
        if self.is_classifier():
            raise Error(
                "Use predict_class or predict_proba for binary classification"
            )
        return self.predict_margin[width](x, num_threads=num_threads)

    def predict_proba[
        width: Int = simd_width_of[Self.dtype]()
    ](self, x: NDArray[Self.dtype], *, num_threads: Int = 0) raises -> NDArray[
        Self.dtype
    ]:
        """Binary class probabilities in columns [negative, positive]."""
        comptime assert Self.dtype.is_floating_point()
        if not self.is_classifier():
            raise Error("predict_proba requires a binary classifier")
        return self._predict[width, Self.dtype, 1](x, num_threads)

    def predict_class[
        width: Int = simd_width_of[Self.dtype]()
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
