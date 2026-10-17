"""Private CPU preparation and execution; Forest is the public interface."""

from std.math import exp
from std.sys import simd_width_of
from std.runtime import initialize_runtime
from std.runtime.asyncrt import parallelism_level
from ._executor import _Executor
from ._validation import validate_inference
from ._symmetric import _Symmetric
from ._packed import _Packed, _OP_LT, _OP_LE, _MIXED
from balsa import Model, NodeType, Operator, TaskType
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


struct _CPUPlan[dtype: DType = DType.float32](Movable):
    """An owned Balsa model with SIMD CPU inference. Input precision must match.

    CPU inference vectorizes independent rows, preserving tree summation order.
    Supports scalar regression/identity and binary classification/sigmoid.
    """

    var _strategy: Int
    var _constant: Bool
    var _constant_total: Scalar[Self.dtype]
    var _model: Model[Self.dtype]
    var _symmetric: _Symmetric[Self.dtype]
    var _packed: _Packed[Self.dtype]
    var _executor: _Executor

    def __init__(
        out self, var model: Model[Self.dtype], *, strategy: Int = _AUTO
    ) raises:
        """Validate support, recover compact CPU trees, then take ownership.

        Pass model^ to transfer an existing model, or model.copy() to retain an
        independently mutable source. Balsa remains responsible for structure.
        """
        _ = validate_inference(model)
        self._symmetric = _Symmetric(model)
        self._strategy = strategy
        self._constant = True
        self._constant_total = 0
        for tree in model.trees:
            self._constant &= tree.node_type[0] == NodeType.LEAF
        if self._constant:
            # Preserve the exact incoming-zero and ordered-sum semantics.
            for tree in model.trees:
                self._constant_total += tree.leaf_value[0]
        self._packed = _Packed(model)
        self._model = model^
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
        return _DIRECT

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

    def _finish[
        width: Int
    ](self, var total: SIMD[Self.dtype, width]) -> SIMD[Self.dtype, width]:
        if self._model.average_tree_output != 0:
            total /= Scalar[Self.dtype](self.num_trees())
        # Treelite stores base scores as Float64, even for float32 forests.
        # Add in that precision before narrowing to the prediction dtype.
        return (total.cast[DType.float64]() + self._model.base_scores[0]).cast[
            Self.dtype
        ]()

    def _predict_default[
        output_dtype: DType, mode: Int
    ](
        self,
        x: NDArray[Self.dtype],
        num_threads: Int,
        threshold: Scalar[Self.dtype] = 0.5,
    ) raises -> NDArray[output_dtype]:
        # Limit the wider default to the native vector sizes measured so far.
        # Small batches retain native vectors to avoid oversized tail gathers.
        comptime if (
            simd_width_of[DType.float32]() == 8
            and simd_width_of[DType.float64]() == 4
        ):
            if x.ndim == 2 and x.shape[0] >= 32:
                # Do not reduce the available task count below an explicit
                # limit. Automatic calls have enough blocks at their cutoff.
                if num_threads <= 1 or (x.shape[0] + 15) // 16 >= num_threads:
                    return self._predict[16, output_dtype, mode](
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
                width, output_dtype, mode, False, _OP_LT, _CONSTANT
            ](x, num_threads, threshold)
        if route == _BINNED:
            return self._predict_impl[
                width, output_dtype, mode, False, _OP_LT, _BINNED
            ](x, num_threads, threshold)
        if route == _DIRECT:
            return self._predict_impl[
                width, output_dtype, mode, False, _OP_LT, _DIRECT
            ](x, num_threads, threshold)
        # Inspect the immutable model once at construction, dispatch once per
        # prediction. Uniform LT/LE kernels have no operator dispatch.
        if self._packed.wide:
            return self._predict_general[width, output_dtype, mode, True](
                x, num_threads, threshold
            )
        return self._predict_general[width, output_dtype, mode, False](
            x, num_threads, threshold
        )

    def _predict_general[
        width: Int, output_dtype: DType, mode: Int, wide: Bool
    ](
        self,
        x: NDArray[Self.dtype],
        num_threads: Int,
        threshold: Scalar[Self.dtype],
    ) raises -> NDArray[output_dtype]:
        if self._packed.op == _OP_LT:
            return self._predict_impl[width, output_dtype, mode, wide, _OP_LT](
                x, num_threads, threshold
            )
        if self._packed.op == _OP_LE:
            return self._predict_impl[width, output_dtype, mode, wide, _OP_LE](
                x, num_threads, threshold
            )
        return self._predict_impl[width, output_dtype, mode, wide, _MIXED](
            x, num_threads, threshold
        )

    def _predict_impl[
        width: Int,
        output_dtype: DType,
        mode: Int,
        wide: Bool,
        op: Int,
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
        var rows = x.shape[0]
        var result: NDArray[output_dtype]
        comptime if mode == 1:
            result = NDArray[output_dtype](Shape(rows, 2))
        else:
            result = NDArray[output_dtype](Shape(rows))
        var output = result.unsafe_ptr()
        var blocks = (rows + width - 1) // width
        var workers: Int
        comptime if compact != 0:
            if num_threads < 0:
                raise Error(
                    "num_threads must be nonnegative (0 selects automatic)"
                )
            # Compact scheduling is serial until the separate concurrency work.
            workers = 1
        else:
            workers = self._workers(rows, blocks, num_threads)

        var bins = NDArray[DType.uint8](Shape(0))
        comptime if compact == 2:
            bins = NDArray[DType.uint8](
                Shape(self.num_features(), max(width, _ROW_TILE))
            )
        var bin_output = bins.unsafe_ptr()

        # Each task owns whole SIMD blocks. Model/input captures are borrowed;
        # The executor joins before result storage or either borrow can expire.
        def work(
            task: Int,
        ) {self, x, output, rows, blocks, workers, threshold, bin_output}:
            var begin = blocks * task // workers
            var end = blocks * (task + 1) // workers
            # Tile inside the existing partition, never partition by tile count.
            # Each concurrent call owns its scratch until the executor joins.
            var tile_blocks = max(1, _ROW_TILE // width)
            var scratch = List[SIMD[Self.dtype, width]](capacity=tile_blocks)
            for _ in range(tile_blocks):
                scratch.append(SIMD[Self.dtype, width](0))
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
                    for group in range(0, self.num_trees(), _TREE_GROUP):
                        var group_end = min(
                            group + _TREE_GROUP, self.num_trees()
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
                                    scratch[i][0] = self._packed.tree_range[
                                        1, wide, op
                                    ](
                                        x,
                                        self.num_features(),
                                        row,
                                        1,
                                        group,
                                        group_end,
                                        scratch[i][0],
                                    )
                                else:
                                    scratch[i] = self._packed.tree_range[
                                        width, wide, op
                                    ](
                                        x,
                                        self.num_features(),
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
                        var z = (
                            Scalar[Self.dtype](self._model.sigmoid_alpha)
                            * value
                        )
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

    def _workers(self, rows: Int, blocks: Int, num_threads: Int) raises -> Int:
        if num_threads < 0:
            raise Error("num_threads must be nonnegative (0 selects automatic)")
        if num_threads == 1 or blocks < 2:
            return 1
        if num_threads == 0 and (rows < 256 or rows * self.num_trees() < 8192):
            return 1
        initialize_runtime()
        var limit = min(8, parallelism_level()) if num_threads == 0 else min(
            num_threads, parallelism_level()
        )
        # Intermediate workloads use fewer tasks to limit scheduling variability.
        if num_threads == 0 and rows * self.num_trees() < 32768:
            limit = min(limit, 4)
        return max(1, min(blocks, limit))

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
