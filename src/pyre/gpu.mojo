"""Native, synchronous GPU tree inference with explicit device storage."""
from balsa import Model, TaskType
from max.gpu.host import DeviceContext, DeviceBuffer, DeviceAttribute
from std.math import ceildiv
from std.sys import size_of
from numojo.core.type_aliases import Shape
from ._validation import validate_inference
from ._gpu_storage import (
    GpuArray,
    upload,
    upload_into,
    _Join,
    _gpu,
    _same_device,
)
from ._gpu_kernel import predict_kernel
from ._layout import OP_LT, node_count, node_words, pack


# A batch is a rank-two GPU array; results use the same checked storage type.
comptime GpuBatch[dtype: DType] = GpuArray[dtype, 2]


struct GpuForest[dtype: DType = DType.float32](Movable):
    """Immutable prepared forest. Calls on the same context must not overlap."""

    var _context: DeviceContext
    var _nodes: DeviceBuffer[DType.uint32]
    var _roots: DeviceBuffer[DType.int32]
    var _count: Int
    var _words: Int
    var _general: Bool
    var _features: Int
    var _trees: Int
    var _classifier: Bool
    var _average: Int32
    var _bias: Float64
    var _alpha: Float32
    var _sms: Int

    def __init__(out self, model: Model[Self.dtype], ctx: DeviceContext) raises:
        _gpu(ctx)
        _ = validate_inference(model)
        var words = node_words(model)
        var count = node_count(model)
        var features = Int(model.num_feature)
        var nh = ctx.enqueue_create_host_buffer[DType.uint32](count * words)
        var rh = ctx.enqueue_create_host_buffer[DType.int32](len(model.trees))
        var used = pack[keyed=True, strict=False](
            model, nh.unsafe_ptr(), rh.unsafe_ptr(), words
        )
        var general = (used & ~(UInt32(1) << OP_LT)) != 0
        var nd = ctx.enqueue_create_buffer[DType.uint32](count * words)
        var rd = ctx.enqueue_create_buffer[DType.int32](len(model.trees))
        with _Join(ctx):
            ctx.enqueue_copy(nd, nh)
            ctx.enqueue_copy(rd, rh)
        self._context = ctx.copy()
        self._nodes = nd^
        self._roots = rd^
        self._count = count
        self._words = words
        self._general = general
        self._features = features
        self._trees = len(model.trees)
        self._classifier = model.task_type == TaskType.BINARY_CLF
        self._average = Int32(model.average_tree_output != 0)
        self._bias = model.base_scores[0]
        self._alpha = model.sigmoid_alpha
        self._sms = Int(ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT))

    def __deinit__(deinit self):
        _ = self._nodes^
        _ = self._roots^
        _ = self._context^

    @staticmethod
    def prepare(
        model: Model[Self.dtype], context: DeviceContext
    ) raises -> Self:
        return Self(model, context)

    def num_features(self) -> Int:
        return self._features

    def num_trees(self) -> Int:
        return self._trees

    def is_classifier(self) -> Bool:
        return self._classifier

    def _check_input(self, x: GpuBatch[Self.dtype]) raises:
        # Kernel row counts are signed 32-bit. Check before output allocation
        # as well as before launch, including zero-feature batches.
        if x._shape[0] > Int(Int32.MAX):
            raise Error("Pyre GPU inference supports at most 2147483647 rows")
        if x._shape[1] != self._features:
            raise Error("Pyre input feature count does not match the forest")

    def _check[
        output_dtype: DType, rank: Int
    ](
        self, x: GpuBatch[Self.dtype], output: GpuArray[output_dtype, rank]
    ) raises:
        self._check_input(x)
        if output._shape[0] != x._shape[0]:
            raise Error("Pyre GPU output row count mismatch")
        comptime if rank == 2:
            if output._shape[1] != 2:
                raise Error("Pyre GPU probabilities require [rows, 2] output")
        _same_device(self._context, x._buffer.context())
        _same_device(self._context, output._buffer.context())
        if x._size and output._size:
            var xb = (
                Int(x._buffer.unsafe_ptr())
                + x._offset * size_of[Scalar[Self.dtype]]()
            )
            var yb = (
                Int(output._buffer.unsafe_ptr())
                + output._offset * size_of[Scalar[output_dtype]]()
            )
            var xe = xb + x._size * size_of[Scalar[Self.dtype]]()
            var ye = yb + output._size * size_of[Scalar[output_dtype]]()
            if xb < ye and yb < xe:
                raise Error("Pyre GPU input and output must not overlap")

    def _chunk_shift(self, rows: Int) -> Int:
        """Rows per block as a power of two, fitted on an RTX 5090 sweep.

        Grow while the block's rows stay within 512 bytes (node data then
        dominates L1), tasks stay moderate, and the grid keeps at least eight
        blocks per multiprocessor.
        """
        comptime item = size_of[Scalar[Self.dtype]]()
        var shift = 0
        while (
            shift < 5
            and (2 << shift) * self._trees <= 2048
            and (2 << shift) * self._features * item <= 512
            and ceildiv(rows, 2 << shift) >= 8 * self._sms
        ):
            shift += 1
        return shift

    def _enqueue[
        output_dtype: DType, rank: Int, mode: Int
    ](
        self,
        x: GpuBatch[Self.dtype],
        mut output: GpuArray[output_dtype, rank],
        threshold: Scalar[Self.dtype] = 0.5,
        chunk_shift: Int = -1,
        block_threads: Int = 0,
    ) raises:
        var rows = x._shape[0]
        if rows == 0:
            return
        var shift = chunk_shift if chunk_shift >= 0 else self._chunk_shift(rows)
        var threads = block_threads
        if threads == 0:
            var tasks = self._trees << shift
            threads = max(32, min(128, ceildiv(tasks, 32) * 32))
            # Few blocks leave the GPU idle: widen them over more trees.
            var blocks = ceildiv(rows, 1 << shift)
            while (
                threads < 1024
                and threads < tasks
                and blocks * threads * 2 <= self._sms * 512
            ):
                threads *= 2
        var shared = threads * size_of[Scalar[Self.dtype]]()
        if self._words == 2:
            self._launch[mode=mode, words=2](
                x, output, threshold, shift, threads, shared
            )
        else:
            self._launch[mode=mode, words=4](
                x, output, threshold, shift, threads, shared
            )

    def _launch[
        output_dtype: DType, rank: Int, //, mode: Int, words: Int
    ](
        self,
        x: GpuBatch[Self.dtype],
        mut output: GpuArray[output_dtype, rank],
        threshold: Scalar[Self.dtype],
        shift: Int,
        threads: Int,
        shared: Int,
    ) raises:
        comptime if Self.dtype == DType.float64 and words == 2:
            raise Error("Pyre GPU float64 nodes require the wide layout")
        else:
            var rows = x._shape[0]
            var nodes = self._nodes.unsafe_ptr().as_imm()
            var roots = self._roots.unsafe_ptr().as_imm()
            var input = x._buffer.unsafe_ptr().unsafe_offset(x._offset).as_imm()
            var result = output._buffer.unsafe_ptr().unsafe_offset(
                output._offset
            )
            if self._general:
                comptime kernel = predict_kernel[
                    Self.dtype, output_dtype, mode, words, True
                ]
                self._context.enqueue_function[kernel](
                    nodes,
                    roots,
                    input,
                    result,
                    Int32(rows),
                    Int32(self._trees),
                    Int32(self._features),
                    Int32(shift),
                    self._average,
                    self._bias,
                    self._alpha,
                    threshold,
                    grid_dim=ceildiv(rows, 1 << shift),
                    block_dim=threads,
                    shared_mem_bytes=shared,
                )
            else:
                comptime kernel = predict_kernel[
                    Self.dtype, output_dtype, mode, words, False
                ]
                self._context.enqueue_function[kernel](
                    nodes,
                    roots,
                    input,
                    result,
                    Int32(rows),
                    Int32(self._trees),
                    Int32(self._features),
                    Int32(shift),
                    self._average,
                    self._bias,
                    self._alpha,
                    threshold,
                    grid_dim=ceildiv(rows, 1 << shift),
                    block_dim=threads,
                    shared_mem_bytes=shared,
                )

    def _into[
        output_dtype: DType, rank: Int, mode: Int
    ](
        self,
        x: GpuBatch[Self.dtype],
        mut output: GpuArray[output_dtype, rank],
        threshold: Scalar[Self.dtype] = 0.5,
    ) raises:
        self._check(x, output)
        with _Join(self._context):
            self._enqueue[output_dtype, rank, mode](x, output, threshold)

    def predict_margin_into(
        self, x: GpuBatch[Self.dtype], mut output: GpuArray[Self.dtype, 1]
    ) raises:
        self._into[Self.dtype, 1, 0](x, output)

    def predict_into(
        self, x: GpuBatch[Self.dtype], mut output: GpuArray[Self.dtype, 1]
    ) raises:
        if self._classifier:
            raise Error(
                "Use predict_class or predict_proba for binary classification"
            )
        self.predict_margin_into(x, output)

    def predict_proba_into(
        self, x: GpuBatch[Self.dtype], mut output: GpuArray[Self.dtype, 2]
    ) raises:
        if not self._classifier:
            raise Error("predict_proba requires a binary classifier")
        self._into[Self.dtype, 2, 1](x, output)

    def predict_class_into(
        self,
        x: GpuBatch[Self.dtype],
        mut output: GpuArray[DType.int64, 1],
        threshold: Scalar[Self.dtype] = 0.5,
    ) raises:
        if not (threshold >= 0 and threshold <= 1):
            raise Error("Classification threshold must lie in [0, 1]")
        if not self._classifier:
            raise Error("predict_class requires a binary classifier")
        self._into[DType.int64, 1, 2](x, output, threshold)

    def predict_margin(
        self, x: GpuBatch[Self.dtype]
    ) raises -> GpuArray[Self.dtype, 1]:
        self._check_input(x)
        var result = GpuArray[Self.dtype, 1](self._context, Shape(x._shape[0]))
        self.predict_margin_into(x, result)
        return result^

    def predict(
        self, x: GpuBatch[Self.dtype]
    ) raises -> GpuArray[Self.dtype, 1]:
        self._check_input(x)
        var result = GpuArray[Self.dtype, 1](self._context, Shape(x._shape[0]))
        self.predict_into(x, result)
        return result^

    def predict_proba(
        self, x: GpuBatch[Self.dtype]
    ) raises -> GpuArray[Self.dtype, 2]:
        self._check_input(x)
        var result = GpuArray[Self.dtype, 2](
            self._context, Shape(x._shape[0], 2)
        )
        self.predict_proba_into(x, result)
        return result^

    def predict_class(
        self, x: GpuBatch[Self.dtype], threshold: Scalar[Self.dtype] = 0.5
    ) raises -> GpuArray[DType.int64, 1]:
        self._check_input(x)
        var result = GpuArray[DType.int64, 1](self._context, Shape(x._shape[0]))
        self.predict_class_into(x, result, threshold)
        return result^
