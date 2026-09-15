"""Native, synchronous GPU tree inference with explicit device storage."""
from balsa import Model, TaskType
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
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

# A batch is a rank-two GPU array; results use the same checked storage type.
comptime GpuBatch[dtype: DType] = GpuArray[dtype, 2]


struct GpuForest[dtype: DType = DType.float32](Movable):
    """Immutable prepared forest. Calls on the same context must not overlap."""

    var _context: DeviceContext
    var _meta: DeviceBuffer[DType.int32]
    var _values: DeviceBuffer[Self.dtype]
    var _roots: DeviceBuffer[DType.int32]
    var _nodes: Int
    var _features: Int
    var _trees: Int
    var _classifier: Bool
    var _average: Int32
    var _bias: Float64
    var _alpha: Float32

    def __init__(out self, model: Model[Self.dtype], ctx: DeviceContext) raises:
        _gpu(ctx)
        _ = validate_inference(model)
        var count = 0
        for tree in model.trees:
            if count > 2147483647 - Int(tree.num_nodes):
                raise Error("Pyre GPU forest exceeds Int32 node indexing")
            count += Int(tree.num_nodes)
        var mh = ctx.enqueue_create_host_buffer[DType.int32](count * 5)
        var vh = ctx.enqueue_create_host_buffer[Self.dtype](count * 2)
        var rh = ctx.enqueue_create_host_buffer[DType.int32](len(model.trees))
        var ml = row_major(count, 5)
        var vl = row_major(count, 2)
        var rl = row_major(len(model.trees))
        var mt = TileTensor(mh, ml)
        var vt = TileTensor(vh, vl)
        var rt = TileTensor(rh, rl)
        comptime assert mt.flat_rank == 2
        comptime assert vt.flat_rank == 2
        comptime assert rt.flat_rank == 1
        var base = 0
        for t in range(len(model.trees)):
            rt[t] = Int32(base)
            for n in range(Int(model.trees[t].num_nodes)):
                var pos = base + n
                var l = model.trees[t].cleft[n]
                var r = model.trees[t].cright[n]
                mt[pos, 0] = l + Int32(base) if l >= 0 else l
                mt[pos, 1] = r + Int32(base) if r >= 0 else r
                mt[pos, 2] = model.trees[t].split_index[n]
                mt[pos, 3] = Int32(model.trees[t].cmp[n])
                mt[pos, 4] = Int32(model.trees[t].default_left[n])
                vt[pos, 0] = model.trees[t].threshold[n]
                vt[pos, 1] = model.trees[t].leaf_value[n]
            base += Int(model.trees[t].num_nodes)
        var md = ctx.enqueue_create_buffer[DType.int32](count * 5)
        var vd = ctx.enqueue_create_buffer[Self.dtype](count * 2)
        var rd = ctx.enqueue_create_buffer[DType.int32](len(model.trees))
        with _Join(ctx):
            ctx.enqueue_copy(md, mh)
            ctx.enqueue_copy(vd, vh)
            ctx.enqueue_copy(rd, rh)
        self._context = ctx.copy()
        self._meta = md^
        self._values = vd^
        self._roots = rd^
        self._nodes = count
        self._features = Int(model.num_feature)
        self._trees = len(model.trees)
        self._classifier = model.task_type == TaskType.BINARY_CLF
        self._average = Int32(model.average_tree_output != 0)
        self._bias = model.base_scores[0]
        self._alpha = model.sigmoid_alpha

    def __deinit__(deinit self):
        _ = self._meta^
        _ = self._values^
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

    def _check[
        output_dtype: DType, rank: Int
    ](
        self, x: GpuBatch[Self.dtype], output: GpuArray[output_dtype, rank]
    ) raises:
        if x._shape[1] != self._features:
            raise Error("Pyre input feature count does not match the forest")
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

    def _enqueue[
        output_dtype: DType, rank: Int, mode: Int
    ](
        self,
        x: GpuBatch[Self.dtype],
        mut output: GpuArray[output_dtype, rank],
        threshold: Scalar[Self.dtype] = 0.5,
    ) raises:
        var rows = x._shape[0]
        if rows == 0:
            return
        var ml = row_major(self._nodes, 5)
        var vl = row_major(self._nodes, 2)
        var rl = row_major(self._trees)
        var xl = row_major(len(x._buffer))
        var yl = row_major(len(output._buffer))
        comptime kernel = predict_kernel[
            Self.dtype,
            output_dtype,
            mode,
            type_of(ml),
            type_of(vl),
            type_of(rl),
            type_of(xl),
            type_of(yl),
        ]
        self._context.enqueue_function[kernel](
            TileTensor(self._meta, ml),
            TileTensor(self._values, vl),
            TileTensor(self._roots, rl),
            TileTensor(x._buffer, xl),
            TileTensor(output._buffer, yl),
            Int64(rows),
            Int64(self._trees),
            self._average,
            self._bias,
            self._alpha,
            threshold,
            Int64(x._offset),
            Int64(output._offset),
            Int64(self._features),
            grid_dim=ceildiv(rows, 128),
            block_dim=128,
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
        var result = GpuArray[Self.dtype, 1](self._context, Shape(x._shape[0]))
        self.predict_margin_into(x, result)
        return result^

    def predict(
        self, x: GpuBatch[Self.dtype]
    ) raises -> GpuArray[Self.dtype, 1]:
        var result = GpuArray[Self.dtype, 1](self._context, Shape(x._shape[0]))
        self.predict_into(x, result)
        return result^

    def predict_proba(
        self, x: GpuBatch[Self.dtype]
    ) raises -> GpuArray[Self.dtype, 2]:
        var result = GpuArray[Self.dtype, 2](
            self._context, Shape(x._shape[0], 2)
        )
        self.predict_proba_into(x, result)
        return result^

    def predict_class(
        self, x: GpuBatch[Self.dtype], threshold: Scalar[Self.dtype] = 0.5
    ) raises -> GpuArray[DType.int64, 1]:
        var result = GpuArray[DType.int64, 1](self._context, Shape(x._shape[0]))
        self.predict_class_into(x, result, threshold)
        return result^
