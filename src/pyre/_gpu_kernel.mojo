"""Baseline GPU traversal: one row per thread, ordered tree accumulation."""
from balsa import Operator
from std.gpu import block_idx, block_dim, thread_idx
from layout import TileTensor, TensorLayout
from std.math import isnan, exp


def predict_kernel[
    dtype: DType,
    output_dtype: DType,
    mode: Int,
    ML: TensorLayout,
    VL: TensorLayout,
    RL: TensorLayout,
    XL: TensorLayout,
    YL: TensorLayout,
](
    meta: TileTensor[DType.int32, ML, ImmutAnyOrigin],
    values: TileTensor[dtype, VL, ImmutAnyOrigin],
    roots: TileTensor[DType.int32, RL, ImmutAnyOrigin],
    x: TileTensor[dtype, XL, ImmutAnyOrigin],
    y: TileTensor[output_dtype, YL, MutAnyOrigin],
    rows: Int64,
    trees: Int64,
    average: Int32,
    bias: Float64,
    alpha: Float32,
    threshold: Scalar[dtype],
    input_offset: Int64,
    output_offset: Int64,
    features: Int64,
):
    comptime assert dtype.is_floating_point()
    comptime assert meta.flat_rank == 2
    comptime assert values.flat_rank == 2
    comptime assert roots.flat_rank == 1
    comptime assert x.flat_rank == 1
    comptime assert y.flat_rank == 1
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= Int(rows):
        return
    var total = Scalar[dtype](0)
    for tree in range(Int(trees)):
        var node = Int(rebind[Int32](roots[tree]))
        while rebind[Int32](meta[node, 0]) >= 0:
            var feature = Int(rebind[Int32](meta[node, 2]))
            var value = rebind[Scalar[dtype]](
                x[Int(input_offset) + row * Int(features) + feature]
            )
            var threshold = rebind[Scalar[dtype]](values[node, 0])
            var op = rebind[Int32](meta[node, 3])
            var left = (
                (op == Int32(Operator.EQ) and value == threshold)
                or (op == Int32(Operator.LT) and value < threshold)
                or (op == Int32(Operator.LE) and value <= threshold)
                or (op == Int32(Operator.GT) and value > threshold)
                or (op == Int32(Operator.GE) and value >= threshold)
            )
            if isnan(value):
                left = rebind[Int32](meta[node, 4]) != 0
            node = Int(rebind[Int32](meta[node, 0 if left else 1]))
        total += rebind[Scalar[dtype]](values[node, 1])
    if average:
        total /= Scalar[dtype](Int(trees))
    var value = Scalar[dtype](Float64(total) + bias)
    comptime if mode != 0:
        var z = Scalar[dtype](alpha) * value
        var e = exp(-abs(z))
        value = 1 / (1 + e) if z >= 0 else e / (1 + e)
    var pos = Int(output_offset)
    comptime if mode == 1:
        y[pos + 2 * row] = rebind[y.ElementType](
            Scalar[output_dtype](1 - value)
        )
        y[pos + 2 * row + 1] = rebind[y.ElementType](
            Scalar[output_dtype](value)
        )
    elif mode == 2:
        y[pos + row] = rebind[y.ElementType](
            Scalar[output_dtype](value > threshold)
        )
    else:
        y[pos + row] = rebind[y.ElementType](Scalar[output_dtype](value))
