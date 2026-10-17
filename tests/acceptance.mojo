"""Shared prediction rules and cases, with CPU/GPU execution adapters."""
from balsa import load, Model, ModelBuilder, TreeBuilder, Operator, TaskType
from max.gpu.host import DeviceContext
from pyre import Forest
from pyre.gpu import GpuForest, upload, _has_fp64
from numojo import NDArray
from numojo.core.type_aliases import Shape
from std.math import nan, inf
from std.testing import assert_equal, assert_raises


def close[
    dtype: DType
](
    context: String,
    row: Int,
    actual: Scalar[dtype],
    expected: Scalar[dtype],
    tolerance: Scalar[dtype] = 0,
    strict: Bool = False,
) raises:
    var matches = actual == expected
    if strict or tolerance != 0:
        var delta = abs(actual - expected)
        matches = delta < tolerance if strict else delta <= tolerance
    if not matches:
        raise Error(
            String(
                t"{context} row={row} actual={actual} expected={expected} tolerance={tolerance}"
            )
        )


def same[
    dtype: DType
](context: String, actual: NDArray[dtype], expected: NDArray[dtype]) raises:
    assert_equal(actual.shape, expected.shape, context)
    var columns = actual.shape[1] if actual.ndim == 2 else 1
    for i in range(actual.size):
        var row = i // columns
        close(context, row, actual.item(i), expected.item(i))


@fieldwise_init
struct Predictions[dtype: DType](Movable):
    var margin: NDArray[Self.dtype]
    var prediction: NDArray[Self.dtype]
    # Ordered predictions at thresholds 0, 0.5, and 1.
    var labels: List[NDArray[DType.int64]]


def cpu_predictions[
    dtype: DType
](
    var model: Model[dtype],
    x: NDArray[dtype],
    binary: Bool,
    context: String,
    threads: Int,
) raises -> Predictions[dtype]:
    var forest = Forest(model^)
    var margin = forest.predict_margin(x, num_threads=threads)
    same(context + " margin scalar", margin, forest.predict_margin[1](x))
    var prediction: NDArray[dtype]
    var labels = List[NDArray[DType.int64]]()
    if binary:
        prediction = forest.predict_proba(x, num_threads=threads)
        for threshold in [Scalar[dtype](0), 0.5, 1]:
            labels.append(
                forest.predict_class(x, threshold, num_threads=threads)
            )
        for threshold in [
            nan[dtype](),
            Scalar[dtype](-0.1),
            Scalar[dtype](1.1),
        ]:
            with assert_raises():
                _ = forest.predict_class(x, threshold, num_threads=threads)
        with assert_raises():
            _ = forest.predict(x, num_threads=threads)
    else:
        prediction = forest.predict(x, num_threads=threads)
    return Predictions(margin^, prediction^, labels^)


def gpu_predictions[
    dtype: DType
](
    model: Model[dtype],
    x: NDArray[dtype],
    binary: Bool,
    context: String,
) raises -> Predictions[dtype]:
    var ctx = DeviceContext()
    var forest = GpuForest.prepare(model, ctx)
    var batch = upload(x, ctx)
    var output = forest.predict_margin(batch)
    var margin = output.to_host()
    output._buffer.enqueue_fill(-99)
    for _ in range(3):
        forest.predict_margin_into(batch, output)
    same(context + " margin reuse", output.to_host(), margin)
    var prediction: NDArray[dtype]
    var labels = List[NDArray[DType.int64]]()
    if binary:
        var p = forest.predict_proba(batch)
        prediction = p.to_host()
        p._buffer.enqueue_fill(-99)
        forest.predict_proba_into(batch, p)
        same(context + " probability reuse", p.to_host(), prediction)
        for threshold in [Scalar[dtype](0), 0.5, 1]:
            var c = forest.predict_class(batch, threshold)
            var host = c.to_host()
            c._buffer.enqueue_fill(-99)
            forest.predict_class_into(batch, c, threshold)
            same(context + " class reuse", c.to_host(), host)
            labels.append(host^)
        for threshold in [
            nan[dtype](),
            Scalar[dtype](-0.1),
            Scalar[dtype](1.1),
        ]:
            with assert_raises():
                _ = forest.predict_class(batch, threshold)
        with assert_raises():
            _ = forest.predict(batch)
    else:
        prediction = forest.predict(batch).to_host()
        output._buffer.enqueue_fill(-99)
        forest.predict_into(batch, output)
        same(context + " regression reuse", output.to_host(), prediction)
    return Predictions(margin^, prediction^, labels^)


def check_model[
    dtype: DType, on_gpu: Bool
](
    name: String,
    var model: Model[dtype],
    values: List[Scalar[dtype]],
    expected: List[Scalar[dtype]],
    predictions: List[Scalar[dtype]],
    tolerance: Scalar[dtype] = 0,
    exact_probability: Bool = False,
    threads: Int = 0,
) raises:
    var binary = model.task_type == TaskType.BINARY_CLF
    var context = name + (" GPU " if on_gpu else " CPU ") + String(dtype)
    var rows = len(expected)
    var x = NDArray[dtype](Shape(rows, Int(model.num_feature)))
    assert_equal(x.size, len(values), context)
    assert_equal(rows, len(predictions), context)
    for i in range(len(values)):
        x.itemset(i, values[i])
    comptime if on_gpu and dtype == DType.float64 and not _has_fp64():
        # Targets without FP64 must reject float64 forests up front.
        with assert_raises(contains="FP64"):
            _ = GpuForest.prepare(model, DeviceContext())
        return
    var result: Predictions[dtype]
    comptime if on_gpu:
        result = gpu_predictions(model, x, binary, context)
    else:
        result = cpu_predictions(model^, x, binary, context, threads)
    assert_equal(result.margin.shape, Shape(rows), context + " margin")
    assert_equal(
        result.prediction.shape,
        Shape(rows, 2) if binary else Shape(rows),
        context + " prediction",
    )
    var probability_tolerance = Scalar[dtype](
        1e-6 if dtype == DType.float32 else 1e-12
    )
    if exact_probability:
        probability_tolerance = 0
    for i in range(rows):
        close(
            context + " margin",
            i,
            result.margin.item(i),
            expected[i],
            tolerance * (1 + abs(expected[i])),
        )
        if binary:
            var p = result.prediction.item(2 * i + 1)
            close(
                context + " probability",
                i,
                p,
                predictions[i],
                probability_tolerance,
                not exact_probability,
            )
            close(
                context + " complement", i, result.prediction.item(2 * i), 1 - p
            )
        else:
            close(
                context + " regression",
                i,
                result.prediction.item(i),
                predictions[i],
                tolerance * (1 + abs(predictions[i])),
            )
    assert_equal(len(result.labels), 3 if binary else 0, context + " classes")
    for t in range(len(result.labels)):
        var threshold = Scalar[dtype](t) * 0.5
        var label = context + " class threshold=" + String(threshold)
        assert_equal(result.labels[t].shape, Shape(rows), label)
        for i in range(rows):
            close(
                label,
                i,
                result.labels[t].item(i),
                Int64(predictions[i] > threshold),
            )


def check_dense_reference[
    dtype: DType, on_gpu: Bool = False
](
    name: String,
    values: List[Scalar[dtype]],
    expected: List[Scalar[dtype]],
    predictions: List[Scalar[dtype]],
    tolerance: Scalar[dtype] = 0,
) raises:
    var model = load[dtype]("build/balsa-reference/tests/fixtures/" + name)
    check_model[dtype, on_gpu](
        name, model^, values, expected, predictions, tolerance
    )


def stump[
    dtype: DType = DType.float32
](
    op: Int8 = Operator.LT,
    default: Bool = False,
    classifier: Bool = False,
    trees: Int = 2,
) raises -> Model[dtype]:
    var builder = ModelBuilder[dtype](
        num_feature=1, base_scores=[0.25], average_tree_output=trees > 1
    )
    for _ in range(trees):
        var tree = TreeBuilder[dtype](num_nodes=3)
        tree.numerical_split(
            0,
            feature=0,
            threshold=0.5,
            left=1,
            right=2,
            op=op,
            default_left=default,
        )
        tree.leaf(1, -1.25)
        tree.leaf(2, 2.75)
        builder.add_tree(tree^.build())
    var model = builder^.build()
    if classifier:
        model.task_type = TaskType.BINARY_CLF
        model.set_postprocessor_name("sigmoid")
        model.sigmoid_alpha = 0.75
    return model^


def goes_left[dtype: DType](v: Scalar[dtype], op: Int8, default: Bool) -> Bool:
    if v != v:
        return default
    if op == Operator.EQ:
        return v == 0.5
    if op == Operator.LT:
        return v < 0.5
    if op == Operator.LE:
        return v <= 0.5
    if op == Operator.GT:
        return v > 0.5
    return v >= 0.5


def check_edges[dtype: DType, on_gpu: Bool = False]() raises:
    var samples = [
        Scalar[dtype](0),
        0.5,
        1,
        nan[dtype](),
        -inf[dtype](),
        inf[dtype](),
        -Scalar[dtype](0),
    ]
    for op in [Operator.EQ, Operator.LT, Operator.LE, Operator.GT, Operator.GE]:
        for default in [False, True]:
            # All previous CPU tails plus GPU block tails, now in both precisions.
            for count in range(36):
                var rows = count if count < 34 else 257 if count == 34 else 129
                var values = List[Scalar[dtype]]()
                var expected = List[Scalar[dtype]]()
                for i in range(rows):
                    var v = samples[i % len(samples)]
                    var left = goes_left(v, op, default)
                    values.append(v)
                    expected.append(Scalar[dtype](-1 if left else 3))
                check_model[dtype, on_gpu](
                    String(t"operator={op} default={default} rows={rows}"),
                    stump[dtype](op, default),
                    values,
                    expected,
                    expected,
                )
    check_binary_edges[dtype, on_gpu]([0, 1, 33, 129, 257])


def check_binary_edges[
    dtype: DType, on_gpu: Bool = False
](sizes: List[Int], threads: Int = 0) raises:
    for bias in [Float64(-10000), Float64(0), Float64(10000)]:
        for rows in sizes:
            var model = stump[dtype](Operator.LT, False, True)
            model.base_scores[0] = bias
            for t in range(len(model.trees)):
                model.trees[t].leaf_value[1] = 0
                model.trees[t].leaf_value[2] = 0
            var values = List[Scalar[dtype]]()
            var margins = List[Scalar[dtype]]()
            var predictions = List[Scalar[dtype]]()
            for _ in range(rows):
                values.append(0)
                margins.append(Scalar[dtype](bias))
                predictions.append(
                    Scalar[dtype](0.0 if bias < 0 else 1.0 if bias > 0 else 0.5)
                )
            check_model[dtype, on_gpu](
                String(t"binary bias={bias} rows={rows} threads={threads}"),
                model^,
                values,
                margins,
                predictions,
                exact_probability=True,
                threads=threads,
            )
