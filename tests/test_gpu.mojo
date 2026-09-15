"""GPU semantic, storage, and GTIL acceptance checks (requires a GPU)."""
from reference_data import check_references
from balsa import load, Model, ModelBuilder, TreeBuilder, Operator, TaskType
from max.gpu.host import DeviceContext
from pyre import Forest
from pyre.gpu import GpuForest, GpuBatch, GpuArray, upload, upload_into
from pyre._gpu_storage import _Join, _same_device
from numojo import NDArray
from numojo.core.type_aliases import Shape, NDArrayStrides
from std.math import nan, inf
from std.testing import assert_equal, assert_true, assert_raises, TestSuite


def test_gtil() raises:
    check_references[True]()


def stump[
    dtype: DType
](op: Int8, default: Bool, classifier: Bool) raises -> Model[dtype]:
    var builder = ModelBuilder[dtype](
        num_feature=1, base_scores=[0.25], average_tree_output=True
    )
    for _ in range(2):
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


def operators[dtype: DType]() raises:
    var ctx = DeviceContext()
    var values = [
        Scalar[dtype](-1),
        Scalar[dtype](0.5),
        Scalar[dtype](1),
        nan[dtype](),
        -inf[dtype](),
        inf[dtype](),
    ]
    for op in [Operator.EQ, Operator.LT, Operator.LE, Operator.GT, Operator.GE]:
        for default in [False, True]:
            var model = stump[dtype](op, default, False)
            var cpu = Forest(model.copy())
            var forest = GpuForest.prepare(model, ctx)
            # Mutation after preparation must not affect device inference.
            model.trees[0].leaf_value[1] = 999
            var moved = forest^
            for rows in [0, 1, 33, 257]:
                var x = NDArray[dtype](Shape(rows, 1))
                for row in range(rows):
                    x.itemset(row, values[row % len(values)])
                var batch = upload(x, ctx)
                var moved_batch = batch^
                var y = moved.predict(moved_batch).to_host()
                var expected = cpu.predict(x)
                assert_equal(y.size, rows)
                for i in range(rows):
                    assert_equal(y.item(i), expected.item(i))


def test_operators_f32() raises:
    operators[DType.float32]()


def test_operators_f64() raises:
    operators[DType.float64]()


def binary[dtype: DType]() raises:
    var ctx = DeviceContext()
    var model = stump[dtype](Operator.LT, False, True)
    for t in range(len(model.trees)):
        model.trees[t].leaf_value[1] = 0
        model.trees[t].leaf_value[2] = 0
    for bias in [Float64(-10000), Float64(0), Float64(10000)]:
        model.base_scores[0] = bias
        var f = GpuForest.prepare(model, ctx)
        for rows in [0, 1, 33, 257]:
            var host = NDArray[dtype](Shape(rows, 1))
            for i in range(rows):
                host.itemset(i, 0)
            var x = upload(host, ctx)
            var p = f.predict_proba(x).to_host()
            assert_equal(p.shape[0], rows)
            assert_equal(p.shape[1], 2)
            for threshold in [
                Scalar[dtype](0),
                Scalar[dtype](0.5),
                Scalar[dtype](1),
            ]:
                var y = f.predict_class(x, threshold).to_host()
                for i in range(rows):
                    var expected = Scalar[dtype](
                        0.0 if bias < 0 else 1.0 if bias > 0 else 0.5
                    )
                    assert_equal(p.item(2 * i + 1), expected)
                    assert_equal(y.item(i), Int64(expected > threshold))
            with assert_raises():
                _ = f.predict_class(x, nan[dtype]())
            with assert_raises():
                _ = f.predict_class(x, -0.1)
            with assert_raises():
                _ = f.predict_class(x, 1.1)
            with assert_raises():
                _ = f.predict(x)


def test_binary_f32() raises:
    binary[DType.float32]()


def test_binary_f64() raises:
    binary[DType.float64]()


def test_offsets_capacity_aliasing() raises:
    var ctx = DeviceContext()
    var model = stump[DType.float32](Operator.LT, False, False)
    var f = GpuForest.prepare(model, ctx)
    var host = NDArray[DType.float32](Shape(12, 1))
    for i in range(12):
        host.itemset(i, Float32(i) * 0.1)
    var view = host.view_with_layout(Shape(9, 1), NDArrayStrides(1, 1), 2)
    var x = upload(view, ctx)
    var expected = f.predict(x).to_host()
    var backing = ctx.enqueue_create_buffer[DType.float32](24)
    backing.enqueue_fill(-99)
    var native_input = ctx.enqueue_create_buffer[DType.float32](12)
    native_input.enqueue_copy_from(host.unsafe_ptr())
    ctx.synchronize()
    var address = Int(native_input.unsafe_ptr())
    var batch = GpuBatch[DType.float32].from_buffer(
        native_input, 9, 1, offset=2, context=ctx
    )
    assert_equal(Int(batch._buffer.unsafe_ptr()), address)
    var output = GpuArray[DType.float32, 1](
        backing, Shape(9), offset=3, context=ctx
    )
    for _ in range(3):
        f.predict_into(batch, output)
    var y = output.to_host()
    for i in range(9):
        assert_equal(y.item(i), expected.item(i))
    var all_values = GpuArray[DType.float32, 1](
        backing, Shape(24), context=ctx
    ).to_host()
    assert_equal(all_values.item(2), Float32(-99))
    assert_equal(all_values.item(12), Float32(-99))
    var alias_output = GpuArray[DType.float32, 1](
        native_input, Shape(9), offset=2, context=ctx
    )
    with assert_raises():
        f.predict_into(batch, alias_output)
    with assert_raises():
        _ = GpuBatch[DType.float32].from_buffer(
            native_input, 12, 1, offset=1, context=ctx
        )
    with assert_raises():
        _ = GpuBatch[DType.float32].from_buffer(
            native_input, 1, 1, offset=-1, context=ctx
        )
    with assert_raises():
        _ = GpuBatch[DType.float32](ctx, Shape(-1, 1))
    with assert_raises():
        _ = GpuBatch[DType.float32](ctx, Shape(Int.MAX, 2))
    with assert_raises():
        _ = GpuBatch[DType.float32](ctx, Shape(1))
    with assert_raises():
        _ = batch.dim(2)
    var wrong_rows = GpuArray[DType.float32, 1](ctx, Shape(1))
    with assert_raises():
        f.predict_into(batch, wrong_rows)
    var wrong_features = GpuBatch[DType.float32](ctx, Shape(1, 2))
    with assert_raises():
        _ = f.predict(wrong_features)
    var strided = host.view_with_layout(Shape(3, 1), NDArrayStrides(2, 1), 0)
    with assert_raises():
        _ = upload(strided, ctx)
    var rank1 = NDArray[DType.float32](Shape(2))
    with assert_raises():
        _ = upload(rank1, ctx)
    with assert_raises():
        _ = f.predict_proba(batch)
    with assert_raises():
        _ = f.predict_class(batch)
    var cpu_context = DeviceContext(api="cpu")
    with assert_raises():
        _ = GpuBatch[DType.float32](cpu_context, Shape(1, 1))
    with assert_raises():
        _same_device(ctx, cpu_context)


def test_preparation_and_independent_contexts() raises:
    var ctx1 = DeviceContext()
    var ctx2 = DeviceContext()
    var model = stump[DType.float64](Operator.GE, True, False)
    var f1 = GpuForest.prepare(model, ctx1)
    var f2 = GpuForest.prepare(model, ctx2)
    var host = NDArray[DType.float64](Shape(1, 1))
    host.itemset(0, 0.5)
    var x1 = upload(host, ctx1)
    var x2 = upload(host, ctx2)
    for _ in range(3):
        assert_equal(f1.predict(x1).to_host().item(0), Float64(-1))
        assert_equal(f2.predict(x2).to_host().item(0), Float64(-1))
    for name in ["category", "vector", "multi_target"]:
        var unsupported = load[DType.float64](
            "build/balsa-reference/tests/fixtures/float64_" + name + ".tl"
        )
        with assert_raises():
            _ = GpuForest.prepare(unsupported, ctx1)
    model.trees[0].cleft[0] = 999
    with assert_raises():
        _ = GpuForest.prepare(model, ctx1)


def test_zero_features_and_join_on_error() raises:
    var ctx = DeviceContext()
    var builder = ModelBuilder(num_feature=0, base_scores=[0.25])
    var tree = TreeBuilder(num_nodes=1)
    tree.leaf(0, 1.5)
    builder.add_tree(tree^.build())
    var model = builder^.build()
    var f = GpuForest.prepare(model, ctx)
    var x = GpuBatch[DType.float32](ctx, Shape(257, 0))
    var y = f.predict(x).to_host()
    assert_equal(y.size, 257)
    assert_equal(y.item(256), Float32(1.75))
    var buffer = ctx.enqueue_create_buffer[DType.float32](1)
    with assert_raises():
        with _Join(ctx):
            buffer.enqueue_fill(7)
            raise Error("injected failure after enqueue")
    var result = GpuArray[DType.float32, 1](
        buffer, Shape(1), context=ctx
    ).to_host()
    assert_equal(result.item(0), Float32(7))


def escaped_result() raises -> GpuArray[DType.float32, 1]:
    var ctx = DeviceContext()
    var model = stump[DType.float32](Operator.LT, False, False)
    var forest = GpuForest.prepare(model, ctx)
    var host = NDArray[DType.float32](Shape(1, 1))
    host.itemset(0, 0)
    var batch = upload(host, ctx)
    return forest.predict(batch)


def test_result_outlives_context_and_forest() raises:
    for _ in range(5):
        var result = escaped_result()
        var moved = result^
        assert_equal(moved.to_host().item(0), Float32(-1))


def reusable_upload[dtype: DType]() raises:
    var ctx = DeviceContext()
    var model = stump[dtype](Operator.LT, True, False)
    var cpu = Forest(model.copy())
    var forest = GpuForest.prepare(model, ctx)
    var backing = ctx.enqueue_create_buffer[dtype](264)
    var output_backing = ctx.enqueue_create_buffer[dtype](270)
    ctx.synchronize()
    var input_address = Int(backing.unsafe_ptr())
    var output_address = Int(output_backing.unsafe_ptr())
    for rows in [0, 1, 33, 257, 3, 0, 257]:
        backing.enqueue_fill(-99)
        output_backing.enqueue_fill(-77)
        ctx.synchronize()
        var host = NDArray[dtype](Shape(rows + 2, 1))
        for i in range(host.size):
            host.itemset(i, Scalar[dtype](Float64(i - 2) * 0.25))
        var view = host.view_with_layout(
            Shape(rows, 1), NDArrayStrides(1, 1), 2
        )
        var expected = cpu.predict(view)
        var batch = GpuBatch[dtype].from_buffer(
            backing, rows, 1, offset=3, context=ctx
        )
        var output = GpuArray[dtype, 1](
            output_backing, Shape(rows), offset=5, context=ctx
        )
        assert_equal(Int(batch._buffer.unsafe_ptr()), input_address)
        assert_equal(Int(output._buffer.unsafe_ptr()), output_address)
        upload_into(view, batch)
        # The producer may overwrite/release host storage as soon as upload ends.
        for i in range(host.size):
            host.itemset(i, 999)
        var moved = batch^
        forest.predict_into(moved, output)
        var y = output.to_host()
        for row in range(rows):
            assert_equal(y.item(row), expected.item(row))
        var all_input = GpuArray[dtype, 1](
            backing, Shape(264), context=ctx
        ).to_host()
        for i in range(264):
            if i >= 3 and i < 3 + rows:
                assert_equal(
                    all_input.item(i), Scalar[dtype](Float64(i - 3) * 0.25)
                )
            else:
                assert_equal(all_input.item(i), Scalar[dtype](-99))
        var all_output = GpuArray[dtype, 1](
            output_backing, Shape(270), context=ctx
        ).to_host()
        for i in range(270):
            if i < 5 or i >= 5 + rows:
                assert_equal(all_output.item(i), Scalar[dtype](-77))
        if rows > 0:
            var overlapping = GpuArray[dtype, 1](
                backing, Shape(rows), offset=3, context=ctx
            )
            with assert_raises():
                forest.predict_into(moved, overlapping)


def test_reusable_upload_f32() raises:
    reusable_upload[DType.float32]()


def test_reusable_upload_f64() raises:
    reusable_upload[DType.float64]()


def test_upload_into_failure_and_reuse() raises:
    var ctx = DeviceContext()
    var batch = GpuBatch[DType.float32](ctx, Shape(3, 1))
    batch._buffer.enqueue_fill(-99)
    ctx.synchronize()
    var address = Int(batch._buffer.unsafe_ptr())
    var rank1 = NDArray[DType.float32](Shape(3))
    var rows = NDArray[DType.float32](Shape(2, 1))
    var features = NDArray[DType.float32](Shape(3, 2))
    var host = NDArray[DType.float32](Shape(6, 1))
    var strided = host.view_with_layout(Shape(3, 1), NDArrayStrides(2, 1), 0)
    with assert_raises():
        upload_into(rank1, batch)
    with assert_raises():
        upload_into(rows, batch)
    with assert_raises():
        upload_into(features, batch)
    with assert_raises():
        upload_into(strided, batch)
    var unchanged = batch.to_host()
    for i in range(3):
        assert_equal(unchanged.item(i), Float32(-99))
    var valid = NDArray[DType.float32](Shape(3, 1))
    valid.itemset(0, 1)
    valid.itemset(1, nan[DType.float32]())
    valid.itemset(2, inf[DType.float32]())
    upload_into(valid, batch)
    var actual = batch.to_host()
    assert_equal(actual.item(0), Float32(1))
    assert_true(actual.item(1) != actual.item(1))
    assert_equal(actual.item(2), inf[DType.float32]())
    assert_equal(Int(batch._buffer.unsafe_ptr()), address)
    var empty = NDArray[DType.float32](Shape(7, 0))
    var zero_features = GpuBatch[DType.float32](ctx, Shape(7, 0))
    upload_into(empty, zero_features)
    assert_equal(zero_features.dim(0), 7)


def escaped_upload() raises -> GpuBatch[DType.float32]:
    var ctx = DeviceContext()
    var host = NDArray[DType.float32](Shape(1, 1))
    host.itemset(0, 42)
    var batch = GpuBatch[DType.float32](ctx, Shape(1, 1))
    upload_into(host, batch)
    return batch^


def test_reused_upload_outlives_host_and_context() raises:
    var batch = escaped_upload()
    var moved = batch^
    assert_equal(moved.to_host().item(0), Float32(42))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
