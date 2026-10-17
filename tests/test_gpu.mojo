"""GPU semantic, storage, and GTIL acceptance checks (requires a GPU)."""
from acceptance import stump, check_edges
from reference_data import check_references
from balsa import load, Model, ModelBuilder, TreeBuilder, Operator
from max.gpu.host import DeviceContext
from pyre import Forest
from pyre.gpu import (
    GpuForest,
    GpuBatch,
    GpuArray,
    upload,
    upload_into,
    _has_fp64,
)
from pyre._gpu_storage import _Join, _same_device
from numojo import NDArray
from numojo.core.type_aliases import Shape, NDArrayStrides
from std.math import nan, inf
from std.testing import assert_equal, assert_true, assert_raises, TestSuite


def test_gtil() raises:
    check_references[True]()


def test_shared_edges() raises:
    check_edges[DType.float32, True]()
    check_edges[DType.float64, True]()


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
        # Trees are summed in a different association than on CPU.
        var e = expected.item(i)
        assert_true(abs(y.item(i) - e) <= Float32(1e-6) * (1 + abs(e)))
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


# Float64 where the target supports it; otherwise float32 keeps coverage.
comptime _WIDE = DType.float64 if _has_fp64() else DType.float32


def test_float64_rejected_without_fp64() raises:
    comptime if not _has_fp64():
        var model = stump[DType.float64](Operator.LT, False, False)
        with assert_raises(contains="FP64"):
            _ = GpuForest.prepare(model, DeviceContext())


def test_preparation_and_independent_contexts() raises:
    var ctx1 = DeviceContext()
    var ctx2 = DeviceContext()
    var model = stump[_WIDE](Operator.GE, True, False)
    var f1 = GpuForest.prepare(model, ctx1)
    var f2 = GpuForest.prepare(model, ctx2)
    model.trees[0].leaf_value[1] = 999
    var moved = f1^
    var host = NDArray[_WIDE](Shape(1, 1))
    host.itemset(0, 0.5)
    var x1 = upload(host, ctx1)
    var x2 = upload(host, ctx2)
    for _ in range(3):
        assert_equal(moved.predict(x1).to_host().item(0), Scalar[_WIDE](-1))
        assert_equal(f2.predict(x2).to_host().item(0), Scalar[_WIDE](-1))
    for name in ["category", "vector", "multi_target"]:
        var unsupported = load[_WIDE](
            "build/balsa-reference/tests/fixtures/"
            + String(_WIDE)
            + "_"
            + name
            + ".tl"
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


def test_oversized_batch_rejected_before_allocation() raises:
    var ctx = DeviceContext()
    var builder = ModelBuilder(num_feature=0)
    var tree = TreeBuilder(num_nodes=1)
    tree.leaf(0, 1.5)
    builder.add_tree(tree^.build())
    var model = builder^.build()
    var forest = GpuForest.prepare(model, ctx)
    # Zero features exercise the real public shape without a huge allocation.
    var x = GpuBatch[DType.float32](ctx, Shape(Int(Int32.MAX) + 1, 0))
    var output = GpuArray[DType.float32, 1](ctx, Shape(1))
    output._buffer.enqueue_fill(-99)
    ctx.synchronize()
    for operation in range(5):
        var raised = False
        try:
            if operation == 0:
                _ = forest.predict(x)
            elif operation == 1:
                _ = forest.predict_margin(x)
            elif operation == 2:
                _ = forest.predict_proba(x)
            elif operation == 3:
                _ = forest.predict_class(x)
            else:
                forest.predict_margin_into(x, output)
        except error:
            raised = True
            assert_equal(
                String(error),
                "Pyre GPU inference supports at most 2147483647 rows",
            )
        assert_true(raised)
    assert_equal(output.to_host().item(0), Float32(-99))


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
    comptime if _has_fp64():
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


struct _Lcg:
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int(self.state >> 33)

    def unit(mut self) -> Float64:
        return Float64(self.next() % 1000000) / 1000000.0


def random_forest[
    dtype: DType
](trees: Int, features: Int, depth: Int, seed: UInt64) raises -> Model[dtype]:
    """Uneven trees with every operator, both defaults and repeated features."""
    var rng = _Lcg(seed)
    var builder = ModelBuilder[dtype](
        num_feature=Int32(features),
        base_scores=[0.125],
        average_tree_output=False,
    )
    var ops = [Operator.EQ, Operator.LT, Operator.LE, Operator.GT, Operator.GE]
    for _ in range(trees):
        # Grow a random shape: node i splits with its children appended.
        var left = List[Int]()
        var right = List[Int]()
        var level = List[Int]()
        left.append(-1)
        right.append(-1)
        level.append(0)
        var i = 0
        while i < len(left):
            if level[i] < depth and (level[i] == 0 or rng.next() % 5 != 0):
                left[i] = len(left)
                right[i] = len(left) + 1
                for _ in range(2):
                    left.append(-1)
                    right.append(-1)
                    level.append(level[i] + 1)
            i += 1
        var tree = TreeBuilder[dtype](num_nodes=len(left))
        for n in range(len(left)):
            if left[n] < 0:
                tree.leaf(n, Scalar[dtype](rng.unit() * 2 - 1))
            else:
                # Thresholds lie on a coarse grid so inputs can hit them exactly.
                tree.numerical_split(
                    n,
                    feature=Int32(rng.next() % features),
                    threshold=Scalar[dtype](Float64(rng.next() % 9) / 4 - 1),
                    left=Int32(left[n]),
                    right=Int32(right[n]),
                    op=ops[rng.next() % len(ops)],
                    default_left=rng.next() % 2 == 0,
                )
        builder.add_tree(tree^.build())
    return builder^.build()


def random_rows[
    dtype: DType
](rows: Int, features: Int, seed: UInt64) raises -> NDArray[dtype]:
    var rng = _Lcg(seed)
    var x = NDArray[dtype](Shape(rows, features))
    for i in range(rows * features):
        var pick = rng.next() % 10
        if pick == 0:
            x.itemset(i, nan[dtype]())
        elif pick < 4:
            x.itemset(i, Scalar[dtype](Float64(rng.next() % 9) / 4 - 1))
        else:
            x.itemset(i, Scalar[dtype](rng.unit() * 2.5 - 1.25))
    return x^


def check_launch_configurations[
    dtype: DType
](trees: Int, features: Int, depth: Int, rows: Int) raises:
    var model = random_forest[dtype](trees, features, depth, 7)
    var x = random_rows[dtype](rows, features, 11)
    var expected = Forest(model.copy()).predict_margin(x, num_threads=1)
    var ctx = DeviceContext()
    var forest = GpuForest.prepare(model, ctx)
    var batch = upload(x, ctx)
    var output = GpuArray[dtype, 1](ctx, Shape(rows))
    var tolerance = Scalar[dtype](1e-5 if dtype == DType.float32 else 1e-12)
    for shift in range(6):
        for threads in [0, 32, 64, 128, 256, 512]:
            if threads != 0 and threads < (1 << shift):
                continue
            output._buffer.enqueue_fill(-99)
            with _Join(ctx):
                forest._enqueue[dtype, 1, 0](batch, output, 0.5, shift, threads)
            var y = output.to_host()
            for i in range(rows):
                var e = expected.item(i)
                if abs(y.item(i) - e) > tolerance * (1 + abs(e)):
                    raise Error(
                        String(
                            t"shift={shift} threads={threads} row={i}"
                            t" got={y.item(i)} expected={e}"
                        )
                    )


def test_launch_configurations() raises:
    check_launch_configurations[DType.float32](37, 5, 7, 70)
    comptime if _has_fp64():
        check_launch_configurations[DType.float64](37, 5, 7, 70)
    check_launch_configurations[DType.float32](300, 3, 4, 3)


def test_wide_float32_nodes() raises:
    # More features than the compact node encoding addresses.
    var model = random_forest[DType.float32](9, 9000, 5, 3)
    var ctx = DeviceContext()
    var forest = GpuForest.prepare(model, ctx)
    assert_equal(forest._words, 4)
    var x = random_rows[DType.float32](9, 9000, 5)
    var expected = Forest(model.copy()).predict_margin(x, num_threads=1)
    var y = forest.predict_margin(upload(x, ctx)).to_host()
    for i in range(9):
        # Trees are summed in a different association than on CPU.
        var e = expected.item(i)
        assert_true(abs(y.item(i) - e) <= Float32(1e-6) * (1 + abs(e)))


def check_threshold_extremes[dtype: DType]() raises:
    var big = Scalar[dtype].MAX_FINITE
    var tiny = Scalar[dtype](1e-30)
    var thresholds = [
        inf[dtype](),
        -inf[dtype](),
        big,
        -big,
        tiny,
        -tiny,
        Scalar[dtype](0),
        -Scalar[dtype](0),
        nan[dtype](),
    ]
    var values = [
        inf[dtype](),
        -inf[dtype](),
        big,
        -big,
        tiny,
        Scalar[dtype](0),
        -Scalar[dtype](0),
        nan[dtype](),
        Scalar[dtype](1),
        -tiny,
    ]
    var ctx = DeviceContext()
    for t in thresholds:
        for op in [
            Operator.EQ,
            Operator.LT,
            Operator.LE,
            Operator.GT,
            Operator.GE,
        ]:
            for default in [False, True]:
                var model = stump[dtype](op, default)
                for k in range(len(model.trees)):
                    model.trees[k].threshold[0] = t
                var x = NDArray[dtype](Shape(len(values), 1))
                for i in range(len(values)):
                    x.itemset(i, values[i])
                var expected = Forest(model.copy()).predict_margin(x)
                var forest = GpuForest.prepare(model, ctx)
                var y = forest.predict_margin(upload(x, ctx)).to_host()
                for i in range(len(values)):
                    if y.item(i) != expected.item(i):
                        raise Error(
                            String(
                                t"threshold={t} op={op} default={default}"
                                t" x={values[i]} got={y.item(i)}"
                                t" expected={expected.item(i)}"
                            )
                        )


def test_threshold_extremes() raises:
    check_threshold_extremes[DType.float32]()
    comptime if _has_fp64():
        check_threshold_extremes[DType.float64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
