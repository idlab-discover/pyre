"""Direct GPU checks against the shared saved GTIL inputs and expectations."""
from balsa import load
from max.gpu.host import DeviceContext
from pyre.gpu import GpuForest, GpuArray, upload
from numojo import NDArray
from numojo.core.type_aliases import Shape
from std.testing import assert_equal, assert_true


def check_gpu_reference[
    dtype: DType
](
    name: String,
    values: List[Scalar[dtype]],
    features: Int,
    expected: List[Scalar[dtype]],
    predictions: List[Scalar[dtype]],
    binary: Bool,
    tolerance: Scalar[dtype],
) raises:
    comptime assert dtype.is_floating_point()
    print("GPU oracle:", name)
    var ctx = DeviceContext()
    var model = load[dtype]("build/balsa-reference/tests/fixtures/" + name)
    var forest = GpuForest.prepare(model, ctx)
    var rows = len(values) // features
    var x = NDArray[dtype](Shape(rows, features))
    for i in range(len(values)):
        x.itemset(i, values[i])
    var batch = upload(x, ctx)
    var raw = forest.predict_margin(batch).to_host()
    var reused = GpuArray[dtype, 1](ctx, Shape(rows))
    for _ in range(3):
        forest.predict_margin_into(batch, reused)
    var again = reused.to_host()
    for i in range(rows):
        assert_true(
            abs(raw.item(i) - expected[i]) <= tolerance * (1 + abs(expected[i]))
        )
        assert_equal(raw.item(i), again.item(i))
    if binary:
        var p = forest.predict_proba(batch).to_host()
        var labels = forest.predict_class(batch).to_host()
        var outp = GpuArray[dtype, 2](ctx, Shape(rows, 2))
        var outc = GpuArray[DType.int64, 1](ctx, Shape(rows))
        forest.predict_proba_into(batch, outp)
        forest.predict_class_into(batch, outc)
        var reusedp = outp.to_host()
        var reusedc = outc.to_host()
        for i in range(rows):
            assert_true(
                abs(p.item(2 * i + 1) - predictions[i])
                < Scalar[dtype](1e-6 if dtype == DType.float32 else 1e-12)
            )
            assert_equal(p.item(2 * i), 1 - p.item(2 * i + 1))
            assert_equal(labels.item(i), Int64(predictions[i] > 0.5))
            assert_equal(reusedp.item(2 * i + 1), p.item(2 * i + 1))
            assert_equal(reusedc.item(i), labels.item(i))
    else:
        var p = forest.predict(batch).to_host()
        forest.predict_into(batch, reused)
        var reusedp = reused.to_host()
        for i in range(rows):
            assert_true(
                abs(p.item(i) - predictions[i])
                <= tolerance * (1 + abs(predictions[i]))
            )
            assert_equal(reusedp.item(i), p.item(i))
