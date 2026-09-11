from balsa import load
from std.testing import assert_equal, assert_true
from std.math import abs
from pyre import Forest
from numojo import NDArray
from numojo.core.type_aliases import Shape


def check_dense_reference[
    dtype: DType
](
    name: String,
    values: List[Scalar[dtype]],
    features: Int,
    expected: List[Scalar[dtype]],
    predictions: List[Scalar[dtype]],
    binary: Bool,
    tolerance: Scalar[dtype] = 0,
) raises:
    comptime assert dtype.is_floating_point()
    var forest = Forest(load[dtype]("build/balsa-reference/tests/fixtures/" + name))
    var rows = len(values) // features
    var x = NDArray[dtype](Shape(rows, features))
    for i in range(len(values)):
        x.itemset(i, values[i])
    var scalar = forest.predict_margin[1](x)
    var simd = forest.predict_margin(x)
    for i in range(rows):
        assert_equal(simd.item(i), scalar.item(i))
        assert_true(
            abs(simd.item(i) - expected[i])
            <= tolerance * (1 + abs(expected[i]))
        )
    if binary:
        var proba = forest.predict_proba(x)
        var labels = forest.predict_class(x)
        for i in range(rows):
            var p = proba.item(2 * i + 1)
            assert_true(
                abs(p - predictions[i])
                < Scalar[dtype](1e-6 if dtype == DType.float32 else 1e-12)
            )
            assert_equal(proba.item(2 * i), 1 - p)
            assert_equal(labels.item(i), Int64(predictions[i] > 0.5))
    else:
        var prediction = forest.predict(x)
        for i in range(rows):
            assert_true(
                abs(prediction.item(i) - predictions[i])
                <= tolerance * (1 + abs(predictions[i]))
            )
