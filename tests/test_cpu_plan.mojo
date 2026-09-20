"""Public inference and internal plan eligibility over trainer-free fixtures."""
from std.math import exp
from std.testing import (
    assert_equal,
    assert_almost_equal,
    assert_raises,
    TestSuite,
)
from balsa import ModelBuilder, TreeBuilder, Operator
from pyre import Forest
from pyre._cpu import _CPUPlan, _GENERAL, _DIRECT, _BINNED, _CONSTANT
from cpu_fixtures import synthetic_model, synthetic_input, scalar_margin


def check_family[
    dtype: DType
](kind: String, trees: Int, depth: Int, features: Int, route: Int) raises:
    comptime assert dtype.is_floating_point()
    for classifier in [False, True]:
        var model = synthetic_model[dtype](
            kind, trees, depth, features, borders=2, classifier=classifier
        )
        var forest = Forest(model.copy())
        assert_equal(forest._cpu.kernel(513), route)
        var generic = _CPUPlan(model.copy(), strategy=_GENERAL)
        for rows in [0, 1, 17, 255, 256, 257, 513]:
            var x = synthetic_input[dtype](rows, features)
            var y = forest.predict_margin(x, num_threads=1)
            var control = generic.predict_margin(x, num_threads=1)
            for row in range(rows):
                var expected = scalar_margin(model, x, row)
                assert_equal(y.item(row), expected)
                assert_equal(control.item(row), expected)
            if classifier:
                var p = forest.predict_proba(x, num_threads=1)
                var c = forest.predict_class(x, 0.37, num_threads=1)
                for row in range(rows):
                    var expected = 1 / (
                        1
                        + exp(
                            -Scalar[dtype](0.75) * scalar_margin(model, x, row)
                        )
                    )
                    assert_almost_equal(
                        p.item(2 * row + 1),
                        expected,
                        atol=1e-6 if dtype == DType.float32 else 1e-12,
                        rtol=1e-6 if dtype == DType.float32 else 1e-12,
                    )
                    assert_equal(c.item(row), Int64(expected > 0.37))
            else:
                var prediction = forest.predict(x, num_threads=1)
                for row in range(rows):
                    assert_equal(prediction.item(row), y.item(row))


def check_shapes[dtype: DType]() raises:
    check_family[dtype]("roots", 129, 0, 0, _CONSTANT)
    check_family[dtype]("roots", 129, 0, 16, _CONSTANT)
    check_family[dtype]("stumps", 129, 1, 16, _BINNED)
    for depth in [4, 8, 12, 16]:
        check_family[dtype]("symmetric", 4, depth, 4, _BINNED)
        check_family[dtype]("asymmetric", 2, depth, 4, _GENERAL)
    check_family[dtype]("chain", 17, 16, 16, _GENERAL)
    check_family[dtype]("mixed", 8, 4, 16, _GENERAL)


def test_fixture_shapes_float32() raises:
    check_shapes[DType.float32]()


def test_fixture_shapes_float64() raises:
    check_shapes[DType.float64]()


def test_plan_controls_and_small_batches() raises:
    var model = synthetic_model[DType.float32]("symmetric", 16, 4, 4, borders=2)
    var plan = _CPUPlan(model.copy())
    assert_equal(plan.kernel(1), _DIRECT)
    assert_equal(plan.kernel(255), _DIRECT)
    assert_equal(plan.kernel(256), _BINNED)
    with assert_raises():
        _ = _CPUPlan(model.copy(), strategy=_CONSTANT)
    with assert_raises():
        _ = _CPUPlan(model.copy(), strategy=99)
    with assert_raises():
        _ = _CPUPlan(
            synthetic_model[DType.float32]("asymmetric", 2, 4, 4),
            strategy=_DIRECT,
        )
    # One split per feature cannot amortize binning and remains direct.
    var stumps = Forest(synthetic_model[DType.float32]("stumps", 16, 1, 16))
    assert_equal(stumps._cpu.kernel(5000), _DIRECT)
    var deep = Forest(synthetic_model[DType.float32]("symmetric", 1, 17, 4))
    assert_equal(deep._cpu.kernel(5000), _GENERAL)


def test_constant_sum_order() raises:
    var builder = ModelBuilder[DType.float32](num_feature=0, base_scores=[0.25])
    for value in [Float32(1e20), Float32(1), Float32(-1e20), Float32(3)]:
        var tree = TreeBuilder[DType.float32](num_nodes=1)
        tree.leaf(0, value)
        builder.add_tree(tree^.build())
    var forest = Forest(builder^.build())
    assert_equal(forest._cpu.kernel(513), _CONSTANT)
    var x = synthetic_input[DType.float32](513, 0)
    var y = forest.predict(x, num_threads=8)
    for row in range(513):
        assert_equal(y.item(row), Float32(3.25))
    with assert_raises():
        _ = forest.predict(x, num_threads=-1)


def check_every_leaf[dtype: DType]() raises:
    # Independent features with zero thresholds make all 65,536 leaf addresses
    # reachable. Rotate features across two trees so bin reuse also applies.
    var model = synthetic_model[dtype]("symmetric", 2, 16, 16, borders=1)
    var direct = _CPUPlan(model.copy(), strategy=_DIRECT)
    var forest = Forest(model^)
    assert_equal(forest._cpu.kernel(65537), _BINNED)
    var x = synthetic_input[dtype](65537, 16, edges=False)
    for row in range(65537):
        for f in range(16):
            x.itemset(
                row * 16 + f, Scalar[dtype](1 if ((row >> f) & 1) else -1)
            )
    var y = forest.predict(x, num_threads=1)
    var dy = direct.predict(x, num_threads=1)
    for row in range(65537):
        var total = Scalar[dtype](0)
        for t in range(2):
            var leaf = 0
            for depth in range(16):
                leaf = (leaf << 1) | ((row >> ((depth * 7 + t * 3) % 16)) & 1)
            total += Scalar[dtype]((leaf * 17 + t * 11) % 97 - 48) / 13
        var expected = Scalar[dtype](
            Float64(total / 2) + Float64(0.123456789123)
        )
        assert_equal(y.item(row), expected)
        assert_equal(dy.item(row), expected)


def test_every_depth16_leaf() raises:
    check_every_leaf[DType.float32]()
    check_every_leaf[DType.float64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
