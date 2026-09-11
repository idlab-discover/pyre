from std.math import nan
from std.testing import assert_equal, assert_raises, TestSuite
from balsa import (
    load,
    load_auto,
    Model,
    ModelBuilder,
    TreeBuilder,
    Operator,
    TaskType,
)
from pyre import Forest
from numojo import NDArray
from numojo.core.type_aliases import Shape
from numojo.core.layout.ndstrides import NDArrayStrides
from reference_data import check_references
from test_accelerator import check_gpu_unimplemented, check_device
from numojo.core.accelerator.device import Device


def test_treelite_references() raises:
    check_references()


def make_stump(
    op: Int8 = Operator.LT, default_left: Bool = False
) raises -> Model[DType.float32]:
    var model = ModelBuilder(num_feature=1, base_scores=[0.25])
    var tree = TreeBuilder(num_nodes=3)
    tree.numerical_split(
        0,
        feature=0,
        threshold=0.5,
        op=op,
        left=1,
        right=2,
        default_left=default_left,
    )
    tree.leaf(1, -1.25)
    tree.leaf(2, 2.75)
    model.add_tree(tree^.build())
    return model^.build()


def test_all_operators_defaults_and_tails() raises:
    for op in [Operator.EQ, Operator.LT, Operator.LE, Operator.GT, Operator.GE]:
        for default in [False, True]:
            var forest = Forest(make_stump(op, default))
            for rows in range(34):
                var x = NDArray[DType.float32](Shape(rows, 1))
                for i in range(rows):
                    var value = Float32(i % 4) * 0.5
                    if i % 4 == 3:
                        value = nan[DType.float32]()
                    x.itemset(i, value)
                var y = forest.predict(x)
                assert_equal(y.size, rows)
                for i in range(rows):
                    var v = x.item(i)
                    var left: Bool
                    if op == Operator.EQ:
                        left = v == 0.5
                    elif op == Operator.LT:
                        left = v < 0.5
                    elif op == Operator.LE:
                        left = v <= 0.5
                    elif op == Operator.GT:
                        left = v > 0.5
                    else:
                        left = v >= 0.5
                    if v != v:
                        left = default
                    assert_equal(y.item(i), Float32(-1 if left else 3))


def test_owned_preparation_and_offset_view() raises:
    var model = make_stump()
    var forest = Forest(model.copy())
    model.trees[0].leaf_value[1] = 999
    var backing = NDArray[DType.float32](Shape(12, 1))
    for i in range(12):
        backing.itemset(i, Float32(i) * 0.1)
    var view = backing.view_with_layout(Shape(9, 1), NDArrayStrides(1, 1), 2)
    assert_equal(view.offset, 2)
    var y = forest.predict(view)
    assert_equal(y.size, 9)
    for i in range(9):
        assert_equal(y.item(i), Float32(-1 if i + 2 < 5 else 3))


def test_invalid_inputs_and_unsupported_models() raises:
    var forest = Forest(load("build/balsa-reference/tests/fixtures/float32_op2_missing0.tl"))
    var rank_one = NDArray[DType.float32](Shape(3))
    with assert_raises():
        _ = forest.predict(rank_one)
    var wrong_features = NDArray[DType.float32](Shape(2, 2))
    with assert_raises():
        _ = forest.predict(wrong_features)
    var backing = NDArray[DType.float32](Shape(8, 1))
    var strided = backing.view_with_layout(Shape(4, 1), NDArrayStrides(2, 1), 0)
    with assert_raises():
        _ = forest.predict(strided)
    with assert_raises():
        _ = forest.predict_proba(backing)
    for name in ["category", "vector", "multi_target"]:
        var model = load("build/balsa-reference/tests/fixtures/float32_" + name + ".tl")
        with assert_raises():
            _ = Forest(model^)
    var unsupported = make_stump()
    unsupported.set_postprocessor_name("unknown-transform")
    with assert_raises():
        _ = Forest(unsupported^)
    var malformed = make_stump()
    malformed.trees[0].cleft[0] = 999
    with assert_raises():
        _ = Forest(malformed^)


def check_owned_model[dtype: DType](var model: Model[dtype]) raises:
    var forest = Forest(model^)
    var x = NDArray[dtype](Shape(1, 1))
    x.itemset(0, 0)
    var result = forest.predict(x)
    assert_equal(result.item(0), Scalar[dtype](-1))


def test_balsa_automatic_precision_and_move() raises:
    for name in ["float32_op2_missing0", "float64_op2_missing0"]:
        var loaded = load_auto("build/balsa-reference/tests/fixtures/" + name + ".tl")
        if loaded.isa[Model[DType.float32]]():
            check_owned_model(loaded^.unwrap[Model[DType.float32]]())
        else:
            check_owned_model(loaded^.unwrap[Model[DType.float64]]())


def test_trained_unsupported_semantics() raises:
    for name in [
        "lightgbm_categorical",
        "sklearn_rf_multioutput_regression",
        "sklearn_rf_multiclass",
        "sklearn_extra_multioutput_classification",
    ]:
        var model = load[DType.float64](
            "build/balsa-reference/tests/fixtures/frameworks/" + name + ".tl"
        )
        with assert_raises():
            _ = Forest(model^)


def test_binary_threshold_tie_and_extremes() raises:
    var model = load("build/balsa-reference/tests/fixtures/float32_sigmoid.tl")
    model.base_scores[0] = 0
    for t in range(len(model.trees)):
        model.trees[t].leaf_value[1] = 0
        model.trees[t].leaf_value[2] = 0
    var forest = Forest(model.copy())
    var x = NDArray[DType.float32](Shape(1, 1))
    x.itemset(0, 0)
    var p = forest.predict_proba(x)
    var label = forest.predict_class(x)
    assert_equal(p.item(1), Float32(0.5))
    assert_equal(label.item(0), Int64(0))
    with assert_raises():
        _ = forest.predict_class(x, nan[DType.float32]())
    with assert_raises():
        _ = forest.predict(x)
    for extreme in [-1000.0, 1000.0]:
        model.base_scores[0] = extreme
        var f = Forest(model.copy())
        var probability = f.predict_proba(x)
        assert_equal(probability.item(1), Float32(0 if extreme < 0 else 1))


def test_accelerator_cpu_and_gpu_rejection() raises:
    check_device[Device.CPU]()
    check_gpu_unimplemented()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
