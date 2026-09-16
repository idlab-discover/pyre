from std.testing import assert_equal, assert_raises, TestSuite
from balsa import load, load_auto, Model
from pyre import Forest
from numojo import NDArray
from numojo.core.type_aliases import Shape
from numojo.core.layout.ndstrides import NDArrayStrides
from acceptance import stump, check_edges
from reference_data import check_references
from test_accelerator import check_gpu_unimplemented, check_device
from numojo.core.accelerator.device import Device


def test_treelite_references() raises:
    check_references()


def test_shared_edges() raises:
    check_edges[DType.float32]()
    check_edges[DType.float64]()


def test_owned_preparation_and_offset_view() raises:
    var model = stump()
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
    var forest = Forest(
        load("build/balsa-reference/tests/fixtures/float32_op2_missing0.tl")
    )
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
        var model = load(
            "build/balsa-reference/tests/fixtures/float32_" + name + ".tl"
        )
        with assert_raises():
            _ = Forest(model^)
    var unsupported = stump()
    unsupported.set_postprocessor_name("unknown-transform")
    with assert_raises():
        _ = Forest(unsupported^)
    var malformed = stump()
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
        var loaded = load_auto(
            "build/balsa-reference/tests/fixtures/" + name + ".tl"
        )
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


def test_accelerator_cpu_and_gpu_rejection() raises:
    check_device[Device.CPU]()
    check_gpu_unimplemented()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
