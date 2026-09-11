"""Deterministic row-parallel inference, public controls, and numeric edges."""

from std.math import nan, inf
from std.testing import assert_equal, assert_raises, TestSuite
from max.algorithm import parallelize
from balsa import load, Model, ModelBuilder, TreeBuilder, Operator, TaskType
from numojo import NDArray
from numojo.core.type_aliases import Shape
from numojo.core.layout.ndstrides import NDArrayStrides
from numojo.core.accelerator.device import Device
from pyre import Forest
from pyre._executor import _Executor
from pyre.accelerator import (
    predict_cpu,
    predict_margin_cpu,
    predict_proba_cpu,
    predict_class_cpu,
    to_host,
    to_accelerator,
)


def _stump[dtype: DType](op: Int8, default_left: Bool) raises -> Model[dtype]:
    var model = ModelBuilder[dtype](num_feature=1, base_scores=[0.25])
    var tree = TreeBuilder[dtype](num_nodes=3)
    tree.numerical_split(
        0,
        feature=0,
        threshold=0.5,
        left=1,
        right=2,
        op=op,
        default_left=default_left,
    )
    tree.leaf(1, -1.25)
    tree.leaf(2, 2.75)
    model.add_tree(tree^.build())
    return model^.build()


def _check_operators[dtype: DType]() raises:
    for op in [Operator.EQ, Operator.LT, Operator.LE, Operator.GT, Operator.GE]:
        for default_left in [False, True]:
            var forest = Forest(_stump[dtype](op, default_left))
            # Exercise empty input, incomplete SIMD blocks, and worker tails.
            for rows in [0, 1, 3, 7, 17, 65, 1031]:
                var x = NDArray[dtype](Shape(rows, 1))
                for i in range(rows):
                    x.itemset(i, Scalar[dtype](i % 4) * 0.5)
                    if i % 4 == 3:
                        x.itemset(i, nan[dtype]())
                var serial = forest.predict_margin[1](x, num_threads=1)
                for threads in [0, 1, 2, 4, 8]:
                    var scalar = forest.predict_margin[1](
                        x, num_threads=threads
                    )
                    var vector = forest.predict_margin[4](
                        x, num_threads=threads
                    )
                    var prediction = forest.predict(x, num_threads=threads)
                    var wide = forest.predict_margin[16](x, num_threads=threads)
                    assert_equal(prediction.size, rows)
                    for i in range(rows):
                        var value = x.item(i)
                        var left: Bool
                        if value != value:
                            left = default_left
                        elif op == Operator.EQ:
                            left = value == 0.5
                        elif op == Operator.LT:
                            left = value < 0.5
                        elif op == Operator.LE:
                            left = value <= 0.5
                        elif op == Operator.GT:
                            left = value > 0.5
                        else:
                            left = value >= 0.5
                        var expected = Scalar[dtype](-1 if left else 3)
                        assert_equal(serial.item(i), expected)
                        assert_equal(scalar.item(i), expected)
                        assert_equal(vector.item(i), expected)
                        assert_equal(wide.item(i), expected)
                        assert_equal(prediction.item(i), expected)


def test_parallel_operators_float32() raises:
    _check_operators[DType.float32]()


def test_parallel_operators_float64() raises:
    _check_operators[DType.float64]()


def test_parallel_offset_view() raises:
    var forest = Forest(_stump[DType.float32](Operator.LT, False))
    var backing = NDArray[DType.float32](Shape(1060, 1))
    for i in range(backing.size):
        backing.itemset(i, Float32(i % 4) * 0.25)
    var view = backing.view_with_layout(Shape(1031, 1), NDArrayStrides(1, 1), 7)
    for threads in [0, 1, 2, 4, 8]:
        var result = forest.predict(view, num_threads=threads)
        for i in range(result.size):
            assert_equal(result.item(i), Float32(-1 if (i + 7) % 4 < 2 else 3))


def _check_trained[dtype: DType](name: String, binary: Bool) raises:
    var forest = Forest(
        load[dtype](
            "build/balsa-reference/tests/fixtures/frameworks/" + name + ".tl"
        )
    )
    var features = forest.num_features()
    # Repeated rows give a small independent serial reference for a large batch.
    var reference_input = NDArray[dtype](Shape(33, features))
    var x = NDArray[dtype](Shape(2053, features))
    for row in range(33):
        for feature in range(features):
            var value = (
                Scalar[dtype]((row * features + feature) % 13 - 6) * 0.25
            )
            if row % 5 == 0 and feature == 1:
                value = nan[dtype]()
            reference_input.itemset(row * features + feature, value)
    for row in range(x.shape[0]):
        for feature in range(features):
            x.itemset(
                row * features + feature,
                reference_input.item((row % 33) * features + feature),
            )
    var reference = forest.predict_margin[1](reference_input, num_threads=1)
    for threads in [0, 1, 2, 4, 8]:
        var margins = forest.predict_margin[4](x, num_threads=threads)
        for row in range(x.shape[0]):
            assert_equal(margins.item(row), reference.item(row % 33))
        if binary:
            var expected = forest.predict_proba(reference_input, num_threads=1)
            var probabilities = forest.predict_proba(x, num_threads=threads)
            var classes = forest.predict_class(
                x, threshold=0.37, num_threads=threads
            )
            for row in range(x.shape[0]):
                for cls in range(2):
                    assert_equal(
                        probabilities.item(2 * row + cls),
                        expected.item(2 * (row % 33) + cls),
                    )
                assert_equal(
                    classes.item(row),
                    Int64(
                        expected.item(2 * (row % 33) + 1) > Scalar[dtype](0.37)
                    ),
                )
        else:
            var prediction = forest.predict(x, num_threads=threads)
            for row in range(x.shape[0]):
                assert_equal(prediction.item(row), reference.item(row % 33))


def test_parallel_trained_models() raises:
    _check_trained[DType.float32]("xgboost_regression", False)
    _check_trained[DType.float64]("catboost_numeric_regression", False)
    _check_trained[DType.float64]("sklearn_gradient_binary", True)
    _check_trained[DType.float64]("sklearn_hist_missing_regression", False)


def test_parallel_classifier_empty_and_thresholds() raises:
    var model = load("build/balsa-reference/tests/fixtures/float32_sigmoid.tl")
    model.base_scores[0] = 0
    for tree in range(len(model.trees)):
        model.trees[tree].leaf_value[1] = 0
        model.trees[tree].leaf_value[2] = 0
    var forest = Forest(model^)
    for rows in [0, 1, 7, 1031]:
        var x = NDArray[DType.float32](Shape(rows, 1))
        for row in range(rows):
            x.itemset(row, 0)
        for threads in [0, 1, 2, 4, 8]:
            var probabilities = forest.predict_proba(x, num_threads=threads)
            assert_equal(probabilities.shape[0], rows)
            assert_equal(probabilities.shape[1], 2)
            for threshold in [Float32(0), Float32(0.5), Float32(1)]:
                var labels = forest.predict_class(
                    x, threshold, num_threads=threads
                )
                assert_equal(labels.size, rows)
                for row in range(rows):
                    assert_equal(probabilities.item(2 * row), Float32(0.5))
                    assert_equal(probabilities.item(2 * row + 1), Float32(0.5))
                    assert_equal(labels.item(row), Int64(threshold < 0.5))


def test_negative_thread_counts() raises:
    var regression = Forest(_stump[DType.float32](Operator.LT, False))
    var classifier = Forest(
        load("build/balsa-reference/tests/fixtures/float32_sigmoid.tl")
    )
    for rows in [0, 3]:
        var x = NDArray[DType.float32](Shape(rows, 1))
        for row in range(rows):
            x.itemset(row, 0)
        with assert_raises():
            _ = regression.predict_margin(x, num_threads=-1)
        with assert_raises():
            _ = regression.predict(x, num_threads=-1)
        with assert_raises():
            _ = classifier.predict_proba(x, num_threads=-1)
        with assert_raises():
            _ = classifier.predict_class(x, num_threads=-1)


def test_float32_base_score_double_precision_addition() raises:
    # Verified with Treelite 4.6.1 GTIL: float32(-1 + double(1 + 2**-24)).
    # Narrowing the base before adding would incorrectly return zero.
    var model = ModelBuilder(
        num_feature=1, base_scores=[1.000000059604644775390625]
    )
    var tree = TreeBuilder(num_nodes=1)
    tree.leaf(0, -1)
    model.add_tree(tree^.build())
    var forest = Forest(model^.build())
    var x = NDArray[DType.float32](Shape(1031, 1))
    for row in range(x.shape[0]):
        x.itemset(row, 0)
    for threads in [0, 1, 2, 4, 8]:
        var scalar = forest.predict_margin[1](x, num_threads=threads)
        var vector = forest.predict_margin[4](x, num_threads=threads)
        for row in range(x.shape[0]):
            assert_equal(scalar.item(row), Float32(0.000000059604644775390625))
            assert_equal(vector.item(row), scalar.item(row))


def _concurrent_predictions(
    forest: Forest[DType.float32],
    x: NDArray[DType.float32],
    errors: Pointer[Int, MutUntrackedOrigin],
):
    # MAX workers reenter prediction on the SAME immutable model and input.
    # Each inner call owns its output; the outer worker owns one error slot.
    def work(caller: Int) {forest, x, errors}:
        try:
            for _ in range(20):
                var prediction = forest.predict(x, num_threads=2)
                for row in range(x.shape[0]):
                    if prediction.item(row) != Float32(
                        -1 if row % 4 < 2 else 3
                    ):
                        errors[unsafe_offset=caller] = 1
        except:
            errors[unsafe_offset=caller] = 1

    parallelize(work, 2, 2)


def test_same_forest_concurrent_callers() raises:
    var forest = Forest(_stump[DType.float32](Operator.LT, False))
    var x = NDArray[DType.float32](Shape(1031, 1))
    for row in range(x.shape[0]):
        x.itemset(row, Float32(row % 4) * 0.25)
    var errors: List[Int] = [0, 0]
    # Synchronous parallelize finishes before this owned storage goes away.
    _concurrent_predictions(
        forest, x, errors.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )
    assert_equal(errors[0], 0)
    assert_equal(errors[1], 0)


def test_nested_executor_reuse() raises:
    # Reenter while the cached stream is executing the outer callback. Waiting
    # for that same stream from its callback would deadlock.
    var executor = _Executor()
    var values = List[Int](length=8, fill=0)
    var output = values.unsafe_ptr()

    def outer(i: Int) {executor, output}:
        def inner(j: Int) {i, output}:
            output[unsafe_offset=2 * i + j] += 1

        try:
            executor.run(inner, 2)
        except:
            output[unsafe_offset=2 * i] = -1000

    for _ in range(20):
        executor.run(outer, 4)
    for value in values:
        assert_equal(value, 20)


def test_move_forest_after_context_warmup() raises:
    var original = Forest(_stump[DType.float32](Operator.LT, False))
    var x = NDArray[DType.float32](Shape(1031, 1))
    for row in range(x.shape[0]):
        x.itemset(row, Float32(row % 4) * 0.25)
    var expected = original.predict(x, num_threads=4)
    var moved = original^
    for threads in [1, 8, 2, 4]:
        var actual = moved.predict(x, num_threads=threads)
        for row in range(x.shape[0]):
            assert_equal(actual.item(row), expected.item(row))


def test_parallel_cpu_accelerator_adapters() raises:
    var forest = Forest(_stump[DType.float32](Operator.LT, False))
    var classifier = Forest(
        load("build/balsa-reference/tests/fixtures/float32_sigmoid.tl")
    )
    for rows in [0, 1, 1031]:
        var host = NDArray[DType.float32](Shape(rows, 1))
        for row in range(rows):
            host.itemset(row, Float32(row % 4) * 0.25)
        var x = to_accelerator[DType.float32, Device.CPU](host)
        var expected_margin = classifier.predict_margin(host, num_threads=1)
        var expected_proba = classifier.predict_proba(host, num_threads=1)
        var expected_class = classifier.predict_class(host, 0.37, num_threads=1)
        for threads in [0, 1, 2, 4, 8]:
            var y = predict_cpu(forest, x, num_threads=threads)
            var margin = predict_margin_cpu(classifier, x, num_threads=threads)
            var proba = predict_proba_cpu(classifier, x, num_threads=threads)
            var labels = predict_class_cpu(
                classifier, x, 0.37, num_threads=threads
            )
            var output = to_host(y)
            var margin_host = to_host(margin)
            var proba_host = to_host(proba)
            var labels_host = to_host(labels)
            assert_equal(output.size, rows)
            assert_equal(margin_host.size, rows)
            assert_equal(proba_host.size, 2 * rows)
            assert_equal(labels_host.size, rows)
            for row in range(rows):
                assert_equal(
                    output.item(row), Float32(-1 if row % 4 < 2 else 3)
                )
                assert_equal(margin_host.item(row), expected_margin.item(row))
                assert_equal(
                    proba_host.item(2 * row), expected_proba.item(2 * row)
                )
                assert_equal(
                    proba_host.item(2 * row + 1),
                    expected_proba.item(2 * row + 1),
                )
                assert_equal(labels_host.item(row), expected_class.item(row))
        with assert_raises():
            _ = predict_cpu(forest, x, num_threads=-1)
        with assert_raises():
            _ = predict_margin_cpu(classifier, x, num_threads=-1)
        with assert_raises():
            _ = predict_proba_cpu(classifier, x, num_threads=-1)
        with assert_raises():
            _ = predict_class_cpu(classifier, x, num_threads=-1)


def test_root_only_zero_features() raises:
    var model = ModelBuilder(num_feature=0, base_scores=[0.25])
    var tree = TreeBuilder(num_nodes=1)
    tree.leaf(0, 1.75)
    model.add_tree(tree^.build())
    var forest = Forest(model^.build())
    for rows in [0, 1, 15, 17, 1031]:
        var x = NDArray[DType.float32](Shape(rows, 0))
        for threads in [1, 4]:
            var output = forest.predict_margin[16](x, num_threads=threads)
            assert_equal(output.size, rows)
            for row in range(rows):
                assert_equal(output.item(row), Float32(2))


def test_divergent_leaf_depths_and_wide_tails() raises:
    # Left exits immediately at every level; right reaches a 31-deep leaf.
    # Finished lanes must never advance through their -1 child indices.
    var model = ModelBuilder(num_feature=1)
    var tree = TreeBuilder(num_nodes=63)
    for depth in range(31):
        tree.numerical_split(
            depth,
            feature=0,
            threshold=Float32(depth),
            left=Int32(32 + depth),
            right=Int32(depth + 1),
            default_left=True,
        )
        tree.leaf(32 + depth, Float32(depth))
    tree.leaf(31, 31)
    model.add_tree(tree^.build())
    var forest = Forest(model^.build())
    for rows in [2, 15, 16, 17, 1031]:
        var x = NDArray[DType.float32](Shape(rows, 1))
        for row in range(rows):
            x.itemset(row, Float32(row % 32) - 0.5)
            if row % 33 == 0:
                x.itemset(row, nan[DType.float32]())
        for threads in [1, 4, 8]:
            var scalar = forest.predict_margin[1](x, num_threads=threads)
            var vector = forest.predict_margin[16](x, num_threads=threads)
            for row in range(rows):
                var expected = Float32(0 if row % 33 == 0 else row % 32)
                assert_equal(scalar.item(row), expected)
                assert_equal(vector.item(row), expected)


def _check_sigmoid_extremes[dtype: DType]() raises:
    for magnitude in [Scalar[dtype](1000), inf[dtype]()]:
        var model = _stump[dtype](Operator.LT, False)
        model.task_type = TaskType.BINARY_CLF
        model.set_postprocessor_name("sigmoid")
        model.base_scores[0] = 0
        model.trees[0].leaf_value[1] = -magnitude
        model.trees[0].leaf_value[2] = magnitude
        var forest = Forest(model^)
        for rows in [1, 7, 17, 1031]:
            var x = NDArray[dtype](Shape(rows, 1))
            for row in range(rows):
                x.itemset(row, Scalar[dtype](row % 2))
            for threads in [1, 4]:
                var p = forest.predict_proba(x, num_threads=threads)
                for threshold in [
                    Scalar[dtype](0),
                    Scalar[dtype](0.5),
                    Scalar[dtype](1),
                ]:
                    var labels = forest.predict_class(
                        x, threshold, num_threads=threads
                    )
                    for row in range(rows):
                        var positive = Scalar[dtype](row % 2)
                        assert_equal(p.item(2 * row), 1 - positive)
                        assert_equal(p.item(2 * row + 1), positive)
                        assert_equal(
                            labels.item(row), Int64(positive > threshold)
                        )


def test_vector_sigmoid_extremes() raises:
    _check_sigmoid_extremes[DType.float32]()
    _check_sigmoid_extremes[DType.float64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
