"""Public width selection and model-based LT dispatch against fixed trees."""

from std.math import exp, inf, nan
from std.testing import assert_equal, assert_almost_equal, TestSuite
from balsa import Model, ModelBuilder, TreeBuilder, Operator, TaskType
from numojo import NDArray
from numojo.core.type_aliases import Shape
from pyre import Forest


def _model[dtype: DType](mixed: Bool, classifier: Bool) raises -> Model[dtype]:
    var builder = ModelBuilder[dtype](
        num_feature=3, base_scores=[0.25], average_tree_output=True
    )
    # Uneven depths, three features, and a non-LT operator late in the tree.
    var tree = TreeBuilder[dtype](num_nodes=7)
    tree.numerical_split(
        0,
        feature=0,
        threshold=0.5,
        left=1,
        right=2,
        op=Operator.LT,
        default_left=True,
    )
    tree.leaf(1, -1.5)
    tree.numerical_split(
        2,
        feature=1,
        threshold=0,
        left=3,
        right=4,
        op=Operator.LT,
        default_left=False,
    )
    tree.leaf(3, -0.5)
    tree.numerical_split(
        4,
        feature=2,
        threshold=1,
        left=5,
        right=6,
        op=Operator.GE if mixed else Operator.LT,
        default_left=True,
    )
    tree.leaf(5, 0.5)
    tree.leaf(6, 2.5)
    builder.add_tree(tree^.build())
    # A later tree can also invalidate whole-forest LT specialization.
    var second = TreeBuilder[dtype](num_nodes=3)
    second.numerical_split(
        0,
        feature=1,
        threshold=0.5,
        left=1,
        right=2,
        op=Operator.LE if mixed else Operator.LT,
        default_left=False,
    )
    second.leaf(1, 2)
    second.leaf(2, 4)
    builder.add_tree(second^.build())
    var model = builder^.build()
    if classifier:
        model.task_type = TaskType.BINARY_CLF
        model.set_postprocessor_name("sigmoid")
        model.sigmoid_alpha = 0.75
    return model^


def _expected[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype], c: Scalar[dtype], mixed: Bool) -> Scalar[
    dtype
]:
    var first: Scalar[dtype]
    if a != a or a < 0.5:
        first = -1.5
    elif b < 0:
        first = -0.5
    elif c != c or (c >= 1 if mixed else c < 1):
        first = 0.5
    else:
        first = 2.5
    var second = Scalar[dtype](2 if (b <= 0.5 if mixed else b < 0.5) else 4)
    return (first + second) / 2 + 0.25


def _check[dtype: DType, width: Int]() raises:
    comptime assert dtype.is_floating_point()
    var values = [
        Scalar[dtype](-1),
        Scalar[dtype](-0.0),
        Scalar[dtype](0),
        Scalar[dtype](0.5),
        Scalar[dtype](1),
        -inf[dtype](),
        inf[dtype](),
        nan[dtype](),
    ]
    for mixed in [False, True]:
        var regression = Forest(_model[dtype](mixed, False))
        var classifier = Forest(_model[dtype](mixed, True))
        for rows in [0, 1, 2, 7, 15, 16, 17, 31, 32, 33, 65]:
            var x = NDArray[dtype](Shape(rows, 3))
            for row in range(rows):
                for feature in range(3):
                    x.itemset(
                        3 * row + feature, values[(row + 3 * feature) % 8]
                    )
            for threads in [1, 4]:
                var y = regression.predict[width](x, num_threads=threads)
                var margin = classifier.predict_margin[width](
                    x, num_threads=threads
                )
                var probability = classifier.predict_proba[width](
                    x, num_threads=threads
                )
                var label = classifier.predict_class[width](
                    x, threshold=0.5, num_threads=threads
                )
                assert_equal(y.size, rows)
                assert_equal(margin.size, rows)
                assert_equal(probability.shape[0], rows)
                assert_equal(probability.shape[1], 2)
                assert_equal(label.size, rows)
                for row in range(rows):
                    var expected = _expected(
                        x.item(3 * row),
                        x.item(3 * row + 1),
                        x.item(3 * row + 2),
                        mixed,
                    )
                    assert_equal(y.item(row), expected)
                    assert_equal(margin.item(row), expected)
                    var p = 1 / (1 + exp(-Scalar[dtype](0.75) * expected))
                    assert_almost_equal(
                        probability.item(2 * row + 1),
                        p,
                        atol=1e-6 if dtype == DType.float32 else 1e-12,
                        rtol=1e-6 if dtype == DType.float32 else 1e-12,
                    )
                    assert_almost_equal(
                        probability.item(2 * row),
                        1 - p,
                        atol=1e-6 if dtype == DType.float32 else 1e-12,
                        rtol=1e-6 if dtype == DType.float32 else 1e-12,
                    )
                    assert_equal(label.item(row), Int64(expected > 0))


def test_public_widths_float32() raises:
    _check[DType.float32, 1]()
    _check[DType.float32, 4]()
    _check[DType.float32, 16]()


def test_public_widths_float64() raises:
    _check[DType.float64, 1]()
    _check[DType.float64, 4]()
    _check[DType.float64, 16]()


def test_operator_scan_includes_later_trees() raises:
    # The first tree is LT, but the second tree's LE tie must go left.
    var model = _model[DType.float32](False, False)
    model.trees[1].cmp[0] = Operator.LE
    var forest = Forest(model^)
    var x = NDArray[DType.float32](Shape(17, 3))
    for i in range(x.size):
        x.itemset(i, 0.5)
    var y = forest.predict[16](x, num_threads=4)
    for row in range(17):
        assert_equal(y.item(row), Float32(1.5))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
