"""Recovered topology and byte bins versus the general traversal in both dtypes."""
from std.math import inf, nan
from std.testing import (
    assert_equal,
    assert_true,
    assert_false,
    assert_raises,
    TestSuite,
)
from balsa import Model, ModelBuilder, TreeBuilder, Operator, TaskType
from numojo import NDArray
from numojo.core.type_aliases import Shape
from numojo.core.layout.ndstrides import NDArrayStrides
from pyre import Forest
from pyre._cpu import _CPUPlan, _GENERAL, _DIRECT


def model[dtype: DType](classifier: Bool = False) raises -> Model[dtype]:
    var builder = ModelBuilder[dtype](
        num_feature=3, base_scores=[0.123456789123], average_tree_output=True
    )
    # Different tree depths, repeated splits, and a constant tree. Nontrivial
    # leaves distinguish reversing split order from reversing serialized nodes.
    for t in range(19):
        var depth = t % 5
        var interior = (1 << depth) - 1
        var tree = TreeBuilder[dtype](num_nodes=2 * interior + 1)
        for node in range(2 * interior + 1):
            if node < interior:
                var level = 0
                while (1 << (level + 1)) - 1 <= node:
                    level += 1
                tree.numerical_split(
                    node,
                    feature=Int32(level % 3),
                    threshold=Scalar[dtype](level % 2) * 0.5,
                    left=Int32(node * 2 + 1),
                    right=Int32(node * 2 + 2),
                    op=Operator.LE,
                    default_left=level % 2 == 0,
                )
            else:
                tree.leaf(
                    node, Scalar[dtype]((node * 7 + t * 11) % 31 - 15) / 7
                )
        builder.add_tree(tree^.build())
    var result = builder^.build()
    if classifier:
        result.task_type = TaskType.BINARY_CLF
        result.set_postprocessor_name("sigmoid")
        result.sigmoid_alpha = 0.75
    return result^


def check[dtype: DType, width: Int]() raises:
    comptime assert dtype.is_floating_point()
    var values: List[Scalar[dtype]] = [
        -1,
        -0.0,
        0,
        0.5,
        1,
        -inf[dtype](),
        inf[dtype](),
        nan[dtype](),
        Scalar[dtype](
            0.4999999701976776 if dtype
            == DType.float32 else 0.49999999999999994
        ),
        Scalar[dtype](
            0.5000000596046448 if dtype == DType.float32 else 0.5000000000000001
        ),
    ]
    for classifier in [False, True]:
        var compact = Forest(model[dtype](classifier))
        var direct = _CPUPlan(model[dtype](classifier), strategy=_DIRECT)
        var generic = _CPUPlan(model[dtype](classifier), strategy=_GENERAL)
        assert_true(compact._cpu._symmetric.enabled)
        assert_true(compact._cpu._symmetric.binned)
        for rows in [0, 1, 2, 15, 16, 17, 31, 32, 33, 255, 256, 257, 511, 513]:
            var backing = NDArray[dtype](Shape(rows + 2, 3))
            for i in range(backing.size):
                backing.itemset(i, values[(i * 7 + i // 3) % len(values)])
            var x = backing.view_with_layout(
                Shape(rows, 3), NDArrayStrides(3, 1), 3
            )
            var expected = generic.predict_margin[width](x, num_threads=1)
            var got = compact.predict_margin[width](x, num_threads=8)
            var raw = direct.predict_margin[width](x, num_threads=1)
            for i in range(rows):
                assert_equal(got.item(i), expected.item(i))
                assert_equal(raw.item(i), expected.item(i))
            if classifier:
                var p = compact.predict_proba[width](x)
                var ep = generic.predict_proba[width](x, num_threads=1)
                for i in range(p.size):
                    assert_equal(p.item(i), ep.item(i))
                for threshold in [
                    Scalar[dtype](0),
                    Scalar[dtype](0.37),
                    Scalar[dtype](1),
                ]:
                    var c = compact.predict_class[width](x, threshold)
                    var ec = generic.predict_class[width](
                        x, threshold, num_threads=1
                    )
                    for i in range(c.size):
                        assert_equal(c.item(i), ec.item(i))


def test_float32() raises:
    check[DType.float32, 1]()
    check[DType.float32, 4]()
    check[DType.float32, 16]()
    check[DType.float32, 32]()


def test_float64() raises:
    check[DType.float64, 1]()
    check[DType.float64, 4]()
    check[DType.float64, 16]()


def test_fallbacks() raises:
    # Every split property participates in eligibility; differences below root
    # must not disappear during packing. A later LT tree also forces fallback.
    for change in range(5):
        var m = model[DType.float32]()
        if change == 0:
            m.trees[3].threshold[2] = 0.75
        elif change == 1:
            m.trees[3].default_left[2] = 1
        elif change == 2:
            m.trees[3].split_index[2] = 2
        elif change == 3:
            m.trees[18].cmp[0] = Operator.LT
        else:
            m.trees[3].threshold[0] = inf[DType.float32]()
        var forest = Forest(m^)
        assert_false(forest._cpu._symmetric.enabled)
    var uneven = TreeBuilder[DType.float32](num_nodes=5)
    uneven.numerical_split(
        0, feature=0, threshold=0, left=1, right=2, op=Operator.LE
    )
    uneven.leaf(1, 1)
    uneven.numerical_split(
        2, feature=0, threshold=0, left=3, right=4, op=Operator.LE
    )
    uneven.leaf(3, 2)
    uneven.leaf(4, 3)
    var builder = ModelBuilder[DType.float32](num_feature=1)
    builder.add_tree(uneven^.build())
    var forest = Forest(builder^.build())
    assert_false(forest._cpu._symmetric.enabled)


def test_border_capacity_and_precision() raises:
    for count in [254, 255]:
        var builder = ModelBuilder[DType.float64](num_feature=1)
        for t in range(2 * count):
            var tree = TreeBuilder[DType.float64](num_nodes=3)
            tree.numerical_split(
                0,
                feature=0,
                threshold=Float64(t % count) * 1e-12 + 0.5,
                left=1,
                right=2,
                op=Operator.LE,
                default_left=False,
            )
            tree.leaf(1, -1)
            tree.leaf(2, 1)
            builder.add_tree(tree^.build())
        var m = builder^.build()
        var forest = Forest(m.copy())
        var generic = _CPUPlan(m^, strategy=_GENERAL)
        assert_true(forest._cpu._symmetric.enabled)
        assert_equal(forest._cpu._symmetric.binned, count == 254)
        var x = NDArray[DType.float64](Shape(257, 1))
        for i in range(257):
            x.itemset(i, Float64(i) * 1e-12 + 0.5)
        x.itemset(256, nan[DType.float64]())
        var y = forest.predict(x, num_threads=1)
        var expected = generic.predict(x, num_threads=1)
        for i in range(257):
            assert_equal(y.item(i), expected.item(i))
        with assert_raises():
            _ = forest.predict(x, num_threads=-1)


def test_permuted_node_numbers() raises:
    # Root, left, right, LL, LR, RL, RR map to arbitrary serialized IDs.
    var ids = [0, 6, 5, 4, 3, 2, 1]
    var builder = ModelBuilder[DType.float32](num_feature=2)
    var tree = TreeBuilder[DType.float32](num_nodes=7)
    for node in range(3):
        tree.numerical_split(
            ids[node],
            feature=Int32(0 if node == 0 else 1),
            threshold=0,
            left=Int32(ids[2 * node + 1]),
            right=Int32(ids[2 * node + 2]),
            op=Operator.LE,
        )
    for node in range(3, 7):
        tree.leaf(ids[node], Float32(node - 2))
    builder.add_tree(tree^.build())
    var forest = Forest(builder^.build())
    assert_true(forest._cpu._symmetric.enabled)
    var x = NDArray[DType.float32](Shape(4, 2))
    for row in range(4):
        x.itemset(2 * row, Float32(row // 2))
        x.itemset(2 * row + 1, Float32(row % 2))
    var y = forest.predict(x)
    for row in range(4):
        assert_equal(y.item(row), Float32(row + 1))


def test_deeper_trees() raises:
    for depth in [9, 10, 16]:
        var interior = (1 << depth) - 1
        var tree = TreeBuilder[DType.float32](num_nodes=2 * interior + 1)
        for node in range(interior):
            var level = 0
            while (1 << (level + 1)) - 1 <= node:
                level += 1
            tree.numerical_split(
                node,
                feature=Int32(level),
                threshold=0,
                left=Int32(2 * node + 1),
                right=Int32(2 * node + 2),
                op=Operator.LE,
            )
        for node in range(interior, 2 * interior + 1):
            tree.leaf(node, Float32(node - interior))
        var builder = ModelBuilder[DType.float32](num_feature=Int32(depth))
        builder.add_tree(tree^.build())
        var m = builder^.build()
        var forest = Forest(m.copy())
        var generic = _CPUPlan(m^, strategy=_GENERAL)
        assert_true(forest._cpu._symmetric.enabled)
        var x = NDArray[DType.float32](Shape(33, depth))
        for i in range(x.size):
            x.itemset(i, Float32((i * 7 + i // depth) % 3 - 1))
        var y = forest.predict(x, num_threads=1)
        var expected = generic.predict(x, num_threads=1)
        for i in range(33):
            assert_equal(y.item(i), expected.item(i))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
