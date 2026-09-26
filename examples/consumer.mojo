"""A standalone consumer: no fixture files or source checkout required."""
from std.testing import assert_equal
from balsa import ModelBuilder, TreeBuilder
from pyre import Forest
from numojo import NDArray
from numojo.core.type_aliases import Shape


def check[dtype: DType]() raises:
    var builder = ModelBuilder[dtype](1)
    var tree = TreeBuilder[dtype](1)
    tree.leaf(0, 2.5)
    builder.add_tree(tree^.build())
    var forest = Forest(builder^.build())
    var x = NDArray[dtype](Shape(3, 1))
    for row in range(3):
        x.itemset(row, Scalar[dtype](row))
    var prediction = forest.predict(x)
    for row in range(3):
        assert_equal(prediction.item(row), Scalar[dtype](2.5))


def main() raises:
    check[DType.float32]()
    check[DType.float64]()
    print("Pyre consumer passed (float32 and float64)")
