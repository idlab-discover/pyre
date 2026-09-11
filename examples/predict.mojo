"""Balsa discovers precision; Pyre consumes the model and evaluates arrays."""
from balsa import Model, load_auto
from pyre import Forest
from numojo import NDArray
from numojo.core.type_aliases import Shape


def predict[dtype: DType](var model: Model[dtype]) raises:
    var forest = Forest(model^)
    var x = NDArray[dtype](Shape(3, 1))
    x.itemset(0, 0.0)
    x.itemset(1, 0.5)
    x.itemset(2, 1.0)
    print(forest.predict(x))


def main() raises:
    var model = load_auto("build/balsa-reference/tests/fixtures/float64_op2_missing0.tl")
    if model.isa[Model[DType.float32]]():
        predict(model^.unwrap[Model[DType.float32]]())
    else:
        predict(model^.unwrap[Model[DType.float64]]())
