"""Native timer and independent correctness gate for synthetic CPU shapes."""
from std.sys import argv
from std.time import perf_counter_ns
from std.benchmark import keep
from std.testing import assert_equal
from pyre import Forest
from pyre._cpu import _CPUPlan, _GENERAL, _DIRECT
from cpu_fixtures import synthetic_model, synthetic_input, scalar_margin
from numojo import NDArray


def measure[
    dtype: DType
](forest: Forest[dtype], x: NDArray[dtype], loops: Int) raises:
    for _ in range(3):
        var y = forest.predict_margin(x, num_threads=1)
        keep(y)
    var start = perf_counter_ns()
    for _ in range(loops):
        var y = forest.predict_margin(x, num_threads=1)
        keep(y)
    print(Float64(perf_counter_ns() - start) / 1e9)


def run[dtype: DType](args: List[String]) raises:
    var kind = args[1]
    var depth = Int(args[3])
    var trees = Int(args[4])
    var features = Int(args[5])
    var rows = Int(args[6])
    var borders = Int(args[7])
    var loops = Int(args[9])
    var model = synthetic_model[dtype](kind, trees, depth, features, borders)
    var x = synthetic_input[dtype](rows, features)
    if args[8] == "gate":
        var owned_model = model.copy()
        var start = perf_counter_ns()
        var forest = Forest(owned_model^)
        var preparation = Float64(perf_counter_ns() - start) / 1e9
        var y = forest.predict_margin(x, num_threads=1)
        for row in range(rows):
            assert_equal(y.item(row), scalar_margin(model, x, row))
        var generic = _CPUPlan(model.copy(), strategy=_GENERAL)
        var gy = generic.predict_margin(x, num_threads=1)
        for row in range(rows):
            assert_equal(gy.item(row), y.item(row))
        if forest._cpu._symmetric.enabled:
            var direct = _CPUPlan(model.copy(), strategy=_DIRECT)
            var dy = direct.predict_margin(x, num_threads=1)
            for row in range(rows):
                assert_equal(dy.item(row), y.item(row))
        var nodes = 0
        for tree in model.trees:
            nodes += Int(tree.num_nodes)
        print(forest._cpu.kernel(rows), nodes, preparation, sep=" ")
    else:
        var forest = Forest(model^)
        if args[8] == "generic":
            forest._cpu._use_strategy(_GENERAL)
        elif args[8] == "direct":
            forest._cpu._use_strategy(_DIRECT)
        elif args[8] != "auto":
            raise Error("Unknown benchmark kernel")
        measure(forest, x, loops)


def main() raises:
    var args = List[String]()
    for arg in argv():
        args.append(String(arg))
    if len(args) != 10:
        raise Error(
            "Usage: worker FAMILY DTYPE DEPTH TREES FEATURES ROWS BORDERS"
            " KERNEL LOOPS"
        )
    if args[2] == "float32":
        run[DType.float32](args)
    elif args[2] == "float64":
        run[DType.float64](args)
    else:
        raise Error("Expected float32 or float64")
