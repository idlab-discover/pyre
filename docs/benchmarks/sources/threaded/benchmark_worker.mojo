"""Native warm inference timer; model/input setup and process startup excluded."""
from std.sys import argv
from std.time import perf_counter_ns
from std.benchmark import keep
from std.math import nan
from balsa import ModelBuilder, TreeBuilder, Operator, TaskType, load
from pyre import Forest
from numojo import NDArray
from numojo.core.type_aliases import Shape


def execute[dtype: DType, width: Int](rows: Int, trees: Int, depth: Int, operation: String, loops: Int, threads: Int) raises:
    var classifier = operation == "proba"
    var builder = ModelBuilder[dtype](
        num_feature=8,
        task_type=TaskType.BINARY_CLF if classifier else TaskType.REGRESSOR,
        postprocessor="sigmoid" if classifier else "identity",
        base_scores=[0.125], average_tree_output=True,
    )
    var nodes = (1 << (depth + 1)) - 1
    var interior = (1 << depth) - 1
    for tree_id in range(trees):
        var tree = TreeBuilder[dtype](num_nodes=nodes)
        for node in range(nodes):
            if node < interior:
                tree.numerical_split(node, feature=Int32((node + tree_id) % 8),
                    threshold=Scalar[dtype]((node * 7 + tree_id * 3) % 19 - 9) / 8,
                    op=Operator.LE if tree_id % 2 else Operator.LT,
                    left=Int32(2 * node + 1), right=Int32(2 * node + 2),
                    default_left=tree_id % 2 == 0)
            else:
                tree.leaf(node, Scalar[dtype]((node * 13 + tree_id * 7) % 31 - 15) / 16)
        builder.add_tree(tree^.build())
    var model = builder^.build()
    if depth == 0:
        model = load[dtype]("build/balsa-reference/tests/fixtures/frameworks/xgboost_regression.tl")
    var forest = Forest(model^)
    var x = NDArray[dtype](Shape(rows, forest.num_features()))
    for i in range(rows * forest.num_features()):
        x.itemset(i, Scalar[dtype]((i * 17 + i // 8 * 11) % 101 - 50) / 32)
        if i % 97 == 0:
            x.itemset(i, nan[dtype]())
    # Warm code/data/allocator and (later) the shared thread pool before timing.
    for _ in range(3):
        if classifier:
            var result = forest.predict_proba(x, num_threads=threads)
            keep(result)
        else:
            var result = forest.predict_margin[width](x, num_threads=threads)
            keep(result)
    var start = perf_counter_ns()
    for _ in range(loops):
        if classifier:
            var result = forest.predict_proba(x, num_threads=threads)
            keep(result)
        else:
            var result = forest.predict_margin[width](x, num_threads=threads)
            keep(result)
    var elapsed = perf_counter_ns() - start
    print(Float64(elapsed) / 1e9)


def dispatch[dtype: DType](rows: Int, trees: Int, depth: Int, operation: String, loops: Int, width: Int, threads: Int) raises:
    if width == 1:
        execute[dtype, 1](rows, trees, depth, operation, loops, threads)
    elif width == 4:
        execute[dtype, 4](rows, trees, depth, operation, loops, threads)
    elif width == 8:
        execute[dtype, 8](rows, trees, depth, operation, loops, threads)
    elif width == 16:
        execute[dtype, 16](rows, trees, depth, operation, loops, threads)
    else:
        raise Error("Unsupported width")


def main() raises:
    var args = argv()
    if len(args) != 9:
        raise Error("Usage: worker ROWS TREES DEPTH DTYPE OP WIDTH THREADS LOOPS")
    if args[4] == "float32":
        dispatch[DType.float32](Int(args[1]), Int(args[2]), Int(args[3]), String(args[5]), Int(args[8]), Int(args[6]), Int(args[7]))
    else:
        dispatch[DType.float64](Int(args[1]), Int(args[2]), Int(args[3]), String(args[5]), Int(args[8]), Int(args[6]), Int(args[7]))
