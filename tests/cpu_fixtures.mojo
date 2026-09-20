"""Deterministic Balsa fixtures shared by CPU correctness tests and benchmarks.

These are structural stress fixtures, not trained models or accuracy benchmarks.
No framework-specific metadata participates in their construction or routing.
"""
from std.math import isnan, inf, nan
from balsa import Model, ModelBuilder, TreeBuilder, NodeType, Operator, TaskType
from numojo import NDArray
from numojo.core.type_aliases import Shape


def synthetic_model[
    dtype: DType
](
    kind: String,
    trees: Int,
    depth: Int,
    features: Int,
    borders: Int = 8,
    classifier: Bool = False,
) raises -> Model[dtype]:
    if trees < 1 or depth < 0 or depth > 17 or borders < 1 or features < 0:
        raise Error("Invalid synthetic forest dimensions")
    if kind != "roots" and features == 0:
        raise Error("Splitting trees require features")
    if (
        kind != "roots"
        and kind != "stumps"
        and kind != "symmetric"
        and kind != "asymmetric"
        and kind != "chain"
        and kind != "mixed"
    ):
        raise Error("Unknown fixture family")
    var builder = ModelBuilder[dtype](
        num_feature=Int32(features),
        base_scores=[0.123456789123],
        average_tree_output=True,
        task_type=TaskType.BINARY_CLF if classifier else TaskType.REGRESSOR,
        postprocessor="sigmoid" if classifier else "identity",
        sigmoid_alpha=0.75,
    )
    for t in range(trees):
        var tree_depth = depth
        var chain = kind == "chain"
        if kind == "roots":
            tree_depth = 0
        elif kind == "stumps":
            tree_depth = 1
        elif kind == "mixed":
            if t % 4 == 0:
                tree_depth = 0
            elif t % 4 == 1:
                tree_depth = 1
            elif t % 4 == 3:
                chain = True
        var interior = tree_depth if chain else (1 << tree_depth) - 1
        var tree = TreeBuilder[dtype](num_nodes=2 * interior + 1)
        for node in range(interior):
            var level = node
            if not chain:
                level = 0
                while (1 << (level + 1)) - 1 <= node:
                    level += 1
            var feature = (level * 7 + t * 3) % features
            if kind == "asymmetric":
                feature = (feature + node) % features
            var threshold = (
                Scalar[dtype]((t // features + level) % borders - borders // 2)
                / 8
            )
            var left = 2 * node + 1
            var right = 2 * node + 2
            if chain:
                left = interior + node
                right = node + 1 if node + 1 < interior else 2 * interior
            tree.numerical_split(
                node,
                feature=Int32(feature),
                threshold=threshold,
                left=Int32(left),
                right=Int32(right),
                op=Operator.LE,
                default_left=(level + t) % 2 == 0,
            )
        for node in range(interior, 2 * interior + 1):
            tree.leaf(
                node,
                Scalar[dtype](((node - interior) * 17 + t * 11) % 97 - 48) / 13,
            )
        builder.add_tree(tree^.build())
    return builder^.build()


def synthetic_input[
    dtype: DType
](rows: Int, features: Int, edges: Bool = True) raises -> NDArray[dtype]:
    comptime assert dtype.is_floating_point()
    var x = NDArray[dtype](Shape(rows, features))
    for row in range(rows):
        for f in range(features):
            # SplitMix64 gives reproducible, decorrelated feature values without
            # a Python or random-runtime dependency. Preparation is untimed.
            var bits = UInt64(row * features + f) + UInt64(0x9E3779B97F4A7C15)
            bits = (bits ^ (bits >> 30)) * UInt64(0xBF58476D1CE4E5B9)
            bits = (bits ^ (bits >> 27)) * UInt64(0x94D049BB133111EB)
            bits ^= bits >> 31
            var value = Scalar[dtype](Int(bits % 4097) - 2048) / 1024
            if edges:
                var code = Int((bits >> 32) % 97)
                if code == 0:
                    value = nan[dtype]()
                elif code == 1:
                    value = inf[dtype]()
                elif code == 2:
                    value = -inf[dtype]()
                elif code == 3:
                    value = -0.0
                elif code == 4:
                    value = 0.5
            x.itemset(row * features + f, value)
    return x^


def scalar_margin[
    dtype: DType
](model: Model[dtype], x: NDArray[dtype], row: Int) raises -> Scalar[dtype]:
    """Independent scalar oracle over original Balsa child links and operators.
    """
    var total = Scalar[dtype](0)
    for tree in model.trees:
        var node = 0
        while tree.node_type[node] != NodeType.LEAF:
            var value = x.item(
                row * Int(model.num_feature) + Int(tree.split_index[node])
            )
            var threshold = tree.threshold[node]
            var op = tree.cmp[node]
            var left = (
                (op == Operator.EQ and value == threshold)
                or (op == Operator.LT and value < threshold)
                or (op == Operator.LE and value <= threshold)
                or (op == Operator.GT and value > threshold)
                or (op == Operator.GE and value >= threshold)
            )
            if isnan(value):
                left = tree.default_left[node] != 0
            node = Int(tree.cleft[node] if left else tree.cright[node])
        total += tree.leaf_value[node]
    if model.average_tree_output != 0:
        total /= Scalar[dtype](len(model.trees))
    return Scalar[dtype](Float64(total) + model.base_scores[0])
