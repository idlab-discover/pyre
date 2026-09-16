"""Account for owned Balsa storage; no inference calls or model copies."""
from std.sys import argv, size_of
from balsa import Model, Tree, load_auto
from balsa.model import Extension
from pyre import Forest
from pyre._executor import _State


def field[T: Copyable](name: String, group: String, values: List[T]):
    print(
        name,
        group,
        len(values) * size_of[T](),
        values.capacity() * size_of[T](),
        sep="\t",
    )


def extensions(name: String, values: List[Extension]):
    field(name, "extensions", values)
    for item in values:
        field(name + ".name", "extensions", item.name)
        field(name + ".payload", "extensions", item.payload)


def audit[dtype: DType](var model: Model[dtype]) raises:
    print("dtype", "float32" if dtype == DType.float32 else "float64", sep="\t")
    print("trees", len(model.trees), sep="\t")
    var nodes = 0
    for tree in model.trees:
        nodes += Int(tree.num_nodes)
    print("nodes", nodes, sep="\t")
    # Inline model contains list headers; tree/extension headers live in their
    # parent list allocations. Count each allocation exactly once.
    print(
        "model.inline",
        "containers",
        size_of[Model[dtype]](),
        size_of[Model[dtype]](),
        sep="\t",
    )
    field("model.trees", "containers", model.trees)
    field("model.num_class", "metadata", model.num_class)
    field("model.leaf_vector_shape", "metadata", model.leaf_vector_shape)
    field("model.target_id", "metadata", model.target_id)
    field("model.class_id", "metadata", model.class_id)
    field("model.postprocessor", "metadata", model.postprocessor)
    field("model.base_scores", "inference", model.base_scores)
    field("model.attributes", "metadata", model.attributes)
    extensions("model.extensions", model.extensions)
    for tree in model.trees:
        field("tree.node_type", "inference", tree.node_type)
        field("tree.cleft", "inference", tree.cleft)
        field("tree.cright", "inference", tree.cright)
        field("tree.split_index", "inference", tree.split_index)
        field("tree.default_left", "inference", tree.default_left)
        field("tree.leaf_value", "inference", tree.leaf_value)
        field("tree.threshold", "inference", tree.threshold)
        field("tree.cmp", "inference", tree.cmp)
        field(
            "tree.category_list_right_child",
            "unsupported_semantics",
            tree.category_list_right_child,
        )
        field("tree.leaf_vector", "unsupported_semantics", tree.leaf_vector)
        field(
            "tree.leaf_vector_begin",
            "unsupported_semantics",
            tree.leaf_vector_begin,
        )
        field(
            "tree.leaf_vector_end",
            "unsupported_semantics",
            tree.leaf_vector_end,
        )
        field("tree.category_list", "unsupported_semantics", tree.category_list)
        field(
            "tree.category_list_begin",
            "unsupported_semantics",
            tree.category_list_begin,
        )
        field(
            "tree.category_list_end",
            "unsupported_semantics",
            tree.category_list_end,
        )
        field("tree.data_count", "statistics", tree.data_count)
        field("tree.data_count_present", "statistics", tree.data_count_present)
        field("tree.sum_hess", "statistics", tree.sum_hess)
        field("tree.sum_hess_present", "statistics", tree.sum_hess_present)
        field("tree.gain", "statistics", tree.gain)
        field("tree.gain_present", "statistics", tree.gain_present)
        extensions("tree.tree_extensions", tree.tree_extensions)
        extensions("tree.node_extensions", tree.node_extensions)
    # This enforces Pyre's supported subset and transfers the same allocations.
    # No prediction/executor context or input/output buffers are created.
    var forest = Forest(model^)
    print(
        "forest.inline",
        "wrapper",
        size_of[Forest[dtype]]() - size_of[Model[dtype]](),
        size_of[Forest[dtype]]() - size_of[Model[dtype]](),
        sep="\t",
    )
    print(
        "executor.state",
        "wrapper",
        size_of[_State](),
        size_of[_State](),
        sep="\t",
    )
    print("validated_trees", forest.num_trees(), sep="\t")


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("Usage: model-memory-worker MODEL.tl")
    var model = load_auto(String(args[1]))
    if model.isa[Model[DType.float32]]():
        audit(model^.unwrap[Model[DType.float32]]())
    else:
        audit(model^.unwrap[Model[DType.float64]]())
