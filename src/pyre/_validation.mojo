"""Shared inference capability checks; Balsa owns structural validity."""

from balsa import Model, NodeType, Operator, TaskType, validate


def validate_inference[dtype: DType](model: Model[dtype]) raises -> Bool:
    """Validate the supported scalar task subset; return whether all splits are LT.
    """
    validate(model)
    if model.num_tree == 0:
        raise Error("Pyre requires at least one tree")
    if (
        model.num_target != 1
        or model.num_class[0] != 1
        or model.leaf_vector_shape[0] != 1
        or model.leaf_vector_shape[1] != 1
    ):
        raise Error("Pyre supports single-output scalar leaves only")
    var transform = model.postprocessor_name()
    if not (
        (model.task_type == TaskType.REGRESSOR and transform == "identity")
        or (model.task_type == TaskType.BINARY_CLF and transform == "sigmoid")
    ):
        raise Error("Pyre supports regression/identity and binary/sigmoid only")
    if len(model.extensions) != 0:
        raise Error("Pyre cannot interpret model extensions")
    for tree_id in range(len(model.trees)):
        if model.target_id[tree_id] != 0 or model.class_id[tree_id] != 0:
            raise Error("Pyre requires scalar tree target/class IDs of zero")
    var lt_only = True
    for tree in model.trees:
        if len(tree.tree_extensions) != 0 or len(tree.node_extensions) != 0:
            raise Error("Pyre cannot interpret tree/node extensions")
        for node in range(Int(tree.num_nodes)):
            if tree.node_type[node] == NodeType.CATEGORICAL:
                raise Error("Pyre does not support categorical splits")
            if tree.node_type[node] != NodeType.LEAF:
                lt_only &= tree.cmp[node] == Operator.LT
            if len(tree.leaf_values(node)) != 0:
                raise Error("Pyre does not support vector leaves")
    return lt_only
