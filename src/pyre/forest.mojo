"""CPU tree-ensemble inference over NuMojo arrays, loaded through Balsa."""

from std.math import exp, isnan
from std.sys import simd_width_of
from balsa import Model, NodeType, Operator, TaskType, validate
from numojo import NDArray
from numojo.core.type_aliases import Shape


struct Forest[dtype: DType = DType.float32](Movable):
    """An owned Balsa model with SIMD CPU inference. Input precision must match.

    CPU inference vectorizes independent rows, preserving tree summation order.
    Supports scalar regression/identity and binary classification/sigmoid.
    """

    var _model: Model[Self.dtype]

    def __init__(out self, var model: Model[Self.dtype]) raises:
        """Validate inference support, then take ownership without repacking.

        Pass model^ to transfer an existing model, or model.copy() to retain an
        independently mutable source. Balsa remains responsible for structure.
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
            or (
                model.task_type == TaskType.BINARY_CLF
                and transform == "sigmoid"
            )
        ):
            raise Error(
                "Pyre supports regression/identity and binary/sigmoid only"
            )
        if len(model.extensions) != 0:
            raise Error("Pyre cannot interpret model extensions")
        for tree_id in range(len(model.trees)):
            if model.target_id[tree_id] != 0 or model.class_id[tree_id] != 0:
                raise Error(
                    "Pyre requires scalar tree target/class IDs of zero"
                )
        for tree in model.trees:
            if len(tree.tree_extensions) != 0 or len(tree.node_extensions) != 0:
                raise Error("Pyre cannot interpret tree/node extensions")
            for node in range(Int(tree.num_nodes)):
                if tree.node_type[node] == NodeType.CATEGORICAL:
                    raise Error("Pyre does not support categorical splits")
                if len(tree.leaf_values(node)) != 0:
                    raise Error("Pyre does not support vector leaves")
        self._model = model^

    def num_features(self) -> Int:
        return Int(self._model.num_feature)

    def num_trees(self) -> Int:
        return len(self._model.trees)

    def is_classifier(self) -> Bool:
        return self._model.task_type == TaskType.BINARY_CLF

    def _check(self, x: NDArray[Self.dtype]) raises:
        if x.ndim != 2:
            raise Error("Pyre input must be rank two [rows, features]")
        if x.shape[1] != self.num_features():
            raise Error("Pyre input feature count does not match the forest")
        if (
            not x.flags.C_CONTIGUOUS
            or (x.shape[0] > 1 and x.strides[0] != self.num_features())
            or (self.num_features() > 1 and x.strides[1] != 1)
        ):
            raise Error("Pyre MVP requires C-contiguous input")

    def _block[
        width: Int
    ](self, x: NDArray[Self.dtype], row: Int, count: Int) -> SIMD[
        Self.dtype, width
    ]:
        comptime assert Self.dtype.is_floating_point()
        var total = SIMD[Self.dtype, width](0)
        var input = x.unsafe_ptr()
        for tree in self._model.trees:
            var ids = SIMD[DType.int32, width](0)
            var done = SIMD[DType.bool, width](fill=False)
            var leaves = SIMD[Self.dtype, width](0)
            while True:
                var pending = False
                var values = SIMD[Self.dtype, width](0)
                var thresholds = SIMD[Self.dtype, width](0)
                var ops = SIMD[DType.int8, width](0)
                var defaults = SIMD[DType.bool, width](fill=False)
                var left = ids
                var right = ids
                comptime for lane in range(width):
                    if lane < count and not done[lane]:
                        var node = Int(ids[lane])
                        if tree.node_type[node] == NodeType.LEAF:
                            leaves[lane] = tree.leaf_value[node]
                            done[lane] = True
                        else:
                            pending = True
                            var feature = Int(tree.split_index[node])
                            values[lane] = input[
                                unsafe_offset=(row + lane) * self.num_features()
                                + feature
                            ]
                            thresholds[lane] = tree.threshold[node]
                            ops[lane] = tree.cmp[node]
                            defaults[lane] = tree.default_left[node] != 0
                            left[lane] = tree.cleft[node]
                            right[lane] = tree.cright[node]
                if not pending:
                    break
                var condition = (
                    (ops.eq(Operator.EQ) & values.eq(thresholds))
                    | (ops.eq(Operator.LT) & values.lt(thresholds))
                    | (ops.eq(Operator.LE) & values.le(thresholds))
                    | (ops.eq(Operator.GT) & values.gt(thresholds))
                    | (ops.eq(Operator.GE) & values.ge(thresholds))
                )
                condition = isnan(values).select(defaults, condition)
                ids = condition.select(left, right)
            total += leaves
        if self._model.average_tree_output != 0:
            total /= Scalar[Self.dtype](self.num_trees())
        return total + Scalar[Self.dtype](self._model.base_scores[0])

    def predict_margin[
        width: Int = simd_width_of[Self.dtype]()
    ](self, x: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        """Raw summed/averaged tree output plus base score, shape [rows]."""
        comptime assert (
            width > 0 and (width & (width - 1)) == 0
        ), "SIMD width must be a positive power of two"
        self._check(x)
        var rows = x.shape[0]
        var result = NDArray[Self.dtype](Shape(rows))
        var output = result.unsafe_ptr()
        for row in range(0, rows, width):
            var count = min(width, rows - row)
            var value = self._block[width](x, row, count)
            comptime for lane in range(width):
                if lane < count:
                    output[unsafe_offset=row + lane] = value[lane]
        return result^

    def predict(self, x: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        """Regression values. Classifiers must use predict_class/proba."""
        if self.is_classifier():
            raise Error(
                "Use predict_class or predict_proba for binary classification"
            )
        return self.predict_margin(x)

    def predict_proba(
        self, x: NDArray[Self.dtype]
    ) raises -> NDArray[Self.dtype]:
        """Binary class probabilities in columns [negative, positive]."""
        comptime assert Self.dtype.is_floating_point()
        if not self.is_classifier():
            raise Error("predict_proba requires a binary classifier")
        var margins = self.predict_margin(x)
        var result = NDArray[Self.dtype](Shape(x.shape[0], 2))
        var src = margins.unsafe_ptr()
        var dst = result.unsafe_ptr()
        for row in range(x.shape[0]):
            var z = (
                Scalar[Self.dtype](self._model.sigmoid_alpha)
                * src[unsafe_offset=row]
            )
            var e = exp(-abs(z))
            var p = (1 / (1 + e)) if z >= 0 else (e / (1 + e))
            dst[unsafe_offset=2 * row] = 1 - p
            dst[unsafe_offset=2 * row + 1] = p
        return result^

    def predict_class(
        self, x: NDArray[Self.dtype], threshold: Scalar[Self.dtype] = 0.5
    ) raises -> NDArray[DType.int64]:
        """Binary labels; probability must be strictly above threshold."""
        if not (threshold >= 0 and threshold <= 1):
            raise Error("Classification threshold must lie in [0, 1]")
        var probabilities = self.predict_proba(x)
        var result = NDArray[DType.int64](Shape(x.shape[0]))
        for row in range(x.shape[0]):
            result.unsafe_ptr()[unsafe_offset=row] = Int64(
                probabilities.unsafe_ptr()[unsafe_offset=2 * row + 1]
                > threshold
            )
        return result^
