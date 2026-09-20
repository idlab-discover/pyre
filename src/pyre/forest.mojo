"""Public CPU inference interface; preparation and routing are private to _cpu."""
from balsa import Model
from numojo import NDArray
from ._cpu import _CPUPlan


struct Forest[dtype: DType = DType.float32](Movable):
    """Own a validated model and CPU execution plan, using NuMojo inputs/outputs.

    Construction inspects structure once. Predictions automatically select a
    supported internal kernel; callers never select a trainer or representation.
    Compact/constant paths are serial; num_threads bounds generic worker tasks.
    """

    var _cpu: _CPUPlan[Self.dtype]

    def __init__(out self, var model: Model[Self.dtype]) raises:
        """Transfer model^, or pass model.copy() to retain an editable source.
        """
        self._cpu = _CPUPlan(model^)

    def num_features(self) -> Int:
        return self._cpu.num_features()

    def num_trees(self) -> Int:
        return self._cpu.num_trees()

    def is_classifier(self) -> Bool:
        return self._cpu.is_classifier()

    def predict_margin(
        self, x: NDArray[Self.dtype], *, num_threads: Int = 0
    ) raises -> NDArray[Self.dtype]:
        """Raw margins [rows], after averaging and base score."""
        return self._cpu.predict_margin(x, num_threads=num_threads)

    def predict_margin[
        width: Int
    ](self, x: NDArray[Self.dtype], *, num_threads: Int = 0) raises -> NDArray[
        Self.dtype
    ]:
        return self._cpu.predict_margin[width](x, num_threads=num_threads)

    def predict(
        self, x: NDArray[Self.dtype], *, num_threads: Int = 0
    ) raises -> NDArray[Self.dtype]:
        """Regression values [rows]; classifiers use predict_class/proba."""
        return self._cpu.predict(x, num_threads=num_threads)

    def predict[
        width: Int
    ](self, x: NDArray[Self.dtype], *, num_threads: Int = 0) raises -> NDArray[
        Self.dtype
    ]:
        return self._cpu.predict[width](x, num_threads=num_threads)

    def predict_proba(
        self, x: NDArray[Self.dtype], *, num_threads: Int = 0
    ) raises -> NDArray[Self.dtype]:
        """Binary probabilities in columns [negative, positive]."""
        return self._cpu.predict_proba(x, num_threads=num_threads)

    def predict_proba[
        width: Int
    ](self, x: NDArray[Self.dtype], *, num_threads: Int = 0) raises -> NDArray[
        Self.dtype
    ]:
        return self._cpu.predict_proba[width](x, num_threads=num_threads)

    def predict_class(
        self,
        x: NDArray[Self.dtype],
        threshold: Scalar[Self.dtype] = 0.5,
        *,
        num_threads: Int = 0,
    ) raises -> NDArray[DType.int64]:
        """Binary labels [rows]; probability must be strictly above threshold.
        """
        return self._cpu.predict_class(x, threshold, num_threads=num_threads)

    def predict_class[
        width: Int
    ](
        self,
        x: NDArray[Self.dtype],
        threshold: Scalar[Self.dtype] = 0.5,
        *,
        num_threads: Int = 0,
    ) raises -> NDArray[DType.int64]:
        return self._cpu.predict_class[width](
            x, threshold, num_threads=num_threads
        )
