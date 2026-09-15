"""Checked native GPU array ownership and completed explicit transfers."""
from max.gpu.host import DeviceContext, DeviceBuffer
from numojo import NDArray
from numojo.core.type_aliases import Shape
from std.os import abort
from std.sys import size_of


struct _Join:
    var context: DeviceContext

    def __init__(out self, context: DeviceContext):
        self.context = context.copy()

    def __enter__(self):
        pass

    def __exit__(self):
        # Also runs when an enqueue raises. Never unwind borrowed storage while
        # the stream may still reference it; a failed join is unrecoverable.
        try:
            self.context.synchronize()
        except error:
            abort("Pyre GPU completion failed: " + String(error))


def _gpu(context: DeviceContext) raises:
    if context.api() == "cpu":
        raise Error("Pyre GPU arrays require a GPU context")


def _same_device(a: DeviceContext, b: DeviceContext) raises:
    if a.api() != b.api() or a.id() != b.id():
        raise Error("Pyre GPU device mismatch")


def _count[dtype: DType, rank: Int](shape: Shape) raises -> Int:
    comptime assert rank == 1 or rank == 2
    if shape.ndim != rank:
        raise Error("Pyre GPU array rank mismatch")
    var count = 1
    for i in range(rank):
        var dim = shape[i]
        if dim < 0:
            raise Error("Pyre GPU dimensions must be nonnegative")
        if dim != 0 and count > Int.MAX // size_of[Scalar[dtype]]() // dim:
            raise Error("Pyre GPU shape exceeds addressable storage")
        count *= dim
    return count


struct GpuArray[dtype: DType, rank: Int = 2](Movable):
    """Owned contiguous GPU storage with checked immutable shape metadata.

    Native buffer handles can be shared outside this object. Callers must not
    mutate or release underlying external storage during a prediction. External
    producers must complete before submission. Methods finish before returning.
    """

    var _context: DeviceContext
    var _buffer: DeviceBuffer[Self.dtype]
    var _shape: Shape
    var _offset: Int
    var _size: Int

    def __init__(out self, context: DeviceContext, shape: Shape) raises:
        _gpu(context)
        var count = _count[Self.dtype, Self.rank](shape)
        self._context = context.copy()
        self._buffer = context.enqueue_create_buffer[Self.dtype](count)
        self._shape = shape.copy()
        self._offset = 0
        self._size = count
        context.synchronize()

    def __init__(
        out self,
        var buffer: DeviceBuffer[Self.dtype],
        shape: Shape,
        offset: Int = 0,
        *,
        context: DeviceContext,
    ) raises:
        _gpu(context)
        _same_device(context, buffer.context())
        var count = _count[Self.dtype, Self.rank](shape)
        if offset < 0 or offset > len(buffer) or count > len(buffer) - offset:
            raise Error("Pyre GPU shape/offset exceeds buffer capacity")
        # Native shared handles retain the allocation; raw non-owning wrappers
        # require the external allocation owner to outlive this array.
        buffer.context().synchronize()
        buffer.reassign_ownership_to(context)
        self._context = context.copy()
        self._buffer = buffer^
        self._shape = shape.copy()
        self._offset = offset
        self._size = count

    def __deinit__(deinit self):
        _ = self._buffer^
        _ = self._context^

    @staticmethod
    def from_buffer(
        var buffer: DeviceBuffer[Self.dtype],
        rows: Int,
        features: Int,
        offset: Int = 0,
        *,
        context: DeviceContext,
    ) raises -> Self:
        comptime assert Self.rank == 2
        return Self(buffer^, Shape(rows, features), offset, context=context)

    def dim(self, axis: Int) raises -> Int:
        if axis < 0 or axis >= Self.rank:
            raise Error("Pyre GPU axis out of range")
        return self._shape[axis]

    def size(self) -> Int:
        return self._size

    def to_host(self) raises -> NDArray[Self.dtype]:
        var result = NDArray[Self.dtype](self._shape)
        var context = self._context.copy()
        with _Join(context):
            if self._size:
                var view = DeviceBuffer[Self.dtype](
                    context,
                    self._buffer.unsafe_ptr().unsafe_offset(self._offset),
                    self._size,
                    owning=False,
                )
                view.enqueue_copy_to(result.unsafe_ptr())
                # Keep the view wrapper alive through completion as well.
                context.synchronize()
        return result^


def _check_upload[dtype: DType](input: NDArray[dtype]) raises:
    if input.ndim != 2:
        raise Error("Pyre input must be rank two [rows, features]")
    if (
        not input.flags.C_CONTIGUOUS
        or (input.shape[0] > 1 and input.strides[0] != input.shape[1])
        or (input.shape[1] > 1 and input.strides[1] != 1)
    ):
        raise Error("Pyre GPU upload requires C-contiguous input")


def upload_into[
    dtype: DType
](input: NDArray[dtype], mut batch: GpuArray[dtype, 2]) raises:
    """Completed upload into existing storage with exactly matching shape.

    Dtype must match statically. Contiguous host views and device offsets are
    supported. To change active rows, construct a new checked GpuBatch wrapper
    over a retained native buffer; this preserves allocation capacity and the
    immutable logical-shape contract. Validation failure leaves storage intact.
    """
    _check_upload(input)
    if input.shape[0] != batch._shape[0] or input.shape[1] != batch._shape[1]:
        raise Error("Pyre GPU upload shape must match the destination")
    _same_device(batch._context, batch._buffer.context())
    if batch._size == 0:
        batch._context.synchronize()
        return
    # Declare the non-owning view outside the join: its wrapper and both owners
    # remain alive until completion, including the enqueue-error path.
    var view = DeviceBuffer[dtype](
        batch._context,
        batch._buffer.unsafe_ptr().unsafe_offset(batch._offset),
        batch._size,
        owning=False,
    )
    with _Join(batch._context):
        view.enqueue_copy_from(input.unsafe_ptr())


def upload[
    dtype: DType
](input: NDArray[dtype], context: DeviceContext) raises -> GpuArray[dtype, 2]:
    """Completed upload of a rank-two contiguous NuMojo array (offsets allowed).
    """
    _check_upload(input)
    var result = GpuArray[dtype, 2](context, input.shape)
    with _Join(context):
        if input.size:
            result._buffer.enqueue_copy_from(input.unsafe_ptr())
    return result^
