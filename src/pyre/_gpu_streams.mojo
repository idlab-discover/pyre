"""Copy streams for pipelined host predictions, cached per prepared forest."""

from std.atomic import Atomic
from std.os import abort
from std.memory import MutPointer
from std.memory.alloc import unsafe_alloc
from max.gpu.host import DeviceContext


@fieldwise_init
struct _Streams(Copyable, Movable):
    """One stream uploads slabs, another downloads results; compute stays on
    the forest's own context."""

    var load: DeviceContext
    var store: DeviceContext

    @staticmethod
    def create(context: DeviceContext) raises -> Self:
        var api = String(context.api())
        var id = Int(context.id())
        return Self(
            DeviceContext(device_id=id, api=api),
            DeviceContext(device_id=id, api=api),
        )


struct _State(Movable):
    var busy: Atomic[DType.int64]
    var streams: Optional[_Streams]

    def __init__(out self):
        self.busy = Atomic[DType.int64](0)
        self.streams = None


struct _StreamCache(Movable):
    # As in `_Executor`: stable storage permits synchronized interior
    # mutation through an immutable forest borrow. Only the lease holder
    # touches `streams`; an overlapping caller creates its own pair.
    var _state: MutPointer[_State, MutUntrackedOrigin]

    def __init__(out self):
        self._state = unsafe_alloc[_State](1)
        self._state.unsafe_write(_State())

    def __deinit__(deinit self):
        self._state.unsafe_deinit_pointee()
        self._state.unsafe_free()

    def acquire(self, context: DeviceContext) raises -> Tuple[_Streams, Bool]:
        """Streams for one call and whether the cached pair was leased."""
        var expected = Int64(0)
        if self._state[].busy.compare_exchange(expected, 1):
            try:
                if not self._state[].streams:
                    self._state[].streams = _Streams.create(context)
                return (self._state[].streams.value().copy(), True)
            except error:
                self._state[].busy.store(0)
                raise error^
        return (_Streams.create(context), False)

    def release(self, leased: Bool):
        if leased:
            self._state[].busy.store(0)


struct _JoinAll:
    """Joins compute and both copy streams, also when an enqueue raises."""

    var compute: DeviceContext
    var streams: _Streams

    def __init__(out self, compute: DeviceContext, streams: _Streams):
        self.compute = compute.copy()
        self.streams = streams.copy()

    def __enter__(self):
        pass

    def __exit__(self):
        # Never unwind host or device storage another stream may still use.
        try:
            self.streams.load.synchronize()
            self.compute.synchronize()
            self.streams.store.synchronize()
        except error:
            abort("Pyre GPU completion failed: " + String(error))
