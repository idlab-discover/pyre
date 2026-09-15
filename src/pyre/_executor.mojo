"""A bounded, lazy CPU context cache with independent overlapping calls."""

from std.atomic import Atomic
from std.os import abort
from std.memory import MutPointer
from std.memory.alloc import unsafe_alloc
from std.utils.numerics import FlushDenormals
from max.gpu.host import DeviceContext
from max.runtime import tracing
from max.runtime.tracing import Trace, TraceLevel


struct _State(Movable):
    var busy: Atomic[DType.int64]
    var context: Optional[DeviceContext]

    def __init__(out self):
        self.busy = Atomic[DType.int64](0)
        self.context = None


struct _Lease:
    var state: MutPointer[_State, MutUntrackedOrigin]
    var cached: Bool

    def __init__(out self, state: MutPointer[_State, MutUntrackedOrigin]):
        self.state = state
        var expected = Int64(0)
        self.cached = state[].busy.compare_exchange(expected, 1)

    def __enter__(self) -> Bool:
        return self.cached

    def __exit__(self):
        if self.cached:
            self.state[].busy.store(0)


struct _Executor(Movable):
    # The stable allocation permits synchronized interior mutation through an
    # immutable Forest borrow. Only the lease holder touches context; a busy
    # caller does not read it. Forest's borrow keeps this storage alive.
    var _state: MutPointer[_State, MutUntrackedOrigin]

    def __init__(out self):
        self._state = unsafe_alloc[_State](1)
        self._state.unsafe_write(_State())

    def __deinit__(deinit self):
        self._state.unsafe_deinit_pointee()
        self._state.unsafe_free()

    def run[Func: def(Int) -> None](self, work: Func, workers: Int) raises:
        var parent_id = tracing.get_current_trace_id[TraceLevel.THREAD]()

        def wrapped(i: Int) {work, parent_id}:
            with FlushDenormals():
                try:
                    with Trace[TraceLevel.THREAD, target=StaticString("cpu")](
                        "task", task_id=i, parent_id=parent_id
                    ):
                        work(i)
                except error:
                    abort(String(error))

        with _Lease(self._state) as cached:
            var context: DeviceContext
            if cached:
                if not self._state[].context:
                    self._state[].context = DeviceContext(api="cpu")
                context = self._state[].context.value()
            else:
                context = DeviceContext(api="cpu")
            # Direct enqueue avoids parallelize's eager default-context
            # construction even when its optional context argument is supplied.
            try:
                # Let the caller perform the final partition while the pool
                # runs the others. Fully queued short calls can spend much
                # longer waiting for the last task to start than doing work.
                # The total number of partitions remains the requested limit.
                if workers > 1:
                    context.enqueue_cpu_range(wrapped, count=workers - 1)
                if workers > 0:
                    wrapped(workers - 1)
                context.synchronize()
            except error:
                # As in MAX parallelize, do not unwind borrowed captures while
                # an unsuccessful join might leave tasks using them.
                abort(String(error))
