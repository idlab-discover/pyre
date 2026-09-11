# MAX CPU executor notes

Verified with Mojo 1.0.0 and `max-core==26.5.0` on 2026-09-11.
The existing `max-core` dependency supplies `max.algorithm`; installing the
Python MAX serving package is unnecessary.

## Context lifetime in Pyre

Pyre now retains one CPU `DeviceContext` per forest after its first threaded
prediction. The cache has an atomic lease: the caller that acquires it can
initialize and use the context; other callers create independent temporary
contexts. No caller waits for the lease. This keeps shared-forest concurrency
and executor reentrancy safe without retaining contexts proportional to peak
concurrency. Repeated overlapping calls can still incur context creation costs.

The cache's small state allocation belongs to the forest and survives moves.
Its optional context starts empty, so constructing a forest or predicting
serially does not initialize a CPU context. Destruction releases both the
context and state allocation. The immutable forest borrow keeps this state alive
until a prediction joins its tasks; only atomic operations access the busy flag,
and only its lease holder reads or writes the cached context.

Pyre directly calls `enqueue_cpu_range` and `synchronize`, retaining the
per-worker denormal handling and optional task tracing used by MAX. Its existing row partition already
assigns exactly one block range to each worker, so another partitioning wrapper
is unnecessary. An enqueue/join failure aborts, as MAX `parallelize` does,
because unwinding could release storage still borrowed by outstanding tasks.

Passing the cached context to MAX 26.5 `parallelize` is insufficient: its
`ctx.or_else(DeviceContext(api="cpu"))` expression constructs the default argument
before selecting the optional value. Direct enqueue avoids that unused context.

Regression coverage includes repeated concurrent predictions, a nested executor
call while the cached stream is executing, and moving a warmed forest before
predicting with different worker counts. Measurements and remaining limits are
in [context reuse results](context-reuse.md).

## MAX reference API and captures

Use the runtime closure overload with modern capture syntax:

```mojo
from max.algorithm import parallelize

# Inside a method, with immutable self/x and a mutable result List:
var output = result.unsafe_ptr()
def work(i: Int) {self, x, output}:
    output[unsafe_offset=i] = self.values[0] + x[i]
parallelize(work, len(x), workers)
```

A standalone version using a struct containing `List[Int]` compiled and returned
correct results. Capturing immutable model/input references and a pointer avoids
copying the model or sharing mutable traversal scratch. Writes must address
nonoverlapping output regions. No legacy closure decorators are needed.

`parallelize` returns after all tasks finish, so borrowed captures and output
storage remain alive throughout execution. In MAX 26.5, the implementation
partitions consecutive work indices into exactly `workers` coarse tasks, whose
sizes differ by at most one. `workers` must be positive. Cap it to useful work;
requesting more workers than indices creates empty tasks. Explicit worker count
limits task concurrency for that call; it does not resize the process pool or
reserve threads exclusively for that caller. Concurrent calls share resources.

## Runtime and process controls

Mojo `main()` initializes the runtime automatically. Shared libraries called by
Python/C/C++ must initialize it before executor use:

```mojo
from std.runtime import initialize_runtime

# Before parallelize in a host-callable entry point:
initialize_runtime()
```

Initialization is idempotent, covers all process threads, and keeps the runtime
alive until process exit. A compiled shared-library probe passed four concurrent
Python `ctypes.CDLL` callers, each making 100 calls with four tasks and separate
1,000-element outputs. All outputs matched integer squares. This is an isolated
executor check, not a proof of every application's concurrent safety.

`from std.runtime.asyncrt import parallelism_level` reports pool parallelism.
On this 32-logical-CPU host it returned 32 normally and 2 when the executable was
started with `taskset -c 0,1`. Set affinity before runtime initialization. The
candidate environment variables `MAX_NUM_THREADS`, `MODULAR_NUM_THREADS`, and
`MOJO_NUM_THREADS`, each set to 2 in a fresh process, did **not** change the
reported value. Do not document them as supported pool controls. Use explicit
per-call workers and process affinity for reproducible measurements.

## Sources and reproducibility

- [MAX 26.5 parallelize source](https://github.com/modular/modular/blob/max/v26.5/max/mojo/max/algorithm/backend/cpu/parallelize.mojo)
  documents synchronous execution and implements coarse task partitioning.
- [Runtime initialization API](https://mojolang.org/docs/std/runtime/asyncrt/initialize_runtime/)
  specifies shared-library initialization and process lifetime. Its older closure
  example should not override the modern syntax above.
- [MAX 26.5 DeviceContext source](https://github.com/modular/modular/blob/max/v26.5/max/mojo/max/gpu/host/device_context.mojo)
  specifies context ownership, enqueue, and synchronization.
- [MAX parallelize API](https://max.modular.com/stable/api/mojo/max/algorithm/backend/cpu/parallelize/parallelize/)
  lists runtime closure and explicit worker overloads.

Ignored local probes are `build/executor_probe.mojo`,
`build/executor_library.mojo`, and `build/executor_capture.mojo`. They were built
with the existing Pixi toolchain directly, setting `MODULAR_HOME` to that
environment's `share/max`, without changing dependency resolution.
