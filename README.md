# Pyre

CPU and GPU tree-ensemble inference in pure Mojo, using **Balsa models** and
**NuMojo arrays**. The native runtime does not call Python or libtreelite.

## Build and test

Requires Pixi, a C linker, and sibling Balsa source at `../balsa` (only its pinned Git commit is read). Tested against
Balsa `737874f3eae9d95a0517397130f445b303d28cc6`, Mojo **1.0.0**, NuMojo **0.10.0**, and MAX core **26.5.0**.
Pixi locks package dependencies. `prepare` extracts that exact Balsa commit into
`build/balsa-reference` using `git archive`; builds, fixtures and oracle generation
use this snapshot even while the sibling Balsa working tree changes. The commit
must be available in the sibling repository. Python and pyperf support development
and benchmarking; inference remains native Mojo.

```sh
pixi install --locked
pixi run run
pixi run check
pixi run test-gpu  # GPU inference, native storage, and legacy transfer tests
```

`check` builds the example, precompiles `build/pyre.mojoc`, and runs native tests.
Consumers compile with `-I src -I build/balsa-reference/src` in this environment.

## Balsa owns loading; Pyre owns inference

```mojo
from balsa import load
from pyre import Forest
from numojo import NDArray
from numojo.core.type_aliases import Shape


def main() raises:
    var forest = Forest(load("build/balsa-reference/tests/fixtures/float32_op2_missing0.tl"))
    var x = NDArray[DType.float32](Shape(3, 1))
    x.itemset(0, 0.0)
    x.itemset(1, 0.5)
    x.itemset(2, 1.0)
    print(forest.predict(x))  # [-1, 3, 3]
```

`Forest(model^)` takes ownership of a Balsa model. Use `Forest(model.copy())`
when retaining an independently mutable source. Pyre validates structure with
Balsa and checks its own supported prediction semantics. It traverses the owned
Balsa tree arrays directly, avoiding a second model representation and repacking.
Training metadata remains in that owned model; there is no compact serving layout
or memory-reduction claim yet. Do not mutate the forest's private model fields.

Use Balsa's `ModelBuilder`/`TreeBuilder` for in-memory construction and
`load[DType.float64]` for known double precision. For unknown precision, use
Balsa's `load_auto`, inspect the returned variant once, then transfer its selected
model using `variant^.unwrap[Model[dtype]]()`. The [runnable example](examples/predict.mojo)
shows this dispatch without copying. Pyre has no separate loader or precision
wrapper.

| Operation | Output |
| --- | --- |
| `forest.predict(x)` | Regression values, `[rows]`, model dtype |
| `forest.predict_margin(x)` | Raw margins after averaging/base score, `[rows]` |
| `forest.predict_proba(x)` | Binary probabilities `[rows, 2]`, model dtype |
| `forest.predict_class(x, threshold=0.5)` | Int64 labels `[rows]`, strict `p > threshold` |

Input is a rank-two, C-contiguous NuMojo `NDArray` with matching precision and
feature count. Empty batches and contiguous offset views work; strided input
requires an explicit contiguous copy. `predict` is regression-only; binary labels
use their own typed operation.

## SIMD and supported semantics

Traversal operates on groups of independent rows using pointer gathers, SIMD
comparisons, child selection, accumulation, averaging and bias. Finished lanes
stay at their leaf and partial batches mask input reads; single-row blocks use
scalar traversal. All four `Forest` prediction methods accept a compile-time
`width` parameter. Without an explicit width, batches of at least 32 rows use
16-row groups on targets whose native SIMD widths are eight Float32 lanes and
four Float64 lanes. With an explicit thread limit greater than one, widening
also requires enough 16-row groups to meet that limit. Other calls use
`simd_width_of[dtype]()`. Explicit widths always override this selection: use
`predict[16](x, num_threads=1)` to force 16-row SIMD groups, or
`predict_margin[1](x, num_threads=1)` for a scalar comparison. Width must be a
positive power of two. A logical width can span multiple hardware registers;
choose it by benchmarking the target CPU and model. Binary sigmoid
and label conversion run inside each worker with one output allocation, without
intermediate margin/probability arrays. Tree summation order is preserved. Base
scores are added in their Balsa Float64 precision before narrowing the result.
At construction, Pyre checks whether every numerical split uses LT. Each
prediction dispatches once to an LT-only or general kernel. Compile-time
specialization removes operator loads and the five-way comparison from the
LT-only kernel; mixed/EQ/LE/GT/GE forests retain general semantics. Leaf metadata
does not affect dispatch. No model repacking or runtime code generation occurs.
Run `pixi run assembly` to inspect vector code generation.

Within each worker's row partition, CPU traversal uses 256-row tiles and consecutive
four-tree groups to reuse model data across rows. Each row's running total carries
through every group in original tree order; averaging, base score and output
transform run once afterward. Scratch is private to each worker call and reused
across its tiles: 1 KiB for Float32 or 2 KiB for Float64 at widths up to 256. Wider
explicit widths use one SIMD block per tile. The policy favors larger batches and
deeper forests; one-row calls can pay extra allocation and loop overhead. Tile sizes
are internal constants, and SIMD selection and worker partitioning are unchanged.


Supported: float32 and float64 numerical scalar leaves; EQ/LT/LE/GT/GE; NaN default
routing; root-only/uneven-depth trees; single-output regression/identity and
binary/sigmoid; averaging, base scores, sigmoid alpha and stable probabilities.
Categorical splits, vector leaves, multiclass/multi-target output, unknown
extensions and other task/transform combinations are rejected. Leaf-ID/per-tree output and autotuning remain future work.

## CPU threading

Every prediction method accepts keyword `num_threads`: `1` forces serial
execution, positive values bound task concurrency for that call, and `0` (the
default) selects a conservative automatic policy: serial below 256 rows or
8,192 row–tree pairs, at most four workers below 32,768 pairs, and at most eight
workers for larger workloads. Worker counts are capped by runtime parallelism
and the number of SIMD blocks. Negative values raise, including
on empty input. Work is split into contiguous groups of whole SIMD blocks; each
worker writes disjoint output rows and reads the immutable forest and input.
For threaded calls, the calling thread runs one partition while the MAX pool
runs the others; the requested task limit includes the caller's partition.
Tree accumulation order is unchanged. Calls finish before returning their arrays,
and simultaneous callers may share a forest with independently owned outputs.
Do not mutate the input or private model fields while prediction is running.

The MAX process pool is reused. Each forest also lazily caches one CPU execution
context for repeated threaded predictions. A call acquires the cache without
waiting; overlapping or nested calls use independent temporary contexts. The
cached context is released with the forest. Serial calls do not create a context.
A per-call limit does not reserve cores or resize the pool; simultaneous calls
share its resources. Set process affinity before
runtime initialization to constrain available CPUs. Pyre initializes the runtime
before parallel execution, including when used from a non-Mojo host. The existing
`max-core==26.5.0` dependency supplies `max.algorithm`; no Python model-serving
package is required.

```mojo
var serial = forest.predict_margin(x, num_threads=1)
var parallel = forest.predict_margin(x, num_threads=4)
```

## Native GPU inference

`pyre.gpu` provides `GpuForest`, `GpuBatch[dtype]` (a rank-two
`GpuArray[dtype, 2]`), `GpuArray[dtype, rank]`, `upload`, and `upload_into`. Balsa still owns
Treelite loading and structural validation. GPU preparation borrows its model,
checks the same supported semantics as CPU, and copies the required fields into
independent resident device arrays. The source model can then be released or
edited. The current flattened layout is a baseline, not an optimal-layout claim.

```mojo
from max.gpu.host import DeviceContext
from pyre.gpu import GpuForest, GpuArray, upload

var context = DeviceContext()
var gpu_forest = GpuForest.prepare(model, context)
var batch = upload(samples, context)  # rank-two contiguous NuMojo NDArray
var result = gpu_forest.predict_margin(batch)  # result remains on GPU
var host_result = result.to_host()  # explicit download to NuMojo
```

The four prediction names and task/output rules match the CPU table above.
Each has a corresponding `*_into(batch, output)` method to reuse an output
allocation; `predict_class_into` accepts `threshold` after `output`.
Create outputs with `GpuArray[dtype, 1](context, Shape(rows))`,
`GpuArray[dtype, 2](context, Shape(rows, 2))` for probabilities, or
`GpuArray[DType.int64, 1]` for labels. `dim(axis)` and `size()` inspect shape.
Inputs and outputs must not overlap; wrong shapes/devices raise before launch.

Native device inputs do not require NuMojo wrapping:
`GpuBatch[dtype].from_buffer(buffer^, rows, features, offset=0, context=context)`
retains the allocation without copying data. For a rank-one result or a result
with an offset, use `GpuArray[dtype, rank](buffer^, Shape(...), offset=...,
context=context)`. Construction checks capacity and dimensions, completes the
buffer's source context, and reassigns the native buffer to the supplied context
on the same device. Shared native handles observe this reassignment.
Only contiguous row-major layouts are currently accepted. Upload accepts
contiguous NuMojo offset views; strided input needs explicit conversion.

`upload_into(samples, batch)` reuses existing device storage and completes before
returning. Both shape and dtype must match; dtype mismatches fail at compile time.
Validation failures leave destination bytes intact. Logical shapes stay immutable:
for changing row counts, retain a native buffer handle and create a new
`GpuBatch.from_buffer` wrapper with the active shape. Its capacity and offset are
checked, and its device allocation is retained. Reuse an output buffer with a
matching active output shape in the same way; the example demonstrates this.

Use an owning `DeviceContext` from native allocation/creation, not a borrowed
context returned by `buffer.context()`. Forests and arrays retain its handle and
release their buffers before it; results can outlive the forest and caller's
context variable. Moving a Pyre array preserves its allocation. If a native
buffer wraps externally owned memory, that external owner must still outlive
all Pyre uses. Do not mutate private fields or shared input storage during calls.

All public operations finish before returning, including the `*_into` methods.
External producers must complete before prediction. Calls sharing a context
must not overlap; independent contexts are supported. GPU inference never stages
inputs to CPU or implicitly downloads results. A failed enqueue drains the
stream before unwinding; a failed synchronization aborts because storage may
still be in use. No cuFile or Python dependency is used by inference or transfers.

Run [the GPU example](examples/predict_gpu.mojo) with `pixi run run-gpu`.
`pixi run test-gpu` requires an available GPU and exercises saved GTIL oracles,
both precisions, all four operations, owned storage, offsets, and cleanup.
`pixi run check` remains the CPU/package check. The GPU path is validated on the
RTX 5090 with the pinned toolchain; other GPU backends are not yet validated.

The [GPU benchmark guide](docs/gpu-benchmarking.md) describes the separate
performance corpus, pinned offline generation, correctness gates and phase
boundaries. After corpus generation, `pixi run benchmark-gpu` builds a native
worker and exports calibrated repeated-process pyperf results under ignored
`docs/private/benchmarks/gpu/`. Use `pixi run python tools/benchmark_gpu.py --help`
for a subset or a new output path.

## Legacy accelerator-array adapters

`pyre.accelerator` provides explicit `to_accelerator[dtype, device](x)` and
`to_host(x)` transfers, and `predict_cpu`, `predict_margin_cpu`,
`predict_proba_cpu`, `predict_class_cpu` for CPU `AcceleratorNDArray` inputs.
These CPU adapter operations reject GPU input before accessing storage, with
Mojo `Error("NotImplementedError: GPU inference requires GPU kernels; CPU fallback is disabled")`.
There is no GPU-to-CPU inference fallback. Rejection tests need no GPU;
`test-gpu` also verifies actual CUDA transfers on the RTX 5090.

The transfer adapter borrows optional storage with `value()` to avoid the copied
storage/pointer issue observed in NuMojo 0.10.0. It accesses pinned NuMojo internals
and must be revalidated on upgrades. Accelerator views and CuPy interchange are
not supported. No Balsa or NuMojo source was patched.

## Verification

Native tests compare SIMD and scalar-width predictions with saved Treelite 4.6.1
GTIL outputs for **12 Balsa checkpoints**: eight synthetic cases plus trained
XGBoost, numeric CatBoost, sklearn gradient-boosting classification, and sklearn
histogram-boosting regression. Probes include NaNs, divergent rows, and exact
root thresholds and their floating-point neighbors. Unsupported trained
categorical/vector models are tested for rejection.

Parallel tests also cover shared-forest concurrent calls, all thread controls,
width 16, zero-feature forests, deep lane divergence and sigmoid extremes.
Specialization tests exercise all four public operations at widths 1/4/16 in
both precisions, mixed operators in later nodes/trees, ties, NaNs and uneven
depths. `pixi run test-specialization` runs this focused suite.

Additional tests use Balsa builders and text accessors, transfer both model
precisions from `load_auto`, and cover all operators/default directions, empty
and partial batches, offset/strided views, ownership, malformed models and ties.
Python is only used by `pixi run oracle-generate` to regenerate expected outputs
from Balsa's checkpoints. Pyre neither copies nor regenerates those checkpoints.
`tests/fixtures/manifest.json` records source hashes, original oracle-generation
revision and tolerances. All twelve fixture hashes match the pinned `737874f`
snapshot; the stored reference predictions remain valid.

See [responsibilities](docs/responsibilities.md) and
[upstream notices](THIRD_PARTY_NOTICES.md).
