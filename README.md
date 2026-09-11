# Pyre

CPU-first tree-ensemble inference in pure Mojo, using **Balsa models** and
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
pixi run test-gpu  # explicit CUDA copies + GPU inference rejection
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
scalar traversal. The default width is `simd_width_of[dtype]()`;
`predict_margin[1](x, num_threads=1)` provides a scalar comparison. Binary sigmoid
and label conversion run inside each worker with one output allocation, without
intermediate margin/probability arrays. Tree summation order is preserved. Base
scores are added in their Balsa Float64 precision before narrowing the result.
Run `pixi run assembly` to inspect vector code generation. No measured speedup
over nvForest is claimed.

Supported: float32 and float64 numerical scalar leaves; EQ/LT/LE/GT/GE; NaN default
routing; root-only/uneven-depth trees; single-output regression/identity and
binary/sigmoid; averaging, base scores, sigmoid alpha and stable probabilities.
Categorical splits, vector leaves, multiclass/multi-target output, unknown
extensions and other task/transform combinations are rejected. GPU kernels, leaf-ID/per-tree output and autotuning remain future work.

## CPU threading

Every prediction method accepts keyword `num_threads`: `1` forces serial
execution, positive values bound task concurrency for that call, and `0` (the
default) selects a conservative automatic policy: serial below 256 rows or
32,768 row–tree pairs, otherwise at most eight workers, capped by runtime
parallelism and the number of SIMD blocks. Negative values raise, including
on empty input. Work is split into contiguous groups of whole SIMD blocks; each
worker writes disjoint output rows and reads the immutable forest and input.
Tree accumulation order is unchanged. Calls finish before returning their arrays,
and simultaneous callers may share a forest with independently owned outputs.
Do not mutate the input or private model fields while prediction is running.

The MAX process pool is reused. A per-call limit does not reserve cores or resize
the pool; simultaneous calls share its resources. Set process affinity before
runtime initialization to constrain available CPUs. Pyre initializes the runtime
before parallel execution, including when used from a non-Mojo host. The existing
`max-core==26.5.0` dependency supplies `max.algorithm`; no Python model-serving
package is required. See [executor details](docs/executor-notes.md).

```mojo
var serial = forest.predict_margin(x, num_threads=1)
var parallel = forest.predict_margin(x, num_threads=4)
```

See [benchmark methodology and raw results](docs/benchmarks/README.md) for the
original baseline and the separate threading and SIMD measurements.

## Accelerator arrays

`pyre.accelerator` provides explicit `to_accelerator[dtype, device](x)` and
`to_host(x)` transfers, and `predict_cpu`, `predict_margin_cpu`,
`predict_proba_cpu`, `predict_class_cpu` for CPU `AcceleratorNDArray` inputs.
Every prediction operation rejects GPU input before accessing storage, with
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

Additional tests use Balsa builders and text accessors, transfer both model
precisions from `load_auto`, and cover all operators/default directions, empty
and partial batches, offset/strided views, ownership, malformed models and ties.
Python is only used by `pixi run oracle-generate` to regenerate expected outputs
from Balsa's checkpoints. Pyre neither copies nor regenerates those checkpoints.
`tests/fixtures/manifest.json` records source hashes, original oracle-generation
revision and tolerances. All twelve fixture hashes match the pinned `737874f`
snapshot; the stored reference predictions remain valid.

See [responsibilities](docs/responsibilities.md), [next steps](docs/porting-plan.md)
and [upstream notices](THIRD_PARTY_NOTICES.md).
