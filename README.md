# Pyre

CPU-first tree-ensemble inference in pure Mojo, using **Balsa models** and
**NuMojo arrays**. The native runtime does not call Python or libtreelite.

## Build and test

Requires Pixi, a C linker, and sibling Balsa source at `../balsa`. Tested against
Balsa `b3a3f1c`, Mojo **1.0.0**, NuMojo **0.10.0**, and MAX core **26.5.0**.
Pixi locks the package dependencies; Balsa remains an explicit local source
dependency and must provide its current model, builder, and precision interfaces.

```sh
pixi install --locked
pixi run run
pixi run check
pixi run test-gpu  # explicit CUDA copies + GPU inference rejection
```

`check` builds the example, precompiles `build/pyre.mojoc`, and runs native tests.
Consumers compile with `-I src -I ../balsa/src` in this environment.

## Balsa owns loading; Pyre owns inference

```mojo
from balsa import load
from pyre import Forest
from numojo import NDArray
from numojo.core.type_aliases import Shape


def main() raises:
    var forest = Forest(load("../balsa/tests/fixtures/float32_op2_missing0.tl"))
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

Traversal operates on groups of independent rows. It gathers node/feature values
per lane and uses SIMD comparisons, child selection, accumulation, averaging and
bias. Divergent paths and partial batches are supported; tree summation order is
preserved. The default width is `simd_width_of[dtype]()`; `predict_margin[1](x)`
provides a scalar-width comparison. Run `pixi run assembly` to inspect vector code
generation. No measured speedup over nvForest is claimed.

Supported: float32 and float64 numerical scalar leaves; EQ/LT/LE/GT/GE; NaN default
routing; root-only/uneven-depth trees; single-output regression/identity and
binary/sigmoid; averaging, base scores, sigmoid alpha and stable probabilities.
Categorical splits, vector leaves, multiclass/multi-target output, unknown
extensions and other task/transform combinations are rejected. Threading, GPU
kernels, leaf-ID/per-tree output and autotuning remain future work.

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

Additional tests use Balsa builders and text accessors, transfer both model
precisions from `load_auto`, and cover all operators/default directions, empty
and partial batches, offset/strided views, ownership, malformed models and ties.
Python is only used by `pixi run oracle-generate` to regenerate expected outputs
from Balsa's checkpoints. Pyre neither copies nor regenerates those checkpoints.
`tests/fixtures/manifest.json` records source hashes, Balsa revision and tolerances.

See [responsibilities](docs/responsibilities.md), [next steps](docs/porting-plan.md)
and [upstream notices](THIRD_PARTY_NOTICES.md).
