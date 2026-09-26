# Pyre

![Pyre: a flame mascot on a furnace](assets/branding/pyre-logo-169.png)

[![CI](https://github.com/idlab-discover/pyre/actions/workflows/ci.yml/badge.svg)](https://github.com/idlab-discover/pyre/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/idlab-discover/pyre)](https://github.com/idlab-discover/pyre/releases)

**CPU and GPU tree-ensemble inference in pure Mojo.** Pyre evaluates
[Balsa](https://github.com/idlab-discover/balsa) models with
[NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo) arrays,
without Python or libtreelite in the inference runtime.

- Float32 and float64 regression and binary classification, with missing-value routing.
- Automatic CPU dispatch for constant, symmetric, and general tree structures.
- Resident GPU inference, explicit transfers, and reusable output buffers.

## Get started

Requires Linux x86-64, [Pixi](https://pixi.sh), Git, and a C linker such as GCC.
The locked environment provides Mojo **1.0.0**, MAX core **26.5.0**, and NuMojo
**0.10.0**. Preparation fetches a pinned Balsa revision automatically.

```sh
git clone https://github.com/idlab-discover/pyre.git
cd pyre
git checkout v0.1.0
pixi install --locked
pixi run run
```

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

Save as `app.mojo` in the repository root and run
`pixi run mojo run -I src -I build/balsa-reference/src app.mojo`.
For use from another project, see the [precompiled release bundle](docs/distribution.md).

## API and defaults

| Operation | Result |
| --- | --- |
| `forest.predict(x)` | Regression values |
| `forest.predict_margin(x)` | Raw margins |
| `forest.predict_proba(x)` | Binary probabilities, `[rows, 2]` |
| `forest.predict_class(x)` | Int64 labels, threshold 0.5 |

Inputs are contiguous `[rows, features]` arrays matching the model's precision.
`Forest(model^)` takes ownership; `Forest(model.copy())` retains your original.
Construction validates the model and prepares its execution strategy once.
Categorical splits, vector leaves, and multiclass/multi-target output are not supported.

CPU calls use `num_threads=0` for conservative automatic scheduling; an explicit
positive value sets a per-call task limit. Constant and compact symmetric paths
currently run serially. `GpuForest` uses explicit device storage and completes
calls before returning; `*_into` methods reuse output allocations. GPU execution
has been validated on an RTX 5090; other GPU targets remain unvalidated.
See the [user guide](docs/usage.md) and [GPU example](examples/predict_gpu.mojo).

## Performance snapshot

Across 41 trained regression models and five batch sizes, Pyre achieved these
geometric-mean speedups against nvForest's default chunk setting:

| Execution | Pyre speedup | Faster cases |
| --- | ---: | ---: |
| CPU, 1 thread | **2.49×** | 182 / 205 |
| CPU, 8 threads | **2.07×** | 189 / 205 |
| GPU | **1.71×** | 195 / 205 |

Pyre also beat the original scikit-learn and LightGBM predictors in every measured
CPU case. Measurements used a Ryzen 9 5950X and RTX 5090, resident inputs,
allocating completed calls, and five independent processes. Pyre's native Mojo
API is compared with Python APIs; these are application-call timings, not isolated
kernel comparisons. Against the best measured nvForest chunk per case, aggregate
speedups were **2.43× / 1.59× / 1.05×**, respectively.
[Methodology and limitations](docs/benchmarks/comparison-0.1.0.md).

Future work will optimize larger batches and larger forests, including parallel
execution of compact symmetric CPU trees. Saturation testing across thread counts,
concurrent callers, and diverse hardware will guide more scalable default strategies.

## License

Pyre is licensed under [Apache-2.0](LICENSE), without LLVM exceptions.
See [third-party notices](THIRD_PARTY_NOTICES.md) for dependency attribution.

## Development

```sh
pixi run check       # CPU correctness, package, and isolated consumer
pixi run test-gpu    # Requires a supported GPU
pixi run release    # Versioned library bundle and checksums
```

- [User guide](docs/usage.md): ownership, supported semantics, threading, and GPU buffers.
- [Distribution and CI](docs/distribution.md): pinned dependencies and release consumption.
- [CPU planning](docs/cpu-planning.md) and [symmetric inference](docs/cpu-symmetric.md).
- [GPU benchmarking](docs/gpu-benchmarking.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

<p align="center">
  <img src="assets/branding/pyre-logo-square.png" alt="Pyre mascot" width="160">
</p>
