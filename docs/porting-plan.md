# Pyre next steps

## Implemented

- Mojo 1.0.0 / NuMojo 0.10.0 / MAX 26.5.0 environment.
- Direct ownership of Balsa models with no duplicate loader, tags, text decoder,
  or packed-node representation. Automatic precision discovery stays in Balsa.
- Float32/float64 SIMD CPU numerical scalar regression and binary classification.
- Balsa-builder tests, twelve upstream GTIL reference checkpoints, and rejection
  checks for unsupported trained-framework models.
- Native package/example/test tasks; explicit accelerator copies and CPU-array
  adapters. GPU predictions raise `NotImplementedError` before touching storage.

The [README](../README.md) defines the supported interface and limitations.
The [responsibility split](responsibilities.md) governs future changes.

## 1. Measure the CPU implementation

Benchmark scalar-width and SIMD traversal over Balsa's trained models and larger
representative ensembles. Separate loading, structural validation, prediction,
and allocation costs. Vary depth, row/feature/tree count and precision. Measure
memory retained by Balsa metadata before deciding whether a compact serving
layout is worth adding. Retain correctness checks through any optimization.

## 2. Implement actual GPU inference

Prepare resident device forest storage from the Balsa model; accept NuMojo device
arrays and execute traversal, aggregation and transforms on GPU. Start with the
CPU-supported subset and compare against the same saved oracles. Validate empty
and partial blocks, NaN/default routing, offsets, ownership, stream ordering and
synchronization. GPU-input/device-output prediction must not stage to CPU.
Replace the current not-implemented path only when GPU execution is verified.

## 3. Expand semantics in complete tested slices

Use Balsa's existing categorical and vector-leaf spans and trained checkpoints
for categorical splits, multiclass, vector leaves and multi-target routing.
Keep format validation in Balsa and prediction-capability checks in Pyre. Add
original leaf-ID and per-tree output modes without changing checkpoint node IDs.

## 4. Broaden integration and tune

Add Python interfaces and CuPy/DLPack interchange once ownership and stream
contracts are tested. Resolve the NuMojo transfer workaround against a tested
upstream release. Measure CPU threading, GPU work mapping, shared memory,
workspace reuse and layouts before introducing autotuning. Compare with pinned
nvForest on identical hardware, inputs and precision, separating transfers and
first-call costs from steady-state execution.

## References

- [Balsa usage](../../balsa/docs/usage.md) and [framework fixtures](../../balsa/docs/framework-testing.md).
- [nvForest source](https://github.com/rapidsai/nvforest/tree/f4e27dbf27ca9ddf3a502133aaaf2a193aa69293).
- [NuMojo source](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/tree/515fb2856f0ecf3d2740a34d958fe168183e1129).
