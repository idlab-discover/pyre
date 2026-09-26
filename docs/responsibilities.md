# Balsa, Pyre and NuMojo

Balsa owns the model format; Pyre owns inference; NuMojo owns native arrays.
Balsa remains independent of Pyre, NuMojo, MAX and GPU execution.

| Owner | Responsibility |
| --- | --- |
| Balsa | Checkpoint read/write, version policy, precision discovery, structural validation, owned model fields and borrowed payload slices |
| Balsa | Named operators/node/task types, strict text accessors, model/tree builders and format fixtures |
| Pyre | Inference capability checks, prediction output rules, numerical transforms and CPU/GPU execution |
| Pyre | SIMD traversal, compact CPU layouts/workspaces, scheduling and performance measurements |
| NuMojo | Array storage, layouts, ownership and transfer primitives |
| Pyre tests | Prediction oracles over Balsa fixtures and end-to-end inference regressions |

## Current integration

`Forest(model^)` consumes a Balsa model. `Forest(model.copy())` retains an
independently editable source when needed. The private CPU plan owns inspection, routing and execution behind `Forest`.
It caches an ordered sum for leaf-only forests and borrows tree fields during SIMD
general traversal. Eligible symmetric CPU forests additionally pack split and leaf
arrays owned by NuMojo; Pyre does not implement checkpoint loading.
It uses Balsa's `Operator`, `NodeType`, `TaskType`, `postprocessor_name()` and
`leaf_values()` to interpret validated data. Original node indices are preserved in the owned Balsa model; compact CPU leaf
indices are recovered from validated child links.

Balsa's `load_auto` already handles precision discovery. Callers inspect its
variant and move the selected model into Pyre through `unwrap`; there is no
second Pyre variant/loader abstraction. Balsa builders construct inference test
models without manual wire-field bookkeeping.

GPU preparation now creates independent resident fields from the same Balsa
model; model-layout decisions and performance measurements belong in Pyre.
Native GPU buffers/layouts need no NuMojo accelerator wrapper.

Structural validity is distinct from execution support. Balsa can preserve
categorical, vector-output or extended models which Pyre currently rejects.
Pyre keeps the owned model private to preserve validation invariants. It retains
Balsa metadata alongside compact CPU arrays. The compact path reduces hot
metadata and traversal work, but increases total retained storage. See
[the CPU design and measurements](cpu-symmetric.md).

## Keeping the division useful

- Codec defects belong in Balsa with minimal checkpoint reproductions. Prediction
  defects, reduction precision and output transforms belong in Pyre.
- Reuse Balsa checkpoints and record their hashes. Pyre generates prediction
  inputs and GTIL expectations, not another model-format fixture suite.
- Keep named format concepts and accessors in Balsa. Do not add Balsa inference
  routines or force its storage to depend on NuMojo.
- Native GPU prediction runs GPU kernels on resident storage. Explicit upload
  and download remain separate; legacy CPU accelerator adapters reject GPU input.
- The NuMojo 0.10.0 storage-borrowing workaround is isolated in Pyre's accelerator
  adapter. It is not a Balsa responsibility.

Balsa's requested ergonomics are now present. No additional Balsa changes are
required for this CPU MVP.
