# CPU structure sweep — 2026-09-20

The private CPU plan selects kernels from validated structure, independently of
model producer. All **120** fixture/dtype/batch gates match the independent scalar
Balsa oracle. The sweep contains **312 timing cells** and **2,808 intervals**.
The final implementation passes `pixi run check` (40 Mojo tests) and five Python
memory-accounting tests.

## Main results

Medians in milliseconds for 5,000 rows, float32. Each cell has three samples in
each of three randomized rounds, calibrated to at least 50 ms. Complete allocating
public margin calls include binning and result disposal. Construction is excluded.

| Family | Trees | Depth | Features | Generic ms | Direct ms | Automatic ms | Generic / auto |
|---|---:|---:|---:|---:|---:|---:|---:|
| asymmetric | 128 | 4 | 16 | 5.985560 | — | 5.982748 | 1.00× |
| asymmetric | 128 | 8 | 16 | 11.906514 | — | 11.947091 | 1.00× |
| asymmetric | 64 | 12 | 16 | 11.177611 | — | 11.282017 | 0.99× |
| asymmetric | 4 | 16 | 16 | 1.177756 | — | 1.150756 | 1.02× |
| chain | 128 | 16 | 16 | 11.472823 | — | 11.450349 | 1.00× |
| mixed | 128 | 8 | 16 | 6.448463 | — | 6.442033 | 1.00× |
| roots | 4096 | 0 | 0 | 16.927584 | 7.811056 | 0.001791 | 9450.13× |
| roots | 4096 | 0 | 64 | 16.009098 | 7.809355 | 0.001807 | 8860.74× |
| stumps | 16 | 1 | 16 | 0.240418 | 0.113460 | 0.113608 | 2.12× |
| stumps | 1024 | 1 | 64 | 15.081530 | 7.337603 | 3.393054 | 4.44× |
| symmetric | 128 | 4 | 16 | 6.011362 | 1.829699 | 0.665445 | 9.03× |
| symmetric | 128 | 8 | 16 | 11.958171 | 3.104657 | 0.890351 | 13.43× |
| symmetric | 4 | 12 | 16 | 0.617733 | 0.140904 | 0.093899 | 6.58× |
| symmetric | 64 | 12 | 16 | 11.409996 | 2.406035 | 0.787755 | 14.48× |
| symmetric | 4 | 16 | 16 | 1.180934 | 0.200332 | 0.113987 | 10.36× |

Float64 symmetric speedups at 5,000 rows are 9.81× at depth 4 / 128 trees,
14.80× at depth 8 / 128 trees, 14.94× at depth 12 / 64 trees, and 9.94× at
depth 16 / four trees. The four-tree depth-12 case gains 6.56×. These are
structural fixtures, not claims about every trained model of a given depth.

Asymmetric, uneven-chain, and mixed forests select general traversal. At 5,000
rows their automatic and forced-generic medians differ by less than 3% in this
run. They execute the same kernel; these small differences are measurement
variation, not evidence of a new asymmetric-tree optimization.

Leaf-only forests cache their ordered sum during construction. Their large
warm-call improvement removes repeated tree evaluation altogether: each call
checks inputs and writes/transforms the constant result. This is distinct from
stumps, which still evaluate their per-tree feature splits. The two stump cases
exercise both direct selection without bin reuse and binned selection with reuse.

## Depth and size interpretation

Each full depth-12 tree has 8,191 nodes; each depth-16 tree has 131,071. The
64-tree depth-12 and four-tree depth-16 configurations have approximately the
same expanded node count (524,224 and 524,284), but different numbers of path
decisions and leaf additions per row. Their absolute times do not isolate depth.
The four-tree depth-12/depth-16 pair holds tree count, feature count and border
budget fixed. No Balsa limits were increased.

The depth-16 correctness test separately visits every one of the 65,536 leaf
addresses in two trees, using both direct and binned execution, float32 and
float64, and a partial final SIMD tile. A depth-17 fixture verifies fallback.

## Protocol and scope

- All fixtures are built in Mojo with Balsa builders. Input values use deterministic
  SplitMix64 with explicit ties, NaNs, infinities and signed zero. No CatBoost or
  other trainer metadata is present. This is not an accuracy benchmark.
- Families span constant roots, different-feature stumps, symmetric/asymmetric
  trees, uneven chains, and mixed forests. Both dtypes and batches 1/32/256/5000
  are measured. Selected routes and node counts are saved in the gates.
- Ryzen 9 5950X, affinity `{2}`; governor/frequency policy unchanged, sibling CPU
  and desktop processes not reserved. No compilation or other benchmark overlaps.
- Each interval uses a fresh native worker, builds and prepares the model/input,
  then performs three warmups before its native timer. The Python harness never
  participates in framework inference.
- Controls select validated private strategies on the same preparation path and
  retain the same storage allocation sequence as automatic execution. They do
  not copy or repack a model just to force a strategy. An earlier interrupted
  exploratory run with extra control copies was discarded; none of its timings
  enter these tables.
- The single preparation-duration observation per gate excludes the source-model
  copy and includes validation/packing/cache construction. It is diagnostic, not
  a calibrated constructor benchmark.
- The current large-batch bin policy benefits the measured eligible cases. This
  sweep does not establish optimal small-batch cutoffs or tune concurrency.
- Public inference signatures, NuMojo ownership, GPU paths and supported model
  semantics are preserved. The original Balsa model remains resident alongside
  packed storage. Constant and symmetric paths are serial.

[All medians](timings.md), [raw intervals](timings.jsonl), [oracle gates](gates.json),
and [source/worker hashes and configuration](provenance.json) are retained.
The provenance records the pre-commit HEAD plus the actual worktree source hashes.
See [CPU planning and reproduction](../../cpu-planning.md) for the implementation
seam, fixture families and commands.

## Existing CatBoost workload confirmation

A focused post-refactor check uses the original 5,000-row, 34-feature,
1,024-depth-8-tree trained fixture. Margins, probabilities and labels match
archived validated outputs exactly; live CatBoost tolerance and label checks
also pass. Automatic margins take 7.612 ms float32 and 8.841 ms float64,
versus 103.256 and 119.557 ms for forced general traversal (13.56× and 13.52×).
Native ndarray margin calls take 5.317 and 5.303 ms.

This is a short confirmation (one randomized round, three samples per cell,
50 ms calibration), not a replacement for the prior 15-sample investigation or
proof of a small performance change versus its historical capture. All three
output calls and both precisions are recorded in [medians](catboost-check/timings.md),
[raw intervals](catboost-check/timings.jsonl), [gates](catboost-check/gates.json),
and [provenance](catboost-check/provenance.json). The confirmation ran after the
structural sweep, with no overlapping measurement or compilation.
