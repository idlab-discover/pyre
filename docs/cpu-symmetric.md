# Compact CPU symmetric-tree inference

The feature branch integrates the compact-tree and shared-binning experiments
from the September 20, 2026 CatBoost investigation into the ordinary `Forest`
constructor and prediction API. Framework construction, packing, preparation,
and inference are pure Mojo. Input, output, packed model arrays, and byte scratch
remain owned by NuMojo. Python is only benchmark orchestration and optional
CatBoost reference evaluation.

## Representation and semantic boundary

After Balsa structural validation, `_Symmetric` recovers each tree level using
child links rather than assuming serialized node order. All nodes at a depth
must share feature, threshold, LE operator, and missing direction. Every path
must terminate at the same depth within that tree; different trees may have
different depths, including zero. Finite thresholds and depth at most 16 are
required. An ineligible tree leaves the entire forest on the general evaluator;
there is no partial packing or change to tree summation order.

Packed splits have one entry per level. Leaf bit `d` selects the right edge at
depth `d`; construction reorders leaves accordingly. Features, thresholds,
defaults, offsets and leaves use typed NuMojo arrays. Thresholds and leaves keep
the original float32/float64 precision. Float64 borders are never narrowed.
Per-call scratch is local; no prepared-input cache or mutable shared bins exist.

For each feature, construction deduplicates and sorts used borders. With at most
254 borders per feature, a byte holds the number of borders strictly below an
input value. Byte 255 denotes NaN; each split applies its own missing direction.
Infinities and exact ties therefore retain LE semantics. More than 254 borders
on any feature disables binning while preserving compact direct evaluation.

Automatic binning currently requires at least 256 rows and fewer unique borders
than split occurrences. It uses feature-major tiles of 256 rows (or one vector
for explicit widths above 256), prepares each tile once, and evaluates four-tree
groups across its SIMD row blocks. This is a conservative initial policy, not a
hardware-independent optimal crossover. All predictions include input checks,
result allocation, preparation, ordered accumulation, and the existing averaging,
base-score, sigmoid and class-threshold handling.

The original Balsa model remains owned alongside packed arrays. This preserves
metadata and the generic fallback, but increases resident memory for eligible
forests. The benefit is a smaller *hot* representation and less dependent work;
this branch makes no total-memory reduction claim. Constructor packing and
validation are not part of warm prediction timings.

## API, ownership, concurrency, and GPU scope

`Forest(model^)`, all four inference APIs, dtype matching, explicit SIMD widths,
output shapes, contiguous offset views, and argument validation remain supported.
The compact evaluator executes serially. Nonnegative `num_threads` values remain
accepted but do not schedule compact workers; negative values still raise.
Existing generic traversal and its scheduler are retained. CPU concurrency policy
and redesign belong to a subsequent session. GPU files and shared validation are
unchanged.

NuMojo ownership does not prevent these optimizations. Owned arrays can hold
compact split metadata, reordered leaves, and contiguous feature-major byte
scratch, with borrowed pointers used only during evaluation. The investigation's
large gains came from a combination of representation, split reuse, and access
layout; it did not isolate their individual contributions. Keeping the owner is
compatible with changing the internal layout. Matching native CatBoost's exact
latency is a measurement question, not an ownership guarantee.

## Measurements

The [integrated sweep](benchmarks/cpu-symmetric/README.md) records a 14.31×
float32 and 15.28× float64 speedup over generic traversal on the main 5,000-row,
34-feature, 1,024-depth-8-tree workload, including binning. See its raw intervals
and provenance for limits on interpretation.

## Validation and reproduction

```sh
pixi run check
pixi run benchmark-symmetric-build
pixi run python tools/benchmark_symmetric.py \
  /path/to/catboost-investigation/investigation build/symmetric-sweep
```

The artifact directory is read-only. The output directory must not already
exist. The harness pins one allowed CPU (default 2; override with `--cpu`), checks
all three output types against archived validated baseline bytes, then calibrates
intervals to at least 100 ms. It randomizes kernel/case order across three rounds
of five measurements. Workers load, pack, prepare inputs and warm before timing;
complete allocating prediction calls and result disposal are timed. Both direct
and binned preparation are inside the measured inference calls.

Use `--calls margin proba class` to sweep output APIs and `--native` from a Python
environment containing CatBoost and NumPy to interleave native ndarray calls and
check native numerical tolerances. The default cases span 34/82 features,
64/256/1024 trees, depth 4/8, float32/float64, and available batches of
1/32/256/1024/5000. Depth 9/10 investigation artifacts require increased Balsa
loader limits and are deliberately excluded; this branch does not change limits.

`tests/test_symmetric.mojo` checks direct/binned/generic equality, original tree
order, constant and varying-depth trees, both dtypes, exact ties and adjacent
values, NaNs/defaults, infinities, signed zero, nontrivial base score and sigmoid
scale, averaging, classification thresholds, input offset views, tile/vector
tails, explicit widths, eligibility fallback, byte capacity overflow, arbitrary serialized node numbering,
and depths 9, 10, and 16.
Existing acceptance, specialization and parallel suites remain part of `check`.

For flat CPU profiles and delayed counters on the same 34-feature workload:

```sh
pixi run python tools/profile_symmetric.py \
  /path/to/catboost-investigation/investigation build/symmetric-profile
```

This requires `perf` access to user-mode hardware events. It runs generic and
automatic kernels serially, records fixed periods of five million user cycles,
and omits the first second. It deliberately avoids frame-pointer callgraphs.

Raw sweep output includes gates, model/input/reference/worker SHA256 hashes,
affinity, native package versions when applicable, individual timings, and a
median table. Optional profiling can run the same worker with a long loop count
under `perf stat` or fixed-period `perf record`; do not overlap compilation or
other benchmark runs with measurement. Broader workload and concurrency sweeps
remain future policy-tuning work.
