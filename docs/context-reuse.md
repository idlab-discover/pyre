# CPU execution-context reuse

Pyre retains one lazy CPU context per forest and leases it atomically for each
threaded prediction. A busy cache causes the caller to create an independent
temporary context. This preserves concurrent predictions and nested executor
calls without a lock held across task completion. Context lifetime and the
MAX API details are described in [executor notes](executor-notes.md).

The comparison baseline is `eab1081`, containing both prior optimizations:
SIMD-width selection (`552c643`) and LT specialization (`eab1081`). Before and
after use the same default SIMD width, checkpoints, input bytes, and worker
limits. Raw captures, reproducible scripts, binary/source hashes, and output
comparisons are in the locally ignored `this-vs-nvforest/context-reuse/` folder.

## Allocation results

On the 4,096-row trained XGBoost case with eight workers, ten warm predictions
averaged 61 system allocations / 2,289 bytes before, versus 5.9 allocations /
324.8 bytes after. All 55 allocations attributed to CPU context creation per
prediction disappear from the warm region. Allocations and frees balance; the
cached context and its small state allocation are retained until the forest is
destroyed. Fractional counts reflect occasional runtime bookkeeping.

The logical Mojo allocator still makes 14 allocations per threaded prediction
and five per serial prediction. Those include output/NDArray metadata and task
state; this change does not remove them. Counts and bytes scale exactly from
10 to 100 calls, with zero region allocations outstanding. Serial system heap
traffic remains zero for both measured cases.

Memcheck reports no invalid accesses or definitely, indirectly, or possibly lost
bytes in any of six captures. The final three after-change captures report zero
errors; one baseline capture reports an uninitialized-value condition inside the
runtime allocator while reading input, before prediction measurement. Earlier
runs of both versions also showed runtime allocator warnings during model load.
The unsuppressed logs distinguish those warnings from prediction behavior.

## Native timing

Seven fresh processes per case, pinned to CPUs 8–15 on the Ryzen 5950X,
measured warm allocate/predict/free without simultaneous builds or Valgrind.
The eight-worker medians below include their observed min–max ranges (µs).

| Case | Before | After | nvForest |
|---|---:|---:|---:|
| batch-f32 | 758.20 (646.46–995.63) | 643.95 (620.02–935.65) | 487.82 (484.55–505.11) |
| batch-f64 | 903.22 (879.51–1221.74) | 917.19 (867.87–1144.56) | 569.29 (563.37–571.20) |
| trained-xgboost | 170.85 (82.52–212.89) | 108.77 (93.22–184.14) | 18.94 (18.73–19.26) |

Median changes were −15.1% for float32 batches, +1.5% for float64 batches,
and −36.3% for trained XGBoost. Ranges overlap substantially; these observations
do not establish a consistent latency gain across workloads. Serial medians
changed by less than 0.7%. The large remaining gap to nvForest on the trained
threaded case shows that removing context construction does not eliminate
MAX submission/join overhead and scheduling variability.

## Correctness and remaining costs

`pixi run check` passes 25 tests, including repeated concurrent predictions,
nested executor reentry, moving a forest after context warmup, all operators,
classification, precisions, widths, and thread controls. All seven original
case/thread combinations match pinned nvForest bitwise before and after.

Callgrind confirms that warm context creation disappears. Serial instruction
counts change by only four instructions per call. Threaded instruction totals
include serialized worker waiting under Valgrind and must not be interpreted as
native execution time.

The cache is bounded to one context. Repeated overlapping callers can still
allocate temporary contexts. A bounded pool may help a measured concurrent
serving workload, but all callers must not join the same active stream. Output
buffer reuse and reducing task submission allocations are separate future work.
