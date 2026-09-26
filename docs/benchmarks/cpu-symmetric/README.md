# Integrated compact CPU measurements — 2026-09-20

Historical capture: implementation committed as `1a749e7`, before the private
CPU-plan refactor. See [current planning and synthetic fixtures](../../cpu-planning.md)
for the subsequent work.

On the captured CatBoost numerical models, automatic compact inference retains
NuMojo ownership and substantially reduces serial prediction latency. The
5,000-row, 34-feature, 1,024-depth-8-tree float32 workload falls from **104.604 ms
to 7.310 ms (14.31×)**. Float64 falls from **128.140 ms to 8.384 ms (15.28×)**.
Native CatBoost ndarray calls are 5.211 ms and 5.202 ms respectively. This closes
most of the measured gap without changing input ownership or the public API.

## Large batches

Median milliseconds per allocating margin call, 5,000 rows; 15 measurements
across three randomized rounds. `auto` includes tile allocation and binning.

| Model | Dtype | Generic | Compact direct | Auto | Native array | Generic / auto |
|---|---|---:|---:|---:|---:|---:|
| f34-d4 | float32 | 48.821 | 15.022 | 5.129 | 3.722 | 9.52× |
| f34-d4 | float64 | 56.384 | 17.818 | 5.367 | 3.731 | 10.51× |
| f34-d8 | float32 | 104.604 | 25.720 | 7.310 | 5.211 | 14.31× |
| f34-d8 | float64 | 128.140 | 31.273 | 8.384 | 5.202 | 15.28× |
| f34-d8-t64 | float32 | 6.152 | 1.600 | 0.609 | 1.729 | 10.10× |
| f34-d8-t64 | float64 | 7.301 | 1.893 | 0.688 | 1.729 | 10.62× |
| f34-d8-t256 | float32 | 24.692 | 6.379 | 1.998 | 2.473 | 12.36× |
| f34-d8-t256 | float64 | 28.853 | 7.682 | 2.358 | 2.462 | 12.24× |
| f82-d8 | float32 | 106.843 | 26.330 | 7.719 | 7.264 | 13.84× |
| f82-d8 | float64 | 130.890 | 32.389 | 8.972 | 7.391 | 14.59× |

Models have 1,024 trees unless suffixed `t64` or `t256`; `f` denotes
feature count and `d` tree depth. Prepared NuMojo input is reused for Pyre;
native ndarray calls include Python/API ingestion overhead. A smaller Pyre time
on small models does not establish a faster native traversal kernel.

## Batch crossover

Float32, 34 features, 1,024 depth-8 trees:

| Rows | Generic ms | Direct ms | Auto ms | Native array ms |
|---:|---:|---:|---:|---:|
| 1 | 0.090 | 0.027 | 0.027 | 0.104 |
| 32 | 0.764 | 0.171 | 0.171 | 0.174 |
| 256 | 5.520 | 1.319 | 0.375 | 0.403 |
| 1024 | 22.108 | 5.237 | 1.507 | 1.163 |
| 5000 | 104.604 | 25.720 | 7.310 | 5.211 |

Automatic dispatch uses direct comparisons below 256 rows. These results
support the initial policy for this corpus; they do not establish its optimality
for other distributions, tree structures, feature counts, or CPUs.

## Probability and class confirmation

A separate randomized sweep of the main 5,000-row case confirms the gain for
other output APIs. Float32 probabilities improve from 104.017 to 7.219 ms and
classes from 107.931 to 7.226 ms; native ndarray calls take 5.285 and 6.050 ms.
Float64 probabilities improve from 135.506 to 8.298 ms and classes from 125.801
to 8.545 ms. These are independent measurements, so small differences from the
margin sweep should not be attributed entirely to output transformation.

[Call medians](calls/timings.md), [raw intervals](calls/timings.jsonl),
[gates](calls/gates.json), and [provenance](calls/provenance.json) are retained.

## Runtime profile

Fixed-period user-cycle profiles confirm `compact=2` on automatic inference and
`compact=0` on the generic control. Both spend approximately 99.9% of sampled
cycles inside the inference worker; no lost samples were reported. Binning and
tree evaluation are inlined there, so these profiles do not separate their costs.

Delayed counters cover similar-duration windows: 45.947 billion cycles and
119.181 billion instructions for the 100-call generic run, versus 47.495 billion
cycles and 174.269 billion instructions for the 1,500-call automatic run. Both
counters ran 100% of the enabled time. The first second is excluded, so these
are diagnostic window totals, **not exact per-call instruction counts**.

[Generic flat profile](profile/generic-flat.txt),
[automatic flat profile](profile/auto-flat.txt),
[generic counters](profile/generic-counters.csv),
[automatic counters](profile/auto-counters.csv), and
[commands/provenance](profile/provenance.json) are retained. Binary recordings
remain under `build/symmetric-profile` and can be reproduced with the profiling
script. Sampling and counter runs were separate and did not overlap compilation
or benchmarks.

## Correctness and protocol

- All 54 kernel/model/dtype/batch gates match the archived, previously GTIL-validated
  baseline byte-for-byte for margins, two-column probabilities, and labels.
- All 18 model/dtype/batch cases also pass live CatBoost margin/probability
  tolerances and exact-label checks before timing.
- `pixi run check` passes; the expanded symmetric suite passes six tests. The
  complete CPU suites contain 35 tests, plus five Python memory-accounting tests.
- Ryzen 9 5950X, CPU affinity `{2}`, serial evaluation; sibling CPU and other
  desktop processes were not reserved. Governor/frequency policy was not changed.
- Mojo 1.0.0, NuMojo 0.10.0, MAX core 26.5.0; optional native references use
  CatBoost 1.2.10 and NumPy 2.5.3 from the investigation environment.
- Each Mojo interval is a fresh worker, with loading, validation, packing, input
  preparation and three warmups outside its timer. Prediction checks, allocation,
  bin preparation, inference, transformation and result disposal are inside.
- Native intervals reuse a model and ndarray in the Python harness process, with
  three warmups per interval. Each interval is calibrated to about 100 ms; actual
  durations and loop counts are retained. No compilation overlaps measurement.
- Generic/direct controls select private internal flags only in the benchmark
  worker. The shipping API exposes no kernel override. Packing still occurs
  before timing in all controls; these are warm inference comparisons.
- The baseline Balsa model remains resident alongside packed NuMojo arrays.
  These timings make no memory-reduction or concurrency-performance claim.
- Depth-9/10 full-size investigation models exceed unchanged loader limits;
  they are excluded from this timing sweep. Smaller synthetic depth-9/10/16
  trees are covered by correctness tests.

Full results: [all median cells](timings.md), [raw intervals](timings.jsonl),
[correctness gates](gates.json), [source/input hashes and provenance](provenance.json).
The provenance records the pre-commit HEAD and worktree source hashes used for
measurement. Original checkpoints and raw inputs remain in the investigation
directory; their hashes are recorded, not their large binary contents.

See [design and reproduction](../../cpu-symmetric.md) for commands and scope.
