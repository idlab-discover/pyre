# Final inference benchmarks

All times are microseconds (µs), means across 3 processes × 5 values.
Two warmups and calibrated loops with minimum 0.1 s per sample; CPUs 8–15.
`auto` uses the public default thread policy. Model/input setup is excluded,
while result allocation and requested probability transforms are included.

| Case | Original | Final serial | Final 8 threads | Final auto | Original / auto |
|---|---:|---:|---:|---:|---:|
| trained-xgboost | 375.073 | 147.692 | 179.300 | 151.189 | 2.48× |
| tiny-f32 | 0.668 | 0.457 | 0.452 | 0.454 | 1.47× |
| tail-f32 | 94.533 | 37.785 | 18.277 | 37.667 | 2.51× |
| batch-f32 | 12095.253 | 4502.689 | 901.672 | 868.617 | 13.92× |
| large-f32 | 134219.221 | 58186.773 | 9424.583 | 9494.633 | 14.14× |
| batch-f64 | 11483.968 | 6530.653 | 1088.975 | 1176.617 | 9.76× |
| deep-f32 | 21530.864 | 9787.201 | 1738.421 | 1836.260 | 11.73× |
| binary-f32 | 11374.493 | 4514.521 | 891.253 | 927.676 | 12.26× |
| scalar-f32 | 12361.138 | 10365.215 | 1533.136 | 1548.256 | 7.98× |
| wide-f32 | 11049.414 | 3733.959 | 664.039 | 712.322 | 15.51× |

## SIMD-stage contribution

Holding thread count at one separates SIMD traversal changes from threading.
Values below compare the thread-only implementation with the final optimization
stage: pointer gathers, a singleton scalar fast path, and fused vector
postprocessing. This stage also fixes Float64 base-score addition before narrowing
float32 output, verified against Treelite 4.6.1. The ratios measure these combined
changes.

| Case | Thread-only serial | Final serial | Speedup |
|---|---:|---:|---:|
| trained-xgboost | 387.656 | 147.692 | 2.62× |
| tiny-f32 | 0.675 | 0.457 | 1.48× |
| batch-f32 | 11807.745 | 4502.689 | 2.62× |
| large-f32 | 134649.050 | 58186.773 | 2.31× |
| batch-f64 | 12245.231 | 6530.653 | 1.88× |
| deep-f32 | 21890.118 | 9787.201 | 2.24× |
| binary-f32 | 12073.354 | 4514.521 | 2.67× |
| wide-f32 | 11612.900 | 3733.959 | 3.11× |

Final traversal uses pointer-gather APIs to assemble node/input lane vectors and
performs packed comparisons, selections and accumulation. Assembly inspection
found packed AVX operations, but no hardware `vgather` instructions: the compiler
lowered gathers to scalar loads and packing on this host. Tail input/leaf accesses
are masked. Binary transforms operate on vectors and classification avoids a
temporary two-column probability array.

The host was shared with concurrent Balsa benchmarks and desktop activity.
Some samples have significant jitter; inspect standard deviations/warnings in
the raw pyperf files before interpreting small differences. Thread scaling and
SIMD gains are workload- and machine-specific.

## Supplemental crossover and first-call latency

Randomized seven-process warm medians, 64 depth-six float32 trees, µs:

| Rows | 1 thread | 2 threads | 4 threads | 8 threads |
|---|---:|---:|---:|---:|
| 128 | 152.4 | 89.7 | 48.7 | 49.7 |
| 256 | 301.2 | 162.6 | 88.6 | 71.9 |
| 512 | 600.2 | 317.0 | 165.6 | 116.2 |
| 1024 | 1196.1 | 613.1 | 318.8 | 178.8 |

Separate first-call sweep, median first / warm latency in µs:

| Rows | Serial first / warm | 8-thread first / warm |
|---|---:|---:|
| 128 | 154.3 / 149.5 | 276.2 / 161.6 |
| 512 | 600.9 / 565.1 | 330.8 / 247.0 |
| 4096 | 4779.3 / 4763.3 | 852.0 / 735.9 |

The separate cold sweep also experienced notably slower small threaded warm
samples than the crossover run. This shared-host variability reinforces the
conservative automatic threshold: cold forced threading loses at 128 rows in
this run. The default selects serial for that case. Raw samples are in
`final-crossover.json` and `final-cold.json`; these supplemental medians are not
pyperf significance tests.

## Process memory

`tools/benchmark_memory.py` records Linux `wait4.ru_maxrss`, the whole-process
peak RSS metric used by `/usr/bin/time`. Three fresh processes per combination
perform three warmups and 100 predictions. Models have 4096 input rows, 64
depth-six float32 trees. Observed min–max peak RSS (MiB):

| Implementation | Margin | Binary probabilities |
|---|---:|---:|
| Original, serial | 19.68–19.89 | 19.60–19.89 |
| Thread-only, 8 threads | 19.63–19.68 | 19.44–19.77 |
| Final, 8 threads | 17.66–19.70 | 17.68–19.83 |

This includes executable/runtime pages, model construction, input/output storage,
allocator behavior and executor state; it does not isolate inference allocations
or reserved virtual thread-stack memory. No material process-RSS growth appeared
in these bounded cases. The ranges do not establish an allocation reduction.
Raw records are in `memory.json`.

## Correctness

The example and package compile, and all 20 native tests pass. Checks cover the
12 saved Treelite reference fixtures, both precisions, every numerical operator,
missing values, SIMD widths 1/4/8/16, tails and empty/offset inputs, zero-feature
root-only forests, deeply divergent paths, sigmoid extremes, invalid controls,
CPU accelerator adapters, and concurrent calls sharing one forest. All saved
fixture hashes match the pinned Balsa commit.
