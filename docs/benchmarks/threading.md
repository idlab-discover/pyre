# Thread-only inference measurements

Same native timer, cases and affinity as the baseline. Three independent processes
with five values each; standard deviation and pyperf warnings remain in JSON.
Concurrent desktop and Balsa load limits interpretation of small differences.

All latencies below are microseconds (µs).

| Case | Original | 1 thread | 2 threads | 4 threads | 8 threads | Original / 8 |
|---|---:|---:|---:|---:|---:|---:|
| trained-xgboost | 375.073 | 387.656 | 200.334 | 121.449 | 218.335 | 1.72× |
| tiny-f32 | 0.668 | 0.675 | 0.683 | 0.653 | 0.648 | 1.03× |
| tail-f32 | 94.533 | 95.397 | 56.990 | 35.836 | 35.937 | 2.63× |
| batch-f32 | 12095.253 | 11807.745 | 6699.506 | 2963.734 | 1739.431 | 6.95× |
| large-f32 | 134219.221 | 134649.050 | 77685.362 | 39400.199 | 20159.502 | 6.66× |
| batch-f64 | 11483.968 | 12245.231 | 6663.566 | 3129.815 | 2029.572 | 5.66× |
| deep-f32 | 21530.864 | 21890.118 | 12528.069 | 6269.548 | 3705.608 | 5.81× |
| binary-f32 | 11374.493 | 12073.354 | 6062.827 | 4000.213 | 2051.420 | 5.54× |
| scalar-f32 | 12361.138 | 13062.832 | 6310.182 | 3936.139 | 2173.279 | 5.69× |
| wide-f32 | 11049.414 | 11612.900 | 6008.275 | 3442.981 | 1684.446 | 6.56× |

## Warm crossover sweep

`tools/benchmark_crossover.py` randomizes seven independent native processes per
combination; each measures 100 warm predictions. These supplemental medians are
not pyperf confidence/significance results. Model: 64 depth-six float32 trees,
width eight, CPUs 8–15. Each entry is µs per prediction.

| Rows | 1 thread | 2 threads | 4 threads | 8 threads |
|---|---:|---:|---:|---:|
| 128 | 354.2 | 196.1 | 106.1 | 86.9 |
| 256 | 701.7 | 430.3 | 223.6 | 132.2 |
| 512 | 1431.1 | 833.2 | 408.6 | 236.7 |
| 1024 | 2845.1 | 1457.7 | 739.1 | 476.5 |

All tested warm crossover sizes benefited on this host. Automatic policy remains
conservative: at least 256 rows and 32768 row/tree pairs, with up to eight
workers. For 64 trees this activates at 512 rows. First-call pool initialization
can dominate small batches; callers prioritizing latency can force one thread.

## First prediction

The supplemental cold worker measures the first prediction after model/input
preparation, before any warmup. It then reports the warm mean of 100 predictions.
Seven independently launched processes per case were randomized; medians in µs:

| Rows | Serial first | 8-thread first | Serial warm | 8-thread warm |
|---|---:|---:|---:|---:|
| 128 | 362.0 | 300.6 | 353.2 | 72.2 |
| 512 | 1433.3 | 429.6 | 1400.5 | 236.5 |
| 4096 | 11483.5 | 1828.2 | 11131.2 | 1786.5 |

Pool creation and initially cold inference add about 200 µs for the smaller
threaded cases. These are whole first predictions, not isolated executor-startup
measurements. Process launch and model preparation remain excluded. Raw samples
are in `threaded-cold.json`; the instrumented source is retained in
`build/threaded/cold_worker.mojo`. `tools/benchmark_cold.py` repeats its sweep.
