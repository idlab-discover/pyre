# Inference benchmark protocol

`tools/prepare_balsa.py` reads Git commit
`737874f3eae9d95a0517397130f445b303d28cc6` using `git archive`, never Balsa's
working tree. All compilation, fixture loading and oracle generation use this
snapshot. The baseline passed all eight existing tests against this API before
inference changes.

Run from the Pyre directory:

```sh
pixi run benchmark-build
pixi run python tools/benchmark.py --processes 3 --values 5 --warmups 2 \
  --min-time .1 --affinity 8-15 -o docs/benchmarks/baseline.json
python -m pyperf compare_to docs/benchmarks/baseline.json OTHER.json --table
```

pyperf calibrates native loop counts and collects five values in each of three
independent worker processes, with two discarded warmups. Each callback launches
a native executable that reports only its internal monotonic timer around
repeated predictions. Process startup, model building/loading, validation,
input generation and three initial warm predictions are excluded. Output
allocation, traversal and requested transforms are included. `keep` prevents
unused results being optimized away. Inputs and models are deterministic.

Cases cover one row, a SIMD tail, 4096/16384 row batches, depths 3/6/8/10,
float32/float64, binary probabilities, explicit widths 1/4/8/16, and the pinned
trained XGBoost regression fixture. Synthetic complete trees have eight
features, divergent paths, varying thresholds/operators, missing values,
and nonconstant leaves. The trained case uses the fixture's four depth-three
trees; command-line depth zero is its load-fixture selector.

The original executable, worker and forest source are retained in ignored
`build/baseline`; JSON records executable/source hashes so recompilation cannot
silently masquerade as the original. To repeat that binary pass
`--binary build/baseline/benchmark-worker`. Copy the source next to any saved
binary as `forest.mojo` for correct source provenance.

`environment.json` records compiler, topology and background load. Affinity
8–15 selects eight distinct physical cores on this host. Concurrent Balsa
benchmarks and desktop processes were active; neither cores nor power budget
were exclusive. pyperf reports insufficient samples for <1% stability. These
multi-process results establish comparative evidence on this machine, not
universal throughput guarantees; repeat or expand sampling before relying on
small differences. No privileged governor/boost changes were made.

## Original implementation

| Case | Mean ± standard deviation |
|---|---:|
| Trained XGBoost, 4096 rows | 375 ± 29 µs |
| One row, 16 depth-three trees | 668 ± 21 ns |
| 33 rows, 64 depth-six trees | 94.5 ± 2.4 µs |
| 4096 rows, 64 depth-six trees, float32 | 12.1 ± 1.1 ms |
| 16384 rows, 128 depth-eight trees | 134 ± 3 ms |
| 4096 rows, 64 depth-six trees, float64 | 11.5 ± 0.2 ms |
| 4096 rows, 64 depth-ten trees | 21.5 ± 0.3 ms |
| Binary probabilities, 4096 rows | 11.4 ± 0.1 ms |
| Explicit scalar width, 4096 rows | 12.4 ± 0.1 ms |
| Explicit width 16, 4096 rows | 11.0 ± 0.2 ms |

The float32 width-eight baseline is only marginally faster than scalar width
on this synthetic workload. Traversal still fills SIMD lanes individually;
parallel comparisons alone do not establish an effective vectorized traversal.

## Rebuild every stage after a clean checkout

Archived source (no binaries) lives in `sources/baseline` and `sources/threaded`.
The final stage uses the current implementation. From the Pyre directory:

```sh
pixi run python tools/build_benchmark_stage.py baseline
pixi run python tools/build_benchmark_stage.py threaded
pixi run python tools/build_benchmark_stage.py final
pixi run python tools/benchmark.py --binary build/threaded/benchmark-worker \
  --threads 8 --processes 3 --values 5 --warmups 2 --min-time .1 \
  --affinity 8-15 -o build/repeated-threaded-8.json
```

The stage builder isolates the archived `Forest` module and compiles it against
the pinned Balsa API and current locked compiler/dependencies. The original
baseline used the same Mojo/MAX runtime versions. Existing `max-core==26.5.0`
already supplied `max.algorithm`; no MAX meta-package was added. Python/pyperf
were added before baseline measurement. Numerical comparisons
should still use the original JSON because machine/compiler conditions can vary.

Cold timing instrumentation is generated explicitly by the same builder:

```sh
pixi run python tools/build_benchmark_stage.py threaded --cold
pixi run python tools/benchmark_cold.py --binary build/threaded/cold-worker \
  --output build/repeated-threaded-cold.json
pixi run python tools/benchmark_crossover.py \
  --binary build/threaded/benchmark-worker --output build/repeated-crossover.json
```

Use `final` instead of `threaded` to reproduce the final cold timer. The first
prediction is inserted before the warmup marker and its elapsed time is reported
separately from the repeated warm predictions. Benchmark-stage snapshots preserve
original source formatting; the initial final executable was compiled just before
a formatting-only source pass, so its source hash identifies equivalent formatted
source rather than byte-identical compiler input.
