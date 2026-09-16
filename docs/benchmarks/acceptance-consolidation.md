# Prediction acceptance consolidation — 2026-09-16

The branch shares prediction expectations and edge cases across CPU and GPU,
removes one support module, and expands coverage without changing production
inference. Same-path builds of the pre-refactor and candidate native benchmark
workers are **byte-identical**, for both CPU and GPU. This is a test refactor,
not an inference speedup.

Comparison revisions:

- **main:** `830525d1fa8724558f9ee1279bc02e5b0eedddef`.
- **pre-refactor:** `2d6b803912360da4f2f8ce1753504109c8c11ddb` (`cpu-validation`).
- **candidate:** the `acceptance-consolidation` commit containing this report,
  forked from pre-refactor. No GPU experiment revision was integrated.

The candidate inherits pre-refactor's four-worker intermediate CPU tier. The
64% reduction in intermediate automatic-call latency against main belongs to
that earlier change, not to acceptance consolidation. Main was not modified.

## Design and size

`tests/acceptance.mojo` replaces `support.mojo` and `gpu_support.mojo` and absorbs
the operator and binary semantic runners from the CPU/GPU suites. Its CPU and
GPU execution functions return typed host predictions to one assertion runner.
CPU model ownership, scalar/default comparison and thread controls stay in the
CPU adapter. GPU preparation, context lifetime, upload, completed readback and
reusable output checks stay in the GPU adapter. Neither production path depends
on the acceptance module.

One stump builder replaces three. One independent branch-expectation function
serves the shared matrix and explicit-width/thread tests. Threaded CPU binary
checks now use the shared binary cases. Device-specific aliasing, capacity,
offset, lifetime, context and join-on-error tests remain separate. The GPU
prepare-then-mutate and move check moved into the context ownership test.
Single-tree CPU numeric-extreme fixtures retain their single-tree construction;
averaged two-tree cases are explicit in the shared fixture builder.

The generated reference interface now supplies only checkpoint name, inputs,
saved margins, saved predictions and tolerance. Feature count and task type
come from the loaded checkpoint. The generator and every caller changed together;
there are no compatibility forwarding helpers. All 12 checkpoints, input arrays,
GTIL numerical expectations and tolerances are retained. Regeneration corrected
the manifest's stale Balsa commit; every checkpoint hash remained unchanged.

Inventory counts all `.mojo` and `.py` files under `src`, `examples`, `tests` and
`tools`. Documentation, raw measurement data and ignored build artifacts are
not implementation code. Generated data is reported separately.

| Inventory | main | pre-refactor | candidate |
| --- | ---: | ---: | ---: |
| Handwritten files | 38 | 38 | 37 |
| Handwritten lines | 4,740 | 4,780 | 4,760 |
| Handwritten nonblank lines | 4,326 | 4,362 | 4,342 |
| Handwritten bytes | 197,381 | 199,256 | 197,628 |
| Generated reference lines | 1,306 | 1,306 | 1,282 |
| Generated reference bytes | 31,011 | 31,011 | 30,705 |
| Combined source lines | 6,046 | 6,086 | 6,042 |
| Combined source bytes | 228,392 | 230,267 | 228,333 |

The size reduction is modest: the refactor removes 20 handwritten lines and
1,628 handwritten bytes, while adding diagnostics and coverage. Against main,
handwritten code is 20 lines larger because of the inherited CPU optimization;
combined source is four lines smaller. The main benefit is concentrated
prediction semantics, not a large line-count reduction. This report and its
raw evidence increase documentation size.

## Correctness and coverage

| Coverage | Previous | Candidate |
| --- | --- | --- |
| GTIL cases | 12 per backend | Same 12, with shape checks and class thresholds 0/0.5/1 |
| Operator cases in core suites | CPU 340 f32; GPU 80 across f32/f64 | 720 per backend across both precisions |
| Operator batch lengths | CPU 0–33; GPU 0/1/33/257 | Both: 0–33, 129, 257 |
| Binary tie/extreme cases | CPU 3 f32; GPU 24 across f32/f64 | 30 per backend, exact probabilities, all three thresholds |
| Threaded binary cases | 20 f32 tie cases | 120 across both precisions, ties and extremes |
| GPU probability reuse | Positive column | Both columns, sentinel-filled outputs |
| CPU scalar/default comparison | GTIL margins | Every shared margin case, including threaded binary cases |

Each core backend runs 762 cases: 720 operator/default/length cases, 30 binary
cases, and 12 GTIL cases. Operator expectations remain independent comparisons;
GPU-to-CPU agreement is not their oracle. Shared inputs include NaNs, infinities,
signed zero and exact thresholds; GTIL trained cases retain adjacent representable
threshold inputs. Margins/regression retain scaled absolute tolerances; GTIL
probabilities retain strict absolute tolerances (f32 `1e-6`, f64 `1e-12`). Binary
edge probabilities remain exact, complements exact, and labels remain Int64
with strict `saved_probability > threshold` semantics.

Final verification:

- `pixi run check`: build/package and **28 CPU tests passed** (7 + 15 + 6).
- `pixi run test-gpu`: **10 GPU tests passed**, plus transfer and example tasks.
- Frozen main: **28 CPU tests passed** (8 + 15 + 5).
- Frozen pre-refactor: **29 CPU and 13 GPU tests passed**, plus transfer/example tasks.
- Two consecutive generations with Treelite 4.6.1 / NumPy 2.2.6 were byte-identical.
- An additional exact-comparison probe rejects `Int64.MIN` labels without
  subtraction overflow.
- Six real CPU/GPU rejection probes failed as intended: wrong saved margin,
  wrong probability, and a probability inside tolerance but across the strict
  class threshold. Diagnostics included case, backend, precision, operation,
  row, actual, expected and tolerance. Examples:

```text
rejection-probe GPU float64 margin row=0 actual=0.0 expected=1.0 tolerance=0.0
rejection-probe CPU float64 class threshold=0.5 row=0 actual=0 expected=1 tolerance=0
```

Fewer named tests reflect merged runners, not fewer cases. Detailed local logs,
rejection probe source and executables remain in `build/acceptance/` in this
worktree. No rejected implementation draft is retained in tracked source.

## Performance evidence

Hardware: Ryzen 9 5950X, RTX 5090, NVIDIA driver 610.57.04. Toolchain: Mojo 1.0.0
(`ed45d567`), MAX 26.5.0, NuMojo 0.10.0, identical Pixi lockfile. Balsa revision:
`737874f3eae9d95a0517397130f445b303d28cc6`.

All workers were rebuilt from frozen sources. Measurements ran sequentially,
with CPU affinity 8–15, alternating main/pre/candidate and reverse order, six
fresh processes per revision/case. No compilation or correctness tests ran
during timing. CPU loops target 0.25 seconds after three warm calls. GPU phases
warm and calibrate to at least 0.15 seconds; allocating and reused completed
calls are separate. Setup, upload, readback and oracle checks are outside those
inference timers. Production preparation is unchanged by this refactor.

There was no separate compute workload in the observed GPU process snapshots;
the desktop compositor/browser remained active. The first CPU pass exhibited
a common speed shift across revisions. Raw samples are retained rather than
filtered; no percentage regression allowance was used.

Median microseconds per call (lower is faster):

| Case | main | pre-refactor | candidate |
| --- | ---: | ---: | ---: |
| CPU tiny scalar | 0.535 | 0.538 | 0.538 |
| CPU tiny automatic | 7.995 | 7.958 | 7.999 |
| CPU intermediate automatic | 120.717 | 43.569 | 43.463 |
| CPU intermediate serial | 120.975 | 120.802 | 120.853 |
| CPU large automatic | 1710.116 | 1705.735 | 1723.647 |
| CPU large eight threads | 1542.643 | 1530.734 | 1531.791 |
| CPU large serial | 11787.900 | 11784.740 | 11784.276 |
| CPU f64 probability, first pass | 60.016 | 63.426 | 65.855 |
| CPU f64 probability, longer confirmation | 54.106 | 54.322 | 54.327 |
| GPU base, 33 rows, reuse | 139.000 | 141.097 | 140.506 |
| GPU base, 33 rows, allocate | 147.814 | 148.863 | 150.148 |
| GPU base, 4096 rows, reuse | 170.802 | 169.065 | 171.457 |
| GPU base, 4096 rows, allocate | 175.621 | 177.778 | 180.315 |
| GPU f64, 4096 rows, reuse | 247.480 | 243.870 | 246.856 |
| GPU f64, 4096 rows, allocate | 254.313 | 257.011 | 257.069 |
| GPU binary-model margins, 129 rows, reuse | 156.395 | 158.358 | 159.960 |
| GPU binary-model margins, 129 rows, allocate | 165.280 | 165.135 | 167.555 |

The GPU worker times margins, including on the binary model; it also validates
probabilities/classes outside the timer. This is representative inference
coverage, not a benchmark of every prediction operation. GPU desktop noise
limits small-difference interpretation.

The float64 confirmation used ten fresh processes per revision and 12,000 calls
per process. The apparent first-pass slowdown disappeared. More decisively,
rebuilding pre-refactor and candidate using the **same staging source and output
paths** produced identical complete executables:

| Worker | SHA-256, identical for pre-refactor and candidate |
| --- | --- |
| CPU | `59e2a8826ddc0e99a1e51efa1cd0bb6bcf2d1eba28901c62435885c82c8278de` |
| GPU | `19729e6443fa17df85fdbd0f663ac65628c406c80c8b519b65006b008718062c` |

Earlier builds used distinct source paths and have distinct binary hashes.
The controlled builds remove that confound. Together with an empty production
source diff from pre-refactor, binary identity establishes that this refactor
has not traded inference performance for consolidation. Small timing deltas
are not evidence of a test-refactor runtime cost or benefit.

## Reproduction

[Provenance](acceptance-provenance.json) records source inventories, model/oracle
hashes and controlled binary hashes. [Raw samples](acceptance-samples.csv) include all 288 first-pass samples and 30
confirmation samples, loop counts, durations and timestamps. Native worker
sources are unchanged: `tools/benchmark_worker.mojo` and
`tools/benchmark_gpu_worker.mojo`. Frozen snapshots and commands used locally
are in `build/acceptance/{main,pre}/`, `build_baselines.py`, `build_control.py`,
`measure.py` and `confirm.py`.

Correctness and regeneration from the candidate checkout:

```sh
pixi run prepare
pixi run check
pixi run test-gpu
uv venv --python .pixi/envs/default/bin/python build/oracle-reproduce
uv pip install --python build/oracle-reproduce/bin/python treelite==4.6.1 numpy==2.2.6
build/oracle-reproduce/bin/python tools/generate_references.py
git diff --exit-code -- tests/reference_data.mojo tests/fixtures/manifest.json
```

To rebuild any frozen revision without changing another worktree, extract its
archive into an empty staging directory; for example:

```sh
mkdir -p build/acceptance-reproduce
git archive 2d6b803912360da4f2f8ce1753504109c8c11ddb | tar -x -C build/acceptance-reproduce
pixi run mojo build -I build/acceptance-reproduce/src -I build/balsa-reference/src build/acceptance-reproduce/tools/benchmark_worker.mojo -o build/cpu-reproduce
pixi run mojo build -I build/acceptance-reproduce/src -I build/balsa-reference/src build/acceptance-reproduce/tools/benchmark_gpu_worker.mojo -o build/gpu-reproduce
```

CPU positional arguments are `ROWS TREES DEPTH DTYPE OP WIDTH THREADS LOOPS`.
Use identical arguments and loop counts per revision, with the six alternating
orders above. First-pass configurations:

| Case | Rows | Trees | Depth | Dtype | Operation | Width | Threads |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| tiny-scalar | 1 | 16 | 4 | float32 | margin | 1 | 1 |
| tiny-auto | 33 | 16 | 4 | float32 | margin | 4 | 0 |
| intermediate-auto | 256 | 32 | 4 | float32 | margin | 4 | 0 |
| intermediate-serial | 256 | 32 | 4 | float32 | margin | 4 | 1 |
| large-auto | 8192 | 64 | 6 | float32 | margin | 4 | 0 |
| large-threads8 | 8192 | 64 | 6 | float32 | margin | 4 | 8 |
| large-serial | 8192 | 64 | 6 | float32 | margin | 4 | 1 |
| f64-proba | 1031 | 32 | 4 | float64 | proba | 4 | 0 |

Example confirmation and GPU invocation (the local corpus is in the main
checkout's `build/gpu-corpus`; use its `manifest.json` to regenerate if needed):

```sh
taskset -c 8-15 build/cpu-reproduce 1031 32 4 float64 proba 4 0 12000
taskset -c 8-15 build/gpu-reproduce ../pyre/build/gpu-corpus/base.tl float32 33 1 allocate,reuse 2026 10 divergent ../pyre/build/gpu-corpus/base.oracle.txt 0.15 1e-6
```

GPU configurations are `base/float32/33`, `base/float32/4096`,
`float64/float64/4096`, `binary-float32/float32/129`; use `1e-12` tolerance for
float64. Identical corpus files, seed 2026, 10% missing values and divergent
sample pattern ensure identical inputs. CPU workers deterministically construct
models and input bytes from the listed arguments.

Provenance hashes:

- Pixi lock: `6a89802b0a771faf646cc0bede12c7466844f500b35f3634293d2336efafa2ee`.
- Mojo executable: `0bce743c710ca91b9513eb6bbe600dcce41892dd2b52f29427b17eeeb97d32fa`.
- Generated references: `90d266bc59e4cb81b58ce140c61d15e33fb903cbd6a7905942d0c11dc9561cb1`.
- Fixture manifest: `8dce57048514b43ebf54e736fd38f04e550583ae5290a5def3eb15348d8ec4ee`.
