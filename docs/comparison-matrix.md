# Reproducible CPU and GPU comparisons

`tools/compare.py` benchmarks the current Pyre working checkout against selected
native predictors and nvForest. It trains a shared corpus once, checks every
prediction against Treelite GTIL, and times completed allocating calls on
resident inputs. Every run rebuilds its own Pyre binary. Changing commits does
not require retraining the corpus.

## Setup

From the Pyre repository root, install the locked Mojo environment and prepare
the pinned Balsa source:

```sh
pixi install --locked
pixi run prepare
uv venv --python .pixi/envs/default/bin/python build/comparison-env
uv pip install --python build/comparison-env/bin/python \
  -r tools/comparison/requirements-cpu.txt
```

Install the trainers you want to compare. Preparation and native prediction
only load the selected model families. The XGBoost estimator wrapper also
requires scikit-learn, even when random forests are not selected:

```sh
uv pip install --python build/comparison-env/bin/python \
  scikit-learn==1.9.1 joblib==1.6.0 xgboost-cpu==3.4.1 \
  catboost==1.2.10 lightgbm==4.7.0
# Optional nvForest reference; its distribution includes CUDA dependencies,
# even when this runner selects CPU prediction.
uv pip install --python build/comparison-env/bin/python nvforest-cu13==26.6.0
```

`tools/comparison/requirements.lock` preserves the complete original comparison
environment. You can use different reference versions; the runner records the
installed versions. The repository's Pixi lock determines the Pyre toolchain.
Do not overwrite an existing environment if another benchmark is using it.

## Choose libraries and workload sizes

For example, compare Pyre with XGBoost and CatBoost:

```sh
build/comparison-env/bin/python tools/compare.py plan \
  --libraries xgboost catboost
build/comparison-env/bin/python tools/compare.py prepare \
  --libraries xgboost catboost --output build/my-corpus
build/comparison-env/bin/python tools/compare.py run \
  --libraries xgboost catboost --corpus build/my-corpus --output build/my-run
build/comparison-env/bin/python tools/compare.py report \
  --results build/my-run --output build/my-report
```

Library choices are `scikit-learn`, `xgboost`, `catboost`, `lightgbm`, and
`nvforest`. Pyre always participates. Selecting XGBoost includes depthwise and
lossguide forests. XGBoost uses learning rate 0.05; lossguide has a
127-leaf budget and depthwise has no leaf-count cap. The synthetic regression
target combines nonlinear terms, contributions from the remaining features,
and Gaussian noise, so larger boosting runs continue to fit useful splits. Adding nvForest compares it on the selected trainer families;
selecting nvForest alone uses all families in the matrix and omits the native
predictors. Omitting `--libraries` uses the matrix's full library selection.

The default `benchmarks/cpu-heavy.json` requests:

- 256 and 1,024 trees, depth limits 8 and 10, 32 numerical features.
- 8,192 training rows and held-out batches of 65,536 and 262,144 rows.
- 16 requested inference threads; eight threads for offline training. Use
  `--threads 1 16` to include single-thread execution in the full matrix.
- Five fresh processes per cell, three intervals per process, calibrated to
  at least 150 ms before collecting samples.
- Native predictors and nvForest chunks automatic, 8 and 32.

The model-size budget excludes combinations whose worst-case expansion exceeds
one million nodes. In particular, a full 1,024-tree depth-10 forest is too large.
The manifest lists every exclusion; actual tree/node/depth statistics accompany
accepted models. LightGBM uses its ordinary 31-leaf budget, so a requested depth
of 10 does not imply full depth-10 trees.

Override axes on the command line, using the same preparation axes for `prepare`
and `run`. A run can select a subset of an existing corpus:

```sh
# Deeper models within the existing loader limit.
build/comparison-env/bin/python tools/compare.py prepare \
  --libraries xgboost catboost --trees 64 --depths 12 \
  --rows 65536 262144 --output build/deep-corpus
build/comparison-env/bin/python tools/compare.py run \
  --libraries xgboost catboost --trees 64 --depths 12 \
  --rows 65536 262144 --threads 1 4 8 16 32 \
  --processes 7 --samples 5 --min-seconds .25 \
  --corpus build/deep-corpus --output build/deep-run
```

Custom JSON matrices can also set training rows, features, seed, training
threads, nvForest chunks, input/node budgets and worker timeouts. Unknown keys,
invalid values and incompatible corpora fail before timing. Use `--cpus` with a
comma-separated CPU list to restrict affinity consistently for every worker.
A thread request cannot exceed that affinity set. Defaults inherit all allowed
CPUs; 16 workers on a 32-logical-CPU machine are not pinned to distinct physical
cores individually.

For a short integration check, use `--matrix benchmarks/cpu-smoke.json` with
fresh corpus and run directories. Its one-process timings are validation only.

## Timing and correctness

Every engine evaluates the same exported model and logical input values at the
same requested thread count. Input hashes identify the shared source arrays;
engine-specific preparation can change their container or storage dtype. Native models and Treelite checkpoints have
recorded hashes. Export validation includes split thresholds and adjacent
representable float32 input values. Each fresh inference process checks every
output row against the saved independent GTIL oracle before warmup or timing.
Nonfinite results, shape mismatches and tolerance violations fail the run.
Float32 uses scaled tolerance 2e-5; float64 uses 1e-12. Native scikit-learn and
XGBoost have a separate 2e-5 conversion tolerance because their APIs can quantize
inputs internally. The benchmark preserves that distinction in each record.

By default, `--native-input prepared` prepares CatBoost's `Pool` once, converts
scikit-learn input to its required contiguous float32 once, and calls XGBoost's
public `Booster.inplace_predict` directly. LightGBM and nvForest receive the
resident NumPy arrays in model precision. All source feature values originated
as float32, so the scikit-learn conversion is exact. Full-output gates still run
after preparation. This matches Pyre's already constructed NuMojo input and
prevents repeated conversion from inflating its apparent traversal advantage.
Model-specific binning remains inside prediction; this is not a prequantized-input
comparison.
Use `--native-input ndarray` to measure the original framework wrapper calls,
including their per-call ingestion costs. That option is a different timing
contract; do not combine modes in one speedup aggregate.

Models and inputs are resident before timing. Each measured call includes output
allocation, prediction, completion and output disposal. Loading, preparation,
corpus training, oracle checks, process startup and compilation are excluded.
Pyre uses native Mojo calls; the other engines use their Python APIs. This
measures application calls, including each binding's overhead, rather than
isolated traversal kernels. Numerical regressors only are included; this is not
an accuracy comparison between different training libraries.

The runner randomizes model/batch/thread groups and reverses engine order on
alternate repetitions. Processes execute serially. An advisory lock prevents
two runners from timing in the same checkout concurrently. Other desktop jobs
remain uncontrolled; provenance records load average and affinity. Compilation
finishes before measurement begins. The run stores source and binary hashes
and rejects source changes detected at completion.

A process warms three calls, doubles loop count until the calibration target,
then records each interval separately. The report takes the median within each
process and the median across processes. Min/max are ranges of process medians,
not individual calls or confidence intervals. Geometric means weight matched
model cases equally within a batch/thread group. Ratios greater than one favor
Pyre. Overlapping observed ranges require caution. nvForest's best chunk is
selected retrospectively from the probe; this is not exhaustive tuning.

## Evidence and rerunning another commit

Run directories must be fresh. A run owns its binary and includes:

- `measurements.parquet`, raw intervals and correctness errors.
- `attempts.parquet`, expected cell/repetition outcomes and diagnostics.
- `raw/`, exact worker requests and stdout/stderr for every attempt.
- `provenance.json`, current commit, worktree diff, source/binary/corpus hashes,
  affinity, hardware, load, package versions and the effective matrix.
- `build-command-cpu.json` / `build-cpu.log` and, for GPU runs,
  `build-command-gpu.json` / `build-gpu.log`, the native compilation records.

Failures stop the runner and preserve evidence. The report refuses missing,
duplicate, unsuccessful or out-of-tolerance cells. It writes per-cell and matched
CSV data, a Markdown table, an SVG/PNG chart, and portable provenance. Throughput
counts full forest predictions per second. It does not count single trees.

After changing Pyre commits, repeat `run` with the same corpus and a new output
directory. It compiles from the new working checkout automatically. Repeat
`report` for that run. Reusing the corpus holds model and input hashes constant;
keep the same reference environment to isolate the Pyre change.

A 16-thread result alone does not prove CPU saturation. Establish a plateau with
thread/batch sweeps, check utilization and profile separately from reported
measurements. Concurrent-call throughput needs a separate experiment. Do not
multiply a one-thread result by the logical CPU count.

## November workload update

The earlier noiseless target let 1,024-round XGBoost models converge to mostly
constant added trees, and unconstrained depthwise/lossguide growth produced
identical structures. The current corpus adds contributions from all features
and Gaussian noise, lowers XGBoost's learning rate, and gives lossguide its own
leaf budget. Reports include nonconstant-tree counts alongside requested and
actual sizes. September and November aggregate results use different corpora;
they must not be divided to claim a Pyre implementation speedup. For a direct
before/after measurement, run both revisions on one saved corpus.

## GPU comparisons

The same runner accepts `benchmarks/gpu-smoke.json` and
`benchmarks/gpu-heavy.json`. Preparation remains offline CPU training; inference
runs on the GPU. Pyre and nvForest consume the same serialized models, input
values and GTIL oracles as the CPU comparison.

Use a separate Python environment so CUDA-enabled XGBoost does not conflict
with `xgboost-cpu` in the CPU environment:

```sh
uv venv --python .pixi/envs/default/bin/python build/comparison-gpu-env
uv pip install --python build/comparison-gpu-env/bin/python \
  -r tools/comparison/requirements-gpu.txt

build/comparison-gpu-env/bin/python tools/compare.py prepare \
  --matrix benchmarks/gpu-smoke.json --output build/gpu-smoke-corpus
build/comparison-gpu-env/bin/python tools/compare.py run \
  --matrix benchmarks/gpu-smoke.json --corpus build/gpu-smoke-corpus \
  --output build/gpu-smoke-run
build/comparison-gpu-env/bin/python tools/compare.py report \
  --results build/gpu-smoke-run --output build/gpu-smoke-report
```

The dependency file includes CuPy's `ctk` extra for the CUDA headers needed by
XGBoost's device-output conversion. `tools/comparison/requirements-gpu.lock`
records the complete validated GPU Python environment; install that file in
place of `requirements-gpu.txt` to reproduce its transitive versions.

The smoke matrix covers all five model families, float32 and float64 exports,
1/33/4,097 rows, automatic and eight-row Pyre blocks, and automatic/eight-row
nvForest chunks. It checks launch tails and report matching. Its short timings
are validation evidence, not performance conclusions.

The heavy GPU matrix increases the CPU targets:

| Axis | CPU heavy | GPU heavy |
| --- | --- | --- |
| Trees | 256, 1,024 | 256, 1,024, 4,096 |
| Depth limits | 8, 10 | 6, 8, 10 |
| Inference rows | 65,536, 262,144 | 262,144, 1,048,576, 4,194,304 |
| Maximum bytes per input array | 256 MiB | 1 GiB |
| Calibration target | 150 ms | 250 ms |
| Per-worker timeout | 600 s | 1,800 s |

Both heavy matrices use five independent processes, three measured intervals,
32 features and 8,192 training rows. The GPU preset uses Pyre's automatic
launch policy and nvForest chunks automatic, 8 and 32. Each prediction includes
output allocation, completion and disposal. Model preparation, input uploads,
first-call compilation, warmup, calibration and correctness downloads are
outside the measured intervals. These are wall-clock API latencies, not CUDA
event kernel-only times or end-to-end host-input latencies.

The one-million-node loader limit remains in force. A full 4,096-tree depth-6
forest fits; a full depth-8 forest does not. Leaf-limited families admit other
combinations. Every conservative exclusion appears in the manifest and report.
Requested trees and depth caps do not imply that training produces that many
nonconstant trees or full-depth trees; inspect the recorded topology.

The input budget bounds one input matrix, not total disk or working memory.
The heavy corpus stores inputs and oracles per model and batch, so allow tens
of GiB of disk plus several GiB of host memory. Training and GTIL oracle
creation can take much longer than inference. Start with a subset to validate
large batches before preparing the full corpus:

```sh
build/comparison-gpu-env/bin/python tools/compare.py prepare \
  --matrix benchmarks/gpu-heavy.json --libraries nvforest \
  --trees 256 --depths 6 --rows 1048576 \
  --output build/gpu-large-corpus
build/comparison-gpu-env/bin/python tools/compare.py run \
  --matrix benchmarks/gpu-heavy.json --libraries nvforest \
  --trees 256 --depths 6 --rows 1048576 \
  --corpus build/gpu-large-corpus --output build/gpu-large-run
```

Use `--matrix benchmarks/gpu-heavy.json` without axis overrides for the full
sweep. Reuse the same preparation axes in `run`. The runner serializes worker
processes and records the GPU identity, driver, starting clocks, memory usage,
temperature and `CUDA_VISIBLE_DEVICES`. It uses the default visible GPU. For a
multi-GPU host, set `CUDA_VISIBLE_DEVICES` to one device consistently before
running; it does not modify clocks or isolate the GPU from desktop activity.

Native resident GPU comparisons currently support XGBoost only. Select
`--libraries xgboost nvforest` to include its CUDA `inplace_predict` with CuPy
inputs. CUDA support and device output are required; there is no CPU fallback.
The worker synchronizes every XGBoost call. Pyre and nvForest already wait for
their prediction stream/handle. Other trainer families remain available through
Pyre and nvForest; selecting their native GPU adapters fails validation before
execution. CatBoost's host-oriented prediction path would need a separate
transfer-inclusive timing phase, and is not mixed into this resident comparison.

Custom matrices may set `pyre_gpu_row_groups` to `null` or powers of two up to
32. A numeric value pins rows per block in the current automatic plan, reducing
rows per thread when necessary. The complete effective plan is recorded in
`engine_settings_json`. Every requested setting gets a correctness gate and a
separate report group; reports do not silently select Pyre's fastest setting.
GPU reports retain nullable CPU thread counts, match references on device, and
write `gpu-comparison.svg` and `gpu-comparison.png` alongside the usual CSVs.
