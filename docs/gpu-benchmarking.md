# Reproducing GPU performance measurements

The performance corpus is separate from the twelve saved GTIL regression
fixtures. Generating it never changes `tests/fixtures` or `tests/reference_data.mojo`.
Python frameworks are offline tools; native inference has no framework dependency.

## Generate

From the Pyre repository, with the sibling Balsa frameworks environment already
installed (NumPy 2.2.6, Treelite 4.6.1, xgboost-cpu 2.1.4, CatBoost 1.2.8):

```sh
pixi run gpu-corpus-build
../balsa/.pixi/envs/frameworks/bin/python tools/generate_gpu_corpus.py
pixi run benchmark-gpu-build
```

The generator refuses an existing output directory. Use `--output build/corpus-2`
to reproduce independently. Compare model/oracle hashes and input hashes in the
two manifests; invocation paths and source/binary hashes are provenance, not model
semantics. `--synthetic-only` omits training. Library versions are checked before
work starts. The Balsa codec/builders and CatBoost offline adapter come from the
immutable commit selected by `tools/prepare_balsa.py`.

The 18 synthetic cases screen one factor at a time around 64 trees, depth 6,
32 features, float32 regression, balanced trees and divergent samples. Tree
counts span 1/16/64/256/1024; depths 2/6/10; features 8/32/128/512; missing rates
0/1/10/50%. Additional cases cover uneven trees, identical/coherent rows,
float64, and binary sigmoid classification in both precisions. Two numeric trained
regressors use 64 trees, maximum depth 6 and 8,192 training rows with 32 features.

These are controlled micro-workloads, not a claim about a production distribution.
Uneven trees vary both across and within trees; their smaller actual node count
is reported, so that comparison does not isolate shape at constant node count.
Coherent samples repeat one complete feature/missing mask across all rows.
Missing rates are generation probabilities; actual rates vary by batch, especially
for coherent samples. Samples are logical C row-major without transposition.

Generation caps each synthetic case at 262,144 nodes and sample payload at 256 MiB
before allocation. The native builder additionally limits dimensions. Manifest
statistics derive from actual trees, not requested depth. Models, oracle files,
training values, source files, binaries and sample batches have hashes. The native
producer exports its largest batch for byte verification against Python; every
smaller hashed batch is a prefix. Native exports are temporary and removed.

## Validate and measure

```sh
# Correctness across all six row counts, including partial GPU blocks.
pixi run python tools/benchmark_gpu.py --validate-only \
  --output docs/private/benchmarks/gpu/parity.json

# Initial screen. One process is exploratory, not repeatability evidence.
pixi run python tools/benchmark_gpu.py --processes 1 --rows 257,16384 \
  --output docs/private/benchmarks/gpu/screen.json

# Confirm selected extremes in at least five fresh processes.
pixi run python tools/benchmark_gpu.py --processes 5 --rows 1,16384,65536 \
  --cases base,trees-1024,depth-10,features-512,float64,pattern-coherent \
  --output docs/private/benchmarks/gpu/confirm.json
```

`--manifest`, `--cases`, `--rows` and `--phase` select a matrix. Row counts must
have hashes in the manifest. `--phase` accepts a comma-separated subset (for
example `reuse,pipeline`); selected phases share one fresh case/repetition process and calibrate independently.
Jobs shuffle deterministically with `--seed`. No concurrent GPU benchmark workers
are launched. Default repetition is five fresh processes per case/row/phase.

Before any timing, every requested case/batch passes whole-batch CPU/GPU margin
parity; classifiers also check both probabilities and the label threshold rule.
Each worker also checks both reused upload paths against CPU and the native pinned
producer against every NuMojo input value. It additionally checks CPU and GPU against saved GTIL results on up to
257 deterministic rows plus root thresholds, neighboring representable values,
infinities and NaNs. The tolerance is `abs(actual-expected) <= tol*(1+abs(expected))`,
with `tol=2e-5` for float32 and `1e-12` for float64. The original regression test
suite retains its tighter per-fixture tolerances. Nonfinite mismatches fail.
Timing workers repeat the same correctness gates before warmup.

Every phase warms once, then doubles iterations until the final measured interval
reaches at least 100 ms. `--loops` sets the initial count; `--min-time` can increase
the duration. Cold fields record first observed operations in each process, not
a guarantee of cold OS caches, idle hardware, or unloaded runtime state.

| Phase | Work included |
| --- | --- |
| `read` | Model file bytes; OS cache uncontrolled |
| `decode` | Copy owned encoded bytes, Balsa decode and structural validation |
| `prepare` | Prepare the loaded Balsa forest, including device allocation/upload |
| `samples` | NuMojo allocation and deterministic native fill |
| `upload` | New native device allocation, H2D copy, completion |
| `reuse` | Completed margin prediction with resident model/input and reused output |
| `allocate` | Margin output allocation and completed prediction |
| `device` | Private launches inside device timer, including stream scheduling gaps |
| `download` | Host output allocation, D2H copy, completion |
| `end_to_end` | Existing host input → allocated upload → prediction → host output |
| `pipeline` | Sample allocation/fill → allocated upload → prediction → host output |

`pipeline` is the complete per-batch operation for this in-memory producer. It
excludes model setup, disk reads and sample decoding. Model reading and decoding
are separate. There is no disk-backed sample producer yet. All timed inference
uses margins; binary probabilities are correctness checks outside timing.

## Evidence and limits

Each run saves pyperf JSON, `.details.json` with raw phase records and summaries,
`.raw.jsonl` with exact commands/stdout, and `.gpu.csv` sampled at 500 ms. Partial
records survive failures. Details include the complete manifest, source/binary/
lock hashes, working-tree status and GPU process snapshots before and after.
Keep these under ignored `docs/private/`.

Summaries contain medians, min/max, median absolute deviation, coefficient of
variation (only for multiple processes), and rows/second. Preparation amortization
reports the call count at which preparation falls below 10% of total cost:
`ceil(9 * preparation_seconds / per_call_seconds)`. This is not a speedup claim.

Byte counts are **logical payload accounting**, including prepared node fields,
input/output sizes and explicit transfer sizes. They exclude allocator rounding,
runtime pools and temporary CPU/oracle allocations. Confirm allocation and copy
counts with separate traces. Resident transfer bytes are a source-derived
expectation until traced. Device intervals are not completed-call latency.

Run measurements without a profiler, sanitizer or sanitizer allocator overrides.
The harness rejects known allocator override names. Do not change global GPU power
or desktop state automatically: telemetry and process snapshots expose the shared
GPU environment. Profile selected cases separately and preserve the exact native
worker command. Instrumented timings do not belong in performance comparisons.

## Repeated ingestion experiment

`upload_into` reuses device storage with the existing immutable logical shape.
Native buffer wrappers handle changing active rows within capacity. The private
`gpu_staging.mojo` producer fills native pinned storage without an NDArray adapter;
this is an experiment, not a public staging API.

| Phase | Work included |
| --- | --- |
| `upload_into` | Existing NuMojo input → existing device batch, completed |
| `pinned_fill` | Fill existing native pinned host storage |
| `pinned_upload` | Existing pinned host storage → existing device batch, completed |
| `pinned_allocate` | Allocate native pinned staging and synchronize |
| `batch_allocate` | Allocate native device batch and synchronize |
| `ingest_a` | Allocate/fill NuMojo input, allocate/upload batch, predict into reused output, download |
| `ingest_b` | Allocate/fill NuMojo input, upload into reused batch, predict into reused output, download |
| `ingest_c` | Fill reused native pinned staging, copy into reused batch, predict into reused output, download |

A/B/C use identical values, precision, row-major feature mapping, resident forest,
reused prediction output and a newly allocated host result. B measures device
reuse first; C additionally changes producer storage. Host staging and device
capacity allocation are setup costs for C; B requires device capacity setup.
Summary amortization uses the median observed first device/staging allocation
cost in each fresh process divided by median per-call saving against A. Warmed
allocation phases also include release; they are reported separately. In this
worker version, cold upload excludes device allocation, which has its own field. No finite amortization is reported if no saving
is measured. These ratios do not establish statistical significance.

```sh
pixi run python tools/benchmark_gpu.py --processes 5 --rows 33,65536 \
  --cases base,features-512,float64,trees-1 \
  --phase reuse,samples,upload,download,upload_into,pinned_fill,pinned_upload,pinned_allocate,batch_allocate,ingest_a,ingest_b,ingest_c \
  --output docs/private/benchmarks/gpu/ingestion.json
```

The worker retains one pinned staging allocation even when measuring other phases,
so process memory includes that reserved payload. Allocation-call counts in details
are logical calls in these operations; native runtime pooling can eliminate CUDA
allocation calls. Traces establish CUDA transfers/allocations, while focused tests
check retained allocation identity, offsets, tails, completion and reuse after
validation failure. Do not conflate logical allocation calls with driver calls.

For separate baseline profiles:

```sh
pixi run python tools/profile_gpu.py --case base --rows 65536 \
  --output docs/private/benchmarks/gpu/profile-base
```

The trace summarizer reports resident kernels after the last validation copy,
including warmup/calibration, and checks that interval for transfers and CUDA
allocation calls. Nsight Systems reports registers per thread; hardware-counter
metrics such as achieved occupancy and branch efficiency require Nsight Compute
permissions. Instrumented worker timing lines are trace context only.

The A/B/C timed loops have fixed active shapes; wrapper construction when changing
rows is additional work, covered by correctness tests but excluded from these
steady-state timings. Pyperf system/process metadata describes the Python
orchestrator, including its RSS, rather than the native worker. Trace memory
summaries report tracked CUDA allocation/free events and their peak live bytes;
these include runtime pools and do not establish total process RSS or all virtual
reservations.

## Hardware counters

With Nsight Compute installed and GPU counter access enabled for the current user:

```sh
pixi run python tools/profile_gpu_counters.py --case base --rows 65536 \
  --output docs/private/benchmarks/gpu/counters/base-65536
pixi run python -m unittest discover -s tests -p 'test_gpu_*.py'
```

This separate profiler selects the first full-batch margin kernel, collects launch,
occupancy, memory, scheduler, warp-stall and branch sections, and lets the worker
finish its CPU/GPU, GTIL and reused-upload checks. It validates model/oracle hashes
and records the selected corpus entry, exact command, binary/source/lock hashes,
GPU snapshots, native report, raw CSV and metrics with their units. The output
prefix must be fresh; failures retain metadata and available logs.

Kernel replay flushes caches on each pass. GPU clocks remain uncontrolled and the
profiler does not change clock settings. These are controlled cold-cache kernel
observations, not warmed resident latency or a speedup measurement. Retain the
uninstrumented repeated-process benchmarks for performance comparisons. Current
source hashes alone do not prove which sources produced a pre-existing binary;
rebuild the worker with `pixi run benchmark-gpu-build` when sources change.
