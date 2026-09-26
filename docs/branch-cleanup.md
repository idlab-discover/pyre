# Branch consolidation and first polish

Historical review. Release preparation subsequently removed the sibling-checkout
requirement and added distribution/CI; see [distribution](distribution.md).

Reviewed 2026-09-26 against local `main` at `dff9d58`. No remote is configured.
Review covered source changes, documented contracts, combined-branch validation,
and local experiment preservation. No performance benchmarks were rerun.

## Branch decisions

| Branch | Reviewed tip | Decision |
| --- | --- | --- |
| `acceptance-consolidation` | `b3f1551` | Already in original main; retire branch name |
| `cpu-policy` | `eab0288` | Already in original main; retire branch name |
| `cpu-validation` | `2d6b803` | Already in original main; retire branch name |
| `gpu-row-tree` | `2bed2c1` | Already in original main; retire branch name |
| `model-memory` | `1df90da` | Already in original main; retire branch name |
| `feat/cpu-symmetric-inference` | `2899792` | Integrate; constant/symmetric CPU execution and structural tests |
| `gpu-optimization` | `e54d624` | Integrate with row-limit fix; packed GPU traversal and reductions |
| `gpu-row-tree-experiment` | `138de29` | Retain separately as historical experiment evidence |

CPU and GPU integration merged cleanly at `a006506` on the temporary
`cleanup/consolidation` branch. The final candidate includes both histories and
the fixes below; main can fast-forward to it. Retired branches' commits remain
reachable through main. The experiment branch adds private benchmark snapshots
and plans, not another production implementation; merging it would place old
experimental sources and reports into the active tree unnecessarily.

## Standards review

No violations of the documented Balsa/Pyre/NuMojo responsibility boundaries were
found. `_CPUPlan` encapsulation and the forwarding `Forest` interface follow the
CPU planning document.

- **Fixed correctness gap:** GPU launch narrowed accepted row counts to `Int32`.
  At 2^31 rows, the negative kernel count could leave output unwritten. Validate
  the supported limit before allocating output and in shared `*_into` validation.
  The regression uses a zero-feature batch, checks all four allocating operations
  and reusable margin output, and requires the precise error without modifying
  destination storage. The README now documents the limit.
- **Deferred nonblocking heuristic:** calibration/sampling logic is duplicated
  between `benchmark_cpu_shapes.py` and `benchmark_symmetric.py`. A later tooling
  pass can share measurement mechanics while retaining their different gates.

Two findings: one fixed correctness issue and one optional duplication cleanup;
zero hard architectural violations.

## Spec review

There is no originating issue or configured issue tracker. This axis used the
README, `docs/cpu-planning.md`, `docs/cpu-symmetric.md`, and the GPU kernel report
as contracts. It is not verification against an unavailable external spec.

The same row-count bug was the sole concrete new-branch blocker and is fixed.
No CPU blocker was identified: structural eligibility, depth limits, bin capacity,
fallbacks, ordered accumulation, ownership and documented serial compact paths
match the contracts. GPU normalization, packing bounds and reductions passed
source review and the acceptance checks. No claim of general floating-point
equivalence beyond tested tolerances is made; GPU summation order differs.

One finding on this axis, fixed. Follow-up review found no correctness blocker
in the fixes.

## Additional polish

- Reproduced the previously documented CPU-only compilation failure with
  `mojo build --target-accelerator=none`. A runtime GPU rejection did not prevent
  transfer code from being instantiated. Compile-time branches now exclude those
  transfers from the four legacy CPU inference adapters. Rejection messages and
  CPU behavior are preserved.
- `test-cpu-only` makes this compilation configuration repeatable even on a GPU
  workstation. `check` now includes it and the Python tooling tests, rebuilding
  their native memory worker first. `check-all` combines CPU and GPU gates.
- Mark PNG assets binary for Git diff/merge handling; fix stale integration prose
  and document the new checks. No assets or dependency pins were changed.

## Folder and archive decisions

Only the primary Pyre worktree remains registered. The old sibling Pyre
experiment/worktree directories are already absent.

| Current path | Decision and evidence |
| --- | --- |
| `../balsa` | Keep: `prepare_balsa.py` requires commit `737874f` from this repository to produce a fresh build snapshot |
| `../nvforest`, `../treelite` | Keep: independent clean upstream repositories, not rolled-in Pyre worktrees; outside this cleanup's deletion scope |
| `../nvidiads` | Keep: contains `nvfp4-specification.md`, unrelated to the integrated branches |
| `this-vs-nvforest/` | Keep: ignored comparison harness, models, inputs and results still referenced by benchmark reports |
| `docs/private/` | Keep: ignored historical plans, measurements and archives are not all represented in active branches |
| `build/` | Keep for now: mixes disposable binaries with local probes, models and validation evidence |

An archive audit compared every regular file byte-for-byte with the current tree
and Git blobs at the corresponding preserved branch tip:

| Saved worktree archive | Regular files | Equal to current tree | Equal to branch only | Other contents |
| --- | ---: | ---: | ---: | ---: |
| `pyre-gpu-row-tree.tar.gz` | 266 | 88 | 15 | 163 |
| `pyre-cpu-validation.tar.gz` | 607 | 89 | 22 | 496 |
| `pyre-acceptance-consolidation.tar.gz` | 2692 | 2673 | 16 | 3 |

These archives live under `docs/private/worktree-archive/`. The first two retain
different source snapshots, private evidence and build artifacts. The three
remaining differences in the acceptance archive are compiled outputs. All
archives are retained; no claim is made that regular-file comparison verifies
directory metadata or symlink preservation. The existing backup stash and
untracked `docs/release-readiness-0.1.0.md` are also retained unchanged.

## Validation

On RTX 5090, driver 615.71.09, with locked Mojo 1.0.0, MAX 26.5.0 and NuMojo 0.10.0:

- `pixi run --locked check`: example build, package precompile, 40 native CPU
  tests, seven additional core acceptance executions with GPU compilation
  disabled, and 11 Python tooling tests.
- `pixi run --locked test-gpu`: 14 inference/storage tests, legacy transfers and
  the runnable GPU example.
- Compute Sanitizer memcheck (full leak check), racecheck, initcheck and synccheck
  each exit 0. Device and host allocation pools were disabled as in the existing
  sanitizer protocol; no API error suppression was used.
- `git diff --check` and formatting of changed Mojo files.

Logs are retained locally in `build/cleanup-review/`; sanitizer command arrays,
exit codes and source/binary hashes are in its `sanitizers/` subdirectory. The
before-fix CPU-only failure and after-fix compile log are retained too.

This is merge readiness, not a release or performance-parity certification.
Distribution still depends on a sibling Balsa checkout. Project licensing,
installed-consumer validation, CI and a reproducible public competitor benchmark
corpus remain follow-up work. The older untracked release assessment is historical
and predates the new GPU kernel and the CPU-only compilation fix.
