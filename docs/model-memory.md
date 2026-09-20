# Model storage accounting

Measure the Balsa storage and compact CPU array payload retained by a newly constructed CPU `Forest`:

```sh
pixi run prepare
pixi run mojo build -I src -I build/balsa-reference/src tools/model_memory_worker.mojo -o build/model-memory-worker
pixi run python tools/model_memory.py
pixi run python -m unittest discover -s tests -p test_model_memory.py
```

The default inputs are the twelve saved oracle checkpoints, verified against
their manifest hashes. Pass checkpoint paths to inspect another corpus; Balsa
discovers precision and Pyre rejects unsupported prediction semantics. Use
`--output PATH` to select the JSON report (default: ignored
`docs/private/model-memory/results.json`). Models run sequentially.

The native worker borrows each owned list and records its logical byte length
and reserved capacity using the compiler's element size. It transfers the model
into `Forest` without copying. No input, output, prediction, or execution context
is created. Reports contain aggregated fields, groups, raw worker output, model
hashes, worker hash, and Balsa/Pyre/tool source hashes.

| Group | Counted storage |
| --- | --- |
| `inference` | Existing traversal arrays, including scalar node tags and general operators, plus base scores |
| `unsupported_semantics` | Category and vector-leaf arrays, including offsets retained in supported numerical models |
| `statistics` | Counts, Hessians, gains and their presence arrays |
| `metadata` | Model attributes, postprocessor text, target/class IDs and shapes |
| `extensions` | Extension list slots, names and payloads; Pyre currently rejects nonempty extensions |
| `containers` | Inline model object and tree-list slots, including their embedded list headers |
| `packed_cpu` | NuMojo compact CPU array payloads; separate from the retained Balsa traversal arrays |
| `wrapper` | Additional inline Forest fields and the executor's initial state allocation |

Each enumerated payload allocation is counted once: headers embedded in a tree or extension are
already included in its parent list's slots. Reserved unused slots are included
only in capacity bytes; child allocations are counted only for live elements.
The initial executor state is measured through Pyre's private `_State` type, so
this development tool must be revalidated when executor ownership changes.

These are source-level storage accounts, **not RSS or allocator measurements**.
NuMojo heap control/shape metadata, allocator headers, allocator size-class
rounding, fragmentation, freed decode
buffers, stack temporaries, process/runtime memory and later execution contexts
are excluded. A serialized checkpoint's file size is reported separately.

The inference group is the payload of the retained Balsa arrays. The separate
packed_cpu group is additional storage, not a replacement or a lower bound. Essential scalar settings (feature count, averaging,
task and sigmoid alpha) are embedded in the model object. Some metadata is needed
for construction-time validation. Removing whole groups in place would violate
Balsa's model invariants; a future serving representation must preserve required
semantics after validation and measure preparation cost and inference performance.
Storage differences alone establish no latency improvement.
