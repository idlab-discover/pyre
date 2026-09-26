"""Gate and sweep CPU kernels on captured CatBoost investigation artifacts.

The worker owns inputs/results with NuMojo and times complete allocating calls.
No Python participates in framework inference. Inputs and baseline gates are
read-only; every output is written to a fresh directory supplied by the caller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", type=Path, default=Path("build/benchmark-symmetric-worker"))
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--cases", nargs="+", default=["f34-d4", "f34-d8", "f34-d8-t64", "f34-d8-t256", "f82-d8"])
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 32, 256, 1024, 5000])
    parser.add_argument("--calls", nargs="+", choices=["margin", "proba", "class"], default=["margin"])
    parser.add_argument("--native", action="store_true", help="Requires CatBoost and NumPy in this Python environment")
    args = parser.parse_args()
    if min(args.rounds, args.samples, args.interval) <= 0 or any(n <= 0 for n in args.rows):
        parser.error("rounds, samples, interval, and rows must be positive")
    os.sched_setaffinity(0, {args.cpu})
    args.output.mkdir(parents=True, exist_ok=False)
    args.worker = args.worker.resolve()
    sources = {str(args.worker): hashlib.sha256(args.worker.read_bytes()).hexdigest()}
    for path in sorted(Path("src/pyre").glob("*.mojo")) + [Path(__file__), Path("tools/benchmark_symmetric_worker.mojo"), Path("tools/cpu_samples.mojo"), Path("pixi.lock")]:
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    cells = []
    gates = []
    native = {}
    native_versions = {}
    if args.native:
        import catboost
        import numpy as np
        native_versions = {"catboost": catboost.__version__, "numpy": np.__version__}
    for case in args.cases:
        directory = args.artifacts / case
        features = int(case.split("-")[0][1:])
        for dtype in ["float32", "float64"]:
            for rows in args.rows:
                input_path = directory / f"x-{dtype}-{rows}.bin"
                if not input_path.exists():
                    continue
                model_path = directory / f"{dtype}.treelite"
                for path in [input_path, model_path]:
                    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                prefix = [str(args.worker), str(model_path), dtype, str(input_path), str(rows), str(features), "1"]
                for kernel in ["auto", "direct", "generic"]:
                    gate = args.output / f"{case}-{dtype}-{rows}-{kernel}"
                    gate.mkdir()
                    subprocess.run(prefix + [str(gate), "1", "margin", kernel], check=True, capture_output=True, text=True)
                    for name in ["margins.bin", "probabilities.bin", "labels.bin"]:
                        reference = directory / f"gate-{dtype}-{rows}" / name
                        expected = reference.read_bytes()
                        actual = (gate / name).read_bytes()
                        if actual != expected:
                            raise AssertionError(f"Exact baseline gate failed: {gate / name}")
                        sources[str(reference)] = hashlib.sha256(expected).hexdigest()
                    gates.append({"case": case, "dtype": dtype, "rows": rows, "kernel": kernel, "exact": True})
                    for call in args.calls:
                        cells.append((case, dtype, rows, kernel, call, prefix))
                if args.native:
                    native_model_path = directory / "model.cbm"
                    sources[str(native_model_path)] = hashlib.sha256(native_model_path.read_bytes()).hexdigest()
                    model = catboost.CatBoostClassifier()
                    model.load_model(str(directory / "model.cbm"))
                    x = np.fromfile(input_path, dtype=dtype).reshape(rows, features)
                    # Match the investigation's tolerances before any timing.
                    gate = args.output / f"{case}-{dtype}-{rows}-auto"
                    margin = np.fromfile(gate / "margins.bin", dtype=dtype)
                    proba = np.fromfile(gate / "probabilities.bin", dtype=dtype).reshape(rows, 2)
                    labels = np.fromfile(gate / "labels.bin", dtype="int64")
                    np.testing.assert_allclose(margin, model.predict(x, prediction_type="RawFormulaVal", thread_count=1), rtol=0, atol=1e-4 if dtype == "float32" else 2e-12)
                    np.testing.assert_allclose(proba, model.predict_proba(x, thread_count=1), rtol=0, atol=2e-6 if dtype == "float32" else 2e-12)
                    np.testing.assert_array_equal(labels, model.predict(x, thread_count=1).reshape(-1))
                    native[case, dtype, rows] = model, x
                    for call in args.calls:
                        cells.append((case, dtype, rows, "native-array", call, None))
    if not cells:
        raise ValueError("No matching input cases")
    provenance = {"sources": sources, "affinity": sorted(os.sched_getaffinity(0)), "native_versions": native_versions,
                  "command": vars(args) | {"artifacts": str(args.artifacts), "output": str(args.output), "worker": str(args.worker)},
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "git_diff": subprocess.check_output(["git", "diff"], text=True),
                  "protocol": "Three warmups per worker/interval; complete allocating calls including disposal; setup excluded; serial; randomized rounds. Gates equal archived validated baseline bytes."}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output / "gates.json").write_text(json.dumps(gates, indent=2) + "\n")

    def interval(cell, loops):
        case, dtype, rows, kernel, call, prefix = cell
        if kernel != "native-array":
            result = subprocess.check_output(prefix + ["-", str(loops), call, kernel], text=True)
            return float(result.strip())
        model, x = native[case, dtype, rows]
        prediction = {"margin": "RawFormulaVal", "proba": "Probability", "class": "Class"}[call]
        for _ in range(3):
            model.predict(x, prediction_type=prediction, thread_count=1)
        start = time.perf_counter()
        for _ in range(loops):
            model.predict(x, prediction_type=prediction, thread_count=1)
        return time.perf_counter() - start

    loops_by_cell = {}
    for i, cell in enumerate(cells):
        loops = 1
        while True:
            elapsed = interval(cell, loops)
            if elapsed >= args.interval:
                break
            loops = max(loops + 1, min(loops * 10, int(loops * args.interval / max(elapsed, 1e-9)) + 1))
        loops_by_cell[i] = loops
    rng = random.Random(20260920)
    records = []
    with (args.output / "timings.jsonl").open("w") as output:
        for repetition in range(args.rounds):
            order = list(range(len(cells)))
            rng.shuffle(order)
            for i in order:
                case, dtype, rows, kernel, call, _ = cells[i]
                loops = loops_by_cell[i]
                for sample in range(args.samples):
                    elapsed = interval(cells[i], loops)
                    record = dict(case=case, dtype=dtype, rows=rows, kernel=kernel, call=call,
                                  repetition=repetition, sample=sample, loops=loops, elapsed_seconds=elapsed,
                                  milliseconds=elapsed * 1000 / loops)
                    records.append(record)
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                print(case, dtype, rows, kernel, call, "round", repetition, flush=True)
    lines = ["| Model | Precision | Rows | Call | Kernel | Median ms |", "|---|---|---:|---|---|---:|"]
    for case, dtype, rows, kernel, call, _ in cells:
        values = [r["milliseconds"] for r in records if (r["case"], r["dtype"], r["rows"], r["kernel"], r["call"]) == (case, dtype, rows, kernel, call)]
        lines.append(f"| {case} | {dtype} | {rows} | {call} | {kernel} | {statistics.median(values):.6f} |")
    (args.output / "timings.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
