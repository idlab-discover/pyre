"""Gate and time trainer-independent CPU fixture families; never trains models."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess

# family, trees, depth, features, distinct-border budget per feature
CASES = [
    ("roots", 4096, 0, 0, 8),
    ("roots", 4096, 0, 64, 8),
    ("stumps", 16, 1, 16, 8),
    ("stumps", 1024, 1, 64, 8),
    ("symmetric", 128, 4, 16, 8),
    ("symmetric", 128, 8, 16, 8),
    ("symmetric", 4, 12, 16, 2),
    ("symmetric", 4, 16, 16, 2),
    ("symmetric", 64, 12, 16, 8),
    ("asymmetric", 128, 4, 16, 8),
    ("asymmetric", 128, 8, 16, 8),
    ("asymmetric", 64, 12, 16, 8),
    ("asymmetric", 4, 16, 16, 8),
    ("chain", 128, 16, 16, 8),
    ("mixed", 128, 8, 16, 8),
]
ROUTES = {0: "generic", 1: "direct", 2: "bins", 3: "constant"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", type=Path, default=Path("build/benchmark-cpu-shapes-worker"))
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 32, 256, 5000])
    parser.add_argument("--families", nargs="+", choices=sorted({c[0] for c in CASES}))
    args = parser.parse_args()
    if min(args.rounds, args.samples, args.interval) <= 0 or any(n <= 0 for n in args.rows):
        parser.error("Timing counts, intervals and rows must be positive")
    os.sched_setaffinity(0, {args.cpu})
    args.output.mkdir(parents=True, exist_ok=False)
    worker = args.worker.resolve()
    sources = [worker, Path(__file__), Path("tools/benchmark_cpu_shapes_worker.mojo"),
               Path("tests/cpu_fixtures.mojo"), Path("pixi.lock"), *sorted(Path("src/pyre").glob("*.mojo"))]
    provenance = {"sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  "affinity": sorted(os.sched_getaffinity(0)), "cases": CASES,
                  "configuration": vars(args) | {"output": str(args.output), "worker": str(worker)},
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "protocol": "Scalar Balsa oracle gates before timing; 3 warmups per interval; independent native workers; calibrated complete allocating public margin calls; construction and input generation excluded; randomized rounds; synthetic structural stress, not model accuracy or native trainer comparison."}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    gates = []
    cells = []

    def invoke(command):
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(f"Worker failed: {command}\n{completed.stdout}\n{completed.stderr}")
        return completed.stdout.strip()

    for family, trees, depth, features, borders in CASES:
        if args.families and family not in args.families:
            continue
        for dtype in ["float32", "float64"]:
            for rows in args.rows:
                prefix = [str(worker), family, dtype, str(depth), str(trees), str(features), str(rows), str(borders)]
                route, nodes, preparation = invoke(prefix + ["gate", "1"]).split()
                metadata = dict(family=family, dtype=dtype, depth=depth, trees=trees, features=features,
                                borders=borders, rows=rows, route=ROUTES[int(route)], nodes=int(nodes))
                gates.append(metadata | {"exact_scalar_oracle": True, "preparation_seconds": float(preparation)})
                for kernel in ["auto", "generic"] + (["direct"] if int(route) != 0 else []):
                    cells.append((metadata | {"kernel": kernel}, prefix))
        print("gated", family, trees, depth, features, flush=True)
    (args.output / "gates.json").write_text(json.dumps(gates, indent=2) + "\n")
    calibrated = []
    for metadata, prefix in cells:
        loops = 1
        while True:
            elapsed = float(invoke(prefix + [metadata["kernel"], str(loops)]))
            if elapsed >= args.interval:
                break
            loops = max(loops + 1, min(loops * 10, int(loops * args.interval / max(elapsed, 1e-9)) + 1))
        calibrated.append(loops)
    samples = defaultdict(list)
    rng = random.Random(20260920)
    with (args.output / "timings.jsonl").open("w") as output:
        for repetition in range(args.rounds):
            order = list(range(len(cells)))
            rng.shuffle(order)
            for i in order:
                metadata, prefix = cells[i]
                loops = calibrated[i]
                for sample in range(args.samples):
                    elapsed = float(invoke(prefix + [metadata["kernel"], str(loops)]))
                    milliseconds = 1000 * elapsed / loops
                    samples[i].append(milliseconds)
                    output.write(json.dumps(metadata | dict(repetition=repetition, sample=sample, loops=loops,
                                                           elapsed_seconds=elapsed, milliseconds=milliseconds)) + "\n")
                    output.flush()
            print("completed round", repetition, flush=True)
    lines = ["| Family | Dtype | Trees | Depth | Features | Rows | Auto route | Kernel | Median ms |",
             "|---|---|---:|---:|---:|---:|---|---|---:|"]
    for i, (m, _) in enumerate(cells):
        lines.append(f"| {m['family']} | {m['dtype']} | {m['trees']} | {m['depth']} | {m['features']} | {m['rows']} | {m['route']} | {m['kernel']} | {statistics.median(samples[i]):.6f} |")
    (args.output / "timings.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
