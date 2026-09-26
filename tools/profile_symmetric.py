"""Serial fixed-period CPU profiles for the investigation's 34-feature model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--worker", type=Path, default=Path("build/benchmark-symmetric-worker"))
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    args.output.mkdir(parents=True, exist_ok=False)
    model = args.artifacts / "f34-d8" / "float32.treelite"
    inputs = args.artifacts / "f34-d8" / "x-float32-5000.bin"
    commands = []
    for kernel, loops in [("generic", 100), ("auto", 1500)]:
        worker = [str(args.worker.resolve()), str(model), "float32", str(inputs),
                  "5000", "34", "1", "-", str(loops), "margin", kernel]
        for kind in ["stat", "record"]:
            if kind == "stat":
                command = ["perf", "stat", "-D", "1000", "-x", ",", "-e", "cycles:u,instructions:u",
                           "-o", str(args.output / f"{kernel}-counters.csv"), "--", *worker]
            else:
                command = ["perf", "record", "-D", "1000", "-e", "cycles:u", "-c", "5000000",
                           "-o", str(args.output / f"{kernel}.data"), "--", *worker]
            commands.append(command)
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            (args.output / f"{kernel}-{kind}.log").write_text(result.stdout + result.stderr)
        report = subprocess.check_output(["perf", "report", "--stdio", "--no-children", "--percent-limit", "1",
                                          "-i", str(args.output / f"{kernel}.data")], text=True)
        (args.output / f"{kernel}-flat.txt").write_text(report)
    (args.output / "provenance.json").write_text(json.dumps({
        "commands": commands, "affinity": sorted(os.sched_getaffinity(0)),
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.worker, model, inputs]},
        "note": "Separate diagnostic windows; first 1000 ms omitted. Flat IP sampling only. Counter totals are not exact per-call costs. No compilation or other benchmark run should overlap.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
