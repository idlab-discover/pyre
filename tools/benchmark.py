"""pyperf-managed native inference: calibrated loops, independent processes.

Example: pixi run benchmark --processes 6 --values 5 --warmups 2 --min-time .15
         --affinity 0-7 -o docs/private/benchmarks/baseline.json
"""
import hashlib
import os
from pathlib import Path
import subprocess
import pyperf

ROOT = Path(__file__).resolve().parents[1]
# name, rows, trees, depth, precision, operation, lane width
CASES = [
    ("trained-xgboost", 4096, 8, 0, "float32", "margin", 8),
    ("tiny-f32", 1, 16, 3, "float32", "margin", 8),
    ("tail-f32", 33, 64, 6, "float32", "margin", 8),
    ("batch-f32", 4096, 64, 6, "float32", "margin", 8),
    ("large-f32", 16384, 128, 8, "float32", "margin", 8),
    ("batch-f64", 4096, 64, 6, "float64", "margin", 4),
    ("deep-f32", 4096, 64, 10, "float32", "margin", 8),
    ("binary-f32", 4096, 64, 6, "float32", "proba", 8),
    ("scalar-f32", 4096, 64, 6, "float32", "margin", 1),
    ("wide-f32", 4096, 64, 6, "float32", "margin", 16),
]

def add_cmdline(cmd, args):
    cmd.extend(["--binary", args.binary, "--threads", str(args.threads)])
    if args.cases:
        cmd.extend(["--cases", args.cases])


def main():
    runner = pyperf.Runner(add_cmdline_args=add_cmdline)
    runner.argparser.add_argument("--binary", default=str(ROOT / "build/benchmark-worker"))
    runner.argparser.add_argument("--threads", type=int, default=1)
    runner.argparser.add_argument("--cases", default="", help="comma-separated case names")
    args = runner.parse_args()
    runner.metadata.update({
        "balsa_commit": "737874f3eae9d95a0517397130f445b303d28cc6",
        "worker_sha256": hashlib.sha256(Path(args.binary).read_bytes()).hexdigest(),
        "forest_sha256": hashlib.sha256((Path(args.binary).parent / "forest.mojo" if (Path(args.binary).parent / "forest.mojo").exists() else ROOT / "src/pyre/forest.mojo").read_bytes()).hexdigest(),
        "requested_threads": args.threads,
        "timing_scope": "warm native predictions, output allocation included; model/input setup excluded",
    })
    selected = set(args.cases.split(",")) if args.cases else None
    for name, rows, trees, depth, dtype, operation, width in CASES:
        if selected and name not in selected:
            continue
        command = [args.binary, str(rows), str(trees), str(depth), dtype, operation, str(width), str(args.threads)]
        def time_func(loops, command=command):
            proc = subprocess.run(command + [str(loops)], cwd=ROOT, text=True, capture_output=True, check=True)
            return float(proc.stdout.strip())
        runner.bench_time_func(name, time_func, metadata={
            "rows": rows, "trees": 4 if depth == 0 else trees, "depth": 3 if depth == 0 else depth,
            "precision": dtype, "operation": operation, "simd_width": width,
        })

if __name__ == "__main__":
    main()
