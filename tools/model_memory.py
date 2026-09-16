"""Measure Balsa allocations retained by Pyre, using a native worker."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"containers", "inference", "metadata", "unsupported_semantics",
          "statistics", "extensions", "wrapper"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_worker(output):
    metadata = {}
    fields = defaultdict(lambda: {"logical_bytes": 0, "capacity_bytes": 0})
    groups = defaultdict(lambda: {"logical_bytes": 0, "capacity_bytes": 0})
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            key, value = parts
            if key not in {"dtype", "trees", "nodes", "validated_trees"} or key in metadata:
                raise ValueError(f"Unexpected metadata: {line}")
            metadata[key] = value if key == "dtype" else int(value)
        elif len(parts) == 4:
            name, group, logical, capacity = parts
            logical, capacity = int(logical), int(capacity)
            if group not in GROUPS or not 0 <= logical <= capacity:
                raise ValueError(f"Invalid field: {line}")
            if name in fields and fields[name]["group"] != group:
                raise ValueError(f"Inconsistent field group: {name}")
            fields[name]["group"] = group
            for target in (fields[name], groups[group]):
                target["logical_bytes"] += logical
                target["capacity_bytes"] += capacity
        else:
            raise ValueError(f"Malformed worker line: {line}")
    if (set(metadata) != {"dtype", "trees", "nodes", "validated_trees"}
            or metadata["dtype"] not in {"float32", "float64"}
            or metadata["trees"] <= 0
            or metadata["nodes"] < metadata["trees"]
            or metadata["validated_trees"] != metadata["trees"]
            or set(groups) != GROUPS):
        raise ValueError("Incomplete or unvalidated worker output")
    totals = {key: sum(group[key] for group in groups.values())
              for key in ("logical_bytes", "capacity_bytes")}
    totals["spare_capacity_bytes"] = totals["capacity_bytes"] - totals["logical_bytes"]
    return dict(**metadata, fields=dict(fields), groups=dict(groups), totals=totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="*", help="Default: saved oracle fixtures")
    parser.add_argument("--worker", type=Path, default=ROOT / "build/model-memory-worker")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/private/model-memory/results.json")
    args = parser.parse_args()
    models = args.models
    expected = {}
    if not models:
        manifest = json.loads((ROOT / "tests/fixtures/manifest.json").read_text())
        models = [ROOT / "build/balsa-reference" / item["balsa_path"] for item in manifest["fixtures"]]
        expected = {str(path.resolve()): item["sha256"]
                    for path, item in zip(models, manifest["fixtures"])}
    worker = args.worker.resolve()
    records = []
    for path in models:
        path = path.resolve()
        digest = sha256(path)
        if str(path) in expected and digest != expected[str(path)]:
            raise ValueError(f"Fixture hash mismatch: {path}")
        command = [str(worker), str(path)]
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        record = parse_worker(result.stdout)
        record.update(path=str(path), sha256=digest, file_bytes=path.stat().st_size,
                      command=command, worker_stdout=result.stdout)
        records.append(record)
    sources = [ROOT / "tools/model_memory_worker.mojo", ROOT / "tools/model_memory.py",
               *sorted((ROOT / "src/pyre").glob("*.mojo")),
               *sorted((ROOT / "build/balsa-reference/src/balsa").glob("*.mojo"))]
    report = dict(schema_version=1, accounting="logical lengths and reserved list capacities; not RSS",
                  platform=platform.platform(), worker_sha256=sha256(worker),
                  balsa_commit=(ROOT / "build/balsa-reference/.reference-commit").read_text().strip(),
                  source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in sources},
                  records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("| Model | Nodes | Accounted capacity (bytes) | Traversal/base payload | Unused semantic payload | Statistics |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for record in records:
        groups = record["groups"]
        print(f"| {Path(record['path']).stem} | {record['nodes']} | {record['totals']['capacity_bytes']} | "
              f"{groups['inference']['logical_bytes']} | {groups['unsupported_semantics']['logical_bytes']} | "
              f"{groups['statistics']['logical_bytes']} |")


if __name__ == "__main__":
    main()
