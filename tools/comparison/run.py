"""Serial, randomized subprocess orchestration and Parquet evidence capture."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import uuid

from .schema import VERSION, identity, parquet, sha

ROOT = Path(__file__).resolve().parents[2]
ROUTES = {0: "generic", 1: "symmetric_direct", 2: "symmetric_bins", 3: "constant", -1: "gpu_packed"}


def cells(manifest, matrix):
    for model, device, engine in itertools.product(manifest["models"], matrix["devices"], matrix["engines"]):
        threads = matrix["cpu_threads"] if device == "cpu" else [None]
        chunks = matrix["nvforest_chunks"] if engine == "nvforest" else [None]
        groups = matrix["pyre_gpu_row_groups"] if engine == "pyre" and device == "gpu" else [None]
        for inp, thread, chunk, group in itertools.product(model["inputs"], threads, chunks, groups):
            if inp["batch_rows"] not in matrix["batch_rows"]:
                continue
            cell = dict(model_id=model["model_id"], engine=engine, device=device,
                        batch_rows=inp["batch_rows"], cpu_threads=thread,
                        nvforest_chunk=chunk, pyre_gpu_row_group=group)
            cell["cell_id"] = identity(cell)
            yield cell, model, inp


def parse_native(output, count):
    metadata, rows = {}, []
    for line in output.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "gate" and len(parts) == 3:
            metadata.update(max_abs_error=float(parts[1]), max_scaled_error=float(parts[2]))
        elif parts[0] == "meta" and len(parts) == 3:
            metadata.update(route=ROUTES[int(parts[1])], effective_row_group=int(parts[2]) or None)
        elif parts[0] == "plan" and len(parts) == 6:
            metadata["gpu_plan"] = dict(zip(
                ["chunk_shift", "threads", "rows_per_thread", "trees_per_thread", "staged"],
                map(int, parts[1:])))
        elif parts[0] == "sample" and len(parts) == 4:
            rows.append(dict(sample=int(parts[1]), loops=int(parts[2]), elapsed_ns=int(parts[3])))
        else:
            raise ValueError(f"Unexpected native worker output: {line}")
    if set(metadata) - {"gpu_plan"} != {"route", "effective_row_group", "max_abs_error", "max_scaled_error"}:
        raise ValueError("Worker omitted oracle or execution metadata")
    if metadata["route"] == "gpu_packed" and "gpu_plan" not in metadata:
        raise ValueError("GPU worker omitted its launch plan")
    result = [metadata | row for row in rows]
    validate_samples(result, count)
    return result


def validate_samples(rows, count):
    if len(rows) != count or [r["sample"] for r in rows] != list(range(count)):
        raise ValueError("Worker sample count/index mismatch")
    for row in rows:
        if any(type(row[k]) is not int or row[k] <= 0 for k in ["loops", "elapsed_ns"]):
            raise ValueError("Invalid interval duration/loop count")
        if any(not math.isfinite(row[k]) or row[k] < 0 for k in ["max_abs_error", "max_scaled_error"]):
            raise ValueError("Invalid correctness measurement")


def command_output(command):
    try:
        p = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=10)
        return {"exit": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def verify_corpus(corpus, manifest):
    for model in manifest["models"]:
        for key in ["model", "native"]:
            if sha(corpus / model[key + "_path"]) != model[key + "_sha256"]:
                raise ValueError(f"Changed {key} artifact: {model['model_id']}")
        for inp in model["inputs"]:
            for key in ["input", "oracle"]:
                if sha(corpus / inp[key + "_path"]) != inp[key + "_sha256"]:
                    raise ValueError(f"Changed {key} artifact: {model['model_id']}")


def run(matrix, corpus, output, cpus=None):
    (ROOT / "build").mkdir(exist_ok=True)
    lock = (ROOT / "build/comparison.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another comparison is already running in this checkout") from None
    corpus, output = Path(corpus).resolve(), Path(output).resolve()
    worker = output / "comparison-worker"
    manifest = json.loads((corpus / "manifest.json").read_text())
    if manifest.get("schema_version") != VERSION:
        raise ValueError("Unsupported corpus schema")
    requested = set(itertools.product(manifest["matrix"]["families"],
                                      manifest["matrix"]["tree_counts"],
                                      manifest["matrix"]["depth_limits"]))
    recorded = [(m["family"], m["requested_tree_count"], m["requested_depth_limit"])
                for m in manifest["models"] + manifest["excluded"]]
    if set(recorded) != requested or len(recorded) != len(requested):
        raise ValueError("Corpus preparation is incomplete or contains duplicate configurations")
    # Allow execution-policy changes, but never silently substitute model/input axes.
    for key in ["seed", "features", "training_rows"]:
        if matrix[key] != manifest["matrix"][key]:
            raise ValueError(f"Corpus matrix differs in {key}; prepare a matching corpus")
    for key in ["families", "tree_counts", "depth_limits", "batch_rows"]:
        if not set(matrix[key]) <= set(manifest["matrix"][key]):
            raise ValueError(f"Requested {key} are absent from the corpus")
    manifest["models"] = [m for m in manifest["models"] if m["family"] in matrix["families"]
                          and m["requested_tree_count"] in matrix["tree_counts"]
                          and m["requested_depth_limit"] in matrix["depth_limits"]]
    if not manifest["models"]:
        raise ValueError("No models selected; inspect corpus exclusions")
    verify_corpus(corpus, manifest)
    if cpus is not None:
        os.sched_setaffinity(0, cpus)
    if "cpu" in matrix["devices"] and max(matrix["cpu_threads"]) > len(os.sched_getaffinity(0)):
        raise ValueError("Requested CPU thread limit exceeds the available affinity set")
    gpu_worker = output / "comparison-gpu-worker"
    output.mkdir(parents=True, exist_ok=False)
    compile_sources = [ROOT / "tools/comparison_gpu_worker.mojo", ROOT / "tools/comparison_worker.mojo", ROOT / "tools/cpu_samples.mojo",
                       ROOT / "tools/prepare_balsa.py", ROOT / "pixi.lock",
                       *sorted((ROOT / "src/pyre").rglob("*.mojo")),
                       *sorted((ROOT / "build/balsa-reference/src").rglob("*.mojo"))]
    before_build = {str(p): sha(p) for p in compile_sources}
    if "pyre" in matrix["engines"]:
        for device in matrix["devices"]:
            source = "comparison_gpu_worker.mojo" if device == "gpu" else "comparison_worker.mojo"
            binary = gpu_worker if device == "gpu" else worker
            flags = [] if device == "gpu" else ["--target-accelerator=none", "-D", "PYRE_CPU_ONLY"]
            build_command = ["pixi", "run", "mojo", "build", *flags,
                             "-I", "src", "-I", "build/balsa-reference/src",
                             "tools/" + source, "-o", str(binary)]
            built = subprocess.run(build_command, cwd=ROOT, text=True, capture_output=True)
            (output / f"build-{device}.log").write_text(built.stdout + built.stderr)
            (output / f"build-command-{device}.json").write_text(json.dumps(build_command))
            built.check_returncode()
        if any(sha(path) != digest for path, digest in before_build.items()):
            raise ValueError("Sources changed while compiling; rerun with a fresh output directory")
    (output / "raw").mkdir()
    run_id = str(uuid.uuid4())
    sources = [Path(__file__), *Path(__file__).parent.glob("*.py"), ROOT / "tools/compare.py",
               ROOT / "tools/comparison_worker.mojo", ROOT / "tools/cpu_samples.mojo",
               ROOT / "pixi.lock", *sorted(Path(__file__).parent.glob("requirements*")),
               *sorted((ROOT / "src/pyre").glob("*.mojo"))]
    sources.extend(compile_sources)
    sources.extend(p for p in [worker, gpu_worker] if p.is_file())
    head = command_output(["git", "rev-parse", "HEAD"])
    provenance = dict(started_utc=datetime.now(timezone.utc).isoformat(), schema_version=VERSION, run_id=run_id, matrix=matrix,
        corpus_manifest_sha256=sha(corpus / "manifest.json"), corpus=str(corpus),
        git_head=head, git_diff=command_output(["git", "diff", "HEAD"]),
        git_status=command_output(["git", "status", "--short"]),
        source_sha256={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in sources},
        platform=platform.platform(), machine=platform.machine(), python=sys.version,
        cpu_affinity=sorted(os.sched_getaffinity(0)), cpu=command_output(["lscpu", "-J"]),
        gpu=command_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv"]),
        packages=command_output([sys.executable, "-c", "import importlib.metadata,json; print(json.dumps(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())))"]),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_state=command_output(["nvidia-smi", "--query-gpu=uuid,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu", "--format=csv"]),
        load_average=os.getloadavg(),
        cpu_frequency_policy={str(p): p.read_text().strip() for p in
            Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor")},
        protocol="Alternating engine order within matched groups; raw intervals; allocating completed resident calls; warmup/calibration excluded; independent process per cell/repetition; full-output GTIL gate; no ranking.")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    planned = list(cells(manifest, matrix))
    attempts, measurements = [], []
    rng = random.Random(matrix["seed"])

    def checkpoint():
        parquet(output / "measurements.parquet", measurements, "measurements")
        parquet(output / "attempts.parquet", attempts, "attempts")

    checkpoint()
    try:
        for repetition in range(matrix["processes"]):
            groups = {}
            for index, (cell, _, _) in enumerate(planned):
                key = (cell["model_id"], cell["device"], cell["batch_rows"], cell["cpu_threads"])
                groups.setdefault(key, []).append(index)
            keys = list(groups)
            rng.shuffle(keys)
            order = [i for key in keys for i in (groups[key] if repetition % 2 == 0 else groups[key][::-1])]
            for index in order:
                cell, model, inp = planned[index]
                base = cell | dict(schema_version=VERSION, run_id=run_id, run_purpose=matrix["purpose"], repetition=repetition)
                name = f"r{repetition}-{cell['cell_id'][:16]}"
                reqpath, stdout, stderr = [output / "raw" / (name + suffix) for suffix in [".request.json", ".stdout", ".stderr"]]
                record = base | dict(request_path=str(reqpath.relative_to(output)),
                                     stdout_path=str(stdout.relative_to(output)), stderr_path=str(stderr.relative_to(output)))
                request = cell | dict(native_input=matrix.get("native_input", "prepared"), model={k: v for k, v in model.items() if k not in {"trees", "inputs"}}, input_path=str(corpus / inp["input_path"]),
                    oracle_path=str(corpus / inp["oracle_path"]), model_path=str(corpus / model["model_path"]),
                    native_path=str(corpus / model["native_path"]), samples=matrix["samples"],
                    min_seconds=matrix["min_seconds"],
                    tolerance=model["native_tolerance"] if cell["engine"] == "native" else model["tolerance"])
                reqpath.write_text(json.dumps(request, indent=2) + "\n")
                env = os.environ | {"OMP_NUM_THREADS": str(cell["cpu_threads"] or 1), "OMP_DYNAMIC": "FALSE",
                    "OPENBLAS_NUM_THREADS": str(cell["cpu_threads"] or 1), "MKL_NUM_THREADS": str(cell["cpu_threads"] or 1)}
                if cell["engine"] == "pyre":
                    command = [str(gpu_worker if cell["device"] == "gpu" else worker), cell["device"], request["model_path"], model["model_dtype"],
                        request["input_path"], str(cell["batch_rows"]), str(model["features"]), request["oracle_path"],
                        str(cell["cpu_threads"] or 1), str(matrix["samples"]), str(matrix["min_seconds"]),
                        str(request["tolerance"]), str(cell["pyre_gpu_row_group"] or 0)]
                else:
                    command = [sys.executable, "-m", "comparison.reference", str(reqpath)]
                try:
                    with stdout.open("w") as out, stderr.open("w") as err:
                        result = subprocess.run(command, cwd=ROOT / "tools", env=env, stdout=out, stderr=err,
                                                timeout=matrix["worker_timeout_seconds"])
                    if result.returncode:
                        diagnostic = (stdout.read_text() + "\n" + stderr.read_text())[-2000:]
                        raise RuntimeError(f"Worker exit {result.returncode}: {diagnostic}")
                    if cell["engine"] == "pyre":
                        rows = parse_native(stdout.read_text(), matrix["samples"])
                        for row in rows:
                            settings = dict(cpu_thread_limit=cell["cpu_threads"],
                                gpu_row_group=cell["pyre_gpu_row_group"],
                                policy="automatic" if cell["pyre_gpu_row_group"] is None else "pinned_gpu_rows_per_block",
                                gpu_plan=row.pop("gpu_plan", None))
                            row.update(api_layer="native_mojo", output_dtype=model["model_dtype"],
                                engine_settings_json=json.dumps(settings, sort_keys=True),
                                engine_version=head.get("stdout", "").strip())
                    else:
                        rows = [json.loads(line) for line in stdout.read_text().splitlines() if line.strip()]
                        if len(rows) == 1 and rows[0].get("status") == "unsupported":
                            attempts.append(record | rows[0])
                            continue
                        validate_samples(rows, matrix["samples"])
                    if any(r["max_scaled_error"] > request["tolerance"] for r in rows):
                        raise ValueError("Worker reported an out-of-tolerance gate")
                    dimensions = {k: model[k] for k in ["framework", "family", "model_dtype", "features",
                        "requested_tree_count", "requested_depth_limit", "tree_count", "max_depth", "node_count", "tree_type"]}
                    for row in rows:
                        measurements.append(base | dimensions | row | dict(phase="resident_allocate_completed",
                            gate_tolerance=request["tolerance"],
                            model_sha256=model["model_sha256"], input_sha256=inp["input_sha256"], oracle_sha256=inp["oracle_sha256"]))
                    attempts.append(record | dict(status="ok", reason=None))
                except (OSError, RuntimeError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
                    attempts.append(record | dict(status="failed", reason=str(error)))
                    print("FAILED", model["model_name"], cell["engine"], str(error), flush=True)
                    return False
                if len(attempts) % 25 == 0:
                    checkpoint()
                print("recorded", f"{len(attempts)}/{len(planned)*matrix['processes']}", repetition, model.get("model_name", cell["model_id"]), cell["engine"], cell["device"], cell["batch_rows"], attempts[-1]["status"], flush=True)
    finally:
        checkpoint()
    changed = [path for path, digest in provenance["source_sha256"].items()
               if sha(ROOT / path) != digest]
    if changed:
        (output / "changed-sources.json").write_text(json.dumps(changed, indent=2))
        raise ValueError("Sources changed during measurement; retain run for diagnosis only")
    return not any(a["status"] == "failed" for a in attempts)
