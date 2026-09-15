"""Manifest-driven, correctness-gated GPU measurements exported as pyperf JSON.

Each phase warms and calibrates to >=100 ms inside a fresh native process.
Device intervals include stream scheduling gaps and are not kernel-only time.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import pyperf

ROOT = Path(__file__).resolve().parents[1]
PHASES = ('device', 'reuse', 'allocate', 'upload', 'download', 'prepare', 'read', 'decode', 'samples', 'end_to_end', 'pipeline',
          'upload_into', 'pinned_fill', 'pinned_upload', 'pinned_allocate', 'batch_allocate', 'ingest_a', 'ingest_b', 'ingest_c')
COLD = ('context', 'read', 'decode', 'prepare', 'samples', 'upload', 'first_prediction', 'download', 'batch_allocate', 'pinned_allocate')


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def parse_worker(raw, requested):
    records = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) not in (11, 13) or fields[0] not in PHASES:
            raise RuntimeError(f'Unexpected worker output: {line}')
        phase, total, loops = fields[0], float(fields[1]), int(fields[2])
        if not math.isfinite(total) or total <= 0 or loops < 1:
            raise RuntimeError(f'Invalid timing: {line}')
        records.append(dict(phase=phase, total_seconds=total, native_loops=loops,
                            seconds_per_call=total / loops, cold_ns=dict(zip(COLD, map(int, fields[3:])))))
    if [r['phase'] for r in records] != list(requested):
        raise RuntimeError('Worker did not return each requested phase exactly once')
    return records


def memory_payload(case, rows):
    width = {'float32': 4, 'float64': 8}[case['dtype']]
    samples = rows * case['features'] * width
    result = rows * width
    forest = case['statistics']['nodes'] * (20 + 2 * width) + case['statistics']['trees'] * 4
    return dict(sample_bytes=samples, margin_bytes=result, prepared_forest_bytes=forest,
                upload_h2d_bytes=samples, download_d2h_bytes=result,
                prepare_h2d_bytes=forest, resident_prediction_transfer_bytes=0,
                pipeline_transfer_bytes=samples+result, pinned_staging_reserved_bytes=samples,
                ingestion_allocation_calls_per_batch={'A': {'sample_host': 1, 'sample_device': 1, 'result_host': 1},
                    'B': {'sample_host': 1, 'sample_device': 0, 'result_host': 1},
                    'C': {'sample_host': 0, 'sample_device': 0, 'result_host': 1}},
                note='Logical payload accounting from current source; excludes allocator rounding, runtime pools and temporary CPU/oracle storage. Transfer counts require separate trace confirmation.')


def summaries(records):
    groups = {}
    for r in records:
        groups.setdefault((r['case'], r['rows'], r['phase']), []).append(r)
    out = []
    for (case, rows, phase), group in sorted(groups.items()):
        values = [r['seconds_per_call'] for r in group]
        median = statistics.median(values)
        out.append(dict(case=case, rows=rows, phase=phase, processes=len(values),
                        median_seconds=median, min_seconds=min(values), max_seconds=max(values),
                        mad_seconds=statistics.median(abs(v-median) for v in values),
                        cv=statistics.stdev(values)/statistics.mean(values) if len(values)>1 else None,
                        rows_per_second=rows/median))
    # Preparation amortization as reuse counts, not a speedup versus another implementation.
    lookup = {(s['case'], s['rows'], s['phase']): s['median_seconds'] for s in out}
    for s in out:
        prepare = lookup.get((s['case'], s['rows'], 'prepare'))
        if prepare is not None and s['phase'] in ('reuse', 'pipeline'):
            s['calls_to_reduce_preparation_to_10pct_of_total'] = math.ceil(9*prepare/s['median_seconds'])
    for s in out:
        if s['phase'] not in ('ingest_b', 'ingest_c'):
            continue
        prefix = (s['case'], s['rows'])
        baseline = lookup.get((*prefix, 'ingest_a'))
        device_setup = lookup.get((*prefix, 'batch_allocate'))
        host_setup = lookup.get((*prefix, 'pinned_allocate')) if s['phase']=='ingest_c' else 0
        if baseline is not None:
            saving = baseline - s['median_seconds']
            s['speedup_vs_ingest_a'] = baseline/s['median_seconds']
            relevant = groups[(s['case'], s['rows'], s['phase'])]
            setup = [r['cold_ns']['batch_allocate'] + (r['cold_ns']['pinned_allocate'] if s['phase']=='ingest_c' else 0)
                     for r in relevant if 'batch_allocate' in r.get('cold_ns', {}) and 'pinned_allocate' in r.get('cold_ns', {})]
            if setup:
                cost = statistics.median(setup)/1e9
                s['median_observed_setup_seconds'] = cost
                s['setup_amortization_calls_vs_a'] = math.ceil(cost/saving) if saving>0 else None
            elif device_setup is not None and host_setup is not None:
                s['warm_allocator_amortization_calls_vs_a'] = math.ceil((device_setup+host_setup)/saving) if saving>0 else None
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, default=ROOT/'build/benchmark-gpu-worker')
    p.add_argument('--manifest', type=Path, default=ROOT/'build/gpu-corpus/manifest.json')
    p.add_argument('--output', type=Path, default=ROOT/'docs/private/benchmarks/gpu/scaling.json')
    p.add_argument('--processes', type=int, default=5)
    p.add_argument('--loops', type=int, default=1, help='Initial calibration loops per phase')
    p.add_argument('--min-time', type=float, default=0.1)
    p.add_argument('--rows', default='1,33,257,4096,16384,65536')
    p.add_argument('--cases', help='Comma-separated manifest case names')
    p.add_argument('--phase', default='all', help='all or comma-separated phases')
    p.add_argument('--seed', type=int, default=20260915, help='Job shuffle seed')
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    if a.processes < 1 or a.loops < 1 or not math.isfinite(a.min_time) or a.min_time < 0.1:
        p.error('processes/loops must be positive and min-time must be >=0.1 seconds')
    rows = list(dict.fromkeys(int(n) for n in a.rows.split(',')))
    if not rows or any(n < 1 for n in rows):
        p.error('rows must be positive')
    phases = PHASES if a.phase == 'all' else tuple(a.phase.split(','))
    if not phases or len(set(phases)) != len(phases) or any(s not in PHASES for s in phases):
        p.error('unknown or repeated phase')
    phases = tuple(phase for phase in PHASES if phase in phases)
    # Do not accidentally benchmark the sanitizer's altered allocator/runtime.
    overrides = {k: v for k, v in os.environ.items() if k.startswith(('MODULAR_', 'MAX_', 'CUDA_'))}
    if any('SANITIZER' in k or 'ALLOCATOR' in k or 'MEMORY_POOL' in k or 'MEMORY_MANAGER' in k for k in overrides):
        p.error(f'remove instrumentation/allocator overrides before measuring: {list(overrides)}')
    manifest_path = a.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 1:
        p.error('unsupported performance corpus schema')
    cases = manifest['cases']
    if a.cases:
        requested = set(a.cases.split(','))
        cases = [f for f in cases if f['name'] in requested]
        if {f['name'] for f in cases} != requested:
            p.error('unknown manifest case')
    binary, output = a.binary.resolve(), a.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    details_path = output.with_suffix('.details.json')
    raw_path = output.with_suffix('.raw.jsonl')
    telemetry_path = output.with_suffix('.gpu.csv')
    if any(f.exists() for f in (output, details_path, raw_path, telemetry_path)):
        p.error('output exists; choose a fresh output path')
    for case in cases:
        if case['dtype'] not in ('float32', 'float64'):
            p.error('unsupported manifest dtype')
        for field, hash_field in [('model_path', 'sha256'), ('oracle_path', 'oracle_sha256')]:
            if sha(manifest_path.parent/case[field]) != case[hash_field]:
                raise ValueError(f'{case["name"]}: {field} hash mismatch')
        if any(str(n) not in case['inputs'] for n in rows):
            p.error(f'{case["name"]}: requested rows absent from hashed corpus')
    sources = sorted((ROOT/'src/pyre').glob('*.mojo')) + sorted((ROOT/'tools').glob('*gpu*'))
    details = dict(command=sys.argv, arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        gpu_before=subprocess.check_output(['nvidia-smi'], text=True), environment=overrides,
        binary_sha256=sha(binary), source_sha256={str(f.relative_to(ROOT)): sha(f) for f in sources if f.is_file()},
        lock_sha256=sha(ROOT/'pixi.lock'), manifest_sha256=sha(manifest_path), corpus=manifest,
        phases=dict(reuse='Completed resident predict_margin_into; reused input/output',
                    device='Private resident launches in device timer; includes stream scheduling gaps',
                    end_to_end='Existing host samples -> allocated upload -> allocated prediction -> host result; excludes sample creation/model setup/disk',
                    pipeline='Sample allocation/fill -> allocated upload -> allocated prediction -> host result; excludes model setup/disk',
                    read='Repeated model file read; OS cache is uncontrolled, not a cold disk measurement',
                    decode='Balsa decode/validation plus owned encoded-buffer copy; no file read',
                    cold='First observed operations in each process, not guaranteed cold OS/GPU caches'),
        records=[], validation=[], status='running')
    def command(case, n, phase):
        g = case['generation']
        return [str(binary), str(manifest_path.parent/case['model_path']), case['dtype'], str(n), str(a.loops), phase,
                str(g['seed']), str(g['missing']), g['pattern'], str(manifest_path.parent/case['oracle_path']),
                str(a.min_time), str(case['tolerance'])]
    runs = {}
    telemetry = None
    with raw_path.open('x') as raw_file, telemetry_path.open('x') as gpu_file:
        try:
            telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,pstate,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used',
                '--format=csv', '--loop-ms=500'], stdout=gpu_file, stderr=subprocess.STDOUT)
            # Gate the entire requested matrix before recording any timed results.
            for case in cases:
                for n in rows:
                    cmd = command(case, n, 'validate')
                    raw = subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=a.timeout)
                    if raw.strip() != f'validated {n} {case["oracle_rows"]}':
                        raise RuntimeError(f'Invalid validation output: {raw}')
                    record = dict(case=case['name'], rows=n, command=cmd, stdout=raw)
                    details['validation'].append(record)
                    raw_file.write(json.dumps(dict(kind='validation', **record))+'\n'); raw_file.flush()
                print(f'Parity passed: {case["name"]} ({len(rows)} batches)', flush=True)
            if not a.validate_only:
                jobs = [(f, n, repeat) for f in cases for n in rows for repeat in range(a.processes)]
                random.Random(a.seed).shuffle(jobs)
                for index, (case, n, repeat) in enumerate(jobs):
                    # Every selected phase runs in this fresh case/repetition
                    # process, with independent warmup and calibration.
                    requests = [('all' if phases == PHASES else ','.join(phases), phases)]
                    for phase, expected in requests:
                        cmd = command(case, n, phase)
                        started = time.time()
                        raw = subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=a.timeout)
                        raw_file.write(json.dumps(dict(kind='timing', command=cmd, stdout=raw, started_unix=started))+'\n'); raw_file.flush()
                        for record in parse_worker(raw, expected):
                            if record['total_seconds'] < a.min_time:
                                raise RuntimeError('Worker did not meet calibration duration')
                            record.update(case=case['name'], rows=n, process=repeat, dtype=case['dtype'],
                                          input_sha256=case['inputs'][str(n)]['sha256'], payload=memory_payload(case,n))
                            key = f'{case["name"]}/rows={n}/{record["phase"]}'
                            runs.setdefault(key, []).append(pyperf.Run([record['seconds_per_call']], metadata={
                                'name': key, 'loops': 1, 'unit': 'second', 'native_loops': record['native_loops']}))
                            details['records'].append(record)
                    print(f'{index+1}/{len(jobs)} {case["name"]} rows={n}', flush=True)
            details['status'] = 'complete'
        except BaseException as error:
            details['status'] = 'failed'
            details['error'] = repr(error)
            if isinstance(error, subprocess.CalledProcessError):
                details['failed_worker'] = dict(command=error.cmd, stdout=error.output)
            raise
        finally:
            if telemetry is not None:
                telemetry.terminate()
                telemetry.wait(timeout=10)
            details['gpu_after'] = subprocess.check_output(['nvidia-smi'], text=True)
            details['summary'] = summaries(details['records'])
            details_path.write_text(json.dumps(details, indent=2)+'\n')
    if runs:
        pyperf.BenchmarkSuite([pyperf.Benchmark(runs[name]) for name in sorted(runs)]).dump(str(output))
    print(details_path)


if __name__ == '__main__':
    main()
