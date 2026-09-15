"""Capture separate Nsight Systems traces of manifest-selected native workloads.

Instrumented worker timings are retained only as trace context, never pyperf data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def summarize(path, phase, timed_loops=None):
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        copies = [dict(r) for r in db.execute('select copyKind, count(*) as count, sum(bytes) as bytes, sum(end-start) as ns from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind')]
        kernels = [dict(r) for r in db.execute('select s.value as name, count(*) as count, avg(k.end-k.start) as mean_ns, min(k.registersPerThread) as registers_min, max(k.registersPerThread) as registers_max from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id group by s.value')]
        api = []
        for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
            if table in tables:
                api.extend(dict(r) for r in db.execute(f'select s.value as name, count(*) as count, sum(a.end-a.start) as ns from {table} a join StringIds s on a.nameId=s.id group by s.value order by ns desc'))
        result = dict(copies=copies, kernels=kernels, api=api, tables=sorted(tables))
        if 'CUDA_GPU_MEMORY_USAGE_EVENTS' in tables:
            result['tracked_memory_events'] = [dict(r) for r in db.execute(
                'select e.label as kind,m.memoryOperationType as operation,count(*) as count,sum(m.bytes) as bytes from CUDA_GPU_MEMORY_USAGE_EVENTS m join ENUM_CUDA_MEM_KIND e on m.memKind=e.id group by m.memKind,m.memoryOperationType')]
            live, peak = {}, {}
            for event in db.execute('select memKind,memoryOperationType,bytes from CUDA_GPU_MEMORY_USAGE_EVENTS order by start'):
                kind, operation, size = event
                if operation not in (0,1):
                    raise RuntimeError('Unknown memory operation type')
                live[kind] = live.get(kind,0) + (size if operation==0 else -size)
                peak[kind] = max(peak.get(kind,0),live[kind])
            result['peak_tracked_bytes_by_kind'] = peak
            result['memory_note'] = 'Operation 0 allocation / 1 free confirmed by correlated cuMemAlloc/cuMemFree calls. Tracked CUDA events include runtime host-pool allocation, not all process RSS or virtual reservations.'
        if phase == 'reuse':
            last_copy = db.execute('select max(end) from CUPTI_ACTIVITY_KIND_MEMCPY').fetchone()[0]
            resident = list(db.execute('select start, end, registersPerThread, gridX, blockX from CUPTI_ACTIVITY_KIND_KERNEL where start > ? order by start', (last_copy,)))
            if not resident:
                raise RuntimeError('Trace has no resident kernel suffix after validation copies')
            start, end = resident[0]['start'], resident[-1]['end']
            durations = [r['end']-r['start'] for r in resident]
            transfers = db.execute('select count(*) from CUPTI_ACTIVITY_KIND_MEMCPY where start >= ? and start <= ?', (start,end)).fetchone()[0]
            allocs = []
            for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
                if table in tables:
                    allocs.extend(dict(r) for r in db.execute(f"select s.value as name, count(*) as count from {table} a join StringIds s on a.nameId=s.id where a.start>=? and a.start<=? and (s.value like '%Alloc%' or s.value like '%Free%') group by s.value", (start,end)))
            result['resident_suffix'] = dict(kernels=len(resident), transfers=transfers, allocation_api=allocs,
                median_kernel_ns=statistics.median(durations), kernel_total_ns=sum(durations),
                elapsed_ns=end-start, kernel_fraction=sum(durations)/(end-start),
                registers_per_thread=resident[0]['registersPerThread'], grid_x=resident[0]['gridX'], block_x=resident[0]['blockX'],
                boundary='All kernels after final validation D2H copy; includes warmup/calibration, excludes setup. Inspect full timeline for host gaps.')
        if phase in ('pipeline', 'ingest_a', 'ingest_b', 'ingest_c') and timed_loops:
            selected = list(db.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL order by start desc limit ?', (timed_loops,)))
            if len(selected) != timed_loops:
                raise RuntimeError('Incomplete timed kernel interval')
            first_kernel, last_kernel = selected[-1]['start'], selected[0]['end']
            first_upload = db.execute('select start from CUPTI_ACTIVITY_KIND_MEMCPY where end<=? order by end desc limit 1', (first_kernel,)).fetchone()[0]
            previous_copy = db.execute('select max(end) from CUPTI_ACTIVITY_KIND_MEMCPY where end<?', (first_upload,)).fetchone()[0]
            begin = previous_copy if previous_copy is not None else first_upload
            end = db.execute('select end from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? order by start limit 1', (last_kernel,)).fetchone()[0]
            transfer = [dict(r) for r in db.execute('select copyKind,count(*) as count,sum(bytes) as bytes,sum(end-start) as ns from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=? group by copyKind', (begin,end))]
            allocs = []
            for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
                if table in tables:
                    allocs.extend(dict(r) for r in db.execute(f"select s.value as name,count(*) as count from {table} a join StringIds s on a.nameId=s.id where a.start>=? and a.start<=? and (s.value like '%Alloc%' or s.value like '%Free%') group by s.value", (begin,end)))
            memory = []
            if 'CUDA_GPU_MEMORY_USAGE_EVENTS' in tables:
                memory = [dict(r) for r in db.execute('select memKind,memoryOperationType as operation,count(*) as count,sum(bytes) as bytes from CUDA_GPU_MEMORY_USAGE_EVENTS where start>=? and start<=? group by memKind,memoryOperationType', (begin,end))]
            result['timed_ingestion'] = dict(iterations=timed_loops, transfers=transfer, allocation_api=allocs, memory_events=memory,
                boundary='Last measured prediction launches plus their H2D/D2H copies; starts at previous iteration final-copy completion and ends at last D2H completion. CUDA interval only; allocator pools may hide logical allocations.')
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'build/gpu-corpus/manifest.json')
    p.add_argument('--binary', type=Path, default=ROOT/'build/benchmark-gpu-worker')
    p.add_argument('--case', required=True)
    p.add_argument('--rows', type=int, required=True)
    p.add_argument('--phase', choices=['reuse','pipeline','ingest_a','ingest_b','ingest_c'], default='reuse')
    p.add_argument('--output', type=Path, required=True, help='Fresh trace prefix')
    a = p.parse_args()
    manifest_path = a.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    case = next(c for c in manifest['cases'] if c['name']==a.case)
    if str(a.rows) not in case['inputs']:
        p.error('rows not present in hashed manifest')
    model, oracle = (manifest_path.parent/case[k] for k in ('model_path','oracle_path'))
    if sha(model)!=case['sha256'] or sha(oracle)!=case['oracle_sha256']:
        raise ValueError('Corpus hash mismatch')
    out, binary = a.output.resolve(), a.binary.resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    if any(out.parent.glob(out.name+'.*')):
        p.error('trace prefix already exists')
    g = case['generation']
    worker = [str(binary), str(model), case['dtype'], str(a.rows), '1', a.phase, str(g['seed']),
              str(g['missing']),g['pattern'],str(oracle),'0.1',str(case['tolerance'])]
    command = ['nsys','profile','--trace=cuda','--sample=none','--cpuctxsw=none','--cuda-memory-usage=true',
               '--output='+str(out),*worker]
    metadata = dict(command=command, binary_sha256=sha(binary), manifest_sha256=sha(manifest_path),
                    gpu=subprocess.check_output(['nvidia-smi'],text=True),
                    nsys=subprocess.check_output(['nsys','--version'],text=True),
                    note='Instrumented run: not a performance result')
    try:
        completed = subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
        out.with_suffix('.log').write_text(completed.stdout)
        completed.check_returncode()
        export = ['nsys','export','--type=sqlite','--output='+str(out.with_suffix('.sqlite')),str(out.with_suffix('.nsys-rep'))]
        metadata['export_command'] = export
        subprocess.run(export,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        timing_lines = [line.split() for line in completed.stdout.splitlines() if line.startswith(a.phase+' ')]
        loops = int(timing_lines[-1][2]) if timing_lines else None
        metadata['summary'] = summarize(out.with_suffix('.sqlite'),a.phase,loops)
    finally:
        out.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(out.with_suffix('.metadata.json'))


if __name__=='__main__':
    main()
