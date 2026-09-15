"""Capture one full-batch margin launch with Nsight Compute, separate from timings.

The first prediction in the worker runs on the requested complete batch, before
its oracle and reused-transfer checks. Kernel replay flushes caches each pass;
these counters describe controlled cold-cache traversal, not warm call latency.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess

from profile_gpu import ROOT, sha

SECTIONS = (
    'LaunchStats', 'Occupancy', 'SpeedOfLight', 'MemoryWorkloadAnalysis',
    'SchedulerStats', 'WarpStateStats', 'SourceCounters',
)


def parse_metrics(raw):
    """Reject empty exports and retain metric units and full kernel identity."""
    lines = raw.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID",')), None)
    if start is None:
        raise ValueError('Nsight Compute export has no CSV metric header')
    records = list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
    if records and 'Metric Name' in records[0]:
        metrics = [r for r in records if r.get('Metric Name') and r.get('Metric Value')]
    else:
        # NCU 2026 raw CSV: metric columns, followed by units and kernel rows.
        if len(records) != 2 or records[0].get('ID') != '':
            raise ValueError('Expected a units row and one kernel row')
        units, values = records
        identity = ('ID', 'Kernel Name', 'Block Size', 'Grid Size')
        metrics = [dict({k: values[k] for k in identity},
                        **{'Metric Name': name, 'Metric Unit': units[name], 'Metric Value': value})
                   for name, value in values.items() if '__' in name and value]
    if not metrics or len({r['ID'] for r in metrics}) != 1:
        raise ValueError('Expected exactly one profiled kernel with metrics')
    if any('pyre__gpu_kernel_predict_kerne' not in r['Kernel Name'] for r in metrics):
        raise ValueError('Export did not select the Pyre prediction kernel')
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'build/gpu-corpus/manifest.json')
    p.add_argument('--binary', type=Path, default=ROOT/'build/benchmark-gpu-worker')
    p.add_argument('--case', required=True)
    p.add_argument('--rows', type=int, required=True)
    p.add_argument('--output', type=Path, required=True, help='Fresh report prefix')
    a = p.parse_args()
    manifest_path, binary, out = a.manifest.resolve(), a.binary.resolve(), a.output.resolve()
    manifest = json.loads(manifest_path.read_text())
    case = next((c for c in manifest['cases'] if c['name'] == a.case), None)
    if case is None or str(a.rows) not in case['inputs'] or a.rows <= 0:
        p.error('case and positive row count must be present in hashed manifest')
    model, oracle = (manifest_path.parent/case[k] for k in ('model_path', 'oracle_path'))
    if sha(model) != case['sha256'] or sha(oracle) != case['oracle_sha256']:
        raise ValueError('Corpus hash mismatch')
    if out.suffix:
        p.error('output must be a suffix-free report prefix')
    out.parent.mkdir(parents=True, exist_ok=True)
    if any(out.parent.glob(out.name+'.*')):
        p.error('report prefix already exists')
    g = case['generation']
    worker = [str(binary), str(model), case['dtype'], str(a.rows), '1', 'validate',
              str(g['seed']), str(g['missing']), g['pattern'], str(oracle), '0.1', str(case['tolerance'])]
    command = ['ncu', '--kernel-name', 'regex:pyre__gpu_kernel_predict_kerne',
               '--launch-count', '1', '--replay-mode', 'kernel', '--cache-control', 'all',
               '--clock-control', 'none', '--export', str(out)]
    for section in SECTIONS:
        command.extend(['--section', section])
    command.extend(worker)
    metadata = dict(command=command, case=case, rows=a.rows,
                    binary_sha256=sha(binary), manifest_sha256=sha(manifest_path),
                    source_sha256={str(f.relative_to(ROOT)): sha(f) for f in
                                   sorted((ROOT/'src/pyre').glob('*.mojo'))},
                    worker_source_sha256=sha(ROOT/'tools/benchmark_gpu_worker.mojo'),
                    tool_sha256=sha(Path(__file__)), lock_sha256=sha(ROOT/'pixi.lock'),
                    git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                    gpu_before=subprocess.check_output(['nvidia-smi'], text=True),
                    ncu=subprocess.check_output(['ncu', '--version'], text=True),
                    note='First full-batch margin launch; kernel replay with cache flushes; clocks uncontrolled. Instrumented data is not a performance result. Source hashes describe current files, not proof of binary provenance.')
    try:
        run = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=600)
        out.with_suffix('.log').write_text(run.stdout)
        metadata['exit_code'] = run.returncode
        run.check_returncode()
        expected = f"validated {a.rows} {case['oracle_rows']}"
        if expected not in run.stdout.splitlines():
            raise RuntimeError('Worker did not complete CPU/GPU and oracle validation')
        export = ['ncu', '--import', str(out.with_suffix('.ncu-rep')), '--page', 'raw', '--csv']
        metadata['export_command'] = export
        raw = subprocess.check_output(export, text=True, stderr=subprocess.PIPE)
        out.with_suffix('.csv').write_text(raw)
        metadata['metrics'] = parse_metrics(raw)
        metadata['validated'] = True
    except Exception as error:
        metadata['error'] = str(error)
        raise
    finally:
        metadata['gpu_after'] = subprocess.check_output(['nvidia-smi'], text=True)
        out.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(out.with_suffix('.metadata.json'))


if __name__ == '__main__':
    main()
