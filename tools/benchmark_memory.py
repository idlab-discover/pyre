"""Linux process peak RSS (includes model/input/setup), separate from timing."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
records = []
for stage, threads in [('baseline', 1), ('threaded', 8), ('final', 8)]:
    for operation in ['margin', 'proba']:
        for repeat in range(3):
            cmd = ['taskset', '-c', '8-15', f'build/{stage}/benchmark-worker',
                   '4096', '64', '6', 'float32', operation, '8', str(threads), '100']
            with tempfile.TemporaryFile(mode='w+') as output:
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=output, stderr=output, text=True)
                _, status, usage = os.wait4(proc.pid, 0)
                proc.returncode = os.waitstatus_to_exitcode(status)
                output.seek(0)
                result = output.read()
                if proc.returncode:
                    raise RuntimeError(f'{cmd}: {result}')
            records.append(dict(stage=stage, threads=threads, rows=4096, trees=64,
                                depth=6, operation=operation, repeat=repeat,
                                process_peak_rss_kib=usage.ru_maxrss))
(ROOT / 'docs/private/benchmarks').mkdir(parents=True, exist_ok=True)
(ROOT / 'docs/private/benchmarks/memory.json').write_text(json.dumps(records, indent=2) + '\n')
for record in records:
    print(record)
