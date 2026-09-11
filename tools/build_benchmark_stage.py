"""Rebuild archived benchmark implementations, optionally timing the first call."""
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['baseline', 'threaded', 'final'])
    parser.add_argument('--cold', action='store_true')
    args = parser.parse_args()
    if args.cold and args.stage == 'baseline':
        parser.error('Cold instrumentation requires the threaded API')
    subprocess.run(['python', 'tools/prepare_balsa.py'], cwd=ROOT, check=True)
    output = ROOT / 'build' / args.stage
    module = output / 'module' / 'pyre'
    module.mkdir(parents=True, exist_ok=True)
    if args.stage == 'final':
        forest = ROOT / 'src/pyre/forest.mojo'
        worker = ROOT / 'tools/benchmark_worker.mojo'
    else:
        source = ROOT / 'docs/benchmarks/sources' / args.stage
        forest = source / 'forest.mojo'
        worker = source / 'benchmark_worker.mojo'
    shutil.copy2(forest, output / 'forest.mojo')
    shutil.copy2(forest, module / 'forest.mojo')
    (module / '__init__.mojo').write_text('from .forest import Forest\n')
    source = worker.read_text()
    if args.cold:
        marker = '    # Warm code/data/allocator and (later) the shared thread pool before timing.'
        assert marker in source, 'Warmup marker changed; review cold instrumentation'
        source = source.replace(marker, '''    var cold_start = perf_counter_ns()
    if classifier:
        var result = forest.predict_proba(x, num_threads=threads)
        keep(result)
    else:
        var result = forest.predict_margin[width](x, num_threads=threads)
        keep(result)
    var cold = perf_counter_ns() - cold_start
''' + marker)
        source = source.replace('    print(Float64(elapsed) / 1e9)',
            '    print(Float64(cold) / 1e9, Float64(elapsed) / Float64(loops) / 1e9)')
    stem = 'cold' if args.cold else 'benchmark'
    generated = output / f'{stem}_worker.mojo'
    generated.write_text(source)
    subprocess.run(['mojo', 'build', '-I', str(module.parent), '-I',
        str(ROOT / 'build/balsa-reference/src'), str(generated), '-o',
        str(output / f'{stem}-worker')], cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
