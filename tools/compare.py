"""Train, validate and benchmark current-checkout CPU/GPU inference on matched models."""
import argparse
import json
from pathlib import Path

from comparison.schema import read_matrix

LIBRARIES = {
    'scikit-learn': ['sklearn_random_forest'],
    'xgboost': ['xgboost_depthwise', 'xgboost_lossguide'],
    'catboost': ['catboost_symmetric'],
    'lightgbm': ['lightgbm_leafwise'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['prepare', 'run', 'plan']:
        p = sub.add_parser(name)
        p.add_argument('--matrix', type=Path, default=Path(__file__).resolve().parents[1] / 'benchmarks/cpu-heavy.json')
        p.add_argument('--libraries', nargs='+', choices=[*LIBRARIES, 'nvforest'],
                       help='Reference libraries; Pyre always participates. nvforest alone uses all matrix families.')
        for option in ['trees', 'depths', 'rows', 'threads']:
            p.add_argument('--' + option, type=int, nargs='+')
        p.add_argument('--native-input', choices=['prepared', 'ndarray'])
        p.add_argument('--processes', type=int)
        p.add_argument('--samples', type=int)
        p.add_argument('--min-seconds', type=float)
        if name != 'plan':
            p.add_argument('--output', type=Path, required=True)
        if name == 'run':
            p.add_argument('--corpus', type=Path, required=True)
            p.add_argument('--cpus', help='Comma-separated CPU IDs, inherited by every worker')
    report = sub.add_parser('report')
    report.add_argument('--results', type=Path, required=True)
    report.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'report':
        from comparison.report import report
        report(args.results, args.output)
        return
    matrix = read_matrix(args.matrix)
    for option, key in [('trees', 'tree_counts'), ('depths', 'depth_limits'),
                        ('rows', 'batch_rows'), ('threads', 'cpu_threads'),
                        ('processes', 'processes'), ('samples', 'samples'), ('min_seconds', 'min_seconds'),
                        ('native_input', 'native_input')]:
        value = getattr(args, option)
        if value is not None:
            matrix[key] = value
    if args.libraries:
        families = [f for lib in args.libraries for f in LIBRARIES.get(lib, [])]
        if families:
            matrix['families'] = list(dict.fromkeys(families))
        matrix['engines'] = ['pyre'] + (['native'] if families else []) + (['nvforest'] if 'nvforest' in args.libraries else [])
    from comparison.schema import validate_matrix
    validate_matrix(matrix)
    if args.command == 'prepare':
        from comparison.corpus import prepare
        prepare(matrix, args.output)
    elif args.command == 'run':
        from comparison.run import run
        cpus = {int(v) for v in args.cpus.split(',')} if args.cpus else None
        if not run(matrix, args.corpus, args.output, cpus=cpus):
            raise SystemExit('Some attempts failed; inspect attempts.parquet and raw logs')
    else:
        print(json.dumps(matrix, indent=2))


if __name__ == '__main__':
    main()
