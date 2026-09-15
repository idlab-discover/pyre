"""Generate a bounded, separate GPU performance corpus with pinned offline tools.

Run with the existing Balsa frameworks environment; see docs/gpu-benchmarking.md.
Synthetic checkpoints use the native Balsa builder executable, never Treelite's
builder. Treelite supplies metadata and independent prediction oracles.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import treelite
import treelite.frontend
import treelite.gtil

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {'numpy': '2.2.6', 'treelite': '4.6.1', 'xgboost-cpu': '2.1.4', 'catboost': '1.2.8'}
ROWS = [1, 33, 257, 4096, 16384, 65536]
SEED = 20260915
MAX_NODES = 262144
MAX_SAMPLE_BYTES = 256 * 1024**2


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def mix(x):
    x = np.asarray(x, dtype=np.uint32).copy()
    x ^= x << np.uint32(13)
    x ^= x >> np.uint32(17)
    x ^= x << np.uint32(5)
    return x


def samples(rows, features, dtype, seed, missing, pattern):
    out = np.empty((rows, features), dtype=dtype)
    for start in range(0, rows, 512):
        stop = min(rows, start + 512)
        index = np.arange(start * features, stop * features, dtype=np.uint32)
        if pattern == 'coherent':
            index %= features
        if pattern == 'legacy':
            values = ((index.astype(np.int64) % 13 - 6) * 0.25).astype(dtype)
        else:
            h = mix(index + seed)
            values = ((h % 4096).astype(np.int32) - 2048).astype(dtype) / 1024
            values[mix(h ^ np.uint32(0x9e3779b9)) % 100 < missing] = np.nan
        out[start:stop] = values.reshape(stop - start, features)
    return out


def screening():
    base = dict(trees=64, depth=6, features=32, dtype='float32', task='regression',
                shape='balanced', missing=0, pattern='divergent', seed=SEED)
    yield 'base', base
    for key, values in [('trees', [1, 16, 256, 1024]), ('depth', [2, 10]),
                        ('features', [8, 128, 512]), ('missing', [1, 10, 50]),
                        ('shape', ['uneven']), ('pattern', ['coherent'])]:
        for value in values:
            yield f'{key}-{value}', dict(base, **{key: value})
    yield 'float64', dict(base, dtype='float64')
    yield 'binary-float32', dict(base, task='binary')
    yield 'binary-float64', dict(base, task='binary', dtype='float64')


def statistics(meta):
    counts, depths, leaf_depths = [], [], []
    for tree in meta['trees']:
        nodes = {n['node_id']: n for n in tree['nodes']}
        counts.append(len(nodes))
        stack = [(0, 0)]
        leaves = []
        while stack:
            node, depth = stack.pop()
            n = nodes[node]
            if 'leaf_value' in n:
                leaves.append(depth)
            else:
                stack.extend([(n['left_child'], depth + 1), (n['right_child'], depth + 1)])
        depths.append(max(leaves))
        leaf_depths.extend(leaves)
    return dict(trees=len(counts), nodes=sum(counts), nodes_per_tree=counts,
                max_depth=max(depths), depth_per_tree=depths,
                min_leaf_depth=min(leaf_depths), mean_leaf_depth=float(np.mean(leaf_depths)))


def trained(directory):
    import xgboost as xgb
    import catboost
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(8192, 32)).astype('float32')
    y = (2*x[:, 0] - x[:, 1] + 0.5*x[:, 2]**2 + np.sin(x[:, 3])).astype('float64')
    x[rng.random(x.shape) < 0.01] = np.nan
    np.savez(directory / 'training.npz', x=x, y=y)
    # Hash logical training arrays: ZIP container metadata is not a model input.
    train_hash = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    common = dict(n_estimators=64, max_depth=6, learning_rate=0.1,
                  n_jobs=1, random_state=SEED, tree_method='hist')
    estimator = xgb.XGBRegressor(**common).fit(x, y)
    model = treelite.frontend.from_xgboost(estimator.get_booster())
    probe = samples(257, 32, 'float32', SEED, 1, 'divergent')
    np.testing.assert_allclose(treelite.gtil.predict(model, probe, nthread=1).reshape(-1),
                               estimator.predict(probe), rtol=2e-5, atol=2e-5)
    yield 'xgboost-64-depth6', model, dict(framework='xgboost-cpu', parameters=common,
                                          training_sha256=train_hash, conversion_tolerance=2e-5)
    params = dict(iterations=64, depth=6, learning_rate=0.1, random_seed=SEED,
                  thread_count=1, verbose=False, allow_writing_files=False, nan_mode='Min')
    estimator = catboost.CatBoostRegressor(**params).fit(x, y)
    source_path = directory / 'catboost-source.json'
    estimator.save_model(source_path, format='json')
    original = json.loads(source_path.read_text())
    source = {k: original[k] for k in ('features_info', 'oblivious_trees', 'scale_and_bias')}
    source_path.write_text(json.dumps(source, sort_keys=True, indent=2) + '\n')
    # Reuse the pinned, source-validated offline numeric adapter.
    spec = importlib.util.spec_from_file_location('balsa_frameworks', ROOT / 'build/balsa-reference/tools/framework_fixtures.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    model = adapter.catboost_numeric_model(source, 32)
    np.testing.assert_allclose(treelite.gtil.predict(model, probe, nthread=1).reshape(-1),
                               estimator.predict(probe), rtol=1e-12, atol=1e-12)
    yield 'catboost-64-depth6', model, dict(framework='catboost', parameters=params,
                                          training_sha256=train_hash, source_sha256=sha(source_path),
                                          conversion_tolerance=1e-12)


def record(name, model_path, config, source, directory, generator, rows):
    model = treelite.Model.deserialize(model_path)
    meta = json.loads(model.dump_as_json())
    dtype, features = meta['threshold_type'], meta['num_feature']
    stats = statistics(meta)
    if stats['nodes'] > MAX_NODES:
        raise ValueError(f'{name}: actual node cap exceeded')
    if dtype != config['dtype'] or features != config['features']:
        raise ValueError(f'{name}: metadata mismatch')
    byte_count = max(rows) * features * np.dtype(dtype).itemsize
    if byte_count > MAX_SAMPLE_BYTES:
        raise ValueError(f'{name}: sample cap exceeded before allocation')
    x = samples(max(rows), features, dtype, config['seed'], config['missing'], config['pattern'])
    # Export native producer bytes once; all row sizes are prefixes. A SHA check
    # verifies the Python oracle producer against the actual worker producer.
    with tempfile.TemporaryDirectory() as tmp:
        native = Path(tmp) / 'samples.bin'
        subprocess.run([str(generator), 'samples', dtype, str(max(rows)), str(features),
                        str(config['seed']), str(config['missing']), config['pattern'], str(native)], check=True)
        if sha(native) != hashlib.sha256(x.tobytes()).hexdigest():
            raise ValueError(f'{name}: native/Python sample bytes differ')
    # Include generated rows plus exact root thresholds, adjacent values,
    # infinities, NaNs and default routing in the independent oracle batch.
    borders = []
    for tree in meta['trees'][:16]:
        root = tree['nodes'][0]
        if 'threshold' not in root:
            continue
        value = np.array(root['threshold'], dtype=dtype)
        for v in (np.nextafter(value, -np.inf, dtype=dtype), value,
                  np.nextafter(value, np.inf, dtype=dtype), -np.inf, np.inf, np.nan):
            row = np.zeros(features, dtype=dtype)
            row[root['split_feature_id']] = v
            borders.append(row)
    probe = np.concatenate([x[:min(257, len(x))], np.asarray(borders, dtype=dtype).reshape(-1, features)])
    margins = treelite.gtil.predict(model, probe, pred_margin=True, nthread=1).reshape(-1)
    predictions = treelite.gtil.predict(model, probe, nthread=1).reshape(-1)
    oracle = directory / f'{name}.oracle.txt'
    with oracle.open('w') as out:
        out.write(f'{len(probe)} {features}\n')
        for row, margin, pred in zip(probe, margins, predictions):
            out.write(' '.join(map(str, [*map(float, row), float(margin), float(pred)])) + '\n')
    return dict(name=name, model_path=model_path.name, sha256=sha(model_path), model_bytes=model_path.stat().st_size,
                dtype=dtype, task=config['task'], features=features, statistics=stats,
                generation=config, source=source, oracle_path=oracle.name, oracle_sha256=sha(oracle),
                oracle_rows=len(probe), tolerance=2e-5 if dtype == 'float32' else 1e-12,
                inputs={str(n): dict(rows=n, bytes=n*features*np.dtype(dtype).itemsize,
                    sha256=hashlib.sha256(x[:n].tobytes()).hexdigest()) for n in rows})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'build/gpu-corpus')
    p.add_argument('--generator', type=Path, default=ROOT / 'build/generate-gpu-model')
    p.add_argument('--rows', default=','.join(map(str, ROWS)))
    p.add_argument('--synthetic-only', action='store_true')
    a = p.parse_args()
    rows = sorted(set(map(int, a.rows.split(','))))
    if not rows or min(rows) < 1 or max(rows) > 65536:
        p.error('rows must lie in [1, 65536]')
    versions = {k: importlib.metadata.version(k) for k in VERSIONS}
    if versions != VERSIONS:
        raise RuntimeError(f'Pinned versions required: {VERSIONS}; found {versions}')
    directory, generator = a.output.resolve(), a.generator.resolve()
    if directory.exists():
        p.error('output exists; use a fresh directory to preserve evidence')
    directory.mkdir(parents=True)
    cases, commands = [], []
    for name, config in screening():
        if config['trees'] * (2**(config['depth']+1)-1) > MAX_NODES:
            raise ValueError('Node cap exceeded before native generation')
        model_path = directory / f'{name}.tl'
        command = [str(generator), 'model', config['dtype'], str(config['trees']), str(config['depth']),
                   str(config['features']), str(config['seed']), config['shape'], config['task'], str(model_path)]
        subprocess.run(command, check=True)
        commands.append(command)
        cases.append(record(name, model_path, config, {'builder': 'balsa.ModelBuilder/TreeBuilder + save'},
                            directory, generator, rows))
        print(f'Generated {name}: native sample bytes verified', flush=True)
    if not a.synthetic_only:
        for name, model, source in trained(directory):
            path = directory / f'{name}.tl'
            model.serialize(path)
            dtype = json.loads(model.dump_as_json())['threshold_type']
            config = dict(dtype=dtype, features=32, seed=SEED, missing=1, pattern='divergent', task='regression')
            cases.append(record(name, path, config, source, directory, generator, rows))
            print(f'Generated {name}: source parity and native sample bytes verified', flush=True)
    source_paths = [Path(__file__), ROOT/'tools/generate_gpu_model.mojo', ROOT/'tools/gpu_samples.mojo',
                    ROOT/'build/balsa-reference/tools/framework_fixtures.py']
    manifest = dict(schema=1, versions=versions, seed=SEED, rows=rows,
                    balsa_commit=(ROOT/'build/balsa-reference/.reference-commit').read_text().strip(),
                    command=sys.argv, native_commands=commands, generator_sha256=sha(generator),
                    source_sha256={str(f.relative_to(ROOT)): sha(f) for f in source_paths},
                    caps=dict(nodes=MAX_NODES, sample_bytes=MAX_SAMPLE_BYTES),
                    sample_layout='C row-major, native little-endian; prefix-stable gpu_samples.mojo',
                    cases=cases)
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(directory/'manifest.json')


if __name__ == '__main__':
    main()
