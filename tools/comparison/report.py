"""Validate complete runs and summarize matched process medians."""
from pathlib import Path
import csv
import json
import math

from .run import cells
from .schema import sha

NAMES = {'sklearn_random_forest': 'scikit-learn RF', 'xgboost_depthwise': 'XGBoost depthwise',
         'xgboost_lossguide': 'XGBoost lossguide', 'catboost_symmetric': 'CatBoost symmetric',
         'lightgbm_leafwise': 'LightGBM leafwise'}


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def summarize(results):
    import pandas as pd
    results = Path(results)
    if (results / 'changed-sources.json').exists():
        raise ValueError('Sources changed during this run; refusing to rank')
    provenance = json.loads((results / 'provenance.json').read_text())
    manifest = json.loads((results / 'corpus-manifest.json').read_text())
    matrix = provenance['matrix']
    attempts = pd.read_parquet(results / 'attempts.parquet')
    raw = pd.read_parquet(results / 'measurements.parquet')
    planned = {c['cell_id'] for c, _, _ in cells(manifest, matrix)}
    expected = {(cell, rep) for cell in planned for rep in range(matrix['processes'])}
    actual = set(zip(attempts.cell_id, attempts.repetition))
    if actual != expected or len(attempts) != len(expected) or not attempts.status.eq('ok').all():
        raise ValueError('Incomplete, duplicated or unsuccessful attempts; refusing to rank this run')
    if raw.duplicated(['cell_id', 'repetition', 'sample']).any():
        raise ValueError('Duplicated measurement samples')
    counts = raw.groupby(['cell_id', 'repetition']).size()
    if set(counts.index) != expected or not counts.eq(matrix['samples']).all():
        raise ValueError('Missing measurement samples')
    if not raw.groupby(['cell_id', 'repetition'])['sample'].apply(lambda s: set(s) == set(range(matrix['samples']))).all():
        raise ValueError('Invalid sample indices')
    if not (raw.max_scaled_error <= raw.gate_tolerance).all():
        raise ValueError('Out-of-tolerance measurements')
    raw['seconds'] = raw.elapsed_ns / raw.loops / 1e9
    if not raw.seconds.map(lambda x: math.isfinite(x) and x > 0).all():
        raise ValueError('Invalid measurement duration')
    medians = raw.groupby(['cell_id', 'repetition']).seconds.median()
    stats = medians.groupby('cell_id').agg(['median', 'min', 'max', 'count']).reset_index()
    meta = raw.drop_duplicates('cell_id').drop(columns=['seconds', 'loops', 'sample', 'elapsed_ns', 'repetition'])
    stats = stats.merge(meta, on='cell_id', validate='one_to_one')
    stats['configuration'] = stats.apply(lambda r:
        f'CPU {int(r.cpu_threads)} threads' if r.device == 'cpu' else
        'GPU automatic' if pd.isna(r.pyre_gpu_row_group) else
        f'GPU {int(r.pyre_gpu_row_group)} rows/block', axis=1)
    stats['mrows_s'] = stats.batch_rows / stats['median'] / 1e6
    active_trees = {m['model_id']: sum(t['nodes'] > 1 for t in m['trees']) for m in manifest['models'] if 'trees' in m}
    stats['nonconstant_trees'] = stats.model_id.map(active_trees)
    match = ['device', 'model_id', 'model_sha256', 'input_sha256', 'oracle_sha256', 'batch_rows', 'cpu_threads', 'phase']
    pyre = stats[stats.engine.eq('pyre')]
    if pyre.empty:
        raise ValueError('Pyre measurements are required for a comparison report')
    pairs = []
    for engine in ['native', 'nvforest']:
        ref = stats[stats.engine.eq(engine)]
        if ref.empty:
            continue
        for setting in (['native'] if engine == 'native' else ['nvforest default', 'nvforest best probed']):
            chosen = ref
            if setting == 'nvforest default':
                chosen = ref[ref.nvforest_chunk.isna()]
                if chosen.empty:
                    continue
            elif setting == 'nvforest best probed':
                chosen = ref.loc[ref.groupby(match, dropna=False)['median'].idxmin()]
            pair = pyre.merge(chosen, on=match, suffixes=('_pyre', '_reference'), validate='many_to_one')
            if len(pair) != len(pyre):
                raise ValueError(f'Missing matched references for {setting}')
            pair['reference'] = setting
            pair['speedup'] = pair.median_reference / pair.median_pyre
            pair['spread_low'] = pair.min_reference / pair.max_pyre
            pair['spread_high'] = pair.max_reference / pair.min_pyre
            pair['range_overlap'] = (pair.min_reference <= pair.max_pyre) & (pair.min_pyre <= pair.max_reference)
            pairs.append(pair)
    if not pairs:
        raise ValueError('No reference measurements')
    pairs = pd.concat(pairs, ignore_index=True)
    return provenance, manifest, stats, pairs


def report(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.hashsalt'] = 'pyre-comparison'
    import pandas as pd
    provenance, manifest, stats, pairs = summarize(results)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output / 'cells.csv', index=False)
    pairs.to_csv(output / 'matched.csv', index=False)
    summary = []
    for (ref, family, device, configuration, threads, rows), group in pairs.groupby(
            ['reference', 'family_pyre', 'device', 'configuration_pyre', 'cpu_threads', 'batch_rows'], dropna=False):
        summary.append(dict(reference=ref, family=family, device=device, configuration=configuration,
                            threads=None if pd.isna(threads) else int(threads), rows=int(rows),
                            speedup=geomean(group.speedup), cells=len(group), overlaps=int(group.range_overlap.sum())))
    summary = pd.DataFrame(summary)
    summary.to_csv(output / 'summary.csv', index=False)
    matrix = provenance['matrix']
    label = '/'.join(d.upper() for d in matrix['devices'])
    stem = '-'.join(matrix['devices']) + '-comparison'
    head = provenance['git_head']['stdout'].strip()
    lines = [f'# Current {label} comparison', '',
             f'Pyre revision `{head}`, rebuilt from the working checkout. Source and binary hashes are in the run provenance. '
             f'{len(stats)} timing cells, {matrix["processes"]} independent processes per cell, '
             f'{matrix["samples"]} intervals per process, calibration target {matrix["min_seconds"]} s. Run purpose: {matrix["purpose"]}. All planned attempts and full-output correctness gates passed.', '',
             'Warm allocating completed calls; loading, conversion, preparation and input creation excluded. Pyre uses native Mojo; references use their Python APIs. Each comparison uses identical model hashes, shared input values, devices and requested CPU thread counts. The native input mode is recorded in the matrix: prepared mode excludes reusable input preparation; ndarray mode includes per-call native ingestion. Engine order alternates within each matched group; group order is randomized. GPU model preparation and input uploads are excluded; each timed call allocates a device output and waits for completion. Output downloads occur only in the correctness gate. This is inference performance, not training speed or model accuracy.', '',
             'Speed factor = reference latency / Pyre latency. Greater than 1 favors Pyre. Aggregates are geometric means over matched models at the displayed batch size. Overlap counts compare the observed ranges of process medians; they are not confidence intervals. nvForest best is selected retrospectively among the probed chunks, with depth-first layout fixed.', '',
             '| Reference | Tree family | Pyre configuration | Rows | Speed factor | Cells | Overlapping ranges |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in summary.itertuples():
        lines.append(f'| {r.reference} | {NAMES[r.family]} | {r.configuration} | {r.rows:,} | {r.speedup:.2f}× | {r.cells} | {r.overlaps} |')
    lines += ['', '## Pyre throughput and observed spread', '',
              '| Family | Trees | Nonconstant trees | Actual max depth | Nodes | Precision | Pyre configuration | Rows | Median ms | Min–max ms | M rows/s |',
              '| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in stats[stats.engine.eq('pyre')].sort_values(['family','tree_count','max_depth','cpu_threads','batch_rows']).itertuples():
        lines.append(f'| {NAMES[r.family]} | {r.tree_count} | {int(r.nonconstant_trees)} | {r.max_depth} | {r.node_count:,} | {r.model_dtype} | {r.configuration} | {r.batch_rows:,} | {r.median*1000:.3f} | {r.min*1000:.3f}–{r.max*1000:.3f} | {r.mrows_s:.2f} |')
    lines += ['', '## Exclusions', '', 'Conservative expanded-node budget, not an inference failure.']
    for excluded in manifest['excluded']:
        lines.append(f'- {excluded["family"]}, {excluded["requested_tree_count"]} trees, depth cap {excluded["requested_depth_limit"]}: {excluded["reason"]}, bound {excluded["upper_bound_nodes"]:,} nodes.')
    lines += ['', '## Provenance', '',
              'CSV files retain every cell, matched ratio, process-median range, input/model/oracle hash, selected nvForest chunk, and Pyre GPU launch plan. Each Pyre row-group setting is reported separately; no best-Pyre selection is made. Raw intervals and per-attempt stdout/stderr remain in the local run directory.', '',
              'These are measurements on a shared desktop with unchanged frequency/governor settings. They do not prove hardware saturation or concurrent-serving capacity. See the benchmark guide for reproduction and timing scope.']
    (output / 'README.md').write_text('\n'.join(lines)+'\n')
    (output / 'provenance.json').write_text(json.dumps({k: provenance[k] for k in ['run_id','started_utc','matrix','git_head','source_sha256','cpu_affinity','cpu','packages','load_average','cpu_frequency_policy','gpu','gpu_state','cuda_visible_devices'] if k in provenance} | {
        'report_source_sha256': sha(Path(__file__)),
        'raw_measurements_sha256': sha(Path(results)/'measurements.parquet'),
        'raw_attempts_sha256': sha(Path(results)/'attempts.parquet'),
        'local_results': str(Path(results).resolve())}, indent=2)+'\n')
    refs = [ref for ref in summary.reference.unique() if ref != 'nvforest default']
    configurations = sorted(summary.configuration.unique())
    fig, axes = plt.subplots(len(configurations), len(refs), figsize=(max(12, 6*len(refs)), 5*len(configurations)), squeeze=False)
    colors = ['#087f8c','#e88c30','#526bb0']
    families = sorted(summary.family.unique())
    batches = sorted(summary.rows.unique())
    for i, ref in enumerate(refs):
        for j, configuration in enumerate(configurations):
            ax = axes[j,i]
            for k, batch in enumerate(batches):
                data = summary[(summary.reference==ref)&(summary.configuration==configuration)&(summary.rows==batch)].set_index('family').reindex(families)
                width = .7 / len(batches)
                bars = ax.barh([n+(k-(len(batches)-1)/2)*width for n in range(len(families))], data.speedup, height=width, color=colors[k % len(colors)], label=f'{batch:,} rows')
                ax.bar_label(bars, fmt='%.2f×', padding=3, fontsize=8)
            ax.set_yticks(range(len(families)), [NAMES[f] for f in families]); ax.invert_yaxis()
            ax.axvline(1,color='#666',linestyle='--',linewidth=1)
            ax.set_xlim(0, ax.get_xlim()[1]*1.18)
            ax.set_title(f'{ref} / Pyre · {configuration}')
            ax.spines[['top','right']].set_visible(False)
            ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0, -.08), ncol=len(batches))
    if matrix['devices'] == ['cpu']:
        cpu_info = json.loads(provenance['cpu']['stdout'])['lscpu']
        hardware = next((entry['data'] for entry in cpu_info if entry['field'] == 'Model name:'), 'CPU')
    else:
        gpu_lines = provenance.get('gpu', {}).get('stdout', '').strip().splitlines()
        hardware = '; '.join(row[0] for row in csv.reader(gpu_lines[1:]) if row) or 'GPU details in provenance'

    fig.text(.5, .98, f'Current Pyre {label} inference · {hardware}\n{matrix['purpose'].capitalize()} run · matched models and inputs · {matrix.get("native_input", "prepared")} references · {matrix["processes"]} processes/cell · higher favors Pyre',fontsize=14, ha='center', va='top')
    fig.tight_layout(rect=[0,.04,1,.89])
    fig.set_layout_engine('none')
    fig.savefig(output/(stem + '.svg'),metadata={'Date':None})
    fig.savefig(output/(stem + '.png'),dpi=160)
    plt.close(fig)
