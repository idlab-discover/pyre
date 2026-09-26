"""Build a versioned Mojo library bundle with dependency pins and checksums."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tomllib

from prepare_balsa import COMMIT

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = tomllib.loads((ROOT / 'pixi.toml').read_text())
    version = config['workspace']['version']
    output = ROOT / 'build/release'
    bundle = output / f'pyre-{version}-linux-64'
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / 'lib').mkdir(parents=True)
    for name in ['pyre.mojoc', 'balsa.mojoc']:
        shutil.copy2(ROOT / 'build' / name, bundle / 'lib' / name)
    for name in ['LICENSE', 'THIRD_PARTY_NOTICES.md', 'pixi.lock']:
        shutil.copy2(ROOT / name, bundle / name)
    shutil.copytree(ROOT / 'LICENSES', bundle / 'LICENSES')
    shutil.copy2(ROOT / 'examples/consumer.mojo', bundle / 'consumer.mojo')
    readme = (ROOT / 'docs/distribution.md').read_text()
    readme = readme.replace('(../examples/predict_gpu.mojo)',
        f'(https://github.com/idlab-discover/pyre/blob/v{version}/examples/predict_gpu.mojo)')
    readme = readme.replace('(usage.md#native-gpu-inference)',
        f'(https://github.com/idlab-discover/pyre/blob/v{version}/docs/usage.md#native-gpu-inference)')
    (bundle / 'README.md').write_text(readme)
    # Keep the dependency specification identical to the checked-in lock.
    source = (ROOT / 'pixi.toml').read_text()
    workspace = source.split('[tasks]')[0]
    dependencies = '[dependencies]' + source.split('[dependencies]', 1)[1]
    (bundle / 'pixi.toml').write_text(workspace + '[tasks]\n'
        'consumer = "mojo run --target-accelerator=none -I lib consumer.mojo"\n\n' + dependencies)
    files = sorted(p for p in bundle.rglob('*') if p.is_file())
    metadata = dict(version=version, balsa_commit=COMMIT,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
        mojo=subprocess.check_output(['mojo', '--version'], text=True).strip(),
        dependencies=config['dependencies'],
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'src').rglob('*.mojo'))},
        files_sha256={str(p.relative_to(bundle)): sha(p) for p in files})
    (bundle / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    archive = output / f'{bundle.name}.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(bundle, arcname=bundle.name)
    (output / 'SHA256SUMS').write_text(f'{sha(archive)}  {archive.name}\n')
    print(archive)


if __name__ == '__main__':
    main()
