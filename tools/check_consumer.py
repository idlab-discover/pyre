"""Compile a consumer outside the repository using only packaged dependencies."""
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def check(package_dir=ROOT / 'build'):
    with tempfile.TemporaryDirectory(prefix='pyre-consumer-') as temporary:
        temporary = Path(temporary)
        for name in ['pyre.mojoc', 'balsa.mojoc']:
            shutil.copy2(package_dir / name, temporary / name)
        shutil.copy2(ROOT / 'examples/consumer.mojo', temporary / 'consumer.mojo')
        subprocess.run(['mojo', 'run', '--target-accelerator=none', '-I', str(temporary),
                        'consumer.mojo'], cwd=temporary, check=True)


if __name__ == '__main__':
    check()
