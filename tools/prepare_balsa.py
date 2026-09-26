"""Materialize the pinned Balsa API from Git, without a sibling checkout."""
import argparse
import io
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

COMMIT = "737874f3eae9d95a0517397130f445b303d28cc6"
REPOSITORY = "https://github.com/idlab-discover/balsa.git"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "build/balsa-reference"


def prepare(destination=DEST, repository=REPOSITORY):
    destination = Path(destination).resolve()
    marker = destination / ".reference-commit"
    if marker.exists() and marker.read_text().strip() == COMMIT:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        temporary = Path(temporary)
        checkout = temporary / "git"
        subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "fetch", "--quiet", "--depth=1",
                        repository, COMMIT], check=True)
        archive = subprocess.check_output(["git", "-C", str(checkout), "archive", COMMIT])
        extracted = temporary / "source"
        extracted.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(extracted, filter="data")
        (extracted / ".reference-commit").write_text(COMMIT + "\n")
        if destination.exists():
            shutil.rmtree(destination)
        extracted.rename(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument("--repository", default=REPOSITORY,
                        help="Git URL or local repository for offline preparation")
    args = parser.parse_args()
    prepare(args.output, args.repository)
