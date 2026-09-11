"""Materialize the immutable Balsa API without consulting its working tree."""
from pathlib import Path
import subprocess
import tarfile
import io

COMMIT = "737874f3eae9d95a0517397130f445b303d28cc6"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "build/balsa-reference"
marker = DEST / ".reference-commit"
if not marker.exists() or marker.read_text().strip() != COMMIT:
    archive = subprocess.check_output(["git", "-C", str(ROOT.parent / "balsa"), "archive", COMMIT])
    DEST.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(DEST, filter="data")
    marker.write_text(COMMIT + "\n")
