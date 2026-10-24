"""Standard Whest .16 validate/package in a new estimator-only staging folder."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

here = Path(__file__).resolve().parent
candidate = here / "candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py"
sha = "3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757"
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == sha
stage = here / "runtime_rectd4_submission_v1"
archive = here / "fullpair_runtime_rectd4_bulkcopies_v1_whest016.tar.gz"
receipt = here / "package_runtime_rectd4_v1.json"
assert not stage.exists() and not archive.exists() and not receipt.exists()
stage.mkdir()
(stage / "estimator.py").write_bytes(candidate.read_bytes())
cli = str(here.parent / "data_gate/.venv/bin/whest")
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
records = []
for command in (
    [cli, "validate", "--estimator", str(stage / "estimator.py"), "--json"],
    [cli, "package", "--estimator", str(stage / "estimator.py"), "--output", str(archive), "--yes", "--json"],
    [cli, "validate-package", str(archive), "--json"],
):
    result = subprocess.run(command, cwd=stage, env=env, capture_output=True, text=True)
    records.append(dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    print(result.stdout, flush=True)
    if result.returncode:
        print(result.stderr, flush=True)
        receipt.write_text(json.dumps(dict(passed=False, records=records), indent=2) + "\n")
        raise SystemExit(result.returncode)
with tarfile.open(archive, "r:gz") as tar:
    assert sorted(tar.getnames()) == ["estimator.py", "manifest.json"]
    assert hashlib.sha256(tar.extractfile("estimator.py").read()).hexdigest() == sha
receipt.write_text(json.dumps(dict(passed=True, source_sha256=sha,
    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), records=records), indent=2) + "\n")
