"""Upload the exact qualified Read5 archive ONLY after explicit archive approval."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path('/Users/matt/arc-white-box-estimation')
read = lambda p: json.loads(Path(p).read_text())
package = read(HERE / 'package.json')
archive = Path(package['archive'])
assert package['validated'] and sorted(package['members']) == ['estimator.py', 'manifest.json']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == package['archive_sha256'] == 'a090a56fef84e31516c4552bc1c634c07f2c8ae9cf7737de25d8ba8fbb1bde72'
assert hashlib.sha256((HERE / 'candidate.py').read_bytes()).hexdigest() == package['candidate_sha256'] == 'd7723068ae269155e0369e4ba322b447222bbb31295d89394c069695ad0e824b'
assert read(HERE / 'unit_v1/result.json')['passed']
assert read(HERE / 'dev_score.json')['development_gate_passed']
score = read(HERE / 'panel_score.json')
assert score['panel_gate_passed'] and score['official_bank_ratio_projection'] < 4e-9
boundary = read(HERE / 'boundary_comparison.json')
assert all(boundary[k] for k in ('native_byte_exact', 'native_flops_exact', 'repeat_byte_exact',
                                 'numpy_free_client', 'boundary_gate_passed'))
contract = read(HERE / 'contract_watch.json')
assert contract['completed'] and contract['whole_group_retired']
env = dict(os.environ, AICROWD_API_KEY=(ROOT / 'key.txt').read_text().strip(),
           PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
assert env['AICROWD_API_KEY']
output = HERE / 'submission.json'
assert not output.exists(), 'An upload attempt already exists; inspect it instead of duplicating.'
command = [str(ROOT / '.venv/bin/whest'), 'submit', str(archive), '--yes',
           '--challenge', 'arc-white-box-estimation-challenge-2026',
           '--description', 'Phase 2 Read5: bounded five-level read kernel; fixed14 NC3.8935e-9; native/client parity and runtime qualified',
           '--json']
with output.open('x') as stream:
    done = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
print(json.dumps(dict(exit_code=done.returncode, receipt=str(output), archive_sha256=package['archive_sha256'])), flush=True)
sys.exit(done.returncode)
