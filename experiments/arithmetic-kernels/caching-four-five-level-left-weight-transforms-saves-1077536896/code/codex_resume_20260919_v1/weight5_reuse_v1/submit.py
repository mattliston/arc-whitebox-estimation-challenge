"""Upload the exact qualified Weight5 archive ONLY after explicit archive approval."""
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
assert hashlib.sha256(archive.read_bytes()).hexdigest() == package['archive_sha256'] == '9b943bbc066881136e698a61240f20370ae0477dd7b95aede11d421b238f1673'
assert hashlib.sha256((HERE / 'candidate.py').read_bytes()).hexdigest() == package['candidate_sha256'] == 'a432b94336a1ec45f9140d2e0a7ed808a8a7826deb72034a7ca185d331c823b9'
assert read(HERE / 'unit_v1/result.json')['passed']
assert read(HERE / 'native_comparison.json')['native_gate_passed']
score = read(HERE / 'panel_score.json')
assert score['panel_gate_passed'] and score['official_bank_ratio_projection'] < 4e-9
boundary = read(HERE / 'boundary_comparison.json')
assert all(boundary[k] for k in ('native_and_parent_byte_exact', 'native_flops_exact', 'repeat_byte_exact',
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
           '--description', 'Phase 2 Weight5: exact weight-transform reuse on Read5; fixed14 NC3.8832e-9; native/client parity and runtime qualified',
           '--json']
with output.open('x') as stream:
    done = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
print(json.dumps(dict(exit_code=done.returncode, receipt=str(output), archive_sha256=package['archive_sha256'])), flush=True)
sys.exit(done.returncode)
