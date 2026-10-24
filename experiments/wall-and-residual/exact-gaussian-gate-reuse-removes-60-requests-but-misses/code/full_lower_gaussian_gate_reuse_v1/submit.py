"""Single qualified archive, using the user's standing explicit authorization."""
import os
import subprocess
from common import HERE, D, CLI, read, write, pin, verify, seal_check

for name in ['native_sources_sealed.json','boundary_sources_sealed.json','release_sources_sealed.json']:
    seal_check(name)
assert read(HERE/'release_completed.json')['completed']
assert read(HERE/'native_qualified.json')['passed'] and read(HERE/'boundary_qualified.json')['passed']
for p in HERE.glob('*_watch.json'):
    g = read(p)
    assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['max_swap_growth_bytes']==0
package = read(HERE/'package.json')
assert package['validated'] and sorted(package['members']) == ['estimator.py','manifest.json']
verify(package['archive']); verify(package['candidate'])
assert package['candidate'] == read(HERE/'build.json')['candidate']
authority = read(D/'board_submission_preauthorization_20260920.json')
assert authority['explicit_preauthorization_received'] and authority['user_statement']=='you are approved for all uploads'
root = CLI.parents[2]
receipt = HERE/'submission.json'
assert not receipt.exists() and not (HERE/'submission_intent.json').exists(), 'Inspect prior attempt; no duplicate upload.'
env = dict(os.environ,AICROWD_API_KEY=(root/'key.txt').read_text().strip(),PYTHONDONTWRITEBYTECODE='1',
           OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
assert env['AICROWD_API_KEY']
description = ('Phase 2 full lower moments and fixed covariance edge model; exact per-layer Gaussian gate reuse; '
               'all 16 outputs byte-identical to parent 331684 on native and ordinary NumPy-free checks; '
               '60 fewer requests per prediction; fixed prediction-only residual and total-wall gates passed')
command = [str(CLI.with_name('whest')),'submit',package['archive']['path'],'--yes','--challenge',
           'arc-white-box-estimation-challenge-2026','--description',description,'--json']
write(HERE/'submission_intent.json',dict(archive=package['archive'],candidate=package['candidate'],
    authorization=pin(D/'board_submission_preauthorization_20260920.json'),submit_source=pin(__file__)))
with receipt.open('x') as f:
    result = subprocess.run(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
write(HERE/'submission_attempt.json',dict(exit_code=result.returncode,receipt=pin(receipt),
    archive=package['archive'],candidate_sha256=package['candidate']['sha256']))
print(dict(exit_code=result.returncode,receipt=str(receipt)),flush=True)
raise SystemExit(result.returncode)
