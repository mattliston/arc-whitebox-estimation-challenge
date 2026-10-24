"""One preauthorized Phase 2 bank attempt, using the exact validated archive."""
import os
import subprocess
import sys
from pathlib import Path
from common import HERE,D,CANDIDATE_SHA,read,verify,verify_pin,pin,write

ROOT=Path('/Users/matt/arc-white-box-estimation')
verify()
package=read(HERE/'package.json')
assert package['validated']
verify_pin(package['archive']);verify_pin(package['candidate'])
assert package['candidate']['sha256']==CANDIDATE_SHA
assert package['members']==['estimator.py','manifest.json'] or sorted(package['members'])==['estimator.py','manifest.json']
assert read(HERE/'panel_score.json')['panel_advancement_passed']
assert read(HERE/'confirmation_score.json')['confirmation_advancement_passed']
boundary=read(HERE/'boundary_comparison.json')
assert all(boundary[k] for k in ['native_byte_exact','native_flops_exact','repeat_byte_exact','numpy_free_client','boundary_gate_passed'])
for name in ('contract_watch.json','package_watch.json','boundary_watch.json'):
    guard=read(HERE/name)
    assert guard['completed'] and guard['whole_group_retired']
    assert guard['max_swap_growth_bytes']==0
authority=read(D/'board_submission_preauthorization_20260919.json')
assert authority['explicit_preauthorization_received']
receipt=HERE/'submission.json'
assert not receipt.exists(), 'Existing attempt must be inspected, never duplicated.'
env=dict(os.environ,AICROWD_API_KEY=(ROOT/'key.txt').read_text().strip(),
         PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
assert env['AICROWD_API_KEY']
description=('Phase 2 full covariance/star/trace support on preserved Lazy; '
             'broader paired panel and eight-network confirmation, complete metered bills, '
             'NumPy-free client parity/runtime and package qualified')
command=[str(ROOT/'.venv/bin/whest'),'submit',package['archive']['path'],'--yes',
         '--challenge','arc-white-box-estimation-challenge-2026',
         '--description',description,'--json']
with receipt.open('x') as stream:
    result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
write(HERE/'submission_attempt.json',dict(exit_code=result.returncode,receipt=pin(receipt),
    archive=package['archive'],candidate_sha256=CANDIDATE_SHA))
print(dict(exit_code=result.returncode,receipt=str(receipt)),flush=True)
raise SystemExit(result.returncode)
