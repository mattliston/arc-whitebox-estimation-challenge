import subprocess
import sys
from common import (HERE, D, P2, CANDIDATE, CANDIDATE_SHA, PARENT_SHA, RUNTIME,
                    read, write, sha, pin, verify, verify_pin, verify_call)

verify()
for item in read(HERE/'confirmation_sources_sealed.json')['pins']:
    verify_pin(item)
assert read(HERE/'panel_score.json')['panel_advancement_passed']
parent = read(D/'lazy_rotation_objective_v1/confirmation/completed.json')
assert parent['candidate_sha256'] == PARENT_SHA
old_scores = {r['rank']:r for r in read(D/'lazy_rotation_objective_v1/confirmation_score.json')['rows']}
assert [r['rank'] for r in parent['records']] == [37,39,42,48,59,63,66,67]
inputs = []
for row in parent['records']:
    rank = row['rank']
    call = row['arms']['candidate']['call']
    verify_call(call)
    truth = P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
    assert sha(truth) == old_scores[rank]['truth_sha256']
    inputs.append(dict(rank=rank, parent=call, truth=pin(truth)))
write(HERE/'confirmation_inputs.json', dict(rows=inputs, references_decoded=False))
out = HERE/'confirmation'
out.mkdir(exist_ok=False)
records = []
for row in inputs:
    rank = row['rank']
    verify()
    target = out/f'candidate_{rank}'
    watch_path = out/f'candidate_{rank}_watch.json'
    command = [sys.executable,str(D/'run_guarded.py'),'--record',str(watch_path),
               '--seconds','100',str(RUNTIME),'-B','-W','ignore',
               str(D/'weight5_reuse_v1/native.py'),str(CANDIDATE),str(target),
               '--ranks',str(rank)]
    with (out/f'candidate_{rank}.log').open('x') as log:
        result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode == 0, (rank,result.returncode)
    watch = read(watch_path)
    assert watch['completed'] and watch['whole_group_retired']
    assert watch['max_swap_growth_bytes'] == 0
    receipt = read(target/'result.json')
    assert receipt['candidate_sha256'] == CANDIDATE_SHA and not receipt['truth_accessed']
    assert len(receipt['records']) == 1
    call = receipt['records'][0]
    verify_call(call)
    assert call['weights_sha256'] == row['parent']['weights_sha256']
    rec = dict(**row, call=call, result=pin(target/'result.json'), watch=pin(watch_path))
    records.append(rec)
    write(out/f'verified_{rank}.json',rec)
    print(dict(rank=rank,flops=call['flops'],wall_s=call['wall_s']),flush=True)
write(out/'completed.json',dict(completed=True,records=records,new_calls=8,
    candidate_sha256=CANDIDATE_SHA,references_decoded=False,
    sources_seal=pin(HERE/'confirmation_sources_sealed.json')))
