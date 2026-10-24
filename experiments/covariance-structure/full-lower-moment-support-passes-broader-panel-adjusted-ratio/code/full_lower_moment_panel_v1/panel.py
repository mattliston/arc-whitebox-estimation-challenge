"""Serial cold full-estimator calls; all reference arrays remain unopened."""
import subprocess
import sys
from common import (HERE, D, CANDIDATE, CANDIDATE_SHA, RUNTIME, read, write,
                    sha, pin, verify, verify_call)

verify()
out = HERE / 'panel'
out.mkdir(exist_ok=False)
records = []
for row in read(HERE / 'inputs.json')['rows']:
    rank = row['rank']
    verify()
    if row['saved_candidate'] is not None:
        call = row['saved_candidate']
        watch_path = None
        result_path = None
    else:
        target = out / f'candidate_{rank}'
        watch_path = out / f'candidate_{rank}_watch.json'
        command = [sys.executable, str(D / 'run_guarded.py'), '--record',
                   str(watch_path), '--seconds', '100', str(RUNTIME), '-B',
                   '-W', 'ignore', str(D / 'weight5_reuse_v1/native.py'),
                   str(CANDIDATE), str(target), '--ranks', str(rank)]
        with (out / f'candidate_{rank}.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0, (rank, result.returncode)
        watch = read(watch_path)
        assert watch['completed'] and watch['whole_group_retired']
        assert watch['max_swap_growth_bytes'] == 0
        result_path = target / 'result.json'
        receipt = read(result_path)
        assert receipt['candidate_sha256'] == CANDIDATE_SHA
        assert not receipt['truth_accessed']
        assert len(receipt['records']) == 1
        call = receipt['records'][0]
    verify_call(call)
    assert call['rank'] == rank
    assert call['weights_sha256'] == row['parent']['weights_sha256']
    rec = dict(rank=rank, call=call, parent=row['parent'], truth=row['truth'],
               saved=row['saved_candidate'] is not None,
               result=pin(result_path) if result_path else None,
               watch=pin(watch_path) if watch_path else None)
    records.append(rec)
    write(out / f'verified_{rank}.json', rec)
    print(dict(rank=rank, saved=rec['saved'], flops=call['flops'],
               cost_ratio=call['flops']/row['parent']['flops'],
               wall_s=call['wall_s']), flush=True)
write(out / 'completed.json', dict(completed=True, records=records,
    candidate_sha256=CANDIDATE_SHA, new_calls=sum(not r['saved'] for r in records),
    sources_seal=pin(HERE/'sources_sealed.json'), references_decoded=False))

