"""Judge the fixed two-network actual endpoint and full metered cost."""
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
import numpy as np

H=Path(__file__).resolve().parent;BASE=H.parent
candidate=BASE/'joint_cov_train2_v1';control=BASE/'both_tail_train2_v6'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((H/'build.json').read_text());assert sha(H/'candidate.py')==build['candidate_sha256']
assert json.loads((H/'control_parity.json').read_text())['passed']
g=json.loads((H/'train2_guard.json').read_text())
assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
rows=[json.loads(x) for x in (candidate/'rows.jsonl').read_text().splitlines()]
baseline={x['rank']:x for x in map(json.loads,(control/'rows.jsonl').read_text().splitlines())}
assert [r['rank'] for r in rows]==[35,100]
checked=[]
for row in rows:
    rank=row['rank'];b=baseline[rank]
    y=np.load(candidate/f'pred_{rank}.npy',allow_pickle=False)
    old=np.load(control/f'pred_{rank}.npy',allow_pickle=False)
    assert y.shape==(16,1024) and np.isfinite(y).all() and np.array_equal(y[:8],old[:8])
    truth=np.load(BASE.parent/f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz',allow_pickle=False)['mc_post'][15]
    mse=math.fsum((float(x)-float(t))**2 for x,t in zip(y[15],truth))/1024
    assert abs(mse/row['mse']-1)<1e-12
    adjusted=mse*max(.1,row['flops']/2**41)
    assert abs(adjusted/row['adjusted']-1)<1e-12
    checked.append(dict(rank=rank,raw=mse,flops=row['flops'],adjusted=adjusted,
                        raw_ratio=mse/b['mse'],bill_ratio=row['flops']/b['flops'],adjusted_ratio=adjusted/b['adjusted'],
                        first8_rows_exact=True,prediction_sha256=sha(candidate/f'pred_{rank}.npy')))
rat=lambda key:math.fsum(r[key] for r in rows)/math.fsum(baseline[r['rank']][key] for r in rows)
ratio=rat('adjusted');wins=sum(r['adjusted_ratio']<1 for r in checked)
out=dict(passed=wins==2 and ratio<=.99,rows=checked,pooled_adjusted_ratio=ratio,
         adjusted_change_pct=100*(ratio-1),pooled_raw_ratio=rat('mse'),raw_change_pct=100*(rat('mse')-1),
         pooled_bill_ratio=rat('flops'),bill_change_pct=100*(rat('flops')-1),adjusted_wins=wins,
         candidate_sha256=sha(H/'candidate.py'),utc_finished=datetime.now(timezone.utc).isoformat(),
         check32_accessed=False,valid40_accessed=False,new_samples=False,old_information_gate_still_failed=True)
with (H/'assessment.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(out,indent=2),flush=True)
