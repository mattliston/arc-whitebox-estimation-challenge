"""One fixed native old-cohort finite direction; reused TRAIN2 only."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
P2=HERE.parent
eps=.001
manifest=json.loads((HERE/'old_cohort_carry_information_manifest_v1.json').read_text())
for arm in manifest['arms']:
    assert hashlib.sha256(Path(arm['path']).read_bytes()).hexdigest()==arm['sha256']
    guard=json.loads((HERE/f"old_cohort_carry_{arm['arm']}_train2_v1_guard.json").read_text())
    assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
base_rows={r['rank']:r for r in map(json.loads,(HERE/'both_tail_train2_v6/rows.jsonl').read_text().splitlines())}
probe_rows={arm:{r['rank']:r for r in map(json.loads,(HERE/f'old_cohort_carry_{arm}_train2_v1/rows.jsonl').read_text().splitlines())}
            for arm in ('plus','minus')}
es=[];ds=[];remainders=[];rows=[]
for rank in (35,100):
    center=np.load(HERE/'both_tail_train2_v6'/f'pred_{rank}.npy')
    plus=np.load(HERE/'old_cohort_carry_plus_train2_v1'/f'pred_{rank}.npy')
    minus=np.load(HERE/'old_cohort_carry_minus_train2_v1'/f'pred_{rank}.npy')
    assert center[:8].tobytes()==plus[:8].tobytes()==minus[:8].tobytes()
    truth=np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
    e=center[15]-truth
    d=(plus[15]-minus[15])/(2*eps)
    r=(plus[15]+minus[15])*.5-center[15]
    curvature=float(np.linalg.norm(r)/np.linalg.norm(eps*d))
    es.append(e);ds.append(d);remainders.append(r)
    rows.append(dict(rank=rank,first8_bytes_exact=True,even_remainder_ratio=curvature,
                     plus_raw_pct=100*(probe_rows['plus'][rank]['mse']/base_rows[rank]['mse']-1),
                     minus_raw_pct=100*(probe_rows['minus'][rank]['mse']/base_rows[rank]['mse']-1),
                     plus_adjusted_pct=100*(probe_rows['plus'][rank]['adjusted']/base_rows[rank]['adjusted']-1),
                     minus_adjusted_pct=100*(probe_rows['minus'][rank]['adjusted']/base_rows[rank]['adjusted']-1)))
e=np.concatenate(es);d=np.concatenate(ds)
theta=float(np.clip(-np.dot(e,d)/np.dot(d,d),-.002,.002))
gain=100*(1-float(np.mean((e+theta*d)**2)/np.mean(e*e)))
for row,e1,d1 in zip(rows,es,ds):
    row['shared_theta_linearized_raw_gain_pct']=100*(1-float(np.mean((e1+theta*d1)**2)/np.mean(e1*e1)))
advance=bool(gain>=2 and all(r['shared_theta_linearized_raw_gain_pct']>0 and r['even_remainder_ratio']<=.20 for r in rows))
record=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope=__doc__,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            protocol_sha256=hashlib.sha256((HERE/'old_cohort_carry_information_protocol_v1.md').read_bytes()).hexdigest(),
            epsilon=eps,shared_clipped_theta=theta,linearized_pooled_raw_gain_pct=gain,
            rows=rows,advance_to_broader_training_fit=advance,
            limitation='One fixed output-direction projection on reused TRAIN35/100. Not a measured candidate at fitted theta, adjusted score, independent validation, or general capacity bound.')
with (HERE/'old_cohort_carry_information_result_v1.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
