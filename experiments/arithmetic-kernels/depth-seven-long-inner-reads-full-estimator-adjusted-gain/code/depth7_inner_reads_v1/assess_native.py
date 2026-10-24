"""Reconcile the full-program follow-up from saved outputs and truth."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

D=Path(__file__).resolve().parent
H=D.parent
P2=H.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
seal=json.loads((D/'pre_endpoint_seal.json').read_text())
for name,value in seal['files'].items():
    assert sha(D/name)==value,name
guard=json.loads((D/'train2_guard.json').read_text())
assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
OUT=H/'depth7_inner_reads_train2_v1'
BASE=H/'both_tail_train2_v6'
readrows=lambda p:{r['rank']:r for r in map(json.loads,(p/'rows.jsonl').read_text().splitlines())}
candidate,baseline=readrows(OUT),readrows(BASE)
manifest=json.loads((OUT/'manifest.json').read_text())
assert manifest['sha256']==sha(D/'candidate.py') and manifest['ranks']==[35,100]
rows=[]
lastcalls=lastchunks=0
for rank in [35,100]:
    truthfile=P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz'
    with np.load(truthfile) as z:
        truth=z['mc_post'][15].astype(np.float64)
    outputs={key:np.load(path/f'pred_{rank}.npy') for key,path in [('candidate',OUT),('baseline',BASE)]}
    reconciled={}
    for key,saved in [('candidate',candidate[rank]),('baseline',baseline[rank])]:
        pred=outputs[key]
        assert pred.shape==(16,1024) and np.isfinite(pred).all()
        mse=math.fsum((float(p)-float(t))**2 for p,t in zip(pred[15],truth))/1024
        adjusted=mse*max(.1,saved['flops']/2**41)
        assert math.isclose(mse,saved['mse'],rel_tol=2e-14,abs_tol=0)
        assert math.isclose(adjusted,saved['adjusted'],rel_tol=2e-14,abs_tol=0)
        reconciled[key]=dict(mse=mse,adjusted=adjusted,flops=saved['flops'])
    c,b=reconciled['candidate'],reconciled['baseline']
    count=candidate[rank]['d7r']
    rows.append(dict(rank=rank,**reconciled,
        changes_pct={key:100*(c[key]/b[key]-1) for key in ['mse','adjusted','flops']},
        adjusted_win=c['adjusted']<b['adjusted'],
        all16_layer_max_absolute_difference=np.max(np.abs(outputs['candidate']-outputs['baseline']),axis=1).tolist(),
        byte_equal_layers=[outputs['candidate'][i].tobytes()==outputs['baseline'][i].tobytes() for i in range(16)],
        routed_calls=count['calls']-lastcalls,routed_chunks=count['chunks']-lastchunks,
        wall_s=candidate[rank]['wall_s'],cpu_s=candidate[rank]['cpu_s'],
        candidate_prediction_sha256=sha(OUT/f'pred_{rank}.npy'),baseline_prediction_sha256=sha(BASE/f'pred_{rank}.npy'),
        truth_sha256=sha(truthfile)))
    lastcalls,lastchunks=count['calls'],count['chunks']
pooled={key:sum(r['candidate'][key] for r in rows)/sum(r['baseline'][key] for r in rows)
        for key in ['mse','adjusted','flops']}
gates=dict(two_adjusted_wins=all(r['adjusted_win'] for r in rows),
    adjusted_gain=pooled['adjusted']<=.985,raw_degradation=pooled['mse']<1.005)
result=dict(candidate_sha256=sha(D/'candidate.py'),pre_endpoint_sha256=sha(D/'pre_endpoint_seal.json'),
    assessment_sha256=sha(Path(__file__)),rows=rows,pooled_ratios=pooled,
    pooled_changes_pct={k:100*(v-1) for k,v in pooled.items()},gates=gates,
    passes_native_gate=all(gates.values()),scores_independently_reconciled=True,
    primitive_gate_passed=False,explicit_post_primitive_followup=True,
    validation_accessed=False,submitted=False,
    timing_scope='Direct local candidate time only; no paired parent timing or platform qualification',
    decision='Expand frozen candidate only on native pass' if all(gates.values()) else
             'Close fixed route; no expansion, further tuning, composition, validation or submission')
with (D/'native_assessment.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({k:result[k] for k in ['pooled_changes_pct','gates','passes_native_gate','decision']},indent=2),flush=True)
