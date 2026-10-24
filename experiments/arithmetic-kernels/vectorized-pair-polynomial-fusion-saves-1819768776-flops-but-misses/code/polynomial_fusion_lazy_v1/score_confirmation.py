"""Fixed eight-network paired final-error confirmation, scored only after sealing."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
D=HERE.parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
build=read(HERE/'build.json')
completed=read(HERE/'confirmation/completed.json')
assert completed['completed'] and completed['candidate_sha256']==build['candidate_sha256']
assert sha(HERE/'candidate.py')==build['candidate_sha256']
assert [r['rank'] for r in completed['records']]==[37,39,42,48,59,63,66,67]
sealed=[]
for record in completed['records']:
    predictions={}
    for name,entry in record['arms'].items():
        call=entry['call']
        watch=read(entry['watch'])
        assert watch['completed'] and watch['whole_group_retired'] and watch['max_swap_growth_bytes']==0
        assert sha(call['output'])==call['output_sha256']
        pred=np.load(call['output'])
        assert pred.shape==(16,1024) and pred.dtype==np.float32 and np.isfinite(pred).all() and (pred>=0).all()
        predictions[name]=pred
    assert np.array_equal(predictions['candidate'][:1],predictions['lazy'][:1])
    delta=predictions['candidate'].astype(np.float64)-predictions['lazy'].astype(np.float64)
    sealed.append(dict(**record,first_layer_byte_exact=True,all_layer_parent_maxabs=float(np.max(np.abs(delta)))))
with (HERE/'confirmation_predictions_sealed.json').open('x') as f:
    json.dump(dict(records=sealed,targets_opened=False,candidate_sha256=build['candidate_sha256']),f,indent=2)
rows=[]
for entry in sealed:
    rank=entry['rank']
    path=D.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
    prior={r['rank']:r for r in read(D/'lazy_rotation_objective_v1/confirmation_score.json')['rows']}
    assert sha(path)==prior[rank]['truth_sha256']
    truth=np.load(path)['mc_post'][-1].astype(np.float64)
    row=dict(rank=rank,truth_sha256=sha(path),all_layer_parent_maxabs=entry['all_layer_parent_maxabs'])
    for name,item in entry['arms'].items():
        call=item['call']
        pred=np.load(call['output'])[-1].astype(np.float64)
        raw=float(np.mean((pred-truth)**2))
        row[name]=dict(raw=raw,flops=call['flops'],adjusted=raw*call['multiplier'])
    row['raw_ratio']=row['candidate']['raw']/row['lazy']['raw']
    row['adjusted_ratio']=row['candidate']['adjusted']/row['lazy']['adjusted']
    rows.append(row)
aggregate={name:{key:float(np.mean([r[name][key] for r in rows])) for key in ('raw','flops','adjusted')}
           for name in ('candidate','lazy')}
pooled=aggregate['candidate']['adjusted']/aggregate['lazy']['adjusted']
wins=sum(r['adjusted_ratio']<1 for r in rows)
result=dict(candidate_sha256=build['candidate_sha256'],rows=rows,aggregate=aggregate,
            pooled_adjusted_ratio=pooled,wins=wins,max_individual_ratio=max(r['adjusted_ratio'] for r in rows),
            confirmation_gate_passed=pooled<=.998 and wins>=6 and max(r['adjusted_ratio'] for r in rows)<=1.01,
            caveat='Fixed old validation ranks unused by this component study; not globally untouched. Ordinary1B reference MSE, no noise correction.')
with (HERE/'confirmation_score.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
