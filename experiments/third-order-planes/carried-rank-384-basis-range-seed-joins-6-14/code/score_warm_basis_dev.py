"""Cold-cost fixed development score after sealing both Warm-basis predictions."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
build=read(HERE/'warm_basis_build.json')
assert sha(HERE/'warm_basis_cost75.py')==build['candidate_sha256']
for name in ('warm_basis_dev_watch.json','warm_basis_cold100_watch.json'):
    w=read(HERE/name)
    assert w['completed'] and w['whole_group_retired'] and w['max_swap_growth_bytes']==0
dev=read(HERE/'warm_basis_dev_v1/result.json')
cold=read(HERE/'warm_basis_cold100_v1/result.json')
assert dev['candidate_sha256']==cold['candidate_sha256']==build['candidate_sha256']
assert dev['records'][0]['output_sha256']==dev['records'][2]['output_sha256']
assert dev['records'][1]['output_sha256']==cold['records'][0]['output_sha256']
pairs=[]
for rank,current,parent_dir in ((35,dev['records'][0],'cost75_dev_v1'),
                                (100,cold['records'][0],'cost75_cold100_v1')):
    parent_meta=read(HERE/parent_dir/'result.json')
    assert parent_meta['candidate_sha256']==build['parent_sha256']
    parent=parent_meta['records'][0]
    assert current['includes_initialization'] and parent['includes_initialization']
    assert current['weights_sha256']==parent['weights_sha256']
    for record in (current,parent):
        assert sha(record['output'])==record['output_sha256']
    a,b=np.load(current['output']),np.load(parent['output'])
    assert np.array_equal(a[:6],b[:6])
    pairs.append((rank,current,parent,a,b))
with (HERE/'warm_basis_dev_predictions_sealed.json').open('x') as f:
    json.dump(dict(candidate_sha256=build['candidate_sha256'],repeat_byte_exact=True,
        warm_cold100_byte_exact=True,pairs=[dict(rank=r,current=c,parent=p) for r,c,p,a,b in pairs]),f,indent=2)
rows=[]
for rank,current,parent,a,b in pairs:
    truth_path=HERE.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
    truth=np.load(truth_path)['mc_post'][-1].astype(np.float64)
    scores={}
    for label,record,pred in (('candidate',current,a),('parent',parent,b)):
        raw=float(np.mean((pred[-1].astype(np.float64)-truth)**2))
        scores[label]=dict(raw=raw,flops=record['flops'],adjusted=raw*record['multiplier'])
    rows.append(dict(rank=rank,truth_sha256=sha(truth_path),**scores,
        layerwise_delta_mse=np.mean((a.astype(np.float64)-b.astype(np.float64))**2,axis=1).tolist(),
        raw_ratio=scores['candidate']['raw']/scores['parent']['raw'],
        adjusted_ratio=scores['candidate']['adjusted']/scores['parent']['adjusted']))
rec=dict(rows=rows,repeat_byte_exact=True,warm_cold100_byte_exact=True,
    both_adjusted_improve=all(r['adjusted_ratio']<1 for r in rows),
    both_cold_flops_lower=all(r['candidate']['flops']<r['parent']['flops'] for r in rows),
    pooled_adjusted_ratio=sum(r['candidate']['adjusted'] for r in rows)/sum(r['parent']['adjusted'] for r in rows))
rec['first6_byte_exact']=True
rec['development_gate_passed']=rec['both_adjusted_improve'] and rec['pooled_adjusted_ratio']<=.995
with (HERE/'warm_basis_dev_score.json').open('x') as f:
    json.dump(rec,f,indent=2)
print(json.dumps(rec),flush=True)
