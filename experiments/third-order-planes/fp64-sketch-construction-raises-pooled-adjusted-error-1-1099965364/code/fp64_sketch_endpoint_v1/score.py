from pathlib import Path
import hashlib,json

HERE=Path(__file__).resolve().parent;RESPONSE=HERE.parent/'fp64_sketch_response_v1'
def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
for p in read(HERE/'sources_sealed.json')['pins']:assert pin(p['path'])==p,p['path']
assert read(RESPONSE/'completed.json')['completed']
qualification=read(RESPONSE/'response_result.json')
assert qualification['repeat_exact'] and qualification['base_first5_original_exact'] and qualification['probes_first8_own_base_exact']
closeout=read(RESPONSE/'closeout.json');assert closeout['resources']['all_groups_retired']
meta=read(HERE/'inputs.json')
write(HERE/'predictions_sealed.json',dict(inputs=pin(HERE/'inputs.json'),response_qualification=pin(RESPONSE/'response_result.json'),
    response_closed_or_complete=pin(RESPONSE/'closeout.json'),labels_decoded=False,
    original_response_gate_passed=qualification['passed']))
import numpy as np
rows=[]
for row in meta['rows']:
    p,c=row['parent'],row['candidate']
    assert c['includes_initialization'] and c['multiplier']==max(.1,c['flops']/2**41)
    assert c['weight5_cache']['numeric_cache_retired']
    parent=np.load(p['output']['path'],allow_pickle=False)
    candidate=np.load(row['candidate_output']['path'],allow_pickle=False)
    assert parent.shape==candidate.shape==(16,1024) and parent.dtype==candidate.dtype==np.float32
    assert np.isfinite(candidate).all() and (candidate>=0).all()
    assert candidate[:5].tobytes()==parent[:5].tobytes()
    with np.load(p['truth']['path'],allow_pickle=False) as f:truth=f['mc_post'][-1].astype(np.float64)
    assert truth.shape==(1024,) and np.isfinite(truth).all()
    scores={}
    for name,array,bill in [('parent',parent,p['flops']),('candidate',candidate,c['flops'])]:
        raw=float(np.mean((array[-1].astype(np.float64)-truth)**2));multiplier=max(.1,bill/2**41)
        scores[name]=dict(raw=raw,flops=bill,multiplier=multiplier,adjusted=raw*multiplier)
    rows.append(dict(rank=row['rank'],**scores,raw_ratio=scores['candidate']['raw']/scores['parent']['raw'],
        cost_ratio=c['flops']/p['flops'],adjusted_ratio=scores['candidate']['adjusted']/scores['parent']['adjusted']))
pooled=sum(r['candidate']['adjusted'] for r in rows)/sum(r['parent']['adjusted'] for r in rows)
result=dict(passed=pooled<1 and all(r['adjusted_ratio']<=1.05 for r in rows),rows=rows,
    pooled_adjusted_ratio=pooled,pooled_raw_ratio=sum(r['candidate']['raw'] for r in rows)/sum(r['parent']['raw'] for r in rows),
    gate='pooled own-cold adjusted<1 and each<=1.05',original_response_gate_passed=qualification['passed'],
    independent_objective_question=True,fit_steps=0,new_estimator_calls=0,new_samples=0,official_score=False,pristine_holdout=False)
write(HERE/'score.json',result)
print(result,flush=True)
