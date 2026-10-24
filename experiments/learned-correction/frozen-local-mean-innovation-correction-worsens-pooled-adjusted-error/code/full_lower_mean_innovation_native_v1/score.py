import numpy as np
from common import HERE,read,write,pin,verify,seal_check,guarded,sha

seal_check();guarded('qualify')
assert read(HERE/'qualification.json')['passed']
seal=read(HERE/'predictions_sealed.json')
verify(seal['qualification']);verify(seal['candidate']);verify(seal['inputs'])
meta=read(HERE/'inputs.json');rows=[]
for pair in seal['pairs']:
    rank=pair['rank'];tp=meta['truths'][str(rank)];verify(tp)
    with np.load(tp['path'],allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
    assert truth.shape==(1024,) and np.isfinite(truth).all()
    a=pair['candidate'];b=pair['parent'];verify(a['prediction']);assert sha(b['output'])==b['output_sha256']
    ca=np.load(a['prediction']['path'],allow_pickle=False);pa=np.load(b['output'],allow_pickle=False)
    scores={}
    for name,array,rec in [('candidate',ca,a),('parent',pa,b)]:
        assert rec['multiplier']==max(.1,rec['flops']/2**41)
        raw=float(np.mean((array[-1].astype(np.float64)-truth)**2))
        scores[name]=dict(raw=raw,adjusted=raw*rec['multiplier'],flops=rec['flops'],multiplier=rec['multiplier'])
    expected=meta['expected_parent'][str(rank)]
    assert abs(scores['parent']['raw']/expected['raw']-1)<=1e-12
    assert abs(scores['parent']['adjusted']/expected['adjusted']-1)<=1e-12
    rows.append(dict(rank=rank,**scores,truth=tp,raw_ratio=scores['candidate']['raw']/scores['parent']['raw'],
        cost_ratio=a['flops']/b['flops'],adjusted_ratio=scores['candidate']['adjusted']/scores['parent']['adjusted'],
        layerwise_delta_mse=np.mean((ca.astype(np.float64)-pa.astype(np.float64))**2,axis=1).tolist()))
pooled=sum(r['candidate']['adjusted'] for r in rows)/sum(r['parent']['adjusted'] for r in rows)
result=dict(rows=rows,pooled_adjusted_ratio=pooled,passed=pooled<1 and all(r['adjusted_ratio']<=1.05 for r in rows),
    gate='pooled adjusted ratio <1 and each <=1.05, complete own-cold calls35/100',
    predictions=pin(HERE/'predictions_sealed.json'),official_score=False,pristine_holdout=False,
    new_fit_steps=0,new_samples=0,original_failed_proxy_gate_unchanged=True)
write(HERE/'score.json',result)
print(dict(passed=result['passed'],pooled_adjusted_ratio=pooled,rows=rows),flush=True)
