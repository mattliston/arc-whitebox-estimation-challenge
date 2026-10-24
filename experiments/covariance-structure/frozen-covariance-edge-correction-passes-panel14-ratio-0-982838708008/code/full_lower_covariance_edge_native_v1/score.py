import numpy as np
from common import HERE,read,write,pin,verify,seal_check,guarded
seal_check();guarded('qualify');assert read(HERE/'qualification.json')['passed']
meta=read(HERE/'inputs.json');s=read(HERE/'predictions_sealed.json')
for k in ['candidate','qualification','inputs']:verify(s[k])
rows=[]
for pair in s['pairs']:
    rank=pair['rank'];tp=meta['truths'][str(rank)];verify(tp)
    with np.load(tp['path'],allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
    p=pair['parent'];c=pair['candidate'];verify(p['capture']);verify(c['prediction'])
    with np.load(p['capture']['path'],allow_pickle=False) as z:pa=z['prediction'].copy()
    ca=np.load(c['prediction']['path'],allow_pickle=False);scores={}
    for name,array,r in [('candidate',ca,c),('parent',pa,p)]:
        assert r['includes_initialization'] and r['multiplier']==max(.1,r['flops']/2**41)
        raw=float(np.mean((array[-1].astype(np.float64)-truth)**2))
        scores[name]=dict(raw=raw,adjusted=raw*r['multiplier'],flops=r['flops'],multiplier=r['multiplier'])
    rows.append(dict(rank=rank,**scores,truth=tp,raw_ratio=scores['candidate']['raw']/scores['parent']['raw'],
        cost_ratio=c['flops']/p['flops'],adjusted_ratio=scores['candidate']['adjusted']/scores['parent']['adjusted'],
        layerwise_delta_mse=np.mean((ca.astype(np.float64)-pa.astype(np.float64))**2,axis=1).tolist()))
pooled=sum(r['candidate']['adjusted'] for r in rows)/sum(r['parent']['adjusted'] for r in rows)
result=dict(rows=rows,pooled_adjusted_ratio=pooled,passed=pooled<1 and all(r['adjusted_ratio']<=1.05 for r in rows),
    gate='complete own-cold36/38: pooled adjusted ratio<1 and each<=1.05',predictions=pin(HERE/'predictions_sealed.json'),
    official_score=False,pristine_holdout=False,new_fit_steps=0,new_samples=0,old_proxy_gate_unchanged=True)
write(HERE/'score.json',result);print(result,flush=True)
