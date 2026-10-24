import resource
import numpy as np
from common import *
seal_check();done('qualify');assert read(HERE/'qualification.json')['passed']
s=read(HERE/'predictions_sealed.json')
for k in ('candidate','qualification'):verify(s[k])
rows=[]
for pair in s['pairs']:
    rank=pair['rank'];verify(pair['truth'])
    with np.load(pair['truth']['path'],allow_pickle=False) as z:
        if 'P' in z.files:assert int(z['P'])==1_000_000_000
        truth=z['mc_post'][-1].astype(np.float64)
    p=pair['parent'];c=pair['candidate'];verify(p['prediction']);verify(c['prediction'])
    pa=np.load(p['prediction']['path'],allow_pickle=False);ca=np.load(c['prediction']['path'],allow_pickle=False);scores={}
    for name,array,r in [('candidate',ca,c),('parent',pa,p)]:
        assert r['includes_initialization'] and r['multiplier']==max(.1,r['flops']/2**41)
        raw=float(np.mean((array[-1].astype(np.float64)-truth)**2))
        scores[name]=dict(raw=raw,adjusted=raw*r['multiplier'],flops=r['flops'],multiplier=r['multiplier'])
    rows.append(dict(rank=rank,**scores,truth=pair['truth'],raw_ratio=scores['candidate']['raw']/scores['parent']['raw'],
        cost_ratio=c['flops']/p['flops'],adjusted_ratio=scores['candidate']['adjusted']/scores['parent']['adjusted'],
        layerwise_delta_mse=np.mean((ca.astype(np.float64)-pa.astype(np.float64))**2,axis=1).tolist()))
pooled=sum(r['candidate']['adjusted'] for r in rows)/sum(r['parent']['adjusted'] for r in rows)
write(HERE/'score.json',dict(completed=True,rows=rows,pooled_adjusted_ratio=pooled,
    passed=pooled<1 and all(r['adjusted_ratio']<=1.05 for r in rows),
    gate='complete own-cold36/38: pooled adjusted ratio<1 and each<=1.05',predictions=pin(HERE/'predictions_sealed.json'),
    official_score=False,pristine_holdout=False,new_fit_steps=0,new_samples=0,old_proxy_gate_unchanged=True,
    stopping_decision_explicitly_revised=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(completed=True,pooled_adjusted_ratio=pooled,passed=read(HERE/'score.json')['passed'],rows=rows),flush=True)
