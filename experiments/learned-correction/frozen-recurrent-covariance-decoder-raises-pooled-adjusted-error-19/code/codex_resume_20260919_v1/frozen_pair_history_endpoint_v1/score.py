import math,resource
import numpy as np
from common import *
seal_check();done('qualify');assert read(HERE/'qualification.json')['passed']
s=read(HERE/'predictions_sealed.json')
for k in ('candidate','qualification'):verify(s[k])
rows=[]
for pair in s['pairs']:
    verify(pair['truth'])
    with np.load(pair['truth']['path'],allow_pickle=False) as z:
        if 'P' in z.files:assert int(z['P'])==1_000_000_000
        truth=z['mc_post'][-1].astype(np.float64)
    scores={};arrays={}
    for name in ('candidate','parent'):
        rec=pair[name];verify(rec['prediction']);a=np.load(rec['prediction']['path'],allow_pickle=False);arrays[name]=a
        assert rec['includes_initialization'] and rec['multiplier']==max(.1,rec['flops']/2**41)
        residual=a[-1].astype(np.float64)-truth;raw=float(np.mean(residual*residual))
        audit=math.fsum(float(v)*float(v) for v in residual)/len(residual)
        err=abs(raw-audit)/max(abs(audit),1e-300);assert err<=2e-12
        scores[name]=dict(raw=raw,adjusted=raw*rec['multiplier'],flops=rec['flops'],multiplier=rec['multiplier'],fsum_relative=err)
    rows.append(dict(rank=pair['rank'],**scores,truth=pair['truth'],
        raw_ratio=scores['candidate']['raw']/scores['parent']['raw'],cost_ratio=scores['candidate']['flops']/scores['parent']['flops'],
        adjusted_ratio=scores['candidate']['adjusted']/scores['parent']['adjusted'],
        layerwise_delta_mse=np.mean((arrays['candidate'].astype(np.float64)-arrays['parent'].astype(np.float64))**2,axis=1).tolist()))
pooled=sum(r['candidate']['adjusted'] for r in rows)/sum(r['parent']['adjusted'] for r in rows)
write(HERE/'score.json',dict(completed=True,rows=rows,pooled_adjusted_ratio=pooled,
    passed=pooled<1 and all(r['adjusted_ratio']<=1.05 for r in rows),
    gate='own complete cold36/38: pooled adjusted ratio<1 and each<=1.05',predictions=pin(HERE/'predictions_sealed.json'),
    official_score=False,pristine_holdout=False,new_fit_steps=0,new_samples=0,old_proxy_gate_still_failed=True,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(completed=True,pooled_adjusted_ratio=pooled,passed=read(HERE/'score.json')['passed'],rows=rows),flush=True)
