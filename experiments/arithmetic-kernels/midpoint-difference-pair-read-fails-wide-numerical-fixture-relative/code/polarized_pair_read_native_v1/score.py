import math,resource
import numpy as np
from common import *
seal_check();done('qualify');assert read(HERE/'qualification.json')['passed'];s=read(HERE/'predictions_sealed.json')
for k in ('candidate','qualification'):verify(s[k])
rows=[]
for pair in s['pairs']:
    verify(pair['truth'])
    with np.load(pair['truth']['path'],allow_pickle=False) as z:
        if 'P' in z.files:assert int(z['P'])==1_000_000_000
        truth=z['mc_post'][-1].astype(np.float64)
    scores={}
    for name in ('candidate','parent'):
        x=pair[name];verify(x['prediction']);a=np.load(x['prediction']['path'],allow_pickle=False)
        assert x['includes_initialization'] and x['multiplier']==max(.1,x['flops']/2**41)
        delta=a[-1].astype(np.float64)-truth;raw=float(np.mean(delta*delta));other=math.fsum(float(v)*float(v) for v in delta)/len(delta)
        er=abs(raw-other)/max(other,1e-300);assert er<=2e-12
        scores[name]=dict(raw=raw,adjusted=raw*x['multiplier'],flops=x['flops'],multiplier=x['multiplier'],fsum_relative=er)
    rows.append(dict(rank=pair['rank'],**scores,raw_ratio=scores['candidate']['raw']/scores['parent']['raw'],
        cost_ratio=scores['candidate']['flops']/scores['parent']['flops'],adjusted_ratio=scores['candidate']['adjusted']/scores['parent']['adjusted']))
pooled=math.fsum(x['candidate']['adjusted'] for x in rows)/math.fsum(x['parent']['adjusted'] for x in rows)
passed=pooled<=.99 and all(x['adjusted_ratio']<=1.02 for x in rows)
write(HERE/'score.json',dict(completed=True,rows=rows,pooled_adjusted_ratio=pooled,passed=passed,
    gate='pooled own-cold adjusted ratio<=.99 and each<=1.02',predictions=pin(HERE/'predictions_sealed.json'),
    official_score=False,pristine_holdout=False,new_fit_steps=0,new_samples=0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(completed=True,pooled_adjusted_ratio=pooled,passed=passed,rows=rows),flush=True)
