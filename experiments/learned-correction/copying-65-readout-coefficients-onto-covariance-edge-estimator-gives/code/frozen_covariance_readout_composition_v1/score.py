import math,resource
import numpy as np
from common import *
seal_check();done('seal_predictions');sealed=read(HERE/'predictions_sealed.json')
assert sealed['completed'] and not sealed['targets_decoded'] and [r['rank'] for r in sealed['rows']]==CONFIRM
rows=[];maximum_audit=0
for entry in sealed['rows']:
    for p in (entry['truth'],entry['parent']['prediction'],entry['call']['prediction'],entry['result'],entry['watch']):verify(p)
    with np.load(entry['truth']['path'],allow_pickle=False) as z:truth=z['mc_post'][-1].astype(float)
    assert truth.shape==(1024,) and np.isfinite(truth).all()
    parent=np.load(entry['parent']['prediction']['path'],allow_pickle=False)[-1].astype(float)
    candidate=np.load(entry['call']['prediction']['path'],allow_pickle=False)[-1].astype(float)
    values={}
    for key,prediction in (('parent',parent),('candidate',candidate)):
        raw=float(np.mean((prediction-truth)**2));independent=math.fsum((float(p)-float(t))**2 for p,t in zip(prediction,truth))/1024
        relative_error=abs(raw-independent)/max(abs(independent),1e-30);maximum_audit=max(maximum_audit,relative_error);assert relative_error<1e-13
        flops=(entry['parent'] if key=='parent' else entry['call'])['flops'];mult=max(.1,flops/2**41)
        values[key]=dict(raw=raw,adjusted=raw*mult,flops=flops,multiplier=mult)
    assert values['parent']['flops']==values['candidate']['flops'] and values['parent']['raw']>0
    rows.append(dict(rank=entry['rank'],**values,raw_ratio=values['candidate']['raw']/values['parent']['raw'],
        adjusted_ratio=values['candidate']['adjusted']/values['parent']['adjusted']))
parent=math.fsum(r['parent']['adjusted'] for r in rows)/len(rows)
candidate=math.fsum(r['candidate']['adjusted'] for r in rows)/len(rows)
ratio=candidate/parent;wins=sum(r['adjusted_ratio']<1 for r in rows);worst=max(r['adjusted_ratio'] for r in rows)
passed=ratio<1 and wins>=5 and worst<=1.10
write(HERE/'score.json',dict(completed=True,passed=passed,pooled_parent_adjusted=parent,pooled_candidate_adjusted=candidate,
    pooled_adjusted_ratio=ratio,wins=wins,worst_adjusted_ratio=worst,rows=rows,gate=dict(pooled_less_than=1,min_wins=5,max_worst=1.10),
    independent_fsum_mse_max_relative=maximum_audit,full_cold_bills_exact=True,readout_and_covariance_training_exclusions=True,
    reused_research_networks_not_pristine=True,new_fit_count=0,predictions_sealed=pin(HERE/'predictions_sealed.json'),
    official_score=False,goal_achieved=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=passed,pooled_adjusted_ratio=ratio,wins=wins,worst_adjusted_ratio=worst),flush=True)
