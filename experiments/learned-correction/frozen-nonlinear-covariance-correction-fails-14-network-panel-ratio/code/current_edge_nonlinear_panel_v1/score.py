import resource
import numpy as np
from common import *
stage=sys.argv[1];assert stage in ('panel','confirmation');seal_check()
done(f'{stage}_qualify');q=read(HERE/f'{stage}_qualified.json');assert q['passed'];verify(q['predictions'])
s=read(HERE/f'{stage}_predictions_sealed.json');roster=PANEL if stage=='panel' else CONFIRM
assert [r['rank'] for r in s['rows']]==roster
rows=[]
for row in s['rows']:
    verify(row['truth'])
    with np.load(row['truth']['path'],allow_pickle=False) as z:
        if 'P' in z.files:assert int(z['P'])==1_000_000_000
        truth=z['mc_post'][-1].astype(np.float64)
    scores={}
    for name,r in [('candidate',row['call']),('parent',row['parent'])]:
        verify(r['prediction']);assert r['includes_initialization'] and r['multiplier']==max(.1,r['flops']/2**41)
        a=np.load(r['prediction']['path'],allow_pickle=False)
        assert a.shape==(16,1024) and a.dtype==np.float32 and np.isfinite(a).all()
        raw=float(np.mean((a[-1].astype(np.float64)-truth)**2))
        scores[name]=dict(raw=raw,adjusted=raw*r['multiplier'],flops=r['flops'],multiplier=r['multiplier'])
    c=scores['candidate'];p=scores['parent']
    rows.append(dict(rank=row['rank'],**scores,truth=row['truth'],raw_ratio=c['raw']/p['raw'],
        cost_ratio=c['flops']/p['flops'],adjusted_ratio=c['adjusted']/p['adjusted']))
totals={}
for arm in ('candidate','parent'):
    total=0.0
    for row in rows:total+=row[arm]['adjusted']
    totals[arm]=total
pooled=totals['candidate']/totals['parent'];wins=sum(r['adjusted_ratio']<1 for r in rows)
worst=max(r['adjusted_ratio'] for r in rows);minimum_wins=8 if stage=='panel' else 5
passed=pooled<1 and wins>=minimum_wins and worst<=1.10
write(HERE/f'{stage}_score.json',dict(completed=True,stage=stage,rows=rows,adjusted_totals=totals,
    pooled_adjusted_ratio=pooled,wins=wins,worst_adjusted_ratio=worst,passed=passed,
    gate=dict(pooled_adjusted_ratio_less_than=1,minimum_wins=minimum_wins,worst_adjusted_ratio_at_most=1.10),
    predictions=pin(HERE/f'{stage}_predictions_sealed.json'),qualification=pin(HERE/f'{stage}_qualified.json'),
    official_score=False,pristine_holdout=False,new_fit_steps=0,new_samples=0,old_proxy_gate_unchanged=True,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(stage=stage,passed=passed,pooled_adjusted_ratio=pooled,wins=wins,worst_adjusted_ratio=worst,rows=rows),flush=True)
