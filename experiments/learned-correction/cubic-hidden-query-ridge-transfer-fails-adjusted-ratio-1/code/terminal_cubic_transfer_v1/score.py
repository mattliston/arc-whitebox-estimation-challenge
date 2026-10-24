import math,resource
import numpy as np
from common import *
seal_check();done('fit')
oof=read(HERE/'oof_predictions_sealed.json');check(oof['feature_seal'])
for m in oof['models']:check(m['parameters']);assert not set(m['fit'])&set(m['held'])
for p in oof['predictions'].values():check(p)
pred={k:np.load(p['path'],allow_pickle=False) for k,p in oof['predictions'].items()}
sealed=read(HERE/'features_sealed.json');records={r['rank']:r for r in sealed['records'] if r['name'].startswith('train')}
meta={r['rank']:r for r in read(HERE/'inputs.json')['rows']};rows=[]
for i,rank in enumerate(oof['train']):
    check(meta[rank]['truth']);r=records[rank];check(r['file'])
    with np.load(meta[rank]['truth']['path'],allow_pickle=False) as z:truth=z['mc_post'][-1].astype(float)
    with np.load(r['file']['path'],allow_pickle=False) as z:parent=z['prediction'][-1]
    scores={}
    for kind,array,extra in [('parent',parent,0),('affine',pred['affine'][i],2_000_000),('extended',pred['extended'][i],r['helper_flops']+2_000_000)]:
        delta=array.astype(float)-truth;raw=math.fsum(float(x)*float(x) for x in delta)/1024
        assert abs(float(np.mean(delta*delta))-raw)<=1e-14*raw
        f=r['parent_flops']+extra;scores[kind]=dict(raw=raw,adjusted=raw*max(.1,f/2**41),accounted_flops=f)
    rows.append(dict(rank=rank,fold=i%5,scores=scores,
        adjusted_parent_ratio=scores['extended']['adjusted']/scores['parent']['adjusted'],
        adjusted_affine_ratio=scores['extended']['adjusted']/scores['affine']['adjusted'],
        raw_parent_ratio=scores['extended']['raw']/scores['parent']['raw']))
def ratio(rs,kind):return sum(r['scores']['extended']['adjusted'] for r in rs)/sum(r['scores'][kind]['adjusted'] for r in rs)
pooled=ratio(rows,'parent');affine=ratio(rows,'affine')
folds=[dict(fold=f,ratio=ratio([r for r in rows if r['fold']==f],'parent')) for f in range(5)]
wins=sum(r['adjusted_parent_ratio']<1 for r in rows);foldwins=sum(f['ratio']<1 for f in folds);worst=max(r['adjusted_parent_ratio'] for r in rows)
passed=pooled<=.99 and affine<=.99 and wins>=12 and foldwins>=4 and worst<=1.20
result=dict(completed=True,passed=passed,pooled_adjusted_parent_ratio=pooled,pooled_adjusted_affine_ratio=affine,
    network_wins=wins,fold_wins=foldwins,worst_adjusted_ratio=worst,folds=folds,rows=rows,
    readout_allowance_flops=2_000_000,helper_bill_measured=True,full_native_candidate_bill_measured=False,
    pristine_holdout=False,official_score=False,independent_fsum_agreed=True,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write(HERE/'cv_result.json',result)
print({k:v for k,v in result.items() if k!='rows'},flush=True)
