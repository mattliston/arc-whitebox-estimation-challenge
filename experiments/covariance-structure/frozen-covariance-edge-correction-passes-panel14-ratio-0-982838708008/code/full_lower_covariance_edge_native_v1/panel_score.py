import argparse
import numpy as np
from common import HERE,read,write,pin,verify,seal_check,guarded,sha
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['panel','confirmation']);stage=ap.parse_args().stage
seal_check()
for p in read(HERE/'panel_sources_sealed.json')['pins']:verify(p)
seal=read(HERE/f'{stage}_predictions_sealed.json')
for k in ['source','sources','inputs']:verify(seal[k])
arrays=[]
for entry in seal['rows']:
    rank=entry['rank'];guarded(f'{stage}_{rank}');p=entry['parent'];c=entry['call']
    verify(p['prediction']);assert sha(c['output'])==c['output_sha256']
    assert p['weights_sha256']==c['weights_sha256'] and c['includes_initialization'] and p['includes_initialization']
    if p['format']=='npz':
        with np.load(p['prediction']['path'],allow_pickle=False) as z:pa=z['prediction'].copy()
    else:pa=np.load(p['prediction']['path'],allow_pickle=False)
    ca=np.load(c['output'],allow_pickle=False)
    assert ca.shape==pa.shape==(16,1024) and ca.dtype==pa.dtype==np.float32
    assert np.isfinite(ca).all() and (ca>=0).all() and ca[:8].tobytes()==pa[:8].tobytes()
    arrays.append((entry,pa,ca))
write(HERE/f'{stage}_qualified_before_targets.json',dict(passed=True,first8_exact_all=True,
    predictions=pin(HERE/f'{stage}_predictions_sealed.json'),targets_not_yet_decoded=True))
rows=[]
for entry,pa,ca in arrays:
    tp=entry['truth'];verify(tp)
    with np.load(tp['path'],allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
    scores={}
    for name,array,r in [('parent',pa,entry['parent']),('candidate',ca,entry['call'])]:
        assert r['multiplier']==max(.1,r['flops']/2**41)
        raw=float(np.mean((array[-1].astype(np.float64)-truth)**2))
        scores[name]=dict(raw=raw,flops=r['flops'],multiplier=r['multiplier'],adjusted=raw*r['multiplier'])
    rows.append(dict(rank=entry['rank'],**scores,raw_ratio=scores['candidate']['raw']/scores['parent']['raw'],
        adjusted_ratio=scores['candidate']['adjusted']/scores['parent']['adjusted'],cost_ratio=entry['call']['flops']/entry['parent']['flops']))
pooled=sum(r['candidate']['adjusted'] for r in rows)/sum(r['parent']['adjusted'] for r in rows)
wins=sum(r['adjusted_ratio']<1 for r in rows);worst=max(r['adjusted_ratio'] for r in rows)
assert len(rows)==(14 if stage=='panel' else 8)
passed=pooled<1 and wins>=(8 if stage=='panel' else 5) and worst<=1.10
result=dict(passed=passed,rows=rows,pooled_adjusted_ratio=pooled,wins=wins,worst_adjusted_ratio=worst,
    mean_parent_adjusted=sum(r['parent']['adjusted'] for r in rows)/len(rows),
    mean_candidate_adjusted=sum(r['candidate']['adjusted'] for r in rows)/len(rows),
    mean_parent_raw=sum(r['parent']['raw'] for r in rows)/len(rows),mean_candidate_raw=sum(r['candidate']['raw'] for r in rows)/len(rows),
    predictions=pin(HERE/f'{stage}_predictions_sealed.json'),official_score=False,pristine_holdout=False,new_fit_steps=0,new_samples=0)
write(HERE/f'{stage}_score.json',result);print(result,flush=True)
