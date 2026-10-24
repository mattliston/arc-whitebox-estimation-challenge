from common import (HERE,D,CANDIDATE_SHA,read,write,pin,verify,verify_pin,verify_call)
import numpy as np

verify()
for item in read(HERE/'confirmation_sources_sealed.json')['pins']:
    verify_pin(item)
receipt = read(HERE/'confirmation/completed.json')
assert receipt['completed'] and receipt['candidate_sha256'] == CANDIDATE_SHA
assert [r['rank'] for r in receipt['records']] == [37,39,42,48,59,63,66,67]
for row in receipt['records']:
    for call in (row['call'],row['parent']):
        verify_call(call)
        pred = np.load(call['output'],allow_pickle=False)
        assert pred.shape == (16,1024) and pred.dtype == np.float32
        assert np.isfinite(pred).all() and (pred>=0).all()
    verify_pin(row['result']);verify_pin(row['watch'])
write(HERE/'confirmation_predictions_sealed.json',dict(
    receipt=pin(HERE/'confirmation/completed.json'),references_decoded=False))
old = {r['rank']:r for r in read(D/'lazy_rotation_objective_v1/confirmation_score.json')['rows']}
rows=[]
for item in receipt['records']:
    rank=item['rank'];verify_pin(item['truth'])
    with np.load(item['truth']['path'],allow_pickle=False) as z:
        truth=z['mc_post'][-1].astype(np.float64)
    row=dict(rank=rank,truth=item['truth'])
    for name,call in [('candidate',item['call']),('parent',item['parent'])]:
        pred=np.load(call['output'],allow_pickle=False)[-1].astype(np.float64)
        raw=float(np.mean((pred-truth)**2))
        row[name]=dict(raw=raw,flops=call['flops'],adjusted=raw*call['multiplier'])
    for key in ('raw','flops','adjusted'):
        assert abs(row['parent'][key]-old[rank]['candidate'][key])<1e-20,(rank,key)
    row['raw_ratio']=row['candidate']['raw']/row['parent']['raw']
    row['adjusted_ratio']=row['candidate']['adjusted']/row['parent']['adjusted']
    rows.append(row)
aggregate={name:{key:float(np.mean([r[name][key] for r in rows]))
                 for key in ('raw','flops','adjusted')}
           for name in ('candidate','parent')}
pooled=aggregate['candidate']['adjusted']/aggregate['parent']['adjusted']
wins=sum(r['adjusted_ratio']<1 for r in rows)
maximum=max(r['adjusted_ratio'] for r in rows)
result=dict(completed=True,rows=rows,aggregate=aggregate,pooled_adjusted_ratio=pooled,
    pooled_raw_ratio=aggregate['candidate']['raw']/aggregate['parent']['raw'],
    wins=wins,max_individual_ratio=maximum,
    confirmation_advancement_passed=pooled<=.99 and wins>=5 and maximum<=1.10,
    parent_reproduced=True,official_score=False,
    caveat='Eight fixed reused research networks, ordinary1B means, no reference-noise correction.')
write(HERE/'confirmation_score.json',result)
print(result,flush=True)
