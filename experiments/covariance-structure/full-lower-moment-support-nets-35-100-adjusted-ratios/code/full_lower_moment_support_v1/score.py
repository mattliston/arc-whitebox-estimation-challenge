"""Fixed two-network objective screen, only after sealing qualified cold outputs."""
from common import HERE,sha,read,write,seal_check,guarded
seal_check()
for name in ('unit','null','dev','cold100'):
    guarded(name)
    if name=='unit':assert read(HERE/'unit_result.json')['passed']
    else:assert read(HERE/(name+'_validation.json'))['passed']
meta=read(HERE/'inputs.json')
dev=read(HERE/'dev/result.json')['records']
cold=read(HERE/'cold100/result.json')['records']
assert [r['rank'] for r in dev]==[35,100,35] and [r['rank'] for r in cold]==[100]
assert dev[0]['output_sha256']==dev[2]['output_sha256']
assert dev[1]['output_sha256']==cold[0]['output_sha256']
pairs=[]
for rank,current,key in ((35,dev[0],'parent_dev'),(100,cold[0],'parent_cold100')):
    parent=read(meta[key]['path'])['records'][0]
    assert parent['rank']==current['rank']==rank
    assert current['includes_initialization'] and parent['includes_initialization']
    assert current['weights_sha256']==parent['weights_sha256']
    for r in (current,parent):
        assert sha(r['output'])==r['output_sha256']
        assert r['multiplier']==max(.1,r['flops']/2**41)
        assert r['weight5_cache']['completed'] and r['weight5_cache']['numeric_cache_retired']
    pairs.append(dict(rank=rank,candidate=current,parent=parent))
write(HERE/'predictions_sealed.json',dict(pairs=pairs,source_seal_sha256=sha(HERE/'sources_sealed.json'),
    inputs_sha256=sha(HERE/'inputs.json'),repeat35_exact=True,warm_cold100_exact=True,
    stage_validation_pins=[dict(path=str(HERE/(n+'_validation.json')),sha256=sha(HERE/(n+'_validation.json')))
                           for n in ('null','dev','cold100')],means_not_yet_decoded=True))
import numpy as np
rows=[]
for pair in pairs:
    rank=pair['rank'];truth_info=meta['truths'][str(rank)]
    assert sha(truth_info['path'])==truth_info['sha256']
    with np.load(truth_info['path']) as f:truth=f['mc_post'][-1].astype(np.float64)
    assert truth.shape==(1024,) and np.isfinite(truth).all()
    scores={};arrays={}
    for label in ('candidate','parent'):
        r=pair[label];a=np.load(r['output'])
        assert a.shape==(16,1024) and a.dtype==np.float32 and np.isfinite(a).all() and (a>=0).all()
        raw=float(np.mean((a[-1].astype(np.float64)-truth)**2))
        scores[label]=dict(raw=raw,flops=r['flops'],multiplier=r['multiplier'],adjusted=raw*r['multiplier'])
        arrays[label]=a
    expected=meta['expected_parent'][str(rank)]
    assert abs(scores['parent']['raw']/expected['raw']-1)<=1e-12
    assert abs(scores['parent']['adjusted']/expected['adjusted']-1)<=1e-12
    assert arrays['candidate'][:4].tobytes()==arrays['parent'][:4].tobytes()
    rows.append(dict(rank=rank,truth_sha256=truth_info['sha256'],**scores,
        raw_ratio=scores['candidate']['raw']/scores['parent']['raw'],
        cold_cost_ratio=scores['candidate']['flops']/scores['parent']['flops'],
        adjusted_ratio=scores['candidate']['adjusted']/scores['parent']['adjusted'],
        layerwise_delta_mse=np.mean((arrays['candidate'].astype(np.float64)-arrays['parent'].astype(np.float64))**2,axis=1).tolist()))
result=dict(rows=rows,development_gate_passed=all(r['adjusted_ratio']<=.95 for r in rows),
    gate='Each complete own-cold adjusted score <=0.95 of accepted parent on both35/100',
    pooled_adjusted_ratio=sum(r['candidate']['adjusted'] for r in rows)/sum(r['parent']['adjusted'] for r in rows),
    pooled_raw_ratio=sum(r['candidate']['raw'] for r in rows)/sum(r['parent']['raw'] for r in rows),
    predictions_seal_sha256=sha(HERE/'predictions_sealed.json'),
    first_four_parent_rows_exact=True,repeat35_exact=True,warm_cold100_exact=True,
    parent_raw_and_adjusted_reproduced=True,fresh_generalization_evidence=False,official_score=False,
    any_training=False,new_physical_samples=0)
write(HERE/'dev_score.json',result)
print(__import__('json').dumps(result),flush=True)

