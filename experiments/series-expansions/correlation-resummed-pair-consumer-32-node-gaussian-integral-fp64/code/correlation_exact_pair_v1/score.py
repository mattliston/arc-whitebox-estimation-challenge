import time,resource,json
from common import HERE,read,write,pin,sha,verify
def main():
    verify('native_sources_sealed.json');meta=read(HERE/'inputs.json');started=time.monotonic()
    for stage in ['null','dev','cold100']:
        g=read(HERE/(stage+'_watch.json'));assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
        assert read(HERE/(stage+'_validation.json'))['passed']
    dev=read(HERE/'dev/result.json')['records'];cold=read(HERE/'cold100/result.json')['records'];pairs=[]
    assert dev[0]['output_sha256']==dev[2]['output_sha256'] and dev[1]['output_sha256']==cold[0]['output_sha256']
    for rank,current,key in [(35,dev[0],'parent_dev'),(100,cold[0],'parent_cold100')]:
        parent=read(meta[key]['path'])['records'][0];assert current['rank']==parent['rank']==rank
        for r in [current,parent]:assert sha(r['output'])==r['output_sha256'] and r['includes_initialization'] and r['multiplier']==max(.1,r['flops']/2**41)
        pairs.append(dict(rank=rank,candidate=current,parent=parent))
    write(HERE/'before_truth.json',dict(pairs=pairs,ordinary_mean_arrays_not_yet_decoded=True,source_seal=pin(HERE/'native_sources_sealed.json')))
    import numpy as np
    rows=[]
    for pair in pairs:
        rank=pair['rank'];truthpin=meta['truths'][str(rank)];assert sha(truthpin['path'])==truthpin['sha256']
        with np.load(truthpin['path'],allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
        scores={};arrays={}
        for label in ['parent','candidate']:
            r=pair[label];a=np.load(r['output'],allow_pickle=False);arrays[label]=a
            mse=float(np.mean((a[-1].astype(np.float64)-truth)**2));scores[label]=dict(raw=mse,flops=r['flops'],multiplier=r['multiplier'],adjusted=mse*r['multiplier'])
        expected=meta['expected_parent'][str(rank)];assert scores['parent']['raw']==expected['raw'] and scores['parent']['adjusted']==expected['adjusted']
        rows.append(dict(rank=rank,**scores,raw_ratio=scores['candidate']['raw']/scores['parent']['raw'],cost_ratio=scores['candidate']['flops']/scores['parent']['flops'],
            adjusted_ratio=scores['candidate']['adjusted']/scores['parent']['adjusted'],layer_delta_mse=np.mean((arrays['candidate'].astype(np.float64)-arrays['parent'])**2,axis=1).tolist()))
    result=dict(completed=True,passed=all(r['adjusted_ratio']<=.90 for r in rows),rows=rows,
        pooled_adjusted_ratio=sum(r['candidate']['adjusted'] for r in rows)/sum(r['parent']['adjusted'] for r in rows),
        gate='Both own-cold adjusted ratios <=.90',fresh_generalization=False,official_score=False,new_samples=0,training=False,
        before_truth=pin(HERE/'before_truth.json'),wall_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE/'score.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
