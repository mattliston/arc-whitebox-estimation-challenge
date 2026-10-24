"""Score only sealed complete native candidate predictions on fixed old public answers."""
import json,resource
from common import HERE,read,write,pin,verify
def main():
    verify();verify('predictions_sealed.json')
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info())
    rows=read(HERE/'candidate/result.json')['records'];c100=read(HERE/'cold100/result.json')['records'][0]
    assert rows[1]['prediction']['sha256']==c100['prediction']['sha256']
    inputs=read(HERE/'inputs.json');out=[]
    for row,parentrow in [(rows[0],inputs['qualification'][0]),(c100,inputs['cold100'])]:
        rank=row['rank']
        with np.load(inputs['truth'][str(rank)]['path'],allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
        pred=np.load(row['prediction']['path'],allow_pickle=False)[-1].astype(np.float64)
        parent=np.load(parentrow['prediction']['path'],allow_pickle=False)[-1].astype(np.float64)
        raw=float(np.mean((pred-truth)**2));base=float(np.mean((parent-truth)**2))
        adjusted=raw*max(.1,row['whole_flops']/2**41);baj=base*max(.1,parentrow['flops']/2**41)
        out.append(dict(rank=rank,raw=raw,parent_raw=base,raw_ratio=raw/base,flops=row['whole_flops'],parent_flops=parentrow['flops'],
                        adjusted=adjusted,parent_adjusted=baj,adjusted_ratio=adjusted/baj))
    pooled=sum(r['adjusted'] for r in out)/sum(r['parent_adjusted'] for r in out)
    gate=all(r['adjusted_ratio']<1 for r in out) and pooled<=.995
    r=dict(completed=True,gate_passed=gate,records=out,pooled_adjusted_ratio=pooled,
           independent_new_holdout=False,phase2_only=True,fitted_parameters=0,new_samples=0,new_upload=False,
           peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE/'score.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':main()
