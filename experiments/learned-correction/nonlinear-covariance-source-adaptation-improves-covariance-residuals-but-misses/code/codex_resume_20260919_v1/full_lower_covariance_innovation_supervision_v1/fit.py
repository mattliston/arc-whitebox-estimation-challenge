import json,resource
from common import *
def main():
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info());verify();verify('statistics_sealed.json')
    data={}
    for rank in TRAIN+VALID:
        with np.load(HERE/f'statistics_{rank:04d}.npz',allow_pickle=False) as z:data[rank]={k:z[k] for k in z.files}
    nc=float(np.mean([data[r]['cov_energy'] for r in TRAIN]));nv=float(np.mean([data[r]['value_energy'] for r in TRAIN]));assert min(nc,nv)>0
    G=np.mean([data[r]['cov_gram']/nc+data[r]['value_gram']/nv for r in TRAIN],axis=(0,1))
    b=np.mean([data[r]['cov_rhs']/nc+data[r]['value_rhs']/nv for r in TRAIN],axis=(0,1))
    scale=np.maximum(np.sqrt(np.maximum(np.diag(G),0)),1e-20);Gn=G/scale[:,None]/scale[None,:];bn=b/scale
    M=Gn+1e-3*np.eye(16);beta=np.linalg.solve(M,bn);res=float(np.linalg.norm(M@beta-bn)/max(np.linalg.norm(bn),1e-30));assert res<=1e-9,res
    delta=beta/scale;delta32=delta.astype(np.float32);assert np.isfinite(delta32).all();q=delta32.astype(np.float64)
    dest=HERE/'payload.npz';np.savez(dest,delta64=delta,delta32=delta32,beta=beta,scale=scale,gram=G,rhs=b)
    write(HERE/'payload_sealed.json',dict(payload=pin(dest),train=list(TRAIN),valid=list(VALID),fit='one shared16-coefficient covariance-innovation/response ridge',ridge=1e-3,solve_relative=res,covariance_norm=nc,response_norm=nv,targets_used_in_fit=list(TRAIN),delta32=q.tolist()))
    rows=[]
    for rank in TRAIN+VALID:
        d=data[rank];layers=[]
        for j,l in enumerate(range(8,15)):
            v={}
            for prefix in ('cov','value'):
                e=float(d[f'{prefix}_energy'][j]);loss=float(e-2*q@d[f'{prefix}_rhs'][j]+q@d[f'{prefix}_gram'][j]@q);assert loss>=-1e-12*e
                v[prefix]=dict(baseline=e,residual=max(loss,0),ratio=max(loss,0)/e)
            layers.append(dict(layer=l,**v))
        pooled={prefix:sum(x[prefix]['residual'] for x in layers)/sum(x[prefix]['baseline'] for x in layers) for prefix in ('cov','value')}
        row=dict(rank=rank,role='TRAIN' if rank in TRAIN else 'VALID',layers=layers,pooled=pooled,passed=pooled['cov']<=.98 and pooled['value']<=.95);rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='layers'}),flush=True)
    passed=all(r['passed'] for r in rows if r['role']=='VALID')
    write(HERE/'result.json',dict(completed=True,passed=passed,rows=rows,payload=pin(dest),solve_relative=res,parameters=16,fit_count=1,native_gradient_qualified=False,complete_adjusted_score=False,ordinary_1B_targets_opened=False,new_samples=0,qualification='Frozen-current-state Gaussian-gain innovation proxy on reused coherent8M public research references; no independent sampling-noise correction or changed native trajectory.',peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(passed=passed)),flush=True)
if __name__=='__main__':main()
