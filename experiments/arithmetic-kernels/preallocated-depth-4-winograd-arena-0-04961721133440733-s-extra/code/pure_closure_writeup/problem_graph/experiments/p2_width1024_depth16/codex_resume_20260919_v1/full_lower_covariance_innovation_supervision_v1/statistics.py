import gc,json,resource,sys,time
from common import *
def main():
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info());verify();seal=verify('captures_sealed.json')
    assert read(HERE/'wide_qualification.json')['passed']
    rank=int(sys.argv[1]);row=next(r for r in seal['rows'] if r['rank']==rank);start=time.monotonic()
    with np.load(row['capture']['path'],allow_pickle=False) as z:cap={k:z[k] for k in z.files}
    t={k[6:]:v for k,v in cap.items() if k.startswith('param_')};W=np.load(weights(rank),allow_pickle=False).astype(np.float64)
    tp=truth(rank)
    with np.load(tp,allow_pickle=False) as z:
        assert int(z['P'])==8_000_000
        tm=z['mc_post'].astype(np.float64);tc=z['mc_post_cov'].astype(np.float64)
        pre_m8=z['mc_pre_m'][8].astype(np.float64);pre_c8=z['mc_pre_cov'][8].astype(np.float64)
    sym=lambda x:(x+x.T)*.5
    mr=float(np.linalg.norm(W[8].T@tm[7]-pre_m8)/np.linalg.norm(pre_m8));cr=float(np.linalg.norm(W[8].T@tc[7]@W[8]-pre_c8)/np.linalg.norm(pre_c8))
    assert max(mr,cr)<=1e-4,(mr,cr)
    arrays={k:[] for k in ('cov_gram','cov_rhs','cov_energy','value_gram','value_rhs','value_energy')};layers=[]
    for l in range(8,15):
        pre={k:cap[f'{k}_{l:02d}'].astype(np.float64) for k in ('sigma','alpha','pdf','gate','mg','C_pre')}
        ep=tm[l-1]-cap[f'm_{l-1:02d}'];Ep=sym(tc[l-1]-cap[f'C_post_{l-1:02d}'])
        actual=sym(tc[l]-cap[f'C_post_{l:02d}']);transport=tangent_cov(pre,ep,Ep,W[l]);U=actual-transport
        d={k:cap[f'edge_{k}_{l:02d}'] for k in ('F','R','S','pf','gf','sf')};B=basis(d,t);X=B.reshape(-1,16)
        wg=W[l+1];resp=cap[f'pdf_{l+1:02d}'].astype(np.float64)/(2*cap[f'sigma_{l+1:02d}'].astype(np.float64))
        y=np.sum(wg*(U@wg),axis=0)*resp;Z=np.empty((1024,16),dtype=np.float64)
        for j in range(16):Z[:,j]=np.sum(wg*(B[:,:,j]@wg),axis=0)*resp
        arrays['cov_gram'].append(X.T@X/(1024**2));arrays['cov_rhs'].append(X.T@U.ravel()/(1024**2));arrays['cov_energy'].append(float(np.mean(U*U)))
        arrays['value_gram'].append(Z.T@Z/1024);arrays['value_rhs'].append(Z.T@y/1024);arrays['value_energy'].append(float(np.mean(y*y)))
        # Independent arbitrary fixed-coefficient loss check on full current row.
        q=np.sin(np.arange(1,17,dtype=np.float64))*.03125
        ce=float(np.mean((U-B@q)**2));ve=float(np.mean((y-Z@q)**2))
        cq=float(arrays['cov_energy'][-1]-2*q@arrays['cov_rhs'][-1]+q@arrays['cov_gram'][-1]@q)
        vq=float(arrays['value_energy'][-1]-2*q@arrays['value_rhs'][-1]+q@arrays['value_gram'][-1]@q)
        assert abs(ce-cq)<=2e-10*max(ce,1e-30) and abs(ve-vq)<=2e-10*max(ve,1e-30)
        layers.append(dict(layer=l,innovation_covariance_energy=arrays['cov_energy'][-1],incoming_tangent_residual_ratio=arrays['cov_energy'][-1]/float(np.mean(actual*actual)),next_response_energy=arrays['value_energy'][-1],covariance_loss_identity_relative=abs(ce-cq)/max(ce,1e-30),response_loss_identity_relative=abs(ve-vq)/max(ve,1e-30)))
        print(json.dumps(dict(rank=rank,**layers[-1])),flush=True)
        del B,X,Z,U,actual,transport,pre,Ep,d;gc.collect()
    dest=HERE/f'statistics_{rank:04d}.npz';np.savez(dest,**{k:np.asarray(v) for k,v in arrays.items()})
    write(HERE/f'statistics_{rank:04d}.json',dict(completed=True,rank=rank,arrays=pin(dest),capture=row['capture'],truth=pin(tp),weights=pin(weights(rank)),affine_mean_relative=mr,affine_covariance_relative=cr,layers=layers,wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
if __name__=='__main__':main()
