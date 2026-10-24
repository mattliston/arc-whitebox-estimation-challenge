import json,time,resource,gc
from common import HERE,TRAIN,read,write,pin,sha,verify,tangent,relative
from model import inputs
def main():
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info())
    verify('learning_sources_sealed.json');verify('captures_sealed.json')
    assert read(HERE/'unit_fnp.json')['passed'] and read(HERE/'unit_torch.json')['passed']
    for n in ['unit_fnp_watch.json','unit_torch_watch.json']:
        g=read(HERE/n);assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    started=time.monotonic();meta=read(HERE/'inputs.json');T=tangent();raw=[];labels=[];scales=[];layers=[];ranks=[];proofs=[]
    for rank in TRAIN:
        receipt=read(HERE/f'captures/rank{rank:04d}/result.json');row=receipt['records'][0];entry=meta['train'][str(rank)]
        assert receipt['completed'] and not receipt['truth_accessed'] and row['all16_parent_bytes_and_bill_exact']
        with np.load(row['capture']['path'],allow_pickle=False) as z:cap={k:z[k] for k in z.files}
        W=np.load(entry['weights']['path'],allow_pickle=False).astype(np.float64)
        with np.load(entry['truth']['path'],allow_pickle=False) as z:
            assert int(z['P'])==8_000_000
            tm=z['mc_post'].astype(np.float64);tC=z['mc_post_cov'][7:15].astype(np.float64)
            pm=z['mc_pre_m'][8:16].astype(np.float64);pC=z['mc_pre_cov'][8:16].astype(np.float64)
        checks=[]
        for l in range(8,16):
            im=l-8;true_mean=W[l].T@tm[l-1];true_C=W[l].T@tC[im]@W[l]
            mr=relative(true_mean,pm[im]);cr=relative(true_C,pC[im]);assert max(mr,cr)<=1e-4
            e=tm[l-1]-cap[f'm_{l-1:02d}'];E=tC[im]-cap[f'C_post_{l-1:02d}'];E=(E+E.T)*.5
            pre={k:cap[f'{k}_{l:02d}'] for k in ('sigma','alpha','pdf','gate','mg')}
            transported,_=T(pre,e,E,W[l],final=True)
            if l==15:
                local=tm[l]-cap['prediction'][l].astype(np.float64)-(cap['m_15']>0)*transported
            else:local=tm[l]-cap[f'm_{l:02d}']-transported
            s=pre['sigma'];assert np.isfinite(local).all() and (s>0).all()
            raw.append(np.concatenate([cap[f'features_{l:02d}'],cap[f'hidden_{l:02d}']],axis=1))
            labels.append(local/s);scales.append(s.copy());layers.append(l);ranks.append(rank)
            checks.append(dict(layer=l,affine_mean_relative=mr,affine_covariance_relative=cr,local_source_mse=float(np.mean(local*local))))
        proofs.append(dict(rank=rank,truth=entry['truth'],capture=row['capture'],layers=checks))
        print(json.dumps(dict(rank=rank,qualified=True,source_mse=[c['local_source_mse'] for c in checks])),flush=True)
        del cap,W,tm,tC,pm,pC,true_C,E;gc.collect()
    raw=np.stack(raw);yraw=np.stack(labels);sigmas=np.stack(scales)
    assert raw.shape==(48,1024,94) and raw.dtype==np.float32 and yraw.shape==(48,1024)
    wide=raw.reshape(-1,94).astype(np.float64)
    mu=np.mean(wide,axis=0).astype(np.float32);sd=np.maximum(np.std(wide,axis=0),1e-6).astype(np.float32)
    yrms=np.asarray(np.sqrt(np.mean(yraw*yraw)),np.float32);assert float(yrms)>1e-10 and np.isfinite(yrms)
    X=np.concatenate([inputs(x[:,:30],x[:,30:],l,mu,sd,np) for x,l in zip(raw,layers)],axis=0).astype(np.float32)
    y=(yraw.astype(np.float32)/yrms).reshape(-1)
    assert X.shape==(49152,102) and y.shape==(49152,) and np.isfinite(X).all() and np.isfinite(y).all()
    out=HERE/'train_data.npz';np.savez(out,X=X,y=y,mu=mu,sd=sd,label_rms=yrms,local_labels=yraw,sigma=sigmas,
        source_layers=np.asarray(layers,np.int32),source_ranks=np.asarray(ranks,np.int32))
    result=dict(completed=True,train_ranks=TRAIN,check_target_access=False,ordinary_1B_access=False,rows=len(y),inputs=102,
        source_cells=48,label_rms=float(yrms),zero_normalized_mse=float(np.mean(y.astype(np.float64)**2)),
        data=pin(out),proofs=proofs,wall_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE/'preparation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='proofs'}),flush=True)
if __name__=='__main__':main()
