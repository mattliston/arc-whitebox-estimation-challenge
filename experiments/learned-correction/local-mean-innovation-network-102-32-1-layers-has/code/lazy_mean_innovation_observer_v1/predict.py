import json,time,resource,gc
from common import HERE,TRAIN,CHECK,read,write,pin,sha,verify,tangent
from model import inputs,forward
def main():
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info())
    verify('learning_sources_sealed.json');verify('fit_terminal_sealed.json');verify('captures_sealed.json')
    fit=read(HERE/'fit_result.json');assert fit['completed'] and fit['passed'] and fit['optimizer_and_leaves_retired'] and fit['reloaded_nontrainable']
    g=read(HERE/'fit_watch.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    assert sha(fit['terminal']['path'])==fit['terminal']['sha256']
    with np.load(fit['terminal']['path'],allow_pickle=False) as z:p={k:z[k] for k in z.files}
    assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in p.values()) and (p['sd']>0).all()
    meta=read(HERE/'inputs.json');qualified=read(HERE/'capture_qualification/result.json')['records'];T=tangent();records=[];started=time.monotonic()
    out=HERE/'forecasts';out.mkdir(exist_ok=False)
    for rank in TRAIN+CHECK:
        if rank in TRAIN:
            rec=read(HERE/f'captures/rank{rank:04d}/result.json')['records'][0];entry=meta['train'][str(rank)]
        else:
            i=0 if rank==35 else 1;rec=qualified[i];entry=meta['qualification'][i]
        with np.load(rec['capture']['path'],allow_pickle=False) as z:cap={k:z[k] for k in z.files}
        W=np.load(entry['weights']['path'],allow_pickle=False).astype(np.float64)
        e=np.zeros(1024,np.float64);E=np.zeros((1024,1024),np.float64);local=[]
        for l in range(8,16):
            x=inputs(cap[f'features_{l:02d}'],cap[f'hidden_{l:02d}'],l,p['mu'],p['sd'],np)
            source=forward(x,p,np)*p['label_rms']*cap[f'sigma_{l:02d}'];assert source.shape==(1024,) and np.isfinite(source).all()
            if l>8:
                pre={k:cap[f'{k}_{l:02d}'] for k in ['sigma','alpha','pdf','gate','mg']}
                if l<15:pre['C_pre']=cap[f'C_pre_{l:02d}']
                e,E=T(pre,e,E,W[l],final=l==15)
                if l==15:e=e*(cap['m_15']>0)
            e=e+source
            assert np.isfinite(e).all() and (E is None or np.isfinite(E).all())
            local.append(source)
        result=cap['prediction'].copy();result[-1]=np.maximum(result[-1].astype(np.float64)+e,0).astype(np.float32)
        assert result.shape==(16,1024) and result.dtype==np.float32 and np.isfinite(result).all() and (result>=0).all()
        assert result[:-1].tobytes()==cap['prediction'][:-1].tobytes()
        path=out/f'rank{rank:04d}.npz';np.savez(path,prediction=result,correction=e,local_sources=np.stack(local))
        row=dict(rank=rank,forecast=pin(path),capture=rec['capture'],weights=entry['weights'],parent_prediction=entry['prediction'],
            first15_rows_exact=True,finite_nonnegative=True,source_mse_by_layer=[float(np.mean(v*v)) for v in local])
        records.append(row);print(json.dumps(row),flush=True)
        del cap,W,E,e,result,local;gc.collect()
    seal=dict(pins=[fit['terminal']]+[r['forecast'] for r in records]+[r['parent_prediction'] for r in records],
        all8_predictions_fixed=True,check_mean_arrays_accessed=False,terminal=fit['terminal'],records=records,
        wall_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        production_flops_unmeasured=True,new_full_estimator_calls=0)
    write(HERE/'forecasts_sealed.json',seal);print(json.dumps({k:v for k,v in seal.items() if k not in ['pins','records']}),flush=True)
if __name__=='__main__':main()
