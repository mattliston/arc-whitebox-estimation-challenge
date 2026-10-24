"""One prespecified ridge fit and network-disjoint public CHECK source screen."""
import json,resource,gc
from common import HERE,TRAIN,CHECK,read,write,pin,verify

def main():
    verify();verify('learning_sealed.json')
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info())
    inputs=read(HERE/'learning_inputs.json')
    qualification=read(HERE/'qualification/result.json');assert qualification['qualified']
    for name in ['qualification']+[f'train_{r}' for r in TRAIN]:
        g=read(HERE/(name+'_watch.json'))
        assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0

    def loadrow(rank,truth=None):
        entry=inputs['rows'][str(rank)]
        with np.load(entry['capture']['path'],allow_pickle=False) as z:
            q=np.stack([z[f'energy_{l:02d}'] for l in range(8,16)]).astype(np.float64)
            alpha=np.stack([z[f'alpha_{l:02d}'] for l in range(8,16)]).astype(np.float64)
            pdf=np.stack([z[f'pdf_{l:02d}'] for l in range(8,16)]).astype(np.float64)
            sigma=np.stack([z[f'sigma_{l:02d}'] for l in range(8,16)]).astype(np.float64)
            feature=np.stack([z[f'features_{l:02d}'] for l in range(8,16)]).astype(np.float64)
        tpath=(truth if truth is not None else entry['truth'])['path']
        with np.load(tpath,allow_pickle=False) as z:
            k3=z['mc_pre_k3'][8:16].astype(np.float64)
            k4=z['mc_pre_k4'][8:16].astype(np.float64)
            samples=int(z['P'])
        assert samples>=8_000_000 and k3.shape==k4.shape==q.shape==(8,1024)
        row=dict(rank=rank,q=q,alpha=alpha,skew=feature[:,:,10],
                 y3=k3/sigma**3-feature[:,:,10],y4=k4/sigma**4-feature[:,:,12],
                 w3=sigma*(-alpha*pdf/6),w4=sigma*((alpha*alpha-1)*pdf/24))
        assert all(np.isfinite(x).all() for k,x in row.items() if k!='rank')
        return row

    train=[loadrow(r) for r in TRAIN]
    qs=np.stack([r['q'] for r in train])
    qmean=np.mean(qs,axis=(0,2));qstd=np.std(qs,axis=(0,2))
    assert np.all(qstd>0);del qs
    def design(row,kind,extended):
        n=row['q'].shape[1];q=(row['q']-qmean[:,None])/qstd[:,None]
        if kind==3:
            X=np.zeros((8,n,18 if extended else 16))
            for l in range(8):X[l,:,2*l]=row['alpha'][l];X[l,:,2*l+1]=row['skew'][l]
            if extended:X[:,:,16]=q*row['alpha'];X[:,:,17]=q*row['skew']
        else:
            X=np.zeros((8,n,9 if extended else 8))
            for l in range(8):X[l,:,l]=1
            if extended:X[:,:,8]=q
        return X.reshape(8*n,-1)
    fit_records=[];coeff={}
    for extended in [False,True]:
        name='extended' if extended else 'control'
        for kind in [3,4]:
            X=np.concatenate([design(r,kind,extended) for r in train])
            y=np.concatenate([r[f'y{kind}'].ravel() for r in train])
            w=np.concatenate([r[f'w{kind}'].ravel() for r in train])
            M=X*w[:,None];target=y*w
            scale=np.sqrt(np.mean(M*M,axis=0));floor=max(float(np.max(scale))*1e-12,1e-30)
            scale=np.maximum(scale,floor);M/=scale
            N=len(target);normal=M.T@M/N+1e-3*np.eye(M.shape[1]);rhs=M.T@target/N
            theta=np.linalg.solve(normal,rhs)
            residual=float(np.linalg.norm(normal@theta-rhs)/max(np.linalg.norm(rhs),1e-30))
            assert residual<=1e-10
            coeff[f'{name}_{kind}']=theta/scale
            fit_records.append(dict(model=name,cumulant=kind,parameters=len(theta),rows=N,ridge=.001,
                                    normal_equation_relative=residual,condition_number=float(np.linalg.cond(normal))))
            del X,y,w,M,target,normal,rhs,theta
    model_path=HERE/'terminal_ridge.npz'
    np.savez(model_path,qmean=qmean,qstd=qstd,**coeff)
    # Reload the frozen coefficients before CHECK labels or feature vectors are decoded.
    del coeff
    with np.load(model_path,allow_pickle=False) as z:
        coeff={k:z[k].copy() for k in z.files if k not in ['qmean','qstd']}
        assert np.array_equal(z['qmean'],qmean) and np.array_equal(z['qstd'],qstd)
    predictions={};records=[]
    def evaluate(row,split):
        losses={};local={}
        for name in ['zero','control','extended']:
            total=np.zeros((8,1024));fields={}
            for kind in [3,4]:
                pred=np.zeros((8,1024)) if name=='zero' else (design(row,kind,name=='extended')@coeff[f'{name}_{kind}']).reshape(8,1024)
                err=(row[f'y{kind}']-pred)*row[f'w{kind}']
                total+=err*err;fields[f'k{kind}']=float(np.mean(err*err))
                if name!='zero':predictions[f'{split}_{row["rank"]}_{name}_k{kind}']=pred
            losses[name]=dict(total=float(np.mean(total)),**fields)
            local[name]=np.mean(total,axis=1).tolist()
        rec=dict(rank=row['rank'],split=split,losses=losses,by_layer_loss=local,
                 extended_over_zero=losses['extended']['total']/losses['zero']['total'],
                 extended_over_control=losses['extended']['total']/losses['control']['total'])
        records.append(rec);return rec
    for row in train:evaluate(row,'TRAIN')
    del train;gc.collect()
    for rank in CHECK:evaluate(loadrow(rank),'CHECK')
    pooled={}
    for split in ['TRAIN','CHECK']:
        selected=[r for r in records if r['split']==split]
        m={k:sum(r['losses'][k]['total'] for r in selected)/len(selected) for k in ['zero','control','extended']}
        pooled[split]=dict(losses=m,extended_over_zero=m['extended']/m['zero'],extended_over_control=m['extended']/m['control'])
    cr=[r for r in records if r['split']=='CHECK'];pc=pooled['CHECK']
    gates=dict(pooled_zero=pc['extended_over_zero']<=.85,pooled_control=pc['extended_over_control']<=.95,
               each_network_no_worse=all(r['extended_over_zero']<=1 and r['extended_over_control']<=1 for r in cr))
    path=HERE/'source_predictions.npz';np.savez(path,**predictions)
    result=dict(completed=True,qualified=all(gates.values()),gates=gates,pooled=pooled,records=records,fit_records=fit_records,
                model=pin(model_path),predictions=pin(path),control_parameters=24,extended_parameters=27,
                train_ranks=TRAIN,check_ranks=CHECK,new_samples=0,ordinary_1B_arrays_accessed=False,
                native_candidate_created=False,previous_failed_fit_reopened=False,
                objective='sum of separately squared third/fourth marginal mean-response residuals; not endpoint error',
                network_disjoint_but_reused_public_check=True,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE/'information.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','fit_records']}),flush=True)
if __name__=='__main__':main()
