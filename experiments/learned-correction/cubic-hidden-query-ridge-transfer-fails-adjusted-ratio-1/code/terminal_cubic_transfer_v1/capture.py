"""One cold parent replay plus separately metered unchanged features."""
import argparse,gc,hashlib,importlib.util,resource,sys,time,types
from common import *
from observer_source import observed
from helper import cubic_hidden_features


def main():
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,required=True);ap.add_argument('--name',required=True)
    args=ap.parse_args();seal_check();done('unit');done('solver_unit')
    assert all(t['num_threads']==1 for t in threadpool_info())
    row={r['rank']:r for r in read(HERE/'inputs.json')['rows']}[args.rank]
    for k in ['weights','parent_result','parent_prediction']:check(row[k])
    source,proof=observed();tap={}
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
    spec=importlib.util.spec_from_file_location('transfer_parent',SOURCE);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    ws=np.load(row['weights']['path'],allow_pickle=False)
    assert ws.shape==(16,1024,1024) and ws.dtype==np.float32
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ing:ww=[fnp.asarray(x) for x in ws]
    assert ing.summary_dict()['flops_used']==0
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{args.rank}')
    started=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        spec.loader.exec_module(mod);mod._CHQ_TAP=tap
        exec(compile(source,str(HERE/'observed.py'),'exec'),mod.__dict__)
        original=mod.gnn_step
        def step(l,x,sigma,w,state,p):
            answer=original(l,x,sigma,w,state,p)
            if l==15:tap['H']=answer[1]
            return answer
        mod.gnn_step=step;estimator=mod.Estimator();pred=estimator.predict(mlp,2**41)
    parent_bill=int(ctx.summary_dict()['flops_used']);elapsed=time.monotonic()-started
    assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['numeric_cache_retired']
    prediction=np.asarray(pred).copy();parent=np.load(row['parent_prediction']['path'],allow_pickle=False)
    assert prediction.shape==(16,1024) and prediction.dtype==np.float32
    assert prediction.tobytes()==parent.tobytes() and parent_bill==row['parent_flops']
    assert np.isfinite(prediction).all() and (prediction>=0).all()
    assert set(tap)=={'P','lam','A','C','W','b','H','k3','sigma','alpha','pdf'}
    arrays={k:np.asarray(v) for k,v in tap.items()}
    assert all(np.isfinite(v).all() for v in arrays.values()) and (arrays['sigma']>0).all()
    def digest(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
    before={k:digest(v) for k,v in arrays.items()}
    gc.collect()
    with flops.BudgetContext(flop_budget=2**35,quiet=True) as hctx:
        values=cubic_hidden_features(fnp,**{k:tap[k] for k in arrays if k!='k3'})
    helper_bill=int(hctx.summary_dict()['flops_used']);assert helper_bill<=20_000_000_000
    actual=[np.asarray(v).copy() for v in values]
    assert before=={k:digest(v) for k,v in arrays.items()}
    assert actual[0].shape==(1024,65) and all(np.isfinite(v).all() for v in actual)
    P=arrays['P'].astype(float);lam=arrays['lam'].astype(float)
    V=arrays['W'].astype(float).T;U=V@arrays['A'].astype(float)
    C=arrays['C'].astype(np.float32).astype(float);B=V@((C+C.T)*.5)@V.T
    X=np.column_stack([np.ones(1024),arrays['H'].astype(float)])
    norm=np.linalg.norm(X,axis=0);r=(X/np.maximum(norm,1e-12)[None,:])/arrays['sigma'][:,None]
    pr=P.T@r;vr=V.T@r;ur=U.T@r;br=B@r;b=arrays['b']
    query=(P*lam[None,:])@(pr*pr)+(U@(vr*vr)+2*V@(vr*ur))/3
    query+=b[:,None]*np.einsum('ic,ic->c',r,br)[None,:]+2*br*(b@r)[None,:]
    ref=(-arrays['alpha']*arrays['pdf']/6)[:,None]*query
    relative=float(np.linalg.norm(actual[0]-ref)/max(np.linalg.norm(ref),1e-30));assert relative<=2e-10
    formal=np.sum(P*P*P*lam[None,:],axis=1)+np.sum(U*V*V,axis=1)+3*b*np.diag(B)
    diag_relative=float(np.linalg.norm(formal-arrays['k3'])/max(np.linalg.norm(arrays['k3']),1e-30));assert diag_relative<=1e-3
    fields=dict(feature=actual[0],affine=arrays['sigma'][:,None]*X,prediction=prediction)
    digests={k:digest(v) for k,v in fields.items()}
    if 'old_feature' in row:
        check(row['old_feature']['file'])
        assert helper_bill==row['old_feature']['helper_flops']
        for k in fields:assert digests[k]==row['old_feature']['array_digests'][k],k
    out=HERE/f'{args.name}.npz';np.savez(out,**fields)
    record=dict(rank=args.rank,name=args.name,file=pin(out),weights=row['weights'],
        parent_flops=parent_bill,helper_flops=helper_bill,includes_initialization=True,
        all16_parent_bytes_and_full_bill_exact=True,cache_retired=True,input_preserved=True,
        feature_digests=digests,reference_relative=relative,formal_diagonal_relative=diag_relative,
        source_proof=proof,targets_decoded=False,parent_wall_s=elapsed,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE/f'{args.name}.json',record)
    print({k:v for k,v in record.items() if k not in ['file','weights','feature_digests','source_proof']},flush=True)
if __name__=='__main__':main()
