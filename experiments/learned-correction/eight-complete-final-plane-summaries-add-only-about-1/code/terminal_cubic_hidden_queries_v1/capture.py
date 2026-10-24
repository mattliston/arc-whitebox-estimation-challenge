import ast,gc,hashlib,importlib.util,json,resource,sys,time,types
from common import *
def observed():
    s=SOURCE.read_text();assert sha(SOURCE)==PARENT_SHA
    n=[x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name=='_predict_binary'][-1]
    old=ast.get_source_segment(s,n);new=old
    edits=[
      ('                pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)',
       '                pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)\n                _CHQ_TAP.update(P=pp,lam=lam,A=star,C=C,W=w32,b=bpre)'),
      ('                m = m + final_hook(l, m, aux, W[l])',
       '                m = m + final_hook(l, m, aux, W[l])\n                _CHQ_TAP.update(k3=k3,sigma=aux["sigma"],alpha=aux["alpha"],pdf=aux["pdf"])')]
    for a,b in edits:assert new.count(a)==1;new=new.replace(a,b,1)
    back=new
    for a,b in reversed(edits):assert back.count(b)==1;back=back.replace(b,a,1)
    assert back==old;ast.parse(new)
    return new,dict(function='_predict_binary',exact_inverse=True,original_sha256=hashlib.sha256(old.encode()).hexdigest(),observed_sha256=hashlib.sha256(new.encode()).hexdigest())
def main():
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info());verify();assert read(HERE/'unit.json')['passed']
    out=HERE/'capture';out.mkdir(exist_ok=False);source,proof=observed();tap={}
    stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
    spec=importlib.util.spec_from_file_location('chq_current_parent',SOURCE);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    refs=read(D/'full_lower_multilayer_mean_observer_v1/parent_triple/result.json')['records'];records=[];first=None
    for call,rank in enumerate((35,100,35)):
        wp=weights(rank);ws=np.load(wp,allow_pickle=False)
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ing:ww=[fnp.asarray(ws[l]) for l in range(16)]
        assert ing.summary_dict()['flops_used']==0
        mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}');started=time.monotonic()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            if call==0:
                spec.loader.exec_module(mod);mod._CHQ_TAP=tap;exec(compile(source,str(HERE/'observed.py'),'exec'),mod.__dict__)
                original=mod.gnn_step
                def step(l,x,sigma,w,state,p):
                    answer=original(l,x,sigma,w,state,p)
                    if l==15:tap['H']=answer[1]
                    return answer
                mod.gnn_step=step;estimator=mod.Estimator()
            pred=estimator.predict(mlp,2**41)
        elapsed=time.monotonic()-started;bill=ctx.summary_dict()['flops_used']
        assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['numeric_cache_retired']
        assert set(tap)=={'P','lam','A','C','W','b','H','k3','sigma','alpha','pdf'}
        arrays={k:np.asarray(v) for k,v in tap.items()};arrays['prediction']=np.asarray(pred)
        assert all(np.isfinite(v).all() for v in arrays.values()) and (arrays['sigma']>0).all()
        ref=refs[call];assert ref['rank']==rank and ref['weights_sha256']==sha(wp) and bill==ref['flops']
        assert np.array_equal(arrays['prediction'],np.load(ref['output'],allow_pickle=False))
        digest={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
        if call==0:first=digest
        if call==2:
            assert digest==first;dest=Path(records[0]['capture']['path'])
        else:
            dest=out/f'call_{call:02d}_rank{rank:04d}.npz';np.savez(dest,**arrays)
        rec=dict(call=call,rank=rank,capture=pin(dest),array_digests=digest,flops=bill,wall_s=elapsed,weights=pin(wp),all16_and_bill_exact=True,cache_retired=True);records.append(rec)
        print(json.dumps({k:v for k,v in rec.items() if k!='array_digests'}),flush=True)
        tap.clear();del arrays,pred,ws,ww,mlp;gc.collect()
    write(out/'result.json',dict(completed=True,records=records,source_proof=proof,truth_accessed=False,extra_parent_operations=0,repeat_all_arrays_exact=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
if __name__=='__main__':main()
