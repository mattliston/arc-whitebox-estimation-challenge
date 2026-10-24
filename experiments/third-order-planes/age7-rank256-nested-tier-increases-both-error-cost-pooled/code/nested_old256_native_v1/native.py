import argparse,importlib.util,resource,sys,time,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import *
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--ranks',required=True);ap.add_argument('--disabled',action='store_true');args=ap.parse_args()
seal_check();done('unit');assert read(HERE/'unit.json')['passed']
assert all(t['num_threads']==1 for t in threadpool_info())
meta={r['rank']:r for r in read(HERE/'inputs.json')['rows']};build=read(HERE/'build.json');check(build['candidate'])
out=HERE/args.out;out.mkdir(exist_ok=False)
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('nested_candidate',build['candidate']['path']);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
records=[];layouts=[]
for index,rank in enumerate(map(int,args.ranks.split(','))):
    row=meta[rank];check(row['weights']);w=np.load(row['weights']['path'],allow_pickle=False)
    assert w.shape==(16,1024,1024) and w.dtype==np.float32
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(x) for x in w]
    assert ingress.summary_dict()['flops_used']==0
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
    layouts.clear();start=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        if index==0:
            spec.loader.exec_module(mod);mod._NESTED_ENABLED=not args.disabled
            original=mod._shared_old384_push
            def push(layer,w,p,previous,newborn,lam,state):
                ans=original(layer,w,p,previous,newborn,lam,state)
                layouts.append(dict(layer=layer,old_columns=int(state['columns']),nested_columns=int(state['nested_columns']),
                    nested_joins=len(state['nested_joins']),first_tier_joins=len(state['joins']),
                    total_columns=int(ans.shape[1]),previous_rows=int(previous.shape[0])))
                return ans
            mod._shared_old384_push=push;estimator=mod.Estimator()
        result=estimator.predict(mlp,2**41)
    wall=time.monotonic()-start;bill=ctx.summary_dict()['flops_used']
    assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
    pred=np.asarray(result).copy();assert pred.shape==(16,1024) and pred.dtype==np.float32 and np.isfinite(pred).all() and (pred>=0).all()
    check(row['prediction']);parent=np.load(row['prediction']['path'],allow_pickle=False)
    assert pred[:8].tobytes()==parent[:8].tobytes()
    if args.disabled:
        assert pred.tobytes()==parent.tobytes() and bill==row['parent']['flops']
        assert all(r['nested_columns']==0 for r in layouts)
    else:
        assert [r['nested_joins'] for r in layouts]==[0]*8+list(range(1,8))+[7]
    path=out/f'call{index:02d}_rank{rank:04d}.npy';np.save(path,pred)
    record=dict(index=index,rank=rank,prediction=pin(path),weights=row['weights'],flops=bill,multiplier=max(.1,bill/2**41),
        wall_s=wall,includes_initialization=index==0,input_flops=0,numeric_cache_retired=True,
        first_eight_parent_layers_exact=True,nested_enabled=not args.disabled,layout=list(layouts),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(out/f'call{index:02d}.json',record);records.append(record)
    print(dict(rank=rank,index=index,flops=bill,wall_s=wall),flush=True)
    del w,ww,mlp,result,pred,parent
write(out/'result.json',dict(records=records,completed=True,candidate=build['candidate'],targets_decoded=False))
