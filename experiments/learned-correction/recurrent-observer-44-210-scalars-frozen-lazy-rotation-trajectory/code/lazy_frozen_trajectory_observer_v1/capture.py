"""Original GNN/scalar reference taps only; no numerical parent changes."""
import argparse,importlib.util,json,resource,sys,time,types
from pathlib import Path
from common import HERE,SOURCE,PARENT_SHA,TRAIN,CHECK,sha,read,seal_check,write
seal_check();assert sha(SOURCE)==PARENT_SHA
parser=argparse.ArgumentParser();parser.add_argument('--qualification',action='store_true');parser.add_argument('--rank',type=int);args=parser.parse_args()
assert args.qualification != (args.rank is not None)
inputs=read(HERE/'inputs.json')
if args.qualification:
    rows=inputs['qualification'];out=HERE/'capture_qualification'
else:
    assert args.rank in TRAIN[1:]+CHECK
    q=read(HERE/'neural_qualification/result.json');g=read(HERE/'neural_qualification_watch.json')
    assert q['passed'] and g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    rows=[inputs['cold'][str(args.rank)]];out=HERE/'captures'/f'rank{args.rank:04d}'
out.mkdir(exist_ok=False)
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
assert all(t['num_threads']==1 for t in threadpool_info())
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('lfto_unchanged_parent',SOURCE);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
held={};records=[];first=None;Pfirst=None
for call,row in enumerate(rows):
    rank=row['rank'];assert sha(row['weights'])==row['weights_sha256'];assert sha(row['prediction'])==row['prediction_sha256']
    weights=np.load(row['weights'],allow_pickle=False)
    assert weights.shape==(16,1024,1024) and weights.dtype==np.float32 and np.isfinite(weights).all()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(weights[l]) for l in range(16)]
    assert ingress.summary_dict()['flops_used']==0
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
    held.clear();held.update(x=[None]*16,sigma=[None]*16,H=[None]*16,delta=[None]*16,layers=[])
    started=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        if call==0:
            spec.loader.exec_module(mod)
            original_step,original_scalar=mod.gnn_step,mod.nonlinear_scalar
            def step(l,x,sigma,w,state,p):
                assert held['x'][l] is None
                held['x'][l]=x;held['sigma'][l]=sigma;held['P']=p;held['layers'].append(l)
                delta,next_state=original_step(l,x,sigma,w,state,p)
                held['H'][l]=next_state;held['delta'][l]=delta
                return delta,next_state
            def scalar(mu,v,k3,k4):
                answer=original_scalar(mu,v,k3,k4);held['last_scalar_host']=answer[0]
                return answer
            mod.gnn_step,mod.nonlinear_scalar=step,scalar
            estimator=mod.Estimator()
        prediction=estimator.predict(mlp,2**41)
    elapsed=time.monotonic()-started;F=ctx.summary_dict()['flops_used']
    assert held['layers']==list(range(16))
    assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
    arrays=dict(features=np.stack([np.asarray(x).copy() for x in held['x']]),
                sigma=np.stack([np.asarray(x).copy() for x in held['sigma']]),
                hidden=np.stack([np.asarray(x).copy() for x in held['H']]),
                delta=np.stack([np.asarray(x).astype(np.float32) for x in held['delta']]),
                final_host=np.asarray(held['last_scalar_host']).copy(),prediction=np.asarray(prediction).copy())
    assert arrays['features'].shape==(16,1024,30) and arrays['features'].dtype==np.float32
    assert arrays['sigma'].shape==(16,1024) and arrays['sigma'].dtype==np.float64
    assert arrays['hidden'].shape==(16,1024,64) and arrays['hidden'].dtype==np.float32
    assert arrays['final_host'].shape==(1024,) and arrays['final_host'].dtype==np.float64
    assert arrays['prediction'].shape==(16,1024) and arrays['prediction'].dtype==np.float32
    assert np.asarray(held['delta'][0]).dtype==np.float64 and not np.asarray(held['delta'][0]).any()
    assert (arrays['sigma']>0).all() and (arrays['prediction']>=0).all()
    for key,value in arrays.items():assert np.isfinite(value).all(),key
    expected=np.maximum(arrays['final_host']+arrays['delta'][-1].astype(np.float64),0).astype(np.float32)
    assert expected.tobytes()==arrays['prediction'][-1].tobytes()
    if row['prediction_kind']=='npy':baseline=np.load(row['prediction'],allow_pickle=False)
    else:
        with np.load(row['prediction'],allow_pickle=False) as z:baseline=z['prediction'].copy()
    assert baseline.dtype==np.float32 and baseline.shape==(16,1024)
    assert arrays['prediction'].tobytes()==baseline.tobytes() and F==row['flops'],(rank,F,row['flops'])
    parameters={k:np.asarray(v).copy() for k,v in held['P'].items()}
    assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in parameters.values())
    if Pfirst is None:
        Pfirst=parameters;parameter_path=out/'parent_P.npz';np.savez(parameter_path,**parameters)
    else:assert all(parameters[k].tobytes()==Pfirst[k].tobytes() for k in Pfirst)
    if args.qualification and rank==35:
        if first is None:first={k:v.copy() for k,v in arrays.items()}
        else:assert all(v.tobytes()==first[k].tobytes() for k,v in arrays.items())
    path=out/f'call_{call:02d}_rank{rank:04d}.npz';np.savez(path,**arrays)
    record=dict(call=call,rank=rank,path=str(path),sha256=sha(path),bytes=path.stat().st_size,parent_flops=F,parent_multiplier=max(.1,F/2**41),includes_initialization=call==0,weights=row['weights'],weights_sha256=row['weights_sha256'],all16_parent_bytes_and_bill_exact=True,final_host_reconstruction_exact=True,numeric_cache_retired=True,original_source_unchanged=True,extra_parent_numerical_operations=0,delta0_zero_canonicalized_FP32=True,parent_wall_s=elapsed,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    records.append(record);print(json.dumps(record),flush=True)
    held.clear();del prediction,weights,ww,mlp,arrays,parameters,baseline
result=dict(completed=True,records=records,parent_P=dict(path=str(parameter_path),sha256=sha(parameter_path)),parent_source_sha256=PARENT_SHA,truth_accessed=False,repeat_all_captured_bytes_exact=bool(args.qualification),new_full_parent_calls=len(rows),capture_source_sha256=sha(__file__),source_seal_sha256=sha(HERE/'capture_sources_sealed.json'),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write(out/'result.json',result)
