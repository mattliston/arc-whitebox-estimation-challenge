"""Native whole-estimator calls; retain only existing local-source references."""
import argparse
import importlib.util
import resource
import sys
import time
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import HERE,read,write,pin,verify,seal_check,guarded

parser=argparse.ArgumentParser()
parser.add_argument('--out',required=True)
parser.add_argument('--ranks',required=True)
parser.add_argument('--disabled',action='store_true')
args=parser.parse_args()
seal_check();guarded('unit_v3');assert read(HERE/'unit_v3.json')['passed']
for p in read(HERE/'parameter_accounting_sources_sealed.json')['pins']:verify(p)
candidate=read(HERE/'build.json')['candidate'];verify(candidate)
assert all(t['num_threads']==1 for t in threadpool_info())
out=HERE/args.out;out.mkdir(exist_ok=False)
meta=read(HERE/'inputs.json')
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('native_mean_endpoint',candidate['path'])
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
held={};records=[]
for index,rank in enumerate(map(int,args.ranks.split(','))):
    weight=meta['weights'][str(rank)];verify(weight)
    w=np.load(weight['path'],allow_pickle=False)
    assert w.shape==(16,1024,1024) and w.dtype==np.float32
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:
        ww=[fnp.asarray(w[l]) for l in range(16)]
    assert ingress.summary_dict()['flops_used']==0
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
    held.clear();started=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        if index==0:
            spec.loader.exec_module(mod)
            mod._LMIO_ENABLED=not args.disabled
            original=mod._lmio_source
            def source(features,hidden,layer,sigma):
                assert layer not in held
                result=original(features,hidden,layer,sigma)
                held[layer]=dict(features=features,hidden=hidden,sigma=sigma,source=result)
                return result
            mod._lmio_source=source
            estimator=mod.Estimator()
        prediction=estimator.predict(mlp,2**41)
    wall=time.monotonic()-started;bill=ctx.summary_dict()['flops_used']
    assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
    assert set(held)==(set() if args.disabled else set(range(8,16)))
    a=np.asarray(prediction).copy()
    assert a.shape==(16,1024) and a.dtype==np.float32 and np.isfinite(a).all() and (a>=0).all()
    path=out/f'call{index:02d}_rank{rank:04d}.npy';np.save(path,a)
    capture=None
    if held:
        arrays={f'{key}_{layer:02d}':np.asarray(v).copy() for layer,fields in held.items() for key,v in fields.items()}
        assert all(np.isfinite(v).all() for v in arrays.values())
        cp=out/f'local{index:02d}_rank{rank:04d}.npz';np.savez(cp,**arrays)
        capture=pin(cp)
        del arrays
    row=dict(index=index,rank=rank,prediction=pin(path),capture=capture,weights=weight,
        flops=bill,multiplier=max(.1,bill/2**41),wall_s=wall,includes_initialization=index==0,
        source_enabled=not args.disabled,numeric_cache_retired=True,ingress_flops=0,
        truth_accessed=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    records.append(row);write(out/f'call{index:02d}.json',row)
    print(dict(index=index,rank=rank,enabled=not args.disabled,flops=bill,wall_s=wall),flush=True)
    held.clear();del prediction,a,w,ww,mlp
write(out/'result.json',dict(completed=True,records=records,candidate=candidate,truth_accessed=False,
    capture_additional_fnp_operations=0,full_estimator_calls=len(records),threads=threadpool_info()))
