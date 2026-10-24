"""Copy one target-free Lazy join; require unchanged whole outputs and bill."""
import hashlib,importlib.util,json,resource,sys,time,types
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info

HERE=Path(__file__).resolve().parent
D=HERE.parent
P2=D.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
seal=read(HERE/'source_sealed.json')
for name,digest in seal['files_sha256'].items():assert sha(HERE/name)==digest,name
parent=D/'lazy_rotation_objective_v1/candidate.py'
assert sha(parent)==seal['parent_sha256']
assert read(HERE/'unit.json')['passed']
assert all(x['num_threads']==1 for x in threadpool_info())
out=HERE/'capture'
out.mkdir(exist_ok=False)
store=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest=read(store/'manifest.json')['roles']['observer_fit']['row_manifests']
weight_path=store/Path(manifest[35]['path']).parent/'weights.npy'
weights=np.load(weight_path)
assert sha(weight_path)=='14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2'
stub=types.ModuleType('whestbench')
stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('observed_lazy',parent)
mod=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=mod
packet={}
metadata={}

def install():
    original=mod._shared_old384_push
    def observed(layer,w32,p,previous,newborn,lam,state):
        if layer==8:
            assert not packet and state['lazy'] is None
            start,width=state['cohorts'][layer-5]
            stop=start+width
            assert start==state['columns'] and start>0
            ix=np.asarray(previous).astype(np.int64)
            packet['Q']=np.asarray(state['Q'])[ix,:].copy()
            packet['F']=np.asarray(state['F']).copy()
            packet['J']=np.asarray(p)[ix,start:stop].copy()
            packet['lambda']=np.asarray(lam)[:stop].copy()
            packet['W']=np.asarray(w32)[ix,:].copy()
            metadata.update(layer=layer,rank=35,previous_active=len(ix),
                            prefix_columns=start,joining_columns=width,total_columns=stop,
                            parent_rank=384,old_factors_materialized=True)
        result=original(layer,w32,p,previous,newborn,lam,state)
        if layer==8:
            packet['parent_pushed']=np.asarray(result)[:,:metadata['total_columns']].copy()
        return result
    mod._shared_old384_push=observed

with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:
    ww=[fnp.asarray(weights[l]) for l in range(16)]
mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name='rank35')
began=time.monotonic()
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
    spec.loader.exec_module(mod)
    install()
    prediction=mod.Estimator().predict(mlp,2**41)
seconds=time.monotonic()-began
pred=np.asarray(prediction,dtype=np.float32).copy()
assert pred.shape==(16,1024) and np.isfinite(pred).all() and (pred>=0).all()
np.save(out/'prediction.npy',pred)
control=read(D/'lazy_rotation_v1/dev/result.json')
row=control['records'][0]
bill=ctx.summary_dict()['flops_used']
assert control['candidate_sha256']==sha(parent)
assert sha(out/'prediction.npy')==row['output_sha256']
assert sha(row['output'])==row['output_sha256']
assert bill==row['flops'] and row['weights_sha256']==sha(weight_path)
assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed']
assert mod._W5_PORT_LAST['numeric_cache_retired']
assert packet and all(np.isfinite(x).all() for x in packet.values())
np.savez(out/'join.npz',**packet)
record=dict(**metadata,parent_sha256=sha(parent),observer_sha256=sha(__file__),
            packet_sha256=sha(out/'join.npz'),packet_bytes=(out/'join.npz').stat().st_size,
            arrays={k:dict(shape=list(x.shape),dtype=str(x.dtype)) for k,x in packet.items()},
            output_sha256=sha(out/'prediction.npy'),all16_parent_bytes_exact=True,
            complete_charged_flops=bill,parent_bill_exact=True,wall_s=seconds,
            weights_sha256=sha(weight_path),input_flops_separate=ingress.summary_dict()['flops_used'],
            cache_retired=True,truth_accessed=False,research_observer_not_deployable=True,
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=threadpool_info())
assert record['packet_bytes']<200_000_000
with (out/'result.json').open('x') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record),flush=True)
