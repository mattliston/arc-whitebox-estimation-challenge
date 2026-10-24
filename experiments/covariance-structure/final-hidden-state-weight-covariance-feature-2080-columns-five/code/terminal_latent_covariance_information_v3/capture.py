"""One cold unchanged parent, passive final hooks, separately metered feature."""
import argparse,importlib.util,json,resource,sys,time,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import HERE,D,SOURCE,PARENT_SHA,STORE,TRAIN,sha,read,seal_check,write
from feature import latent_covariance
parser=argparse.ArgumentParser();parser.add_argument('rank',type=int);args=parser.parse_args();rank=args.rank
assert rank in TRAIN[1:];seal_check();assert sha(SOURCE)==PARENT_SHA
assert all(x['num_threads']==1 for x in threadpool_info())
q=read(HERE/'qualification/result.json');assert q['completed'] and q['feature_repeat_exact']
guard=read(HERE/'qualification_watch.json');assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
prior=read(D/'twohop_terminal_v1/train_capture/manifest.json');row=next(r for r in prior['rows'] if r['rank']==rank);base=row['record']
assert sha(row['completion'])==row['completion_sha256'] and sha(base['capture'])==base['capture_sha256']
g=read(row['watch']);assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
assert base['candidate_sha256']==PARENT_SHA and base['includes_initialization']
rows=read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests'];wp=(STORE/rows[rank]['path']).parent/'weights.npy';assert sha(wp)==base['weights_sha256']
out=HERE/'train_capture'/f'rank{rank:04d}';out.mkdir(exist_ok=False)
weights=np.load(wp,allow_pickle=False)
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(weights[l]) for l in range(16)]
mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('tlc_lazy_parent',SOURCE);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
held={};started=time.monotonic()
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
    spec.loader.exec_module(mod)
    original_cov,original_scalar,original_step=mod._cov_compact_diagonal,mod.nonlinear_scalar,mod.gnn_step
    def cov(w,C,indices,alive):
        held.update(C=C,alive=alive,W=w);return original_cov(w,C,indices,alive)
    def scalar(mu,v,k3,k4):
        held.update(mu=mu,v=v);return original_scalar(mu,v,k3,k4)
    def step(l,x,sigma,w,state,p):
        if l==15:held.update(H14=state,sigma=sigma)
        answer=original_step(l,x,sigma,w,state,p)
        if l==15:held['hidden']=answer[1]
        return answer
    mod._cov_compact_diagonal,mod.nonlinear_scalar,mod.gnn_step=cov,scalar,step
    prediction=mod.Estimator().predict(mlp,2**41)
parent_wall=time.monotonic()-started;Fparent=ctx.summary_dict()['flops_used']
assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
pred=np.asarray(prediction).copy();assert pred.shape==(16,1024) and pred.dtype==np.float32 and np.isfinite(pred).all() and (pred>=0).all()
with np.load(base['capture'],allow_pickle=False) as z:
    assert np.array_equal(pred,z['prediction'])
    assert np.array_equal(np.asarray(held['hidden']),z['hidden']) and np.array_equal(np.asarray(held['sigma']).astype(np.float32),z['sigma'])
assert Fparent==base['parent_flops'],(Fparent,base['parent_flops'])
with flops.BudgetContext(flop_budget=5_000_000_000,quiet=True) as feature_ctx:
    features=latent_covariance(fnp,*[held[k] for k in ['C','H14','alive','W','mu','v']])
Ffeature=feature_ctx.summary_dict()['flops_used'];assert Ffeature<5_000_000_000
feat=np.asarray(features).copy();assert feat.shape==(1024,2080) and feat.dtype==np.float32 and np.isfinite(feat).all()
path=out/'capture.npz';np.savez(path,features=feat,hidden=np.asarray(held['hidden']).copy(),sigma=np.asarray(held['sigma']).copy(),prediction=pred)
record=dict(call=0,rank=rank,capture=str(path),capture_sha256=sha(path),parent_flops=Fparent,parent_multiplier=max(.1,Fparent/2**41),feature_flops=Ffeature,weights_sha256=sha(wp),includes_initialization=True,parent_all16_and_bill_exact=True,parent_hidden_and_sigma_exact=True,numeric_cache_retired=True,parent_wall_s=parent_wall,feature_wall_s=feature_ctx.summary_dict()['wall_time_s'],input_flops=ingress.summary_dict()['flops_used'],truth_accessed=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write(out/'result.json',dict(completed=True,record=record,source_sha256=sha(__file__),feature_source_sha256=sha(HERE/'feature.py'),source_seal_sha256=sha(HERE/'sources_sealed.json'),truth_accessed=False))
print(json.dumps(record),flush=True)
