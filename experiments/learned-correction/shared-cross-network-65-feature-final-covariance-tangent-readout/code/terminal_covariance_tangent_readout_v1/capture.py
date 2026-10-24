"""One unchanged cold parent with old passive taps, then research-only J."""
import argparse,importlib.util,json,resource,sys,time,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import HERE,D,SOURCE,PARENT_SHA,STORE,TRAIN,sha,read,seal_check,write
from passive_source import observed_source
from feature import features
parser=argparse.ArgumentParser();parser.add_argument('rank',type=int);rank=parser.parse_args().rank
assert rank in TRAIN[1:];seal_check();assert sha(SOURCE)==PARENT_SHA
assert all(t['num_threads']==1 for t in threadpool_info())
qual=read(HERE/'qualification/result.json');guard=read(HERE/'qualification_watch.json')
assert qual['completed'] and qual['old_qualified_features_exact'] and qual['feature_repeat_exact']
assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
old=read(D/'twohop_terminal_v1/train_capture/manifest.json');row=next(r for r in old['rows'] if r['rank']==rank);base=row['record']
assert sha(row['completion'])==row['completion_sha256'] and sha(base['capture'])==base['capture_sha256']
g=read(row['watch']);assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
assert base['candidate_sha256']==PARENT_SHA and base['includes_initialization']
rows=read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests'];wp=(STORE/rows[rank]['path']).parent/'weights.npy';assert sha(wp)==base['weights_sha256']
out=HERE/'train_capture'/f'rank{rank:04d}';out.mkdir(exist_ok=False)
weights=np.load(wp,allow_pickle=False)
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ingress:ww=[fnp.asarray(weights[l]) for l in range(16)]
assert ingress.summary_dict()['flops_used']==0
mlp=types.SimpleNamespace(width=1024,depth=16,weights=ww,seed=0,name=f'rank{rank}')
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('tctr_lazy_parent',SOURCE);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
source,proof=observed_source();held={};layer=[None];started=time.monotonic()
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
    spec.loader.exec_module(mod);mod._TCI_TAP=held
    exec(compile(source,str(HERE/'passive_observed.py'),'exec'),mod.__dict__)
    original_cov,original_scalar,original_step,original_silu=mod._cov_compact_diagonal,mod.nonlinear_scalar,mod.gnn_step,mod.silu
    def cov(w,c,indices,alive):
        held.update(W=w,C=c,indices=indices,alive=alive)
        return original_cov(w,c,indices,alive)
    def scalar(mu,v,k3,k4):
        answer=original_scalar(mu,v,k3,k4);held.update(mu=mu,v=v,k3=k3,k4=k4,host=answer[0])
        return answer
    def step(l,x,sigma,w,state,p):
        layer[0]=l
        if l==15:held.update(H14=state,sigma=sigma,P=p)
        answer=original_step(l,x,sigma,w,state,p)
        if l==15:held.update(H15=answer[1],delta=answer[0])
        layer[0]=None
        return answer
    def silu(x):
        if layer[0]==15:held['pre1' if 'pre1' not in held else 'pre2']=x
        return original_silu(x)
    mod._cov_compact_diagonal,mod.nonlinear_scalar,mod.gnn_step,mod.silu=cov,scalar,step,silu
    prediction=mod.Estimator().predict(mlp,2**41)
parent_wall=time.monotonic()-started;Fparent=ctx.summary_dict()['flops_used']
assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
a={k:np.asarray(v).copy() for k,v in held.items() if k!='P'}
a.update({'P_'+k:np.asarray(v).copy() for k,v in held['P'].items()});a['prediction']=np.asarray(prediction).copy()
for k,v in a.items():assert np.isfinite(v).all(),k
with np.load(base['capture'],allow_pickle=False) as z:
    assert np.array_equal(a['prediction'],z['prediction'])
    assert np.array_equal(a['H15'],z['hidden']) and np.array_equal(a['sigma'].astype(np.float32),z['sigma'])
assert Fparent==base['parent_flops'],(Fparent,base['parent_flops'])
held.clear();del prediction,weights,ww,mlp
began=time.monotonic();J,B=features(a);feature_wall=time.monotonic()-began
path=out/'capture.npz';np.savez(path,features=J,hidden=a['H15'],sigma=a['sigma'],prediction=a['prediction'])
record=dict(call=0,rank=rank,capture=str(path),capture_sha256=sha(path),parent_flops=Fparent,parent_multiplier=max(.1,Fparent/2**41),weights_sha256=sha(wp),includes_initialization=True,parent_all16_and_bill_exact=True,parent_hidden_and_sigma_exact=True,numeric_cache_retired=True,parent_wall_s=parent_wall,feature_wall_s=feature_wall,extra_parent_operations=0,feature_is_unmetered_research_only=True,input_flops=0,truth_accessed=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write(out/'result.json',dict(completed=True,record=record,source_proof=proof,source_sha256=sha(__file__),feature_source_sha256=sha(HERE/'feature.py'),tangent_source_sha256=sha(HERE/'tangent.py'),source_seal_sha256=sha(HERE/'sources_sealed.json'),truth_accessed=False))
print(json.dumps(record),flush=True)
