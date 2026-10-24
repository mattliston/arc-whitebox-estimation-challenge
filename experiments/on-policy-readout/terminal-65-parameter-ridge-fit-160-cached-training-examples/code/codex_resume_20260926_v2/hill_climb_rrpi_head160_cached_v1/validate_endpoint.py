"""Single-process ordered VALID2 -> VALID8, stopped at the frozen two-case gate."""
from pathlib import Path
import hashlib, json, os, sys, time, types
import numpy as np
import flopscope as flops
from validation_common import pin, checked, once, metrics, gates, configuration
D=Path(__file__).resolve().parent;H=D.parent;P2=H.parent
assert len(sys.argv)==3
cfg=Path(sys.argv[1]).resolve();c,index,base=configuration(cfg,sys.argv[2])
assert c['driver']==pin(__file__)
assert sys.executable==c['runtime'] and np.__version__=='2.4.6'
assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'))
candidate=Path(c['candidate']['path'])
out=Path(c['output']);assert out.parent==D and not out.exists();out.mkdir()
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
mod=types.ModuleType('candidate_eval');mod.__file__=str(candidate);sys.modules[mod.__name__]=mod
exec(compile(checked(c['candidate']),str(candidate),'exec'),mod.__dict__)
class MLP:
 def __init__(self,weights,rank):
  self.width,self.depth,self.weights,self.seed,self.name=1024,16,weights,0,f'phase2-{rank}'
est=mod.Estimator();t0=time.monotonic()
est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
setup=dict(setup_s=time.monotonic()-t0,pb_log=[list(x) for x in getattr(mod,'_PB_LOG',[])])
assert setup['setup_s']<=4.1
counts={x[0]:x[1] for x in setup['pb_log']}
for k,v in c['setup_counts'].items():assert counts.get(k)==v,(k,counts)
once(out/'manifest.json',dict(candidate=c['candidate'],config=pin(cfg),driver=pin(__file__),requested_ranks=c['ranks'],runtime=sys.executable,numpy=np.__version__,setup=setup,thread_env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},single_estimator=True,cache_order_matches_baseline=True))
print(json.dumps(dict(event='setup',**setup)),flush=True)
seen=[]
with (out/'rows.jsonl').open('x') as log:
 for i,rank in enumerate(c['ranks']):
  data=c['data'][i];assert data['rank']==rank==index['data'][i]['rank']
  weights=np.load(__import__('io').BytesIO(checked(data['weights'])),allow_pickle=False)
  assert weights.shape==(16,1024,1024) and np.isfinite(weights).all()
  t0=time.monotonic();cpu0=time.process_time()
  with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
   y=np.asarray(est.predict(MLP([weights[l] for l in range(16)],rank),2**41),dtype=np.float64)
  wall=time.monotonic()-t0;cpu=time.process_time()-cpu0
  assert y.shape==(16,1024) and np.isfinite(y).all()
  used=int(ctx.summary_dict()['flops_used']);assert 0<used<=2**41
  with np.load(__import__('io').BytesIO(checked(data['truth'])),allow_pickle=False) as truthfile:
   truth=truthfile['mc_post'][15].astype(np.float64)
  parent=np.load(__import__('io').BytesIO(checked(index['data'][i]['prediction'])),allow_pickle=False)
  assert parent.shape==y.shape and parent.dtype==y.dtype and parent[0].tobytes()==y[0].tobytes()
  assert float(np.mean((parent[15]-truth)**2))==base[i]['mse']
  mse=float(np.mean((y[15]-truth)**2));util=used/2**41
  ubc=dict(getattr(mod,'_UBC_STATS',{}))
  assert ubc.get('retired') is True and ubc.get('retained_cache_bytes')==ubc.get('live_cache_bytes')==0
  row=dict(rank=rank,mse=mse,util=util,adjusted=mse*max(.1,util),flops=used,wall_s=wall,cpu_s=cpu,ubc=ubc,ynt=dict(getattr(mod,'_YNT_STATS',{})))
  np.save(out/f'pred_{rank}.npy',y)
  seen.append(row);log.write(json.dumps(row)+'\n');log.flush()
  print(json.dumps(row),flush=True)
  del weights,y,truth,parent
  if len(seen)==2:
   m=metrics(base[:2],seen);g=gates(m,c['gates2'])
   prefix=dict(**m,gates=g,accuracy_gates_passed=all(g.values()),resource_qualification_pending=True,admitted=False)
   once(out/'prefix2.json',prefix)
   if not all(g.values()):break
once(out/'completed.json',dict(completed=True,n=len(seen),ranks=[r['rank'] for r in seen],config=pin(cfg),candidate=c['candidate'],early_stopped=len(seen)==2,resource_qualification_pending=True,admitted=False))
print(json.dumps(dict(event='validation_execution_complete',n=len(seen))),flush=True)

