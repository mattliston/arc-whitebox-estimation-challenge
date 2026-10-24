"""[opus-wall] in-process paired pricing judge: setup once, predict seeds in order (first cold, then warm); per-read probe rows,
total FLOPs, charged ops, thread CPU and wall per network."""
import json, os, sys, time, types
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _budget
O = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(O/'analyst/fresh'))
import loader
candidate, seeds, out = Path(sys.argv[1]).resolve(), [int(s) for s in sys.argv[2].split(',')], Path(sys.argv[3]).resolve()
assert not out.exists()
assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('candidate_cert');mod.__file__=str(candidate);sys.modules[mod.__name__]=mod
exec(compile(candidate.read_text(),str(candidate),'exec',dont_inherit=True),mod.__dict__)
mod._PB_WALL_S=float(os.environ.get('PB_WALL','1e9'))
est=mod.Estimator();t0=time.perf_counter();c0=time.thread_time()
est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
setup=dict(wall_s=time.perf_counter()-t0,cpu_s=time.thread_time()-c0,pb_log=json.loads(json.dumps(mod.__dict__.get('_PB_LOG'),default=str)))
original=_budget.BudgetContext._charge_op
def counted(self,op_name,flop_cost,*a,**kw):
    before=self._flops_used
    r=original(self,op_name,flop_cost,*a,**kw)
    mod._WALLP_CNT['flops']+=self._flops_used-before;mod._WALLP_CNT['ops']+=1
    return r
_budget.BudgetContext._charge_op=counted
runs=[]
for seed in seeds:
    w=loader.weights(seed);truth=loader.truth(seed,check_weights=False)['mc_post']
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=f'fresh-{seed}')
    del mod._WALLP_ROWS[:];mod._WALLP_CNT.update(flops=0,ops=0);t0=time.perf_counter();c0=time.thread_time()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        y=np.asarray(est.predict(mlp,2**41),dtype=np.float64)
    runs.append(dict(seed=seed,wall_s=time.perf_counter()-t0,cpu_s=time.thread_time()-c0,flops=int(ctx.summary_dict()['flops_used']),ops=mod._WALLP_CNT['ops'],
                     mse_final=float(((y-truth)**2).mean(axis=1)[-1]),reads=[list(r) for r in mod._WALLP_ROWS]))
    print(json.dumps(dict(seed=seed,flops=runs[-1]['flops'],cpu=runs[-1]['cpu_s'],reads=len(runs[-1]['reads']))),flush=True)
_budget.BudgetContext._charge_op=original
with out.open('x') as f: json.dump(dict(candidate=str(candidate),setup=setup,runs=runs),f)
print('done',flush=True)
