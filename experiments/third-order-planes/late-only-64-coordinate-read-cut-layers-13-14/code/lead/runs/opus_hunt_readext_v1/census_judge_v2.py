"""[opus-hunt] Warm FLOP census: one process, one Estimator, predict seed A (cold) then seed B (warm).
Every charged FLOP is attributed to the candidate-module call chain and current layer globals."""
import collections, hashlib, json, os, sys, time, types
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _budget
O = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(O/'analyst/fresh'))
import loader
candidate, seeds, out = Path(sys.argv[1]).resolve(), [int(s) for s in sys.argv[2].split(',')], Path(sys.argv[3]).resolve()
assert not out.exists()
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
assert all(os.environ.get(k)=='1' for k in THREADS) and os.environ['PB_ALL']=='1'
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('candidate_cert');mod.__file__=str(candidate);sys.modules[mod.__name__]=mod
exec(compile(candidate.read_text(),str(candidate),'exec',dont_inherit=True),mod.__dict__)
mod._PB_WALL_S=float(os.environ.get('PB_WALL','1e9'))
est=mod.Estimator();t0=time.monotonic()
est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
setup_s=time.monotonic()-t0
CF=str(candidate);original=_budget.BudgetContext._charge_op
AGG=None
def g(name):
    v=mod.__dict__.get(name)
    try: return int(v[0]) if v is not None and v[0] is not None else None
    except Exception: return None
def counted(self,op_name,flop_cost,*a,**kw):
    before=self._flops_used
    r=original(self,op_name,flop_cost,*a,**kw)
    d=self._flops_used-before
    if AGG is not None and d:
        f=sys._getframe(1);chain=[]
        while f is not None:
            if f.f_code.co_filename==CF and (not chain or chain[-1]!=f.f_code.co_name): chain.append(f.f_code.co_name)
            f=f.f_back
        key=(tuple(chain),op_name,g('_RS_LAYER'),g('_UVP_CUR'))
        AGG[key][0]+=d;AGG[key][1]+=1
    return r
_budget.BudgetContext._charge_op=counted
res=[]
for seed in seeds:
    w=loader.weights(seed);truth=loader.truth(seed,check_weights=False)['mc_post']
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=f'fresh-{seed}')
    AGG=collections.defaultdict(lambda:[0,0]);t0=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        y=np.asarray(est.predict(mlp,2**41),dtype=np.float64)
    mse=((y-truth)**2).mean(axis=1)
    res.append(dict(seed=seed,flops=int(ctx.summary_dict()['flops_used']),wall_s=time.monotonic()-t0,mse_final=float(mse[-1]),
                    agg=[[list(k[0]),k[1],k[2],k[3],v[0],v[1]] for k,v in sorted(AGG.items(),key=lambda kv:-kv[1][0])]))
    print(json.dumps(dict(seed=seed,flops=res[-1]['flops'],attributed=sum(v[0] for v in AGG.values()))),flush=True)
_budget.BudgetContext._charge_op=original
with out.open('x') as f: json.dump(dict(candidate=str(candidate),sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),setup_s=setup_s,runs=res),f)
print('done',flush=True)
