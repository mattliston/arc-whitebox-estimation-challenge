"""[opus-wall] in-process: setup once, predict seeds (first cold, then warm); dump product census rows."""
import json, os, sys, types
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _budget
O = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(O/'analyst/fresh'))
import loader
candidate, seeds, out = Path(sys.argv[1]).resolve(), [int(s) for s in sys.argv[2].split(',')], Path(sys.argv[3]).resolve()
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('candidate_cert');mod.__file__=str(candidate);sys.modules[mod.__name__]=mod
exec(compile(candidate.read_text(),str(candidate),'exec',dont_inherit=True),mod.__dict__)
mod._PB_WALL_S=1e9
est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
original=_budget.BudgetContext._charge_op
def counted(self,op_name,flop_cost,*a,**kw):
    b=self._flops_used;r=original(self,op_name,flop_cost,*a,**kw);mod._MMP_CNT['flops']+=self._flops_used-b;mod._MMP_CNT['ops']+=1;return r
_budget.BudgetContext._charge_op=counted
res=[]
for seed in seeds:
    w=loader.weights(seed);mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=f'fresh-{seed}')
    del mod._MMP_ROWS[:];mod._MMP_CNT.update(flops=0,ops=0)
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        est.predict(mlp,2**41)
    res.append(dict(seed=seed,flops=int(ctx.summary_dict()['flops_used']),ops=mod._MMP_CNT['ops'],rows=[list(r) for r in mod._MMP_ROWS]))
    print(json.dumps(dict(seed=seed,flops=res[-1]['flops'],ops=res[-1]['ops'],products=len(res[-1]['rows']))),flush=True)
with out.open('x') as f: json.dump(res,f)
