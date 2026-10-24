"""Small algebra certificate for the separate, unshipped pair-order prototype."""
from pathlib import Path
import json
import sys
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

HERE=Path(__file__).resolve().parent
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
mod=types.ModuleType('pair_order_test');mod.__file__=str(HERE/'candidate_pair_order_research_v2.py')
sys.modules[mod.__name__]=mod
exec(compile(Path(mod.__file__).read_text(),mod.__file__,'exec'),mod.__dict__)
rng=np.random.default_rng(20260927)
arr=lambda shape:rng.normal(size=shape).astype(np.float32)
rows=[]
for lazy in (False,True):
    with flops.BudgetContext(flop_budget=2**30,quiet=True):
        state=dict(columns=26,cohorts=[(0,6),(6,8),(14,12)],rank=6,lazy=None,F=None)
        if lazy:
            base,join,rotation=arr((4,14)),arr((6,12)),arr((6,4))
            dense=np.concatenate((rotation.astype(np.float64)@base.astype(np.float64),join.astype(np.float64)),axis=1)
            state['lazy']=dict(layer=13,prefix=14,joined=12,base_rank=4,rank=6,
                base=fnp.asarray(base),join=fnp.asarray(join),map=fnp.asarray(rotation))
        else:
            dense=arr((6,26)).astype(np.float64)
            state['F']=fnp.asarray(dense.astype(np.float32))
        L2=arr((9,26))**2
        # Arbitrary coefficients test the stronger algebraic permutation claim.
        lam=arr((26,))
        expected=(L2.astype(np.float64)*lam.astype(np.float64))@dense.T
        got=np.asarray(mod._bpo_inner_sq(fnp.asarray(L2),fnp.asarray(lam),state),dtype=np.float64)
        again=np.asarray(mod._bpo_inner_sq(fnp.asarray(L2),fnp.asarray(lam),state),dtype=np.float64)
        np.testing.assert_allclose(got,expected,rtol=3e-5,atol=1e-4)
        np.testing.assert_array_equal(got,again)
        rows.append(dict(lazy=lazy,max_abs_error=float(np.max(np.abs(got-expected))),cache_repeat_identical=True))
record=dict(passed=True,cases=rows,stats=mod._BPO_STATS,status='algebra only; no native accuracy or cost claim')
with (HERE/'pair_order_algebra_v2.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
