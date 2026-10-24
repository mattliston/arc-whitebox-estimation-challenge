"""[opus-roster] kernel bill curve: exec a candidate (probe_products style), run setup, then inside a BudgetContext call the production
product entry points on random f32 operands for real shape classes with one dimension swept, recording charged flops (and requests)
per call. usage: curve_v1.py <candidate.py> <out.json>   env: BLAS threads 1"""
import json, sys, types, traceback
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _budget
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
candidate, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(); assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
ops = [0]; original = _budget.BudgetContext._charge_op
def counted(self, *a, **kw):
    ops[0] += 1; return original(self, *a, **kw)
_budget.BudgetContext._charge_op = counted
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
rng = np.random.default_rng(0)
# (entry, base (m,k,n), swept dim index, sweep values)
CASES = [
    ('_mm', (1024, 992, 6144), 1, list(range(960, 1025, 8))),            # depth-7 colpack push, k = previous roster (kpad)
    ('_mm', (960, 928, 5696), 0, list(range(896, 1025, 8))),             # packet push, m = roster
    ('_mm', (960, 928, 5696), 1, list(range(896, 961, 8))),              # packet push, k = roster
    ('_mm', (960, 960, 1024), 0, list(range(896, 1025, 8))),             # single product, m = roster
    ('_mm', (896, 864, 736), 2, list(range(672, 769, 8))),               # single product, n = roster
    ('_shared_old384_mm', (928, 6144, 384), 0, list(range(864, 1025, 8))),   # old-bank read, m = roster
    ('_shared_old384_mm', (384, 928, 6144), 1, list(range(864, 961, 8))),    # old-bank read, k = roster
    ('_shared_old384_mm', (768, 384, 5920), 0, list(range(704, 833, 8))),    # old-tier product, m = roster
]
rows = []
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    for entry, base, idx, sweep in CASES:
        f = mod.__dict__[entry]
        for v in sweep:
            dims = list(base); dims[idx] = v; m, k, n = dims
            a = fnp.asarray(rng.standard_normal((m, k)).astype(np.float32)); b = fnp.asarray(rng.standard_normal((k, n)).astype(np.float32))
            f0, o0 = int(ctx.summary_dict()['flops_used']), ops[0]
            try:
                r = f(a, b); err = None
                del r
            except Exception as e:
                err = repr(e)[:160]
            fl = int(ctx.summary_dict()['flops_used']) - f0
            rows.append(dict(entry=entry, m=m, k=k, n=n, swept='mkn'[idx], flops=fl, ratio=fl / (2 * m * k * n), ops=ops[0] - o0, err=err))
            print('%-18s m %4d k %4d n %5d swept %s  flops %12d  ratio %.3f  ops %4d  %s' % (entry, m, k, n, 'mkn'[idx], fl, fl / (2 * m * k * n), ops[0] - o0, err or ''), flush=True)
            del a, b
            mod._BH_HOLD.clear() if '_BH_HOLD' in mod.__dict__ else None
json.dump(rows, out.open('x'), indent=0)
