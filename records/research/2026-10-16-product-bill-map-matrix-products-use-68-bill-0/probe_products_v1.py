"""[opus-solo] per-product bill probe: exec a candidate, wrap its product entry points, run setup + one predict (PB_ALL) on one seed and
record every product call: entry name, true operand shapes, depth L when given, billed flops delta (ctx.summary_dict), requests delta
(charged ops). Writes JSON rows. usage: probe_products_v1.py <candidate.py> <seed> <out.json>   env: PB_ALL=1, BLAS threads=1"""
import json, os, sys, time, types
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _budget
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'analyst/fresh')); import loader
candidate, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3]).resolve(); assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
CTX = [None]; ROWS = []; DEPTH = [0]
ops = [0]; original = _budget.BudgetContext._charge_op
def counted(self, *a, **kw):
    ops[0] += 1; return original(self, *a, **kw)
_budget.BudgetContext._charge_op = counted
def shape(x):
    try: return list(int(v) for v in x.shape)
    except Exception: return None
def wrap(name):
    f = mod.__dict__.get(name)
    if not callable(f): return False
    def w(*args, **kwargs):
        c = CTX[0]
        if c is None or DEPTH[0] > 0: return f(*args, **kwargs)
        DEPTH[0] += 1
        f0, o0 = int(c.summary_dict()['flops_used'] or 0), ops[0]; t0 = time.monotonic()
        try: return f(*args, **kwargs)
        finally:
            DEPTH[0] -= 1
            arrs = [shape(a) for a in args if shape(a) is not None]
            ints = [a for a in args if isinstance(a, int)]
            lay = mod.__dict__.get('_PHYS_LAYER', [None])[0]
            ROWS.append(dict(fn=name, shapes=arrs[:3], ints=ints[:4], layer=(int(lay) if isinstance(lay, (int, np.integer)) else None),
                             flops=int(c.summary_dict()['flops_used'] or 0) - f0, ops=ops[0] - o0, s=round(time.monotonic() - t0, 4)))
    mod.__dict__[name] = w; return True
NAMES = ['_mm', '_shared_old384_mm', '_square_product', '_packed_product_4', '_packed_product_5', '_packed_product_cached', '_push6_padded', '_push5_product',
         '_packed_read5', '_ipt_product_ab', '_hzg_product', '_hz7_product', '_d7_product_kpad', '_cp_d6_shared', '_gsafe_two_strip', '_gsafe5_full', '_sym_upper', '_psdm_sym_upper', '_join_rows_mm']
wrapped = [n for n in NAMES if wrap(n)]
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:   # the candidate's identity check of its dense-path globals: re-record the wrapped objects
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
w = loader.weights(seed)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=seed, name=f'probe-{seed}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    CTX[0] = ctx
    y = est.predict(mlp, 2**41)
    total = int(ctx.summary_dict()['flops_used'])
CTX[0] = None
json.dump(dict(candidate=str(candidate), seed=seed, wrapped=wrapped, total_flops=total, total_ops=ops[0], rows=ROWS), out.open('x'), indent=0)
print(json.dumps(dict(seed=seed, wrapped=wrapped, total_flops=total, rows=len(ROWS), product_flops=sum(r['flops'] for r in ROWS), product_ops=sum(r['ops'] for r in ROWS))))
