"""[opus-roster] roster census: exec a candidate (probe_products style), wrap fnp.nonzero to log every roster count (caller site, layer, nonzero
count) and wrap the product entry points to log every product (fn, shapes, layer, flops), run setup + one predict (PB_ALL) on one seed.
usage: census_nz_v1.py <candidate.py> <seed> <out.json>   env: PB_ALL=1, BLAS threads 1"""
import json, sys, types
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'analyst/fresh')); import loader
candidate, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3]).resolve(); assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
CTX = [None]; NZ = []; PRODUCTS = []; DEPTH = [0]
def site():
    fr = sys._getframe(2)
    while fr is not None and fr.f_code.co_filename != str(candidate): fr = fr.f_back
    return (fr.f_code.co_name + ':' + str(fr.f_lineno)) if fr is not None else '?'
def layer():
    lay = mod.__dict__.get('_PHYS_LAYER', [None])[0]
    return int(lay) if isinstance(lay, (int, np.integer)) else None
_nz0 = fnp.nonzero
def nz(x, *a, **k):
    r = _nz0(x, *a, **k)
    if CTX[0] is not None:
        NZ.append(dict(site=site(), layer=layer(), n=int(x.shape[0]) if len(x.shape) == 1 else list(map(int, x.shape)), nz=int(r[0].shape[0])))
    return r
fnp.nonzero = nz
_sel0 = mod.select
def _sel(score, floor):
    r = _sel0(score, floor)
    if CTX[0] is not None:
        fr = sys._getframe(1)
        NZ.append(dict(site='select-caller:%s:%d' % (fr.f_code.co_name, fr.f_lineno), layer=layer(), n=int(score.shape[0]), nz=int(r[0].shape[0])))
    return r
mod.select = _sel
def shape(x):
    try: return list(int(v) for v in x.shape)
    except Exception: return None
def wrap(name):
    f = mod.__dict__.get(name)
    if not callable(f): return False
    def w(*args, **kwargs):
        c = CTX[0]
        if c is None or DEPTH[0] > 0: return f(*args, **kwargs)
        DEPTH[0] += 1; f0 = int(c.summary_dict()['flops_used'] or 0)
        try: return f(*args, **kwargs)
        finally:
            DEPTH[0] -= 1
            PRODUCTS.append(dict(fn=name, shapes=[shape(a) for a in args if shape(a) is not None][:3], layer=layer(), flops=int(c.summary_dict()['flops_used'] or 0) - f0))
    mod.__dict__[name] = w; return True
NAMES = ['_mm', '_shared_old384_mm', '_square_product', '_packed_product_4', '_packed_product_5', '_packed_product_cached', '_push6_padded', '_push5_product',
         '_packed_read5', '_ipt_product_ab', '_hzg_product', '_hz7_product', '_d7_product_kpad', '_cp_d6_shared', '_gsafe_two_strip', '_gsafe5_full', '_sym_upper', '_psdm_sym_upper', '_join_rows_mm']
wrapped = [n for n in NAMES if wrap(n)]
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
w = loader.weights(seed)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=seed, name=f'probe-{seed}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    CTX[0] = ctx; y = est.predict(mlp, 2**41); total = int(ctx.summary_dict()['flops_used'])
CTX[0] = None
json.dump(dict(candidate=str(candidate), seed=seed, wrapped=wrapped, total_flops=total, nonzero=NZ, products=PRODUCTS), out.open('x'), indent=0)
print(json.dumps(dict(seed=seed, total_flops=total, nonzero_calls=len(NZ), products=len(PRODUCTS), rem=mod.__dict__.get("_REM_STATS"), sm40_calls=mod.__dict__.get("_ASTRA_SM40_STATS", {}).get("calls"), hz=mod.__dict__.get("_HZ_STATS"), hzg=mod.__dict__.get("_HZG_STATS"))))
for r in NZ: print(r)
