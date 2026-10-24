"""Native 2-predict cache diagnostic (one process: setup, cold net, warm net). Counts _cache_put entry creations per cache and the
charged-op sites per predict. Pure bookkeeping (no extra flopscope ops). usage: diag2_cache_judge_v1.py <candidate> <seedA> <seedB> <out.json>"""
import hashlib, json, os, sys, time, types, collections
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _budget
O = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(O / 'analyst/fresh')); import loader
cand, sa, sb, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]).resolve()
assert not out.exists() and all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'))
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_cert'); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), 'exec', dont_inherit=True), mod.__dict__)
PUTS = collections.Counter(); orig_put = mod._cache_put
def counted_put(cache, key, value):
    PUTS[id(cache)] += 1
    return orig_put(cache, key, value)
mod._cache_put = counted_put
SITE = collections.Counter(); CF = str(cand); orig = _budget.BudgetContext._charge_op
def charged(self, *a, **kw):
    f = sys._getframe(1)
    while f is not None and f.f_code.co_filename != CF: f = f.f_back
    SITE[(a[0] if a else kw.get('op_name'), f.f_code.co_name if f is not None else '?')] += 1
    return orig(self, *a, **kw)
mod._PB_WALL_S = float(os.environ.get('PBW', '1e9'))
est = mod.Estimator(); t0 = time.monotonic()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
names = {id(v): k for k, v in mod.__dict__.items() if isinstance(v, dict)}
caps = {k: getattr(mod, '_CAP_BY_ID', {}).get(id(v)) for k, v in mod.__dict__.items() if isinstance(v, dict) and id(v) in getattr(mod, '_CAP_BY_ID', {})}
res = dict(candidate=str(cand), setup_s=time.monotonic() - t0, setup_puts={names.get(i, str(i)): n for i, n in PUTS.items()}, predicts=[], caps=caps)
_budget.BudgetContext._charge_op = charged
for s in (sa, sb):
    PUTS.clear(); SITE.clear(); w = loader.weights(s)
    mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{s}')
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    sizes = {k: len(v) for k, v in mod.__dict__.items() if isinstance(v, dict) and id(v) in PUTS}
    res['predicts'].append(dict(seed=s, flops=int(ctx.summary_dict()['flops_used']), ops=sum(SITE.values()),
        puts={names.get(i, str(i)): n for i, n in PUTS.most_common()}, sizes=sizes,
        sites=[[k[0], k[1], n] for k, n in SITE.items()], pred_sha=hashlib.sha256(y.astype(np.float32).tobytes()).hexdigest()[:16]))
_budget.BudgetContext._charge_op = orig
out.write_text(json.dumps(res)); print(json.dumps({'ops': [p['ops'] for p in res['predicts']], 'puts_warm': res['predicts'][1]['puts']})[:1500])
