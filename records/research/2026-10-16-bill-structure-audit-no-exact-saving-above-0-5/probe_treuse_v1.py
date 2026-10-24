"""[opus-hunt2] duplicate-transform probe: exec a candidate, wrap the forward transforms (_ipt_forward_a_ab/_b_ab, _hzt_forward_a, _hzd_forward_b)
and the Smirnov side builder, record (python id of the operand, shape, layer) per call, and count repeated forwards of the SAME operand object
within a network (= transform work that an A/B-reuse cache could skip). usage: probe_treuse_v1.py <candidate.py> <seed> <out.json>  env PB_ALL=1"""
import collections, json, sys, types
from pathlib import Path
import numpy as np
import flopscope as flops
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'analyst/fresh')); import loader
candidate, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3]).resolve(); assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
LIVE = [False]; CALLS = []
def wrap(name, argpos=0):
    f = mod.__dict__.get(name)
    if not callable(f): return False
    def w(*a, **k):
        if LIVE[0]:
            x = a[argpos]; lay = mod.__dict__.get('_PHYS_LAYER', [None])[0]
            CALLS.append((name, id(x), tuple(int(v) for v in getattr(x, 'shape', ())), int(lay) if isinstance(lay, (int, np.integer)) else None,
                          int(flops_ctx[0].summary_dict()['flops_used'] or 0)))
        return f(*a, **k)
    mod.__dict__[name] = w; return True
flops_ctx = [None]
wrapped = [n for n in ('_ipt_forward_a_ab', '_ipt_forward_b_ab', '_hzt_forward_a', '_hzd_forward_b', '_rs_load_a', '_rs_load_b') if wrap(n)]
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
w = loader.weights(seed)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=seed, name=f'probe-{seed}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    flops_ctx[0] = ctx; LIVE[0] = True; y = est.predict(mlp, 2**41); total = int(ctx.summary_dict()['flops_used'])
LIVE[0] = False
# repeats: same (name, id, shape) seen more than once in the network
cnt = collections.Counter((n, i, s) for n, i, s, lay, f in CALLS)
rep = {k: v for k, v in cnt.items() if v > 1}
by_name = collections.Counter(); rep_by_name = collections.Counter(); rep_elems = collections.Counter()
for (n, i, s), v in cnt.items():
    by_name[n] += v
    if v > 1:
        rep_by_name[n] += v - 1; rep_elems[n] += (v - 1) * int(np.prod(s)) if s else 0
json.dump(dict(candidate=str(candidate), seed=seed, total_flops=total, wrapped=wrapped, calls=len(CALLS), by_name=dict(by_name), repeats_by_name=dict(rep_by_name), repeat_elements=dict(rep_elems),
               calls_list=[list(c) for c in CALLS]), out.open('x'))
print(json.dumps(dict(seed=seed, total_flops=total, calls=len(CALLS), by_name=dict(by_name), repeats=dict(rep_by_name), repeat_elements=dict(rep_elems))))
