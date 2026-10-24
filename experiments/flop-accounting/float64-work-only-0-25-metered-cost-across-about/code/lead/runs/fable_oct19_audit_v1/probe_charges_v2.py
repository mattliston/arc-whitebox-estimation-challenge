"""[opus-solo] FLOP charge census: exec a candidate, wrap flopscope's BudgetContext._charge_op, run setup + one predict (PB_ALL) on one seed and
record EVERY charge: op_name, flop_cost, shapes, dtypes, caller site (first frame inside the candidate), layer. Also logs select() rosters
(nz, kept count). Aggregates by op_name and by (op_name, site). usage: probe_charges_v1.py <candidate.py> <seed> <out.json>  env: PB_ALL=1, BLAS 1"""
import collections, json, sys, types
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
mod._FAST_RESCUE_CAP_S = 1e9   # [fable-lead 2026-10-19] never let the fast rescue fire inside the census (the probe slows Python; the rescue swaps L12-14 old-tier products for dense matmuls)
LIVE = [False]; ROWS = []; SEL = []
original = _budget.BudgetContext._charge_op
def site():
    fr = sys._getframe(2)
    while fr is not None and fr.f_code.co_filename != str(candidate): fr = fr.f_back
    return (fr.f_code.co_name + ':' + str(fr.f_lineno)) if fr is not None else '?'
def charged(self, op_name, flop_cost, subscripts, shapes, **kw):
    if LIVE[0]:
        lay = mod.__dict__.get('_PHYS_LAYER', [None])[0]
        ROWS.append((op_name, int(flop_cost), subscripts, [list(map(int, s)) for s in shapes] if shapes else None,
                     [str(d) for d in kw.get('dtypes') or ()], site(), int(lay) if isinstance(lay, (int, np.integer)) else None))
    return original(self, op_name, flop_cost, subscripts, shapes, **kw)
_budget.BudgetContext._charge_op = charged
_sel0 = mod.select
def sel(score, floor):
    r = _sel0(score, floor)
    if LIVE[0]:
        lay = mod.__dict__.get('_PHYS_LAYER', [None])[0]
        SEL.append((int(lay) if isinstance(lay, (int, np.integer)) else None, int(score.shape[0]), int(r[0].shape[0]), site()))
    return r
mod.select = sel
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
w = loader.weights(seed)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=seed, name=f'probe-{seed}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    LIVE[0] = True; y = est.predict(mlp, 2**41); total = int(ctx.summary_dict()['flops_used'])
LIVE[0] = False
by_op = collections.defaultdict(lambda: [0, 0]); by_site = collections.defaultdict(lambda: [0, 0, set()])
for op, f, sub, sh, dt, st, lay in ROWS:
    by_op[op][0] += 1; by_op[op][1] += f
    k = (op, st); by_site[k][0] += 1; by_site[k][1] += f; by_site[k][2].add((tuple(tuple(s) for s in sh) if sh else (), tuple(dt)))
ops = sorted(((k, v[0], v[1]) for k, v in by_op.items()), key=lambda t: -t[2])
sites = sorted(((k[0], k[1], v[0], v[1], [[list(map(list, s)), list(d)] for s, d in list(v[2])[:3]]) for k, v in by_site.items()), key=lambda t: -t[3])
json.dump(dict(candidate=str(candidate), seed=seed, total_flops=total, n_charges=len(ROWS), by_op=ops, by_site=sites, select=SEL,
               rows=[list(r) for r in ROWS]), out.open('x'), indent=0)
print(json.dumps(dict(seed=seed, total_flops=total, charges=len(ROWS), selects=len(SEL))))
print('--- by op (flops share, requests)')
for k, n, f in ops[:40]: print('%-28s %7.3f GF (%5.2f%%)  calls %6d' % (k, f / 1e9, 100 * f / total, n))
print('--- top sites')
for op, st, n, f, sh in sites[:45]: print('%-14s %-40s calls %5d  %7.3f GF (%5.2f%%)  %s' % (op, st, n, f / 1e9, 100 * f / total, sh[:2]))
print('--- select rosters (layer, len, kept)'); print(SEL)
