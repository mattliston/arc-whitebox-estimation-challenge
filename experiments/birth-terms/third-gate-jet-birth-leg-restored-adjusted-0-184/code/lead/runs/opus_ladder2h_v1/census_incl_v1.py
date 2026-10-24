"""[opus-reqdiet] in-process WARM census: setup once, predict the seed twice, and for the 2nd (warm) predict attribute every flopscope
charge (BudgetContext._charge_op = one request) INCLUSIVELY to each distinct candidate-file function on the Python stack.
usage: census_incl_v1.py <candidate.py> <seed> <out.json>"""
import collections, json, sys, types
from pathlib import Path
import flopscope as flops
from flopscope import _budget
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'analyst/fresh')); import loader
cand, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3]); assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('cand_census'); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
est = mod.Estimator(); est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
CF = str(cand); ON = [False]; incl = collections.defaultdict(lambda: [0, 0]); total = [0, 0]
orig = _budget.BudgetContext._charge_op
def counted(self, *a, **k):
    u0 = self._flops_used; r = orig(self, *a, **k)
    if ON[0]:
        b = self._flops_used - u0; total[0] += 1; total[1] += b; seen = set(); f = sys._getframe(1)
        while f is not None:
            c = f.f_code
            if c.co_filename == CF and c.co_name not in seen:
                seen.add(c.co_name); e = incl[c.co_name]; e[0] += 1; e[1] += b
            f = f.f_back
    return r
_budget.BudgetContext._charge_op = counted
w = loader.weights(seed); mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')
runs = []
for k in range(2):
    ON[0] = (k == 1)
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx: est.predict(mlp, 2**41)
    runs.append(int(ctx.summary_dict()['flops_used']))
rows = sorted(((n, v[0], v[1]) for n, v in incl.items()), key=lambda t: -t[1])
with out.open('x') as fh: json.dump(dict(candidate=str(cand), seed=seed, flops=runs, warm_requests=total[0], warm_bill=total[1], incl=rows), fh)
print('warm requests', total[0], 'warm bill', total[1], 'flops', runs)
for n, c, b in rows[:60]: print(f'{n:45s} req {c:6d} ({100*c/total[0]:5.1f}%)  bill {100*b/total[1]:6.2f}%')
