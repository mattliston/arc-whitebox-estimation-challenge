"""[opus-regroup] exact billed-FLOP profile by candidate source line: setup once, predict seed cold then warm; every charged op is attributed
to the innermost stack frame inside the candidate file (line, function), with op name, shapes, dtypes. Writes JSON for the WARM predict."""
import json, sys, types, collections
from pathlib import Path
import flopscope as flops
from flopscope import _budget
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'analyst/fresh')); import loader
cand, seed, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]); assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('cand_prof'); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
est = mod.Estimator(); est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
CF = str(cand); REC = [None]; orig = _budget.BudgetContext._charge_op
def charged(self, op_name, flop_cost, subscripts, shapes, **kw):
    before = self._flops_used; r = orig(self, op_name, flop_cost, subscripts, shapes, **kw)
    if REC[0] is not None:
        f = sys._getframe(1); chain = []
        while f is not None:
            if f.f_code.co_filename == CF:
                chain.append((f.f_lineno, f.f_code.co_name))
                if len(chain) == 3: break
            f = f.f_back
        key = (chain[0] if chain else (0, '?'), op_name)
        e = REC[0][key]; e[0] += self._flops_used - before; e[1] += 1
        if len(e[2]) < 2: e[2].append([str(s) for s in shapes] + [str(kw.get('dtypes'))])
        if len(e[3]) < 2 and chain[1:]: e[3].append(chain[1:])
    return r
_budget.BudgetContext._charge_op = charged
w = loader.weights(seed); mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')
res = {}
for phase in ('cold', 'warm'):
    REC[0] = collections.defaultdict(lambda: [0, 0, [], []])
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        est.predict(mlp, 2**41); total = int(ctx.summary_dict()['flops_used'])
    rows = sorted(([k[0][0], k[0][1], k[1], v[0], v[1], v[2], v[3]] for k, v in REC[0].items()), key=lambda r: -r[3])
    res[phase] = dict(total=total, attributed=sum(r[3] for r in rows), rows=rows)
    print(phase, total, 'attributed', res[phase]['attributed'], 'sites', len(rows), flush=True)
out.write_text(json.dumps(dict(candidate=str(cand), seed=seed, **res)))
