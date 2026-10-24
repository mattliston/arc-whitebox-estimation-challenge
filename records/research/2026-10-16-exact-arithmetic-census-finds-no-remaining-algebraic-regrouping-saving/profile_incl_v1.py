"""[opus-regroup] INCLUSIVE billed-FLOP profile (warm predict): each charged op's FLOPs credited once to every distinct (line, function) frame
of the candidate file on its stack; plus entry-level product shapes for outer _mm / _shared_old384_mm / _tsr_mm calls by caller line."""
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
CF = str(cand); ON = [False]; INC = collections.Counter(); SELF = collections.Counter(); CTX = [None]
orig = _budget.BudgetContext._charge_op
def charged(self, op_name, flop_cost, subscripts, shapes, **kw):
    before = self._flops_used; r = orig(self, op_name, flop_cost, subscripts, shapes, **kw)
    if ON[0]:
        d = self._flops_used - before; f = sys._getframe(1); seen = set(); first = True
        while f is not None:
            if f.f_code.co_filename == CF:
                k = (f.f_lineno, f.f_code.co_name)
                if first: SELF[k] += d; first = False
                if k not in seen: seen.add(k); INC[k] += d
            f = f.f_back
    return r
_budget.BudgetContext._charge_op = charged
ENTRY = collections.defaultdict(lambda: [0, 0, set()]); DEPTH = [0]
def wrap(name):
    inner = getattr(mod, name)
    def w(a, b, *args, **kw):
        if not ON[0] or DEPTH[0]: 
            DEPTH[0] += 1
            try: return inner(a, b, *args, **kw)
            finally: DEPTH[0] -= 1
        f = sys._getframe(1); caller = (f.f_lineno, f.f_code.co_name) if f.f_code.co_filename == CF else (0, '?')
        before = CTX[0]._flops_used; DEPTH[0] += 1
        try: return inner(a, b, *args, **kw)
        finally:
            DEPTH[0] -= 1; e = ENTRY[(caller, name)]; e[0] += CTX[0]._flops_used - before; e[1] += 1
            if len(e[2]) < 3: e[2].add((tuple(int(x) for x in a.shape), tuple(int(x) for x in b.shape)))
    setattr(mod, name, w)
for nm in ('_mm', '_shared_old384_mm', '_tsr_mm'):
    if callable(getattr(mod, nm, None)): wrap(nm)
w = loader.weights(seed); mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')
for phase in ('cold', 'warm'):
    ON[0] = phase == 'warm'
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        CTX[0] = ctx; est.predict(mlp, 2**41); total = int(ctx.summary_dict()['flops_used'])
    print(phase, total, flush=True)
rows = sorted(([k[0], k[1], v, SELF[k]] for k, v in INC.items()), key=lambda r: -r[2])
ent = sorted(([k[0][0], k[0][1], k[1], v[0], v[1], sorted(v[2])] for k, v in ENTRY.items()), key=lambda r: -r[3])
out.write_text(json.dumps(dict(candidate=str(cand), seed=seed, total=total, inclusive=rows, entries=ent)))
