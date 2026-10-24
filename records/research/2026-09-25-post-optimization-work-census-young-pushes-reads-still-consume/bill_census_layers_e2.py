#!/usr/bin/env python3
"""TAG fable-engineer2 (lead 09:3x, S27 bill census): fable-porter's bill_census_layers.py (PT-38) plus (a) Estimator.setup() FIRST, the platform
order (setup's own charges are kept apart under net 'setup'), (b) the flopscope op name of every charge, (c) matmul operand SHAPES folded into the op field ('matmul|(7^L, r, k)x(7^L, k, c)'), (d) the INNERMOST candidate-file
frame (function:line) under the live loop (treuse wrappers unwrapped/skipped), so kernel roles (leaf matmul / transform adds / copies / binds) can be split. Every charged flop is
keyed by (net, layer, loop line, callee1..4 outermost first, innermost frame, op); per layer it snapshots lam / old-tier columns / old frame
shapes / live rows at the layer's first op (as the porter's tool). .venv in process, 1 BLAS thread; net order as given (first = cold after
setup). usage: bill_census_layers_e2.py <cand.py> <ranks e.g. 35,100> <out.json>"""
import sys, json, types, collections
from pathlib import Path
import numpy as np, flopscope as flops
from flopscope import _budget
P2 = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16')
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); name = str(cand); ranks = [int(x) for x in sys.argv[2].split(',')]; OUT = Path(sys.argv[3])
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("census_layers_e2_mod"); mod.__file__ = name; sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), name, "exec"), mod.__dict__)
def _unwrap(f):
    while hasattr(f, '__wrapped__'):   # treuse (E-73) wraps _predict_binary in _tr_clearing's w(); the loop is the innermost original
        f = f.__wrapped__
    return f
LIVE = _unwrap(mod._predict_binary).__code__
WRAPPERS = {'w', 'wrapped'}   # wrapper frames (treuse's w, flopscope-style wrapped) are skipped in callee chains
cur = dict(net=None); cost = collections.Counter(); ops = collections.Counter(); snaps = {}
def shp(x):
    try: return list(x.shape)
    except Exception: return None
def snap(P):
    L = P.f_locals; so = L.get('shared_old') or {}
    lam, li = L.get('lam'), L.get('live_indices')
    return dict(cols=(int(lam.shape[0]) if lam is not None else None), old_cols=(int(so['columns']) if 'columns' in so else None),
                Q=shp(so.get('Q')), F=shp(so.get('F')), core=shp(so.get('core')), pre_basis=shp(so.get('pre_basis')),
                live=(int(li.shape[0]) if li is not None and hasattr(li, 'shape') else None))
_orig = _budget.BudgetContext._charge_op
def charged(self, *args, **kwargs):
    before = self._flops_used; r = _orig(self, *args, **kwargs); c = self._flops_used - before
    opn = args[0] if args else kwargs.get('op_name', '?')
    if opn in ('matmul', 'dot') and len(args) > 3:   # leaf products: record operand shapes (a tree leaf is (7^L, r, k) @ (7^L, k, c) -> depth L, dims)
        try: opn = opn + '|' + 'x'.join(str(tuple(int(d) for d in sh)) for sh in args[3][:2])
        except Exception: pass
    f = sys._getframe(1); chain = []; P = None
    while f is not None:
        if f.f_code is LIVE: P = f; break
        if f.f_code.co_filename == name and f.f_code.co_name not in WRAPPERS: chain.append(f)
        f = f.f_back
    nm = lambda i: (chain[-i].f_code.co_name + ':' + str(chain[-i].f_lineno)) if len(chain) >= i else ''
    inner = (chain[0].f_code.co_name + ':' + str(chain[0].f_lineno)) if chain else ''
    if P is None:
        key = (cur['net'], -1, 0, nm(1), nm(2), nm(3), nm(4), inner, opn)
    else:
        l = P.f_locals.get('l')
        key = (cur['net'], l, P.f_lineno, nm(1), nm(2), nm(3), nm(4), inner, opn)
        sk = (cur['net'], l)
        if sk not in snaps: snaps[sk] = snap(P)
    cost[key] += c; ops[key] += 1
    return r
_budget.BudgetContext._charge_op = charged
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, nm): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, nm
est = mod.Estimator(); totals = {}
cur['net'] = 'setup'
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version="2", scratch_dir=None, submission_dir=None, seed=0))
print(json.dumps(dict(setup_charged=sum(v for k, v in cost.items() if k[0] == 'setup'), pb_log=[list(x) for x in getattr(mod, '_PB_LOG', [])])), flush=True)
for r in ranks:
    cur['net'] = r; Wt = np.load(STORE / Path(man[r]["path"]).parent / "weights.npy")
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        est.predict(MLP([Wt[l] for l in range(16)], f"census-{r}"), 2**41)
    totals[r] = ctx.summary_dict()["flops_used"]
    print(json.dumps(dict(net=r, flops=totals[r], util=totals[r] / 2**41, attributed=sum(v for k, v in cost.items() if k[0] == r))), flush=True)
rows = [dict(net=k[0], layer=k[1], line=k[2], callee1=k[3], callee2=k[4], callee3=k[5], callee4=k[6], inner=k[7], op=k[8], flops=v, ops=ops[k]) for k, v in cost.items()]
src = cand.read_text().split('\n'); lines = sorted({k[2] for k in cost if k[2]})
OUT.write_text(json.dumps(dict(cand=name, ranks=ranks, totals=totals, U=2 * 1024 ** 3, rows=rows,
    snaps=[dict(net=k[0], layer=k[1], **v) for k, v in sorted(snaps.items(), key=lambda kv: (str(kv[0][0]), -1 if kv[0][1] is None else kv[0][1]))],
    loop_src={str(n): src[n - 1].strip()[:160] for n in lines}), indent=0))
print(f'wrote {OUT}: {len(rows)} keys')
