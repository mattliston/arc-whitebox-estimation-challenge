#!/usr/bin/env python3
"""TAG fable-porter (PT-38, lead 21:0x "S21 bill census by function, per layer, in U"): native predictions of <cand.py> on <ranks> in ONE
process (.venv, 1 BLAS thread; the first rank is cold, later ranks warm). Every charged flop is keyed by (net, layer, loop line, callee1,
callee2): layer and loop line come from the live _predict_binary frame (its local l and current line), callee1..callee4 are the outermost
candidate-file frames under it (function:line), outermost first. Per layer it also snapshots: len(lam) (all CP columns),
shared_old['columns'] (old-tier columns), shapes of the old tier's Q/F/core/pre_basis, len(live_indices) -- at the layer's FIRST op, i.e. the state after layer l-1
(births of layer l = (cols(l+1) - cols(l)) / 2). Writes raw rows to <out.json>.
usage: bill_census_layers.py <cand.py> <ranks e.g. 35,100> <out.json>"""
import sys, json, types, collections
from pathlib import Path
import numpy as np, flopscope as flops
from flopscope import _budget
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); name = str(cand); ranks = [int(x) for x in sys.argv[2].split(',')]; OUT = Path(sys.argv[3])
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("census_layers_mod"); mod.__file__ = name; sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), name, "exec"), mod.__dict__)
LIVE = mod._predict_binary.__code__   # the live (last-defined) loop
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
    f = sys._getframe(1); chain = []; P = None
    while f is not None:
        if f.f_code is LIVE: P = f; break
        if f.f_code.co_filename == name: chain.append(f)
        f = f.f_back
    if P is None:
        key = (cur['net'], -1, 0, (chain[-1].f_code.co_name + ':' + str(chain[-1].f_lineno)) if chain else '', '', '', '')
    else:
        l = P.f_locals.get('l'); nm = lambda i: (chain[-i].f_code.co_name + ':' + str(chain[-i].f_lineno)) if len(chain) >= i else ''
        key = (cur['net'], l, P.f_lineno, nm(1), nm(2), nm(3), nm(4))
        sk = (cur['net'], l)
        if sk not in snaps: snaps[sk] = snap(P)   # first op of layer l = the state after layer l-1 (lam concat and joins included)
    cost[key] += c; ops[key] += 1
    return r
_budget.BudgetContext._charge_op = charged
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, nm): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, nm
est = mod.Estimator(); totals = {}
for r in ranks:
    cur['net'] = r; Wt = np.load(STORE / Path(man[r]["path"]).parent / "weights.npy")
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        est.predict(MLP([Wt[l] for l in range(16)], f"census-{r}"), 2**41)
    totals[r] = ctx.summary_dict()["flops_used"]
    print(json.dumps(dict(net=r, flops=totals[r], util=totals[r] / 2**41, attributed=sum(v for k, v in cost.items() if k[0] == r))), flush=True)
rows = [dict(net=k[0], layer=k[1], line=k[2], callee1=k[3], callee2=k[4], callee3=k[5], callee4=k[6], flops=v, ops=ops[k]) for k, v in cost.items()]
src = cand.read_text().split('\n'); lines = sorted({k[2] for k in cost if k[2]})
OUT.write_text(json.dumps(dict(cand=name, ranks=ranks, totals=totals, U=2 * 1024 ** 3, rows=rows,
    snaps=[dict(net=k[0], layer=k[1], **v) for k, v in sorted(snaps.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1]))], loop_src={str(n): src[n - 1].strip()[:160] for n in lines}), indent=0))
print(f'wrote {OUT}: {len(rows)} keys')
