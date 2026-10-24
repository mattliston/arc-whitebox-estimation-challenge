#!/usr/bin/env python3
"""TAG fable-engineer (E-42): where does a cold net's EXTRA bill go? Two predicts of the same net in one fresh process; every charged op is
attributed (flopscope BudgetContext._charge_op) to its first candidate frame (function); per-function bill of predict 1 minus predict 2.
usage: cold_premium_census.py <candidate.py> <rank> <tag>   -> prints the functions whose bill differs, and the total"""
import sys, json, types, collections
from pathlib import Path
import numpy as np
import flopscope as flops, flopscope._budget as fb
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); rank = int(sys.argv[2]); tag = sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("cold_premium_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
cand_name = str(cand); acc = collections.Counter()
orig = fb.BudgetContext._charge_op
def charge(self, op_name, flop_cost, *a, **k):
    b0 = self._flops_used
    r = orig(self, op_name, flop_cost, *a, **k)
    f = sys._getframe(1); site = "-"
    while f is not None:
        if f.f_code.co_filename == cand_name:
            site = f.f_code.co_name; break
        f = f.f_back
    acc[(site, op_name)] += self._flops_used - b0
    return r
fb.BudgetContext._charge_op = charge
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
est = mod.Estimator(); per = []; bills = []
for rep in range(2):
    acc.clear()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        est.predict(MLP([W[l] for l in range(16)], f"cp-{rank}-{rep}"), 2**41)
    per.append(dict(acc)); bills.append(ctx.summary_dict()["flops_used"])
diff = collections.Counter()
for k in set(per[0]) | set(per[1]):
    d = per[0].get(k, 0) - per[1].get(k, 0)
    if d: diff[k] = d
print(json.dumps(dict(rank=rank, cold=bills[0], repeat=bills[1], premium_pct=round((bills[0] / bills[1] - 1) * 100, 3))))
byfn = collections.Counter()
for (fn, op), d in diff.items(): byfn[fn] += d
for fn, d in byfn.most_common(40):
    ops = ", ".join(f"{op} {v/bills[0]*100:.4f}%" for (f2, op), v in diff.most_common() if f2 == fn)[:160]
    print(f"   {d/bills[0]*100:8.4f}%  {fn:32s} {ops}")
