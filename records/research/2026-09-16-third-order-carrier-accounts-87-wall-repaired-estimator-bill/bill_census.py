#!/usr/bin/env python3
"""TAG fable-wall-repair (W-J): BILL census of one native prediction — every charged flop attributed to the innermost call site inside the
candidate file (function + line) by hooking BudgetContext._charge_op. Prints the bill by function (and by a coarse component map) and the
top 40 lines.  usage: bill_census.py <candidate.py> <rank>  -> <cand>_bill_r<rank>.json"""
import sys, json, types, importlib.util, collections
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
from flopscope import _budget
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, rank = Path(sys.argv[1]).resolve(), int(sys.argv[2])
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("bill_cand", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
by_site = collections.Counter(); by_func = collections.Counter(); by_op = collections.Counter(); cand_name = str(cand)
KERNEL = {"_packed_product_cached", "_packed_product_3", "_packed_product_4", "_packed_product_5", "_packed_product_6", "_square_product", "_mm", "_shared_old384_mm", "_exact_view", "_a_frame", "_b_frame", "_o_frame", "_packed_frame", "_cp3_bind_arena_views", "<genexpr>", "_p1_d6_832_product", "_resident_gram512"}
ATTR_CALLER = len(sys.argv) > 3 and sys.argv[3] == "callers"
_orig = _budget.BudgetContext._charge_op
def charged(self, *args, **kwargs):
    before = self._flops_used
    r = _orig(self, *args, **kwargs)
    cost = self._flops_used - before
    if cost:
        f = sys._getframe(1); site = None
        while f is not None:
            if f.f_code.co_filename == cand_name and not (ATTR_CALLER and f.f_code.co_name in KERNEL):
                site = (f.f_lineno, f.f_code.co_name); break
            f = f.f_back
        by_site[site] += cost; by_func[site[1] if site else None] += cost
        op = kwargs.get("op_name") or (args[0] if args else None); by_op[str(op)[:40]] += cost
    return r
_budget.BudgetContext._charge_op = charged
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
class MLP:
    def __init__(self, weights): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, f"bill-{rank}"
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    mod.Estimator().predict(MLP([W[l] for l in range(16)]), 2**41)
tot = ctx.summary_dict()["flops_used"]; print(f"{cand.name} r{rank}: bill {tot:.4e} (util {tot/2**41:.4f}); attributed {sum(by_site.values())/tot*100:.1f}%")
print("by function (% of bill):");
for k, v in by_func.most_common(30): print(f"  {100*v/tot:6.2f}%  {k}")
print("top lines:")
for (site, v) in by_site.most_common(40): print(f"  {100*v/tot:6.2f}%  line {site[0] if site else '-'}  {site[1] if site else '-'}")
json.dump(dict(total=tot, by_site={f"{k[0]} {k[1]}" if k else "-": v for k, v in by_site.items()}, by_func={str(k): v for k, v in by_func.items()}, by_op=dict(by_op)),
          open(HERE / f"{cand.stem}_bill_r{rank}{'_callers' if ATTR_CALLER else ''}.json", "w"), indent=1)
