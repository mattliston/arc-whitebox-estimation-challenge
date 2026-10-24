#!/usr/bin/env python3
"""TAG g22-card-census (T-118d): FLOP census of the D11 card (estimator_cap.py) by function + op, one net, full budget context.
Wraps flopscope BudgetContext._charge_op and attributes each charge to the innermost frame inside the card file."""
import collections, importlib.util, json, sys
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import BudgetContext
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
CARD = P2 / "public_d11_util_halving_v1/estimator_cap.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
rank = int(sys.argv[1]) if len(sys.argv) > 1 else 35
SITE = collections.Counter(); ORIG = BudgetContext._charge_op
def traced(self, op_name, flop_cost, subscripts, shapes, **kw):
    before = self._flops_used; ORIG(self, op_name, flop_cost, subscripts, shapes, **kw); cost = self._flops_used - before
    f = sys._getframe(1); fn = "??"; line = 0
    while f is not None:
        if f.f_code.co_filename == str(CARD): fn = f.f_code.co_name; line = f.f_lineno; break
        f = f.f_back
    SITE[(fn, op_name, tuple(tuple(s) for s in shapes) if shapes else ())] += cost
BudgetContext._charge_op = traced
spec = importlib.util.spec_from_file_location("card", CARD); m = importlib.util.module_from_spec(spec); sys.modules["card"] = m; spec.loader.exec_module(m)
man = json.loads((STORE / "manifest.json").read_text()); e = man["roles"]["observer_fit"]["row_manifests"][rank]; rmp = STORE / e["path"]; row = json.loads(rmp.read_text())
W = np.load(rmp.parent / "weights.npy", mmap_mode="r"); mlp = MLP(width=1024, depth=16, weights=[W[l] for l in range(16)], seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=f"census-{rank}")
with threadpool_limits(limits=4):
    with BudgetContext(flop_budget=2**41, wall_time_limit_s=1200.0, quiet=True) as ctx:
        m._predict(mlp)
tot = ctx.flops_used; print(f"total billed {tot:.3e} = {tot/2**41:.4f} of the budget")
byfn = collections.Counter()
for (fn, op, sh), c in SITE.items(): byfn[fn] += c
print("by function:"); [print(f"  {100*c/tot:5.1f}%  {fn}") for fn, c in byfn.most_common(14)]
print("top op x shape sites:"); [print(f"  {100*c/tot:5.1f}%  {fn} {op} {sh}") for (fn, op, sh), c in SITE.most_common(16)]
