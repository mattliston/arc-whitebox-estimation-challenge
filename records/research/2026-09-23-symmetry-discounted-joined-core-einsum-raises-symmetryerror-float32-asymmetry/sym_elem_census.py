#!/usr/bin/env python3
"""TAG fable-engineer (E-41): which NON-product ops work on symmetric matrices (flopscope bills ops on a SymmetricTensor at the
orbit quotient, about half). Hooks flopscope's _call_numpy: each counted op's cost is read from its op record; the op is classed
'product' (matmul/einsum/dot/tensordot), 'sym' (some 2-D square operand or out= with side >= 256 is symmetric to 1e-5 relative),
'square' (square operand, not symmetric) or 'other'. Attributed to the first candidate frame. In-process, one process, nets back to
back. usage: sym_elem_census.py <candidate.py> <ranks> <tag>   -> fable_engineer_v1/sym_elem_census_<tag>.json"""
import sys, json, types, collections
from pathlib import Path
import numpy as np
import flopscope as flops, flopscope._budget as fb
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; tag = sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("sym_elem_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
cand_name = str(cand)
PRODUCT = {"matmul", "einsum", "dot", "tensordot", "inner", "outer", "vdot"}
acc = collections.defaultdict(lambda: [0, 0])     # (class, site, op) -> [ops, flops]
seen = set()
def is_sym(x):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[0] != x.shape[1] or x.shape[0] < 256:
        return None
    v = x.view(np.ndarray)
    if v.dtype.kind not in "fc":
        return None
    a = v.astype(np.float64, copy=False)
    s = float(np.abs(a).max()) if a.size else 0.0
    return bool(s > 0 and float(np.abs(a - a.T).max()) <= 1e-5 * s)
orig = fb._call_numpy
def hooked(fn, *args, **kwargs):
    budget = fb.get_active_budget()
    timer = budget._current_op_timer if budget is not None else None
    idx = getattr(timer, "_op_index", None)
    if idx is not None and (id(budget), idx) not in seen:
        seen.add((id(budget), idx))
        rec = budget._op_log[idx]
        name = rec.op_name
        cls = "product" if name.split(".")[-1] in PRODUCT else "other"
        if cls == "other":
            flags = [is_sym(x) for x in list(args) + [kwargs.get("out")]]
            if any(f is True for f in flags): cls = "sym"
            elif any(f is False for f in flags): cls = "square"
        f = sys._getframe(1); site = "-"
        while f is not None:
            if f.f_code.co_filename == cand_name:
                site = f"{f.f_code.co_name}:{f.f_lineno}"; break
            f = f.f_back
        a = acc[(cls, site, name)]; a[0] += 1; a[1] += rec.flop_cost
    return orig(fn, *args, **kwargs)
for m in list(sys.modules):
    if m.startswith("flopscope") and getattr(sys.modules[m], "_call_numpy", None) is orig:
        sys.modules[m]._call_numpy = hooked
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator(); result = []
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    acc.clear(); seen.clear()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        est.predict(MLP([W[l] for l in range(16)], f"se-{rank}"), 2**41)
    bill = ctx.summary_dict()["flops_used"]
    by = collections.Counter()
    for (cls, site, name), (n, c) in acc.items(): by[cls] += c
    rows = sorted(((k, v[0], v[1]) for k, v in acc.items() if k[0] == "sym"), key=lambda x: -x[2])
    rec = dict(rank=rank, bill=bill, util=round(bill / 2**41, 5), by_class={k: round(v / bill * 100, 3) for k, v in by.items()},
               sym_rows=[(list(k), n, int(c), round(c / bill * 100, 4)) for k, n, c in rows])
    result.append(rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "sym_rows"}), flush=True)
    for k, n, c, pct in rec["sym_rows"][:25]:
        print(f"   {pct:7.4f}%  {n:5d} ops  {k[1]:40s} {k[2]}", flush=True)
json.dump(result, open(HERE / f"sym_elem_census_{tag}.json", "w"), indent=1)
