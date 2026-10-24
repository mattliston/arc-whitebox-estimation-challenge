#!/usr/bin/env python3
"""TAG fable-engineer (E-29): BILL census by dtype x call site of one candidate (in-process flopscope, server numpy env).
Wraps BudgetContext._charge_op: every billed op is attributed to the innermost candidate frame 'function:line', its op name and
its billing dtype (flopscope's own resolve_billing_dtype on the declared dtypes; float64 is billed at 2x float32). Records flops,
calls, and the largest operand/output size per site. usage: dtype_census.py <candidate.py> <ranks> <tag> -> dtype_census_<tag>.json"""
import sys, json, types, collections, math
from pathlib import Path
import numpy as np
import flopscope as flops, flopscope._budget as fb
from flopscope._dtype_billing import resolve_billing_dtype, rate_for
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; tag = sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("dtype_census_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
cand_name = str(cand)
acc = collections.defaultdict(lambda: [0, 0, 0, None, 0.0])   # (dtype, site, op) -> [calls, flops, max elems, max shape, rate]
orig = fb.BudgetContext._charge_op
def charged(self, op_name, flop_cost, subscripts, shapes, **kw):
    before = self._flops_used
    r = orig(self, op_name, flop_cost, subscripts, shapes, **kw)
    billed = self._flops_used - before
    dts = kw.get("dtypes")
    res = resolve_billing_dtype(dts) if dts is not None else None
    dname = res.name if res is not None else "neutral"
    f = sys._getframe(1); site = "-"
    while f is not None:
        if f.f_code.co_filename == cand_name:
            site = f"{f.f_code.co_name}:{f.f_lineno}"; break
        f = f.f_back
    mx, msh = 0, None
    for s in (shapes or ()):
        try:
            n = math.prod(int(x) for x in s)
        except TypeError:
            continue
        if n > mx:
            mx, msh = n, tuple(int(x) for x in s)
    a = acc[(dname, site, op_name)]; a[0] += 1; a[1] += billed
    if mx > a[2]:
        a[2], a[3] = mx, msh
    a[4] = rate_for(res) if res is not None else 1.0
    return r
fb.BudgetContext._charge_op = charged
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator(); out = []
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    acc.clear()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"dc-{rank}"), 2**41), dtype=np.float64)
    bill = ctx.summary_dict()["flops_used"]
    by_dtype = collections.Counter(); by_func = collections.defaultdict(collections.Counter)
    for (dn, site, op), (n, fl, mx, sh, rate) in acc.items():
        by_dtype[dn] += fl; by_func[site.split(":")[0]][dn] += fl
    f64 = by_dtype.get("float64", 0)
    rec = dict(rank=rank, bill=bill, util=round(bill / 2**41, 5), by_dtype={k: [v, round(v / bill * 100, 3)] for k, v in by_dtype.most_common()},
               f64_share_pct=round(f64 / bill * 100, 3), f64_excess_pct=round(f64 / 2 / bill * 100, 3),
               top_f64=[(site, op, n, fl, round(fl / bill * 100, 3), mx, sh) for (dn, site, op), (n, fl, mx, sh, rate) in
                        sorted(acc.items(), key=lambda kv: -kv[1][1]) if dn == "float64"][:60],
               funcs=[(fn, dict(c)) for fn, c in sorted(by_func.items(), key=lambda kv: -kv[1].get("float64", 0))[:40]])
    out.append(rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("top_f64", "funcs")}), flush=True)
    print("  top float64 sites (site, op, calls, flops, % of bill, max elems, shape):")
    for row in rec["top_f64"][:30]:
        print("   ", row)
json.dump(out, open(HERE / f"dtype_census_{tag}.json", "w"), indent=1)
