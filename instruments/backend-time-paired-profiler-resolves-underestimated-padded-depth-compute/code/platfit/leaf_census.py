#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-LC): matmul shape census of whole-net predictions (in-process flopscope): every charged matmul with its
batch count (number of stacked matrices) and m, k, n, from BudgetContext._charge_op's shapes. Tests whether a candidate adds many
tiny batched leaf products (per-matrix call overhead is not in the bill). Nets run back to back (first = cold).
usage: leaf_census.py <cand.py> <ranks> <tag> -> platfit/leaf_<tag>.json"""
import sys, json, types, time, collections
from pathlib import Path
import numpy as np, flopscope as flops
from flopscope import _budget
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, ranks, tag = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(',')], sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("leaf_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
rec = collections.Counter(); first = [True]
_orig = _budget.BudgetContext._charge_op
def _charge(self, op_name, flop_cost, subscripts, shapes, **kw):
    if op_name == "matmul" and shapes:
        a, b = shapes[0], shapes[1] if len(shapes) > 1 else ()
        if first[0]:
            print("first matmul shapes example:", shapes, flush=True); first[0] = False
        a = tuple(int(x) for x in a); b = tuple(int(x) for x in b)
        if len(a) >= 2 and len(b) >= 1:
            batch = 1
            for x in a[:-2]: batch *= x
            m, k = a[-2], a[-1]; n = b[-1] if len(b) >= 2 else 1
            rec[(batch, m, k, n)] += 1
    return _orig(self, op_name, flop_cost, subscripts, shapes, **kw)
_budget.BudgetContext._charge_op = _charge
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, nm): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, nm
est = mod.Estimator(); out = []
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); rec.clear(); t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        est.predict(MLP([W[l] for l in range(16)], f"lc-{rank}"), 2**41)
    rows = [dict(batch=b, m=m, k=k, n=n, calls=c) for (b, m, k, n), c in rec.items()]
    mats = sum(r["batch"] * r["calls"] for r in rows)
    tiny = sum(r["batch"] * r["calls"] for r in rows if max(r["m"], r["k"], r["n"]) <= 16)
    small = sum(r["batch"] * r["calls"] for r in rows if 16 < max(r["m"], r["k"], r["n"]) <= 32)
    out.append(dict(rank=rank, flops=int(ctx.summary_dict()["flops_used"]), matmul_calls=sum(r["calls"] for r in rows), matrices=mats, tiny_le16=tiny, small_17_32=small, rows=rows))
    print(json.dumps(dict(rank=rank, util=round(out[-1]["flops"] / 2**41, 6), matmul_calls=out[-1]["matmul_calls"], matrices=mats, tiny_le16=tiny, small_17_32=small,
                          wall=round(time.time() - t0, 1))), flush=True)
json.dump(out, open(HERE / f"leaf_{tag}.json", "w"))
