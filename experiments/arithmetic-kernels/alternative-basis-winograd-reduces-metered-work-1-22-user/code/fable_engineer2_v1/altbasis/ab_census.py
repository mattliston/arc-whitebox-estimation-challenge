#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-AB): tree-product census of S13 (in-place tree Winograd kernel) for PAPER-PRICING Karstadt-Schwartz
alternative-basis Strassen-Winograd. Wraps _ipt_forward_a / _ipt_forward_b / _ipt_leaf_unwind / _bb_product (P-batched) and records every
tree product side as (side, depth L, rows, cols, P). Per net: total bill, billed add/subtract/copyto volume, and the census.
usage: ab_census.py <cand.py> <ranks> <tag>  -> altbasis/ab_census_<tag>.json"""
import sys, json, types, time, collections
from pathlib import Path
import numpy as np, flopscope as flops
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, ranks, tag = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("ab_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
rec = collections.Counter()
fa, fb, fu, fbb = mod._ipt_forward_a, mod._ipt_forward_b, mod._ipt_leaf_unwind, mod._bb_product
def wa(a, L, m, k): rec[("A", L, m, k, 1)] += 1; return fa(a, L, m, k)
def wb(b, L, k, n): rec[("B", L, k, n, 1)] += 1; return fb(b, L, k, n)
def wu(TA, TB, out, L, m, n): rec[("C", L, m, n, 1)] += 1; return fu(TA, TB, out, L, m, n)
def wbb(As, Bs, outs, L):
    P = len(As); m, k, n = int(As[0].shape[0]), int(As[0].shape[1]), int(Bs[0].shape[1])
    rec[("A", L, m, k, P)] += 1; rec[("B", L, k, n, P)] += 1; rec[("C", L, m, n, P)] += 1
    return fbb(As, Bs, outs, L)
mod._ipt_forward_a, mod._ipt_forward_b, mod._ipt_leaf_unwind, mod._bb_product = wa, wb, wu, wbb
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, nm): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, nm
est = mod.Estimator(); out = []
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); rec.clear(); t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        est.predict(MLP([W[l] for l in range(16)], f"ab-{rank}"), 2**41)
    ops = ctx.summary_dict()["operations"]
    vol = {k: int(ops[k].get("flop_cost", 0)) for k in ("add", "subtract", "copyto", "matmul") if k in ops}
    out.append(dict(rank=rank, flops=int(ctx._flops_used), vol=vol, calls=int(sum(v.get("calls", 0) for v in ops.values())),
                    census=[list(k) + [v] for k, v in sorted(rec.items())]))
    print(json.dumps(dict(rank=rank, util=round(ctx._flops_used / 2**41, 6), vol={k: round(v / 1e9, 2) for k, v in vol.items()},
                          tree_sides=sum(rec.values()), wall=round(time.time() - t0, 1))), flush=True)
json.dump(out, open(HERE / f"ab_census_{tag}.json", "w"))
