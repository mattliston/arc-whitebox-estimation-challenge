#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-D6B): MSE / bill price of running S15's batched P=2 1024^3 products (_mm_batch -> _bb_product, the
covariance/star products, 28 per net) at depth 6 instead of 5, in process. Measurement harness only: the candidate file is not edited;
_bb_product is wrapped so that a depth-5 P=2 1024^3 call runs at depth 6 with _IPT_REGION[6] = (60236288,)*3 and private tree caches
during the call (every other product, region and cache is S15's). Writes read6/par4_<tag>.log and ../par4_<tag>.npz in parity.py's format
(compare with read6/parity_compare.py ab_S14 <tag>).  usage: depth6_batched_parity.py <S15.py> <ranks> <tag>"""
import sys, json, types, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; E = HERE.parent; P2 = E.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, ranks, tag = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("d6b_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
assert getattr(mod, "_AB_ON", False), "candidate lacks --altbasis"
batched = mod._bb_product; CACHES = ("_IPT_A", "_IPT_B", "_IPT_C", "_IPT_BA", "_IPT_BB", "_IPT_BC", "_AB_VIEWS")
REGION = (60236288, 60236288, 60236288); MINE = {c: mod._CapLRU() for c in CACHES}; moved = [0]
def w_batched(As, Bs, outs, L):
    if L != 5 or len(As) != 2 or any(tuple(x.shape) != (1024, 1024) for x in list(As) + list(Bs)):
        return batched(As, Bs, outs, L)
    saved = {c: getattr(mod, c) for c in CACHES}; old = mod._IPT_REGION[6]
    for c in CACHES: setattr(mod, c, MINE[c])
    mod._IPT_REGION[6] = REGION
    try:
        moved[0] += 1; return batched(As, Bs, outs, 6)
    finally:
        mod._IPT_REGION[6] = old
        for c, v in saved.items(): setattr(mod, c, v)
mod._bb_product = w_batched
est = mod.Estimator(); man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
saved = {}; logf = open(E / "read6" / f"par4_{tag}.log", "w")
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    t0 = time.time(); moved[0] = 0
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"par-{rank}"), 2**41))
    f = int(ctx.summary_dict()["flops_used"]); p64 = pred.astype(np.float64)
    rec = dict(cand=cand.name + "+d6bb", rank=rank, flops=f, util=round(f / 2**41, 6), mse=float(((p64[15] - truth[15]) ** 2).mean()),
               wall=round(time.time() - t0, 1), moved=moved[0])
    saved[f"pred_{rank}"] = pred; saved[f"flops_{rank}"] = np.int64(f)
    print(json.dumps(rec), file=logf, flush=True); print(json.dumps(rec), flush=True)
np.savez(E / f"par4_{tag}.npz", **saved); print("DONE", tag, flush=True)
