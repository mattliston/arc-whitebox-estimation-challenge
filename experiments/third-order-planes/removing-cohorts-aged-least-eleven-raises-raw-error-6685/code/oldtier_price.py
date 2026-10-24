#!/usr/bin/env python3
"""TAG fable-oldtier-economics (OT-1): the RAW PRICE of old third-order content on the deployed 331824 card, by cohort age.
For each age A, the signed cube-column weights (lam) of every cohort whose age (layer - birth layer) >= A are ZEROED right after
the bank push at each layer (columns are still pushed and read, so the bill is unchanged; only the content is removed).
A=16 = control (nothing zeroed); A=5 = the whole shared old tier removed (cohorts join the tier at age 5).
usage: oldtier_price.py <candidate.py> <ranks e.g. 35,100> <ages e.g. 16,5,7,9,11>  -> oldtier_price_<cand>.json (+ per-run prints)"""
import sys, json, types, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; ages = [int(a) for a in sys.argv[3].split(",")]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
src0 = cand.read_text()
HOOK_LINE = "            pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)\n"
assert src0.count(HOOK_LINE) >= 1
i = src0.rfind(HOOK_LINE)  # the LAST (live) _predict_binary
src = src0[:i] + HOOK_LINE + "            lam = _ablate_lam(l, lam, shared_old)\n" + src0[i + len(HOOK_LINE):]
src += '''

_ABLATE_AGE = 16
def _ablate_lam(l, lam, state):
    cohorts = state["cohorts"]
    assert len(cohorts) == l, (len(cohorts), l)
    K = int(lam.shape[0]); keep = np.ones(K, dtype=np.float32); hit = 0
    for b, (start, width) in enumerate(cohorts):
        if l - b >= _ABLATE_AGE and width > 0:
            keep[start:start + width] = 0.0; hit += width
    if hit == 0:
        return lam
    return lam * fnp.asarray(keep, dtype=fnp.float32)
'''
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
mod = types.ModuleType("oldtier_price_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
mod.np = np
exec(compile(src, str(cand), "exec"), mod.__dict__)
est = mod.Estimator()
out = {}
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    for A in ages:
        mod._ABLATE_AGE = A; t0 = time.time()
        try:
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"ot-{rank}"), 2**41), dtype=np.float64)
            bill = ctx.summary_dict()["flops_used"]; util = bill / 2**41
            mse = float(((pred[15] - truth[15]) ** 2).mean()); per_layer = [float(((pred[l] - truth[l]) ** 2).mean()) for l in range(16)]
            rec = dict(age=A, rank=rank, util=round(util, 5), mse=mse, adjusted=mse * max(0.1, util), per_layer_mse=per_layer, wall=round(time.time() - t0, 1))
        except Exception as ex:
            import traceback; rec = dict(age=A, rank=rank, error=repr(ex)[:300], tb=traceback.format_exc()[-1500:])
        out[f"{A}:{rank}"] = rec; print(json.dumps({k: v for k, v in rec.items() if k != "per_layer_mse"}), flush=True)
        json.dump(out, open(HERE / f"oldtier_price_{cand.stem}.json", "w"), indent=1)
