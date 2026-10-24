#!/usr/bin/env python3
"""TAG fable-oldtier-economics (OT-10): value of the STAR residual (2,1)-plane carrier (12% of the bill). Variants on the deployed
331824 card: control | star0 = star_pre zeroed at every layer (the carrier is still pushed, bill unchanged) | star0_late = zeroed
for layers >= 8 only | star0_early = zeroed for layers < 8 only.  usage: star_ablate.py <candidate.py> <ranks> <variants>"""
import sys, json, types, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; variants = sys.argv[3].split(",")
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
src0 = cand.read_text()
H = "            K21 = cp_pre + star_pre + fnp.diag(Bpre)[:, None] * bpre[None, :] + 2 * Bpre * bpre[:, None]\n"
i = src0.rfind(H); assert i > 0
src = src0[:i] + "            star_pre = _star_gate(l, star_pre)\n" + H + src0[i + len(H):]
src += '''

_STAR_MODE = "control"
def _star_gate(l, star_pre):
    if _STAR_MODE == "star0" or (_STAR_MODE == "star0_late" and l >= 8) or (_STAR_MODE == "star0_early" and l < 8):
        return star_pre * 0.0
    return star_pre
'''
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
mod = types.ModuleType("star_ablate_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod; mod.np = np
exec(compile(src, str(cand), "exec"), mod.__dict__)
est = mod.Estimator(); out = {}
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    for v in variants:
        mod._STAR_MODE = v; t0 = time.time()
        try:
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"st-{rank}"), 2**41), dtype=np.float64)
            util = ctx.summary_dict()["flops_used"] / 2**41; mse = float(((pred[15] - truth[15]) ** 2).mean())
            rec = dict(variant=v, rank=rank, util=round(util, 5), mse=mse, adjusted=mse * max(0.1, util), wall=round(time.time() - t0, 1))
        except Exception as ex:
            import traceback; rec = dict(variant=v, rank=rank, error=repr(ex)[:300], tb=traceback.format_exc()[-1000:])
        out[f"{v}:{rank}"] = rec; print(json.dumps(rec), flush=True)
        json.dump(out, open(HERE / f"star_ablate_{cand.stem}.json", "w"), indent=1)
