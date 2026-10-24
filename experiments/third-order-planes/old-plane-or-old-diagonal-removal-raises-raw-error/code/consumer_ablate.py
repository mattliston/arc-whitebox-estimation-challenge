#!/usr/bin/env python3
"""TAG fable-oldtier-economics (OT-5): WHICH CONSUMER of the old tier carries its value? Variants on the deployed 331824 card:
 control | oldplane0 = the old tier's (2,1) plane read returns zero (diagonal k3 read and radial source keep the old columns) |
 olddiag0 = the old columns are zeroed in the diagonal k3 read and the omitted-radial source only (plane read intact).
usage: consumer_ablate.py <candidate.py> <ranks> <variants e.g. control,oldplane0,olddiag0> -> consumer_ablate_<cand>.json"""
import sys, json, types, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; variants = sys.argv[3].split(",")
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
src0 = cand.read_text()
# plane read: the LAST _shared_old384_read definition returns young_plane + old_plane
A = "    return young_plane + old_plane\n"; i = src0.rfind(A); assert i > 0
src = src0[:i] + "    return young_plane + (old_plane * 0.0 if _OLDPLANE0 else old_plane)\n" + src0[i + len(A):]
# diagonal read + omitted radial source in the live loop use lam directly: substitute a masked lam
B = "                diagonal = fnp.sum(_cp_diagonal_cube(pp.astype(fnp.float64)) * lam.astype(fnp.float64), axis=1)\n"
j = src.rfind(B); assert j > 0
src = src[:j] + "                diagonal = fnp.sum(_cp_diagonal_cube(pp.astype(fnp.float64)) * _diag_lam(lam, shared_old).astype(fnp.float64), axis=1)\n" + src[j + len(B):]
C = "                source_sum = omitted_radial_source(pp, lam, diagonal, omitted, alive, aux, host)\n"
k = src.rfind(C); assert k > 0
src = src[:k] + "                source_sum = omitted_radial_source(pp, _diag_lam(lam, shared_old), diagonal, omitted, alive, aux, host)\n" + src[k + len(C):]
src += '''

_OLDPLANE0 = False
_OLDDIAG0 = False
def _diag_lam(lam, state):
    oc = int(state["columns"])
    if not _OLDDIAG0 or oc == 0:
        return lam
    keep = np.ones(int(lam.shape[0]), dtype=np.float32); keep[:oc] = 0.0
    return lam * fnp.asarray(keep, dtype=fnp.float32)
'''
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
mod = types.ModuleType("consumer_ablate_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod; mod.np = np
exec(compile(src, str(cand), "exec"), mod.__dict__)
est = mod.Estimator(); out = {}
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    for v in variants:
        mod._OLDPLANE0 = (v == "oldplane0"); mod._OLDDIAG0 = (v == "olddiag0"); t0 = time.time()
        try:
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"ca-{rank}"), 2**41), dtype=np.float64)
            util = ctx.summary_dict()["flops_used"] / 2**41; mse = float(((pred[15] - truth[15]) ** 2).mean())
            rec = dict(variant=v, rank=rank, util=round(util, 5), mse=mse, adjusted=mse * max(0.1, util), wall=round(time.time() - t0, 1))
        except Exception as ex:
            import traceback; rec = dict(variant=v, rank=rank, error=repr(ex)[:300], tb=traceback.format_exc()[-1200:])
        out[f"{v}:{rank}"] = rec; print(json.dumps(rec), flush=True)
        json.dump(out, open(HERE / f"consumer_ablate_{cand.stem}.json", "w"), indent=1)
