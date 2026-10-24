#!/usr/bin/env python3
"""TAG fable-oldtier-economics (OT-4): IMPORTANCE-PRUNED BIRTHS on the deployed 331824 card. At every layer the kink units that
would receive a signed cube-column pair are ranked by importance = birth_j * |bp_j|^2 (kink density x gate^2 x covariance-column
energy = the Frobenius scale of the unit's born star atom) and only the top KEEP fraction (rounded to 32) is born. Fewer columns
cut the young push, the old tier and all plane reads together, so both util and raw move.  usage: birth_prune.py <candidate.py>
<ranks e.g. 35,100> <keep fractions e.g. 1.0,0.75,0.5,0.35>  -> birth_prune_<cand>.json"""
import sys, json, types, time
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; keeps = [float(x) for x in sys.argv[3].split(",")]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
src0 = cand.read_text()
H = "            a = birth / delta\n"
i = src0.rfind(H); assert i > 0
src = src0[:i] + H + "            selected, births = _prune_births(selected, births, birth, bp, t, s)\n" + src0[i + len(H):]
src += '''

_KEEP = 1.0
def _tonp(a):
    return np.asarray(a.tolist() if hasattr(a, "tolist") else a)
_IMP_OUT = None
_IMP_LAYER = [0]
def _prune_births(selected, births, birth, bp, t, s):
    sel = _tonp(selected).astype(np.int64)
    if len(sel) == 0:
        return selected, births
    b2 = (_tonp(bp)[:, sel] ** 2).sum(axis=0); bt = _tonp(birth)[sel]; tt = _tonp(t)[sel]; ss = _tonp(s)[sel]; u = tt + ss
    # Frobenius norm of the unit's born tensor birth*[3 Sym(b b e) + 3u Sym(b e e) + (t^2+ts+s^2) e e e]
    imp = bt * np.sqrt(3 * b2 ** 2 + 3 * u ** 2 * b2 + (tt ** 2 + tt * ss + ss ** 2) ** 2)
    if _IMP_OUT is not None:
        np.savez(str(_IMP_OUT / f"imp_{_IMP_LAYER[0]:02d}.npz"), sel=sel, imp=imp); _IMP_LAYER[0] += 1
    if _KEEP >= 1.0:
        return selected, births
    keep = max(32, (int(round(len(sel) * _KEEP)) + 31) // 32 * 32)
    keep = min(keep, len(sel))
    order = np.argsort(-imp, kind="stable")[:keep]
    new_sel = sel[order]
    mask = np.zeros(int(births.shape[0]), dtype=bool); mask[new_sel] = True
    return fnp.asarray(new_sel), fnp.asarray(mask)
'''
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
mod = types.ModuleType("birth_prune_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod; mod.np = np
exec(compile(src, str(cand), "exec"), mod.__dict__)
import os
if os.environ.get("IMP_OUT"):
    mod._IMP_OUT = Path(os.environ["IMP_OUT"]); mod._IMP_OUT.mkdir(exist_ok=True)
est = mod.Estimator(); out = {}
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    for keep in keeps:
        mod._KEEP = keep; t0 = time.time()
        try:
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"bp-{rank}"), 2**41), dtype=np.float64)
            util = ctx.summary_dict()["flops_used"] / 2**41; mse = float(((pred[15] - truth[15]) ** 2).mean())
            rec = dict(keep=keep, rank=rank, util=round(util, 5), mse=mse, adjusted=mse * max(0.1, util), wall=round(time.time() - t0, 1))
        except Exception as ex:
            import traceback; rec = dict(keep=keep, rank=rank, error=repr(ex)[:300], tb=traceback.format_exc()[-1200:])
        out[f"{keep}:{rank}"] = rec; print(json.dumps(rec), flush=True)
        json.dump(out, open(HERE / f"birth_prune_{cand.stem}.json", "w"), indent=1)
