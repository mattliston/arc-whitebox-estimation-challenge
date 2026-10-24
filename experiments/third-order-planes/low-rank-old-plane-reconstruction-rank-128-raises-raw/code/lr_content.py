#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-LR5, low-rank screen step 2b): content ratios (Frobenius norms, per layer, one net, learned ON, measurement
only) of the screened classes against what they are added to:
  star    : ||star_pre|| / ||cp_pre + star_pre||   (star_pre = _flms_cov_star's star result restricted to the read's live units; cp_pre = the bank-read plane
            returned by _cp_read_young_sq in the SAME layer (the star call precedes the reads); K21 = cp_pre + star_pre + rank-1 mean terms)
  s1      : ||(2/3) s1|| / ||star_pre||              (s1 = pw w, the star's dense product)
  oldtier : ||old_plane|| / ||cp_pre||               (old_plane = inner(old part) @ pre_basis[live]^T, recomputed here)
  base    : ||common|| / ||common + directions t||   (the W5 newborn_common product vs the newborn columns it seeds)
usage: lr_content.py <cand.py> <rank> <tag>"""
import sys, json, types
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, rank, tag = Path(sys.argv[1]).resolve(), int(sys.argv[2]), sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("lrc_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
fro = lambda x: float(np.linalg.norm(np.asarray(x, dtype=np.float64)))
REC = {"star": [], "s1": [], "oldtier": [], "base": []}; LAST = {}
_cs = mod._flms_cov_star
def cov_star(w, C, A):
    # the star call precedes the same layer's reads: keep the result, pair it with that layer's read plane in read()
    Cpre, result = _cs(w, C, A); LAST["star"] = fro(result); LAST["star_arr"] = np.asarray(result, np.float64)
    return Cpre, result
mod._flms_cov_star = cov_star
_mb = mod._mm_batch
def mm_batch(pairs):
    outs = _mb(pairs)
    if len(pairs) == 2 and tuple(pairs[1][0].shape) == (1024, 1024): LAST["s1"] = fro(outs[1]) * 2 / 3
    return outs
mod._mm_batch = mm_batch
_rd = mod._cp_read_young_sq
def read(left, young_right, lam, live_indices, state):
    plane = _rd(left, young_right, lam, live_indices, state); oc = int(state['columns'])
    if oc:
        inner = mod._cp_slr_inner_sq(left[:, :oc], lam[:oc], state)
        old_plane = mod._shared_old384_mm(inner, state['pre_basis'][live_indices, :].T)
        REC["oldtier"].append(fro(old_plane) / max(fro(plane), 1e-300))
    if "star_arr" in LAST:
        li = np.asarray(live_indices); S = LAST.pop("star_arr")[np.ix_(li, li)]; cp = np.asarray(plane, np.float64)
        if S.shape == cp.shape: REC["star"].append(fro(S) / max(fro(cp + S), 1e-300))
    return plane
mod._cp_read_young_sq = read
_push = mod._push_bank_shared_newborn
def push(w, p, previous, newborn):
    out = _push(w, p, previous, newborn)
    if newborn is not None:
        old_columns, selected, base, t, s = newborn
        common = np.asarray(mod._PHYS_HANDLES['newborn_common'], np.float64)
        dt = np.asarray(w[selected, :].T, np.float64) * np.asarray(t, np.float64)
        REC["base"].append(fro(common) / max(fro(common + dt), 1e-300))
    return out
mod._push_bank_shared_newborn = push
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
_orig_cs_inner = cov_star
def cov_star_s1(w, C, A):
    out = _orig_cs_inner(w, C, A)
    if "s1" in LAST: REC["s1"].append(LAST.pop("s1") / max(LAST["star"], 1e-300))
    return out
mod._flms_cov_star = cov_star_s1
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    mod.Estimator().predict(MLP([W[l] for l in range(16)], f"lrc-{rank}"), 2**41)
json.dump(dict(rank=rank, **REC), open(HERE / f"lr_content_{tag}.json", "w"))
for k, v in REC.items():
    print(f"{k:8s} {len(v):2d} layers: median {np.median(v) if v else float('nan'):.4f}  range {min(v) if v else float('nan'):.4f}..{max(v) if v else float('nan'):.4f}")
