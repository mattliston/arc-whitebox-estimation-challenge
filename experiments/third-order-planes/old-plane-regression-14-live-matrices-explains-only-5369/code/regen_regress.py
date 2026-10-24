#!/usr/bin/env python3
"""TAG fable-oldtier-economics (OT-6): can the OLD tier's (2,1) plane be REGENERATED from live state? In-sample (one net, per layer)
least squares of K21_old[response, live] on a dictionary of live matrices (Cpre, Cpre^2, Cpre^3, young plane, star_pre, the b-terms,
gate/variance-scaled variants, and the memoryless transport of the previous layer's old plane). Reports the explained energy
(R^2) per layer; an upper bound for any per-layer scalar-coefficient regeneration.  usage: regen_regress.py <bank_dir> <rank> <layers>"""
import sys, json
from pathlib import Path
import numpy as np
D = Path(sys.argv[1]); rank = int(sys.argv[2]); layers = [int(x) for x in sys.argv[3].split(",")]
P2 = D.resolve().parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
Wall = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy").astype(np.float64)
def old_plane_full(l):
    z = np.load(D / f"layer_{l:02d}.npz"); P = z["pp"].astype(np.float64); w = z["lam"].astype(np.float64); oc = int(z["old_columns"])
    if oc == 0: return None, z
    Po, wo = P[:, :oc], w[:oc]
    return (Po ** 2 * wo[None, :]) @ Po.T, z
out = {}
for l in layers:
    K21o, z = old_plane_full(l)
    if K21o is None: continue
    rows, cols = z["response"], z["live"]
    st = np.load(D / f"state_{l:02d}.npz"); C = st["Cpre"].astype(np.float64); S = st["star_pre"].astype(np.float64)
    mu = st["mu"].astype(np.float64); bpre = st["bpre"].astype(np.float64); Bd = st["Bpre_diag"].astype(np.float64); g = st["gate"].astype(np.float64)
    vv = np.diag(C); sd = np.sqrt(np.maximum(vv, 1e-12)); rho = C / np.outer(sd, sd)
    # young plane
    P = z["pp"].astype(np.float64); w = z["lam"].astype(np.float64); oc = int(z["old_columns"])
    K21y = (P[:, oc:] ** 2 * w[None, oc:]) @ P[:, oc:].T
    feats = {"Cpre": C, "Cpre2": C * C, "Cpre3": C * C * C, "young": K21y, "star": S,
             "bterm": np.outer(Bd, bpre), "Bb": C * bpre[:, None], "vv_outer": np.outer(vv, vv), "v_rho": vv[:, None] * rho,
             "mu_C": mu[:, None] * C, "g_C_g": g[:, None] * C * g[None, :], "rho2_v": rho * rho * np.outer(vv, sd),
             "Cpre2_scaled": (C * C) * (1.0 / np.maximum(sd, 1e-6))[None, :]}
    # memoryless transport of the previous layer's old plane: (W.W)^T (g^2 K g) W  (previous gates unknown here -> use the previous
    # layer's response gate as the linear gain proxy)
    Kprev, zp = old_plane_full(l - 1)
    if Kprev is not None:
        sp = np.load(D / f"state_{l-1:02d}.npz"); gp = sp["gate"].astype(np.float64); W = Wall[l]
        Kp = (gp[:, None] ** 2) * Kprev * gp[None, :]
        feats["memoryless_prev"] = (W * W).T @ Kp @ W
        feats["memoryless_prev_cross"] = 2 * (W.T @ (Kp * (W @ W.T) ) @ W) if False else W.T @ (Kp @ W) * 0  # placeholder (kept zero)
        del feats["memoryless_prev_cross"]
    y = K21o[np.ix_(rows, cols)].ravel()
    names = list(feats); X = np.stack([feats[k][np.ix_(rows, cols)].ravel() for k in names], axis=1)
    def r2(cols_idx):
        Xs = X[:, cols_idx]; beta, *_ = np.linalg.lstsq(Xs, y, rcond=None); res = y - Xs @ beta
        return 1.0 - float((res ** 2).sum() / (y ** 2).sum()), beta
    rec = {"layer": l, "n_old_cols": oc, "norm_old_plane": float(np.linalg.norm(y)), "single": {}, "all": None}
    for i, k in enumerate(names):
        rec["single"][k] = round(r2([i])[0], 4)
    rec["all"], beta = r2(list(range(len(names)))); rec["all"] = round(rec["all"], 4)
    if "memoryless_prev" in names:
        rec["all_but_memoryless"] = round(r2([i for i, k in enumerate(names) if k != "memoryless_prev"])[0], 4)
    print(json.dumps(rec), flush=True); out[l] = rec
json.dump(out, open(D / f"regen_regress.json", "w"), indent=1)
