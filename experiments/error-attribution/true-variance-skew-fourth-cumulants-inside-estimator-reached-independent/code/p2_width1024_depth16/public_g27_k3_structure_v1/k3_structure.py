#!/usr/bin/env python3
"""STRUCTURE of the deep k3 error Δk3 = truth_k3 − card_k3 (layers 2..15, 8 panel nets, G26 features):
(a) per-layer scale (panel-mean rms ratio s_l); (b) regime dependence: mean of the post-scale relative residual
r = (truth − card/s_l)/(card/s_l) per bin of α = m/σ (also g3 = k3/v^1.5, and v) per layer, across nets (mean, sd);
(c) residual after (a)+(b): variance shares, held-out ridge on tape fields + incoming weight norms.
Writes K3_STRUCTURE.json and the LOO α-bin tables f_l(α) per held-out net."""
import json, sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; G26 = HERE.parent / "public_g26_k3_corrector_features_v1"; G19 = HERE.parent / "public_g19_attribution_v1"
NETS = [35, 100, 150, 200, 250, 300, 400, 500]; NB = 10
F = {R: np.load(G26 / f"features_rank{R:04d}.npz") for R in NETS}
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())
def wnorm(R, l):
    entry = man["roles"]["observer_fit"]["row_manifests"][R]; W = np.load((STORE / entry["path"]).parent / "weights.npy", mmap_mode="r")
    return np.linalg.norm(np.asarray(W[l], dtype=float), axis=0)          # incoming norm of unit i at layer l = ||W[l][:, i]||
# (a) panel scale per layer
s = {}
for l in range(2, 16):
    s[l] = float(np.mean([np.sqrt((F[R]["card_k3"][l - 1] ** 2).mean()) / np.sqrt((F[R]["truth_k3"][l] ** 2).mean()) for R in NETS]))
# per-unit tables
rows = {}
for R in NETS:
    for l in range(2, 16):
        c = F[R]["card_k3"][l - 1].astype(float); t = F[R]["truth_k3"][l].astype(float); m = F[R]["card_m"][l - 1].astype(float); v = F[R]["card_v"][l - 1].astype(float)
        cs = c / s[l]; r = (t - cs) / np.where(np.abs(cs) > 1e-12, cs, 1e-12)
        rows[(R, l)] = {"c": c, "t": t, "cs": cs, "r": r, "alpha": m / np.sqrt(v), "g3": c / v ** 1.5, "v": v, "wn": wnorm(R, l), "raw": F[R]["raw20"][l].astype(float)}
# global bin edges per layer (pooled quantiles) for alpha, g3, v
edges = {}
for var in ("alpha", "g3", "v"):
    for l in range(2, 16):
        pooled = np.concatenate([rows[(R, l)][var] for R in NETS]); edges[(var, l)] = np.quantile(pooled, np.linspace(0, 1, NB + 1))
def binmeans(var, l, nets):
    e = edges[(var, l)]; out = []
    for R in nets:
        x = rows[(R, l)][var]; d = rows[(R, l)]; idx = np.clip(np.searchsorted(e, x, side="right") - 1, 0, NB - 1)
        num = d["t"] - d["cs"]; den = d["cs"]
        # robust within-bin LS slope: f_b = sum_bin (t - cs) cs / sum_bin cs^2 (immune to near-zero-k3 units)
        out.append(np.array([float(np.dot(num[idx == b], den[idx == b]) / max(np.dot(den[idx == b], den[idx == b]), 1e-30)) if (idx == b).any() else 0.0 for b in range(NB)]))
    return np.array(out)     # (nets, NB)
res = {"s_l": s, "bins": {}, "variance_shares": {}, "heldout_ridge_R2": {}}
for l in range(2, 16):
    rec = {}
    for var in ("alpha", "g3", "v"):
        B = binmeans(var, l, NETS); rec[var] = {"mean": B.mean(0).round(4).tolist(), "sd_across_nets": B.std(0).round(4).tolist()}
    res["bins"][l] = rec
# (c) variance shares: per net/layer, Δk3 energy explained by (a), by (a)+(b, alpha bins LOO), remainder; held-out ridge on residual
for l in [8, 10, 12, 14, 15]:
    sh = []; r2s = []
    for R in NETS:
        others = [q for q in NETS if q != R]
        d = rows[(R, l)]; dk = d["t"] - d["c"]; e0 = float((dk ** 2).sum())
        after_a = d["t"] - d["cs"]; ea = float((after_a ** 2).sum())
        fb = binmeans("alpha", l, others).mean(0); e = edges[("alpha", l)]; idx = np.clip(np.searchsorted(e, d["alpha"], side="right") - 1, 0, NB - 1)
        pred_b = d["cs"] * fb[idx]; after_ab = after_a - pred_b; eab = float((after_ab ** 2).sum())
        sh.append({"net": R, "share_removed_by_scale": 1 - ea / e0, "share_removed_by_scale_plus_alphabins": 1 - eab / e0})
        # held-out ridge for the residual after (a)+(b): features = [alpha, g3, v, log wn, raw20 (20)] pooled over the other nets
        def feats(q):
            dd = rows[(q, l)]; return np.column_stack([dd["alpha"], dd["g3"], dd["v"], np.log(dd["wn"]), dd["raw"]])
        def target(q):
            dd = rows[(q, l)]; fbq = fb; idxq = np.clip(np.searchsorted(e, dd["alpha"], side="right") - 1, 0, NB - 1)
            return (dd["t"] - dd["cs"]) - dd["cs"] * fbq[idxq]
        Xtr = np.vstack([feats(q) for q in others]); ytr = np.concatenate([target(q) for q in others])
        mu, sdv = Xtr.mean(0), Xtr.std(0) + 1e-12; Xn = (Xtr - mu) / sdv; ym = ytr.mean()
        beta = np.linalg.solve(Xn.T @ Xn + 1.0 * len(ytr) * np.eye(Xn.shape[1]) * 1e-3, Xn.T @ (ytr - ym))
        Xte = (feats(R) - mu) / sdv; yte = target(R); pred = Xte @ beta + ym
        r2s.append(1 - float(((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()))
    res["variance_shares"][l] = {"scale_mean": float(np.mean([x["share_removed_by_scale"] for x in sh])), "scale_plus_alpha_mean": float(np.mean([x["share_removed_by_scale_plus_alphabins"] for x in sh])), "per_net": sh}
    res["heldout_ridge_R2"][l] = {"mean": float(np.mean(r2s)), "per_net": [round(x, 3) for x in r2s]}
    print(f"L{l}: Δk3 energy removed by scale {res['variance_shares'][l]['scale_mean']:.3f}; by scale+α-bins(LOO) {res['variance_shares'][l]['scale_plus_alpha_mean']:.3f}; held-out ridge R² on the remainder (α,g3,v,log|w|,raw20): {res['heldout_ridge_R2'][l]['mean']:.3f} {res['heldout_ridge_R2'][l]['per_net']}")
    print(f"   f_l(α) bins mean {res['bins'][l]['alpha']['mean']}  sd {res['bins'][l]['alpha']['sd_across_nets']}")
    print(f"   f_l(g3) bins mean {res['bins'][l]['g3']['mean']}")
json.dump(res, open(HERE / "K3_STRUCTURE.json", "w"), indent=1)
# LOO injection tables for r35 and r100: (a) scale only; (a)+(b) alpha bins from the other 7 nets
for R in (35, 100):
    others = [q for q in NETS if q != R]
    ta, tab = {}, {}
    for l in range(1, 16):
        c = F[R]["card_k3"][l - 1].astype(float)
        if l == 1: ta[f"dk3_{l}"] = np.zeros(1024); tab[f"dk3_{l}"] = np.zeros(1024); continue
        ta[f"dk3_{l}"] = c * (1.0 / s[l] - 1.0)
        fb = binmeans("alpha", l, others).mean(0); e = edges[("alpha", l)]; idx = np.clip(np.searchsorted(e, rows[(R, l)]["alpha"], side="right") - 1, 0, NB - 1)
        tab[f"dk3_{l}"] = ta[f"dk3_{l}"] + (c / s[l]) * fb[idx]
    np.savez(HERE / f"dk3_scale_rank{R:04d}.npz", **ta); np.savez(HERE / f"dk3_scale_alphabins_LOO_rank{R:04d}.npz", **tab)   # robust-slope version
print("tables written; s_l:", {l: round(x, 4) for l, x in s.items()})
