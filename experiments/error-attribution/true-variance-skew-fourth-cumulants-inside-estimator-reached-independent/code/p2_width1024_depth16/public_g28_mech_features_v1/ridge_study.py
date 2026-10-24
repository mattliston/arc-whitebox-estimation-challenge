#!/usr/bin/env python3
"""Leave-one-net-out ridge R² per layer for Δk3 = truth_k3 − card_k3 and Δv = truth_v − card_v (G26 targets) from
feature FAMILIES built from the card's internal state (mechfeat_retro-{R}.npz) + the G26 baseline features.
Families: F0 baseline [α, g3, v, log w_in, raw20]; Fa CP3 [cp3_diag, cp3 blocks by age (last 4 birth layers + older),
star_diag, cp3_energy, K21 row/col norms]; Fb CP4 [cp4_d, cp4 K31/K22 rownorm, pre_k4, cp4 inherited/birth/dropped d,
K22/K31 rownorm]; Fc pair/weights [K21 rownorm, w_in_norm, gate-weighted column sums Σ_j gate_j W_ji, Σ_j gate_j W_ji²];
Fd pushed previous-layer cumulants [Σ_j W_ji³ post_k3_target_j(l−1), Σ_j W_ji⁴ post_k4_target_j(l−1), Σ_j W_ji² post_var_j(l−1)].
Targets are scaled per layer by the layer rms of the card state; ridge λ chosen on the training nets by 2-fold CV over a grid."""
import json, sys, os, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; G26 = HERE.parent / "public_g26_k3_corrector_features_v1"
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())
NETS = [R for R in [35, 100, 150, 200, 250, 300, 400, 500] if (HERE / f"mechfeat_retro-{R}.npz").exists()]
LAYERS = [int(x) for x in os.environ.get("LAYERS", "3,5,8,10,12,14").split(",")]
def W_of(R):
    e = man["roles"]["observer_fit"]["row_manifests"][R]; return np.load((STORE / e["path"]).parent / "weights.npy", mmap_mode="r")
data = {}
for R in NETS:
    F = np.load(G26 / f"features_rank{R:04d}.npz"); M = np.load(HERE / f"mechfeat_retro-{R}.npz"); W = W_of(R)
    per = {}
    for l in LAYERS:
        i = l - 1
        m = F["card_m"][i].astype(float); v = F["card_v"][i].astype(float); k3 = F["card_k3"][i].astype(float); k4 = F["card_k4"][i].astype(float)
        alpha = m / np.sqrt(v); g3 = k3 / v ** 1.5; raw = F["raw20"][l].astype(float)
        Wl = np.asarray(W[l], dtype=float); win = np.linalg.norm(Wl, axis=0)
        gate_prev = M[f"gate_{l - 1}"] if f"gate_{l - 1}" in M.files else np.ones(1024)
        blocks = M[f"cp3_diag_blocks_{l}"]; nb = l   # blocks 0..l-1 are birth layers 0..l-1 (block b born at layer b)
        age = np.zeros((4, 1024))
        for a in range(4):
            b = nb - 1 - a
            if b >= 0: age[a] = blocks[b]
        older = blocks[:max(nb - 4, 0)].sum(0) if nb > 4 else np.zeros(1024)
        F0 = np.column_stack([alpha, g3, v, np.log(win), raw])
        Fa = np.column_stack([M[f"cp3_diag_{l}"], age.T, older, M[f"star_diag_{l}"], np.log1p(M[f"cp3_energy_{l}"] * 1e6), M[f"K21_rownorm_{l}"], M[f"K21_colnorm_{l}"]])
        Fb = np.column_stack([M[f"cp4_d_{l}"], M[f"cp4_K31_rownorm_{l}"], M[f"cp4_K22_rownorm_{l}"], M[f"pre_k4_{l}"], M.get(f"cp4_inherited_d_{l - 1}", np.zeros(1024)) if hasattr(M, "get") else (M[f"cp4_inherited_d_{l - 1}"] if f"cp4_inherited_d_{l - 1}" in M.files else np.zeros(1024)), (M[f"cp4_k4_birth_{l - 1}"] if f"cp4_k4_birth_{l - 1}" in M.files else np.zeros(1024)), (M[f"cp4_dropped_d_{l - 1}"] if f"cp4_dropped_d_{l - 1}" in M.files else np.zeros(1024)), M[f"K22_rownorm_{l}"], M[f"K31_rownorm_{l}"]])
        Fc = np.column_stack([win, gate_prev @ Wl, gate_prev @ (Wl ** 2), (gate_prev ** 2) @ (Wl ** 2)])
        pk3 = M[f"post_k3_target_{l - 1}"] if f"post_k3_target_{l - 1}" in M.files else np.zeros(1024)
        pk4 = M[f"post_k4_target_{l - 1}"] if f"post_k4_target_{l - 1}" in M.files else np.zeros(1024)
        pv = M[f"post_var_{l - 1}"] if f"post_var_{l - 1}" in M.files else np.zeros(1024)
        Fd = np.column_stack([(Wl ** 3).T @ pk3, (Wl ** 4).T @ pk4, (Wl ** 2).T @ pv, (Wl ** 3).T @ (pk3 * gate_prev)])
        yk3 = (F["truth_k3"][l].astype(float) - k3) / np.sqrt((k3 ** 2).mean()); yv = (F["truth_v"][l].astype(float) - v) / np.sqrt((v ** 2).mean())
        per[l] = {"F0": F0, "Fa": Fa, "Fb": Fb, "Fc": Fc, "Fd": Fd, "yk3": yk3, "yv": yv}
    data[R] = per
def ridge_r2(train, test, cols, target):
    Xtr = np.vstack([np.column_stack([data[R][l][c] for c in cols]) for R in train]); ytr = np.concatenate([data[R][l][target] for R in train])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12; Xn = (Xtr - mu) / sd; ym = ytr.mean(); yc = ytr - ym
    # lambda by 2-fold CV over the training nets
    best = None
    for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        sc = []
        for fold in (0, 1):
            tr = [R for k, R in enumerate(train) if k % 2 != fold]; te = [R for k, R in enumerate(train) if k % 2 == fold]
            Xt = np.vstack([np.column_stack([data[R][l][c] for c in cols]) for R in tr]); yt = np.concatenate([data[R][l][target] for R in tr])
            Xe = np.vstack([np.column_stack([data[R][l][c] for c in cols]) for R in te]); ye = np.concatenate([data[R][l][target] for R in te])
            Xtn = (Xt - mu) / sd; Xen = (Xe - mu) / sd; b = np.linalg.solve(Xtn.T @ Xtn + lam * len(yt) * np.eye(Xtn.shape[1]), Xtn.T @ (yt - yt.mean()))
            pr = Xen @ b + yt.mean(); sc.append(1 - ((ye - pr) ** 2).sum() / ((ye - ye.mean()) ** 2).sum())
        if best is None or np.mean(sc) > best[0]: best = (np.mean(sc), lam)
    lam = best[1]; b = np.linalg.solve(Xn.T @ Xn + lam * len(ytr) * np.eye(Xn.shape[1]), Xn.T @ yc)
    Xte = (np.column_stack([data[test][l][c] for c in cols]) - mu) / sd; yte = data[test][l][target]; pr = Xte @ b + ym
    return 1 - ((yte - pr) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
fams = {"F0 baseline": ["F0"], "+Fa CP3": ["F0", "Fa"], "+Fb CP4": ["F0", "Fb"], "+Fc pair/weights": ["F0", "Fc"], "+Fd pushed prev": ["F0", "Fd"], "ALL": ["F0", "Fa", "Fb", "Fc", "Fd"]}
res = {}
for target in ("yk3", "yv"):
    print(f"=== target {target} (LOO ridge R², mean over held-out nets; nets={NETS})")
    for l in LAYERS:
        row = {}
        for name, cols in fams.items():
            r2 = [ridge_r2([q for q in NETS if q != R], R, cols, target) for R in NETS]; row[name] = (float(np.mean(r2)), float(np.min(r2)))
        res[f"{target}_L{l}"] = row
        print(f"L{l}: " + " | ".join(f"{k} {v[0]:.3f} (min {v[1]:.3f})" for k, v in row.items()))
json.dump(res, open(HERE / "RIDGE_STUDY.json", "w"), indent=1)
