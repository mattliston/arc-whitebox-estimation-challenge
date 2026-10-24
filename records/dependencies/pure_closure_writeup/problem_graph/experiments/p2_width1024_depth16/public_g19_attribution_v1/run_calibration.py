#!/usr/bin/env python3
"""TAG g19b-calibration-v1: per-layer gains on the sidecar deltas.
Columns: for each layer l=1..15 and each term t in {k4, k3}, the TRANSPORTED
endpoint image of that single-layer single-term delta (30 columns, each a
1024-vector at L16 per row). Fit ridge to e_true[15] jointly on given rows;
report in-sample and leave-one-row-out recovered fractions and corrected MSE.
Usage: run_calibration.py 35 100"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_attribution import edg2_relu_mean


def columns_for(rank):
    d = np.load(HERE / f"states_rank{rank:04d}.npz")
    s = np.load(HERE / f"sidecar_states_rank{rank:04d}.npz")
    hm, hv = d["host_m"], d["host_v"]
    hk3, hk4 = d["host_k3"], d["host_k4"]
    e_true = d["mc_post"] - d["host_post"]
    mc_se = d["mc_se"]
    raw20 = s["raw20"]
    sk3, sk4, sv = s["s_k3"], s["s_k4"], s["s_v"]
    # weights for transport
    P2 = HERE.parent
    STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
             "sealed_target_blind_feature_store_v3")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    rmp = STORE / entry["path"]
    W = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
    Wd = [np.asarray(W[l], dtype=np.float64) for l in range(16)]

    def transport_from(l0, delta):
        r = delta
        for l in range(l0 + 1, 16):
            r = raw20[l, :, 5] * (r @ Wd[l])
        return r

    cols = []
    names = []
    for l in range(1, 16):
        i = l - 1
        base = edg2_relu_mean(hm[i], hv[i], hk3[i], hk4[i])
        d_k4 = edg2_relu_mean(hm[i], hv[i], hk3[i], sk4[i]) - base
        d_k3 = edg2_relu_mean(hm[i], hv[i], sk3[i], hk4[i]) - base
        d_v = edg2_relu_mean(hm[i], sv[i], hk3[i], hk4[i]) - base
        for tag, dd in (("k4", d_k4), ("k3", d_k3), ("v", d_v)):
            cols.append(transport_from(l, dd))
            names.append(f"L{l}_{tag}")
    X = np.stack(cols, axis=1)          # [1024, 45]
    return X, e_true[15], float(np.sum(mc_se[15] ** 2)), names


def fit_and_score(ranks):
    data = {r: columns_for(r) for r in ranks}
    names = data[ranks[0]][3]
    lam = 1e-3

    def solve(train_ranks):
        Xs = np.concatenate([data[r][0] for r in train_ranks], axis=0)
        ys = np.concatenate([data[r][1] for r in train_ranks], axis=0)
        G = Xs.T @ Xs
        G += lam * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
        return np.linalg.solve(G, Xs.T @ ys)

    out = {"names": names}
    beta_all = solve(ranks)
    for r in ranks:
        X, y, noise, _ = data[r]
        den = float(np.sum(y ** 2))
        num_in = float(np.sum((y - X @ beta_all) ** 2))
        beta_loo = solve([q for q in ranks if q != r]) if len(ranks) > 1 else beta_all
        num_loo = float(np.sum((y - X @ beta_loo) ** 2))
        unit = np.ones(X.shape[1])
        num_unit = float(np.sum((y - X @ unit) ** 2))
        out[str(r)] = {
            "host_L16_mse": den / 1024.0,
            "uncalibrated_mse": num_unit / 1024.0,
            "insample_mse": num_in / 1024.0,
            "loo_mse": num_loo / 1024.0,
            "recovered_loo_nc": 1.0 - max(num_loo - noise, 0.0)
                                      / max(den - noise, 1e-30),
        }
    out["beta_joint"] = {n: round(float(b), 4)
                         for n, b in zip(names, beta_all)}
    return out


if __name__ == "__main__":
    ranks = [int(a) for a in sys.argv[1:]] or [35, 100]
    res = fit_and_score(ranks)
    (HERE / "CALIBRATION_RESULT.json").write_text(
        json.dumps(res, indent=1) + "\n")
    for r in ranks:
        print(r, json.dumps(res[str(r)]))
