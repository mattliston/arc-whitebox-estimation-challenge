#!/usr/bin/env python3
"""TAG g19carrier-value-map: exact per-(layer, channel) attribution of
the certified 18.7% chain (LOO_GRADE_tensor). The chain is LINEAR in the
per-layer supply deltas: r_ch = sum_l T_l dds_ch,l with T_l the frozen
deploy-tangent transport. Per net (LOO betas frozen from the other 7):
  value[l,ch]   = 2 beta_ch <resid, T_l dds_ch,l> - cross terms (exact
                  leave-cell-out capture drop, reported)
  cum curves    = capture of the best-k cells (greedy)
Output: CHAIN_VALUE_MAP.json + per-net npz. Pure numpy replay on the
banked panel files; no new forwards. Usage: run_chain_value_map.py
<rank> [rank ...]  (default: full 8-net panel)"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_attribution import edg2_relu_mean

RANKS_ALL = [35, 100, 150, 200, 250, 300, 400, 500]
CH = ("k4", "k3", "v")


def load_net(rank):
    lite = HERE / f"states_rank{rank:04d}_lite.npz"
    if lite.exists():
        st = dict(np.load(lite))
    else:
        d = np.load(HERE / f"states_rank{rank:04d}.npz")
        st = {k: d[k] for k in ("host_post", "host_m", "host_v",
                                "host_k3", "host_k4")}
    s = np.load(HERE / f"sidecar_states_rank{rank:04d}.npz")
    tr = np.load(HERE / f"mps_truth_rank{rank:04d}_P8M.npz")
    dep = np.load(HERE / f"deployed_rank{rank:04d}.npy")
    tp = np.load(HERE / f"tensor_planes_rank{rank:04d}.npz")
    dv = np.load(HERE / f"dvhat_rank{rank:04d}.npz")
    P2 = HERE.parent
    STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
             "sealed_target_blind_feature_store_v3")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    W = np.load((STORE / entry["path"]).parent / "weights.npy",
                mmap_mode="r")
    Wd = [np.asarray(W[l], dtype=np.float64) for l in range(16)]
    return st, s["raw20"], tr, dep, tp, dv, Wd


def per_cell_contribs(rank):
    """r_contrib[ch][l] = transported end-frame vector from layer l's
    delta alone (chain linearity)."""
    st, raw20, tr, dep, tp, dv, Wd = load_net(rank)
    contrib = {k: {} for k in CH}
    for l in range(1, 16):
        i = l - 1
        hm, hv = st["host_m"][i], st["host_v"][i]
        hk3, hk4 = st["host_k3"][i], st["host_k4"][i]
        base = edg2_relu_mean(hm, hv, hk3, hk4)
        k3h = tp[f"k3u_{l}"].astype(np.float64) \
            if f"k3u_{l}" in tp.files else hk3
        k4h = tp[f"k4u_{l}"].astype(np.float64) \
            if f"k4u_{l}" in tp.files else hk4
        dvh = dv[f"L{l}"].astype(np.float64) if f"L{l}" in dv.files \
            else np.zeros(1024)
        dds = {"k4": edg2_relu_mean(hm, hv, hk3, k4h) - base,
               "k3": edg2_relu_mean(hm, hv, k3h, hk4) - base,
               "v": edg2_relu_mean(hm, np.maximum(hv + dvh, 1e-30),
                                   hk3, hk4) - base}
        for k in CH:
            contrib[k][l] = dds[k].copy()
        # advance every previously injected contribution one layer
        phi = raw20[l, :, 5]
        for k in CH:
            for l0 in list(contrib[k]):
                if l0 < l:
                    contrib[k][l0] = phi * (contrib[k][l0] @ Wd[l])
    resid = tr["mc_post"][15] - dep[15]
    noise = float(np.sum(tr["mc_se"][15] ** 2))
    return contrib, resid, noise


def loo_betas(data, hold):
    G = np.zeros((3, 3))
    bv = np.zeros(3)
    for r, (R, resid, _) in data.items():
        if r == hold:
            continue
        G += R @ R.T
        bv += R @ resid
    return np.linalg.solve(G + 1e-12 * np.eye(3), bv)


def main():
    ranks = [int(a) for a in sys.argv[1:]] or RANKS_ALL
    per_net = {}
    rv_data = {}
    for r in ranks:
        contrib, resid, noise = per_cell_contribs(r)
        R = np.stack([sum(contrib[k].values()) for k in CH])
        rv_data[r] = (R, resid, noise)
        per_net[r] = (contrib, resid, noise)
        if os.environ.get("DUMP_CONTRIB") == "1":
            np.savez(HERE / f"contrib_tensor_rank{r:04d}.npz",
                     **{f"{k}_L{l}": v for k in CH
                        for l, v in contrib[k].items()})
        # consistency check vs banked rvec
        try:
            rv = np.load(HERE / f"rvec_tensor_rank{r:04d}.npz")
            errs = [float(np.linalg.norm(R[j] - rv[k]) /
                          max(1e-30, np.linalg.norm(rv[k])))
                    for j, k in enumerate(CH)]
            print(f"r{r} replay-vs-banked rel err",
                  [f"{e:.2e}" for e in errs])
        except FileNotFoundError:
            pass
    # value map with per-net LOO betas (exactly the certified frame)
    cells = [(l, k) for k in CH for l in range(1, 16)]
    val_ip = {c: [] for c in cells}    # inner-product value share
    drop = {c: [] for c in cells}      # exact leave-cell-out drop
    caps = []
    for r in ranks:
        contrib, resid, noise = per_net[r]
        beta = loo_betas(rv_data, r)
        corr = sum(beta[j] * sum(contrib[k].values())
                   for j, k in enumerate(CH))
        before = float(resid @ resid) - noise
        after = float((resid - corr) @ (resid - corr)) - noise
        cap = 100 * (1 - after / before)
        caps.append(cap)
        for (l, k) in cells:
            j = CH.index(k)
            c_lk = beta[j] * contrib[k].get(l, np.zeros(1024))
            # inner-product (covariance) share of the captured energy
            val_ip[(l, k)].append(
                100 * float(2 * resid @ c_lk - c_lk @ (2 * corr - c_lk))
                / before)
            a2 = float((resid - (corr - c_lk))
                       @ (resid - (corr - c_lk))) - noise
            drop[(l, k)].append(cap - 100 * (1 - a2 / before))
    out = {"mean_capture_check": round(float(np.mean(caps)), 1),
           "cells": {}}
    for (l, k) in cells:
        out["cells"][f"L{l}_{k}"] = {
            "value_share_mean": round(float(np.mean(val_ip[(l, k)])), 2),
            "loo_drop_mean": round(float(np.mean(drop[(l, k)])), 2),
            "loo_drop_sd": round(float(np.std(drop[(l, k)])), 2)}
    ranked = sorted(out["cells"].items(),
                    key=lambda kv: -kv[1]["loo_drop_mean"])
    out["top_cells"] = [[k, v["loo_drop_mean"]] for k, v in ranked[:12]]
    (HERE / "CHAIN_VALUE_MAP.json").write_text(
        json.dumps(out, indent=1) + "\n")
    print(json.dumps({"capture_check": out["mean_capture_check"],
                      "top": out["top_cells"]}, indent=1))


if __name__ == "__main__":
    main()
