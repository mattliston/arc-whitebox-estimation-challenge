#!/usr/bin/env python3
"""TAG g19g-estimated-consumption-v1 (PHASE 2): consume the v4 engine's
ESTIMATED state deltas through the ceiling machinery with the FIXED
universal betas (k4 0.16, k3 0.33, v 0.82 - the 10-net LOO values from
PERFECT_STATE_CEILING; no fitting here). Deltas: k3hat/k4hat = v4 carried
minus host carried; vhat = the chain's dv_hat (dvhat_rank*.npz).
Output: corrected deployed L16 MSE per net vs the deployed baseline.
Usage: run_estimated_consumption.py <rank> [rank2 ...]"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_attribution import edg2_relu_mean

BETAS = {"k4": 0.16, "k3": 0.33, "v": 0.82}


def one_net(rank):
    lite = HERE / f"states_rank{rank:04d}_lite.npz"
    if lite.exists():
        st = dict(np.load(lite))
    else:
        d = np.load(HERE / f"states_rank{rank:04d}.npz")
        st = {k: d[k] for k in ("host_post", "host_m", "host_v",
                                 "host_k3", "host_k4")}
    s = np.load(HERE / f"sidecar_states_rank{rank:04d}.npz")
    raw20 = s["raw20"]
    tr = np.load(HERE / f"mps_truth_rank{rank:04d}_P8M.npz")
    dep = np.load(HERE / f"deployed_rank{rank:04d}.npy")
    import os
    supply = os.environ.get("SUPPLY", "tensor")
    if supply == "oracle":
        tp = dv = None
    else:
        tpn = "tensor_planes" if supply == "tensor" else "hub_planes"
        sfx = "" if supply == "tensor" else "_hub"
        tsx = os.environ.get("TP_SUFFIX", "")
        tp = np.load(HERE / f"{tpn}_rank{rank:04d}{tsx}.npz")
        dsx = os.environ.get("DV_SUFFIX", sfx + tsx)
        dv = np.load(HERE / f"dvhat_rank{rank:04d}{dsx}.npz")
    P2 = HERE.parent
    STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
             "sealed_target_blind_feature_store_v3")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    W = np.load((STORE / entry["path"]).parent / "weights.npy", mmap_mode="r")
    Wd = [np.asarray(W[l], dtype=np.float64) for l in range(16)]

    r = {k: np.zeros(1024) for k in BETAS}
    coll_on = os.environ.get("COLL_CHANNEL") == "1"
    if coll_on:
        r["coll"] = np.zeros(1024)
        zcv = np.load(HERE / f"host_cov_rank{rank:04d}_base.npz")
        from build_gate1_supplies import tracker
        K3S, _Ttrk = tracker(rank)
    for l in range(1, 16):
        i = l - 1
        hm, hv = st["host_m"][i], st["host_v"][i]
        hk3, hk4 = st["host_k3"][i], st["host_k4"][i]
        base = edg2_relu_mean(hm, hv, hk3, hk4)
        if supply == "truetangent":
            k3h = tr["mc_pre_k3"][l].astype(np.float64)
            k4h = tr["mc_pre_k4"][l].astype(np.float64)
            dvt = np.load(HERE / f"dvhat_rank{rank:04d}_true.npz")
            dvh = dvt[f"L{l}"].astype(np.float64) if f"L{l}" in dvt.files \
                else np.zeros(1024)
        elif supply == "oracle":
            k3h = tr["mc_pre_k3"][l].astype(np.float64)
            k4h = tr["mc_pre_k4"][l].astype(np.float64)
            dvh = tr["mc_pre_v"][l].astype(np.float64) - hv
        else:
            k3h = tp[f"k3u_{l}"].astype(np.float64) \
                if f"k3u_{l}" in tp.files else hk3
            k4h = tp[f"k4u_{l}"].astype(np.float64) \
                if f"k4u_{l}" in tp.files else hk4
            dvh = dv[f"L{l}"].astype(np.float64) if f"L{l}" in dv.files \
                else np.zeros(1024)
        dds = {"k4": edg2_relu_mean(hm, hv, hk3, k4h) - base,
               "k3": edg2_relu_mean(hm, hv, k3h, hk4) - base,
               "v": edg2_relu_mean(hm, np.maximum(hv + dvh, 1e-30), hk3,
                                   hk4) - base}
        if coll_on and 2 <= l <= 14 and f"pre_{l}" in tp.files \
                and l - 1 < zcv["cov"].shape[0]:
            # collective mean channel: dds = (dk3s/6) b^3 (-al pdf)/vperp
            mw = tr["mc_pre_m"][l].astype(np.float64) if False else None
            mcl = hm  # closure state frame (deploy-native)
            Ccl = zcv["cov"][l - 1].astype(np.float64)
            wv = mcl / max(np.linalg.norm(mcl), 1e-30)
            sig2 = max(wv @ Ccl @ wv, 1e-30)
            bv = (Ccl @ wv) / np.sqrt(sig2)
            vperp = np.maximum(np.diag(Ccl) - bv * bv, 1e-30)
            sdp = np.sqrt(vperp)
            alp = mcl / sdp
            pdfp = np.exp(-0.5 * alp * alp) / np.sqrt(2 * np.pi)
            impl = float((wv * wv) @ tp[f"pre_{l}"].astype(np.float64)
                         @ wv)
            dk3s = float(K3S.get(l, 0.0)) - impl
            dds["coll"] = (dk3s / 6.0) * (bv ** 3) * (-alp * pdfp) \
                / vperp
        elif coll_on:
            dds["coll"] = np.zeros(1024)
        phi = raw20[l, :, 5]
        for k in r:
            if os.environ.get("CONSUME") == "finalonly":
                r[k] = dds[k]      # deltas are already accumulated states
            else:
                r[k] = phi * (r[k] @ Wd[l]) + dds[k]
    if os.environ.get("DUMP_R") == "1":
        tag = os.environ.get(
            "RVEC_TAG", f"{supply}{os.environ.get('TP_SUFFIX', '')}")
        np.savez(HERE / f"rvec_{tag}_rank{rank:04d}.npz", **r)
    resid = tr["mc_post"][15] - dep[15]
    noise = float(np.sum(tr["mc_se"][15] ** 2))
    corr_vec = sum(BETAS[k] * r[k] for k in BETAS)
    # the coll channel enters via DUMP_R / LOO fitting only (no fixed
    # beta): excluded from the fixed-beta corr_vec on purpose
    before = (float(np.sum(resid ** 2)) - noise) / 1024
    after = (float(np.sum((resid - corr_vec) ** 2)) - noise) / 1024
    # per-channel ablation
    out = {"deployed_mse_nc": before, "corrected_mse_nc": after,
           "reduction_pct": round(100 * (1 - after / before), 1)}
    for k in BETAS:
        a1 = (float(np.sum((resid - BETAS[k] * r[k]) ** 2)) - noise) / 1024
        out[f"only_{k}"] = round(100 * (1 - a1 / before), 1)
    return out


def main():
    ranks = [int(a) for a in sys.argv[1:]] or [35]
    res = {}
    for rank in ranks:
        res[str(rank)] = one_net(rank)
        print(rank, json.dumps(res[str(rank)]))
    import os
    tag = (os.environ.get("SUPPLY", "tensor")
           + os.environ.get("TP_SUFFIX", "")
           + os.environ.get("DV_SUFFIX", ""))
    (HERE / f"ESTIMATED_CONSUMPTION_{tag}.json").write_text(
        json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
