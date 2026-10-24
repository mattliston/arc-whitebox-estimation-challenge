#!/usr/bin/env python3
"""TAG g19carrier-t0-chain: does the rank-1 (t=0) Mehler truncation of
the tangent chain keep the certified value?  Chain:
  eps_{l+1} = A_l^T [eps_l + inj_l] A_l,   A_l = diag(e1_l) W_{l+1}
  inj_l     = (dK21_l o a2 b1^T)/2 + transpose  (+ t=0 pair channels)
vs the full Mehler chain (run_tangent_v2, dv_corr ~0.8 @ L15 r35).
If YES: the value read of dK21_l is a sum of bilinear plane reads in
the propagated basis => trackable by R scalar-ridge closures (the K3S
tracker generalization) at ~free bill.  Also reports the t=0 chain's
dv_hat correlation WITH the full-chain dv_hat (agreement) and vs truth.
Usage: run_t0_chain_check.py <rank> [pairs01]"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_tangent_v2 import c1_std_factors

PAIRS = len(sys.argv) > 2 and sys.argv[2] == "1"


def main():
    rank = int(sys.argv[1])
    d = (np.load(HERE / f"states_rank{rank:04d}_lite.npz")
         if (HERE / f"states_rank{rank:04d}_lite.npz").exists()
         and not (HERE / f"states_rank{rank:04d}.npz").exists()
         else np.load(HERE / f"states_rank{rank:04d}.npz"))
    z = np.load(HERE / f"host_cov_rank{rank:04d}_base.npz")
    tp = np.load(HERE / f"tensor_planes_rank{rank:04d}.npz")
    T = np.load(HERE / f"mps_cov2_rank{rank:04d}.npz")
    dvfull = np.load(HERE / f"dvhat_rank{rank:04d}.npz")
    P2 = HERE.parent
    STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
             "sealed_target_blind_feature_store_v3")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    W = np.load((STORE / entry["path"]).parent / "weights.npy",
                mmap_mode="r")
    eps = np.zeros((1024, 1024))
    out = {}
    for l in range(1, 15):
        i = l - 1
        m_, v_ = d["host_m"][i], d["host_v"][i]
        c1, e, sd = c1_std_factors(m_, v_)
        e1 = c1[1] / sd                      # dM11/drho t=0 std factor
        a2 = (c1[2] / sd ** 2)               # G(2,1) left factor
        b1 = e1                              # G(2,1) right factor
        inj = np.zeros((1024, 1024))
        if f"pre_{l}" in tp.files:
            dK21 = (tp[f"pre_{l}"].astype(np.float64)
                    - z["k21"][i].astype(np.float64))
            inj = (dK21 * np.outer(a2, b1)) / 2.0
            inj = inj + inj.T
        if PAIRS and f"K22_{l}" in tp.files \
                and i < z["k22"].shape[0]:
            dK22 = (tp[f"K22_{l}"].astype(np.float64)
                    - z["k22"][i].astype(np.float64))
            dK31 = (tp[f"K31_{l}"].astype(np.float64)
                    - z["k31"][i].astype(np.float64))
            np.fill_diagonal(dK22, 0.0)
            np.fill_diagonal(dK31, 0.0)
            a3 = c1[3] / sd ** 3
            i22 = (dK22 / 4.0) * np.outer(a2, a2)
            i31 = (dK31 / 6.0) * np.outer(a3, b1)
            inj = inj + i22 + i31 + i31.T
        A = np.asarray(W[l + 1], dtype=np.float64) * e1[:, None]
        eps = A.T @ (eps + inj) @ A
        dv_hat = np.diag(eps)
        dv_true = T["mc_pre_v"][l + 1] - d["host_v"][l]
        dv_f = (dvfull[f"L{l+1}"].astype(np.float64)
                if f"L{l+1}" in dvfull.files else None)

        def cc(a, b):
            return float(np.sum(a * b)
                         / np.sqrt(max(np.sum(a * a) * np.sum(b * b),
                                       1e-300)))
        out[l + 1] = {
            "dv_corr_truth": round(cc(dv_hat, dv_true), 3),
            "dv_beta": round(float(np.sum(dv_hat * dv_true)
                                   / max(np.sum(dv_hat ** 2), 1e-300)),
                             3),
            "corr_with_fullchain": (round(cc(dv_hat, dv_f), 3)
                                    if dv_f is not None else None)}
        if l + 1 in (9, 12, 14, 15):
            print(rank, f"L{l+1}", json.dumps(out[l + 1]))
    (HERE / f"T0_CHAIN_rank{rank:04d}{'_pairs' if PAIRS else ''}.json"
     ).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
