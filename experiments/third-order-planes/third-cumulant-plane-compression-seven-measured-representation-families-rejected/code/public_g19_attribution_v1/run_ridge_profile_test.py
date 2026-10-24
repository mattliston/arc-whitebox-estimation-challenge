#!/usr/bin/env python3
"""TAG g19carrier-ridge-profile: can the value-basis reads of dK21 be
reconstructed from RIDGE-CONDITIONAL structure?  For value ridges
(columns of the t0-chain conjugation product B_l = A_l...A_14, the
deploy-known basis), the read is u^T dK21 v with u=a2 o b, v=b1 o b.
Identity: (dK21 v)_i = delta cum(z_i, z_i, y), y = v^T z.  Ridge model:
  p_i ~= c1 beta_i^2 + c2 beta_i (+ c3 v_i + c4 v_i k3u_i)
with beta = C v / sigma_y^2 the host conditional slope (deploy-known),
c's free per (l, ridge) [the amplitudes a tracker would carry].
Reports per layer: median profile R2 (2- and 4-shape), and the R2 of
the value READ scalars u^T p across ridges. High R2 => carrier =
R scalar trackers + host second-order state.  Usage: <rank>"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_tangent_v2 import c1_std_factors

NK = 24        # ridges sampled per layer (fixed stride over columns)


def main():
    rank = int(sys.argv[1])
    d = (np.load(HERE / f"states_rank{rank:04d}_lite.npz")
         if (HERE / f"states_rank{rank:04d}_lite.npz").exists()
         and not (HERE / f"states_rank{rank:04d}.npz").exists()
         else np.load(HERE / f"states_rank{rank:04d}.npz"))
    z = np.load(HERE / f"host_cov_rank{rank:04d}_base.npz")
    tp = np.load(HERE / f"tensor_planes_rank{rank:04d}.npz")
    P2 = HERE.parent
    STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
             "sealed_target_blind_feature_store_v3")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    W = np.load((STORE / entry["path"]).parent / "weights.npy",
                mmap_mode="r")
    # backward conjugation products: Q_l = A_l Q_{l+1}, A_l = diag(e1) W
    e1s = {}
    for l in range(1, 15):
        i = l - 1
        c1, e, sd = c1_std_factors(d["host_m"][i], d["host_v"][i])
        e1s[l] = (c1[1] / sd, c1[2] / sd ** 2)
    Q = np.eye(1024)
    Qs = {}
    for l in range(14, 0, -1):
        A = np.asarray(W[l + 1], dtype=np.float64) * e1s[l][0][:, None]
        Q = A @ Q
        Qs[l] = Q.copy()
    ks = np.arange(0, 1024, 1024 // NK)[:NK]
    out = {}
    for l in range(4, 14):
        i = l - 1
        if f"pre_{l}" not in tp.files:
            continue
        dK = (tp[f"pre_{l}"].astype(np.float64)
              - z["k21"][i].astype(np.float64))
        C = z["cov"][i].astype(np.float64)
        e1, a2 = e1s[l]
        hk3 = d["host_k3"][i].astype(np.float64)
        B = Qs[l][:, ks]                        # (1024, NK) ridge cols
        r2p2, r2p4, reads_t, reads_m = [], [], [], []
        for a in range(NK):
            b = B[:, a]
            bn = float(np.linalg.norm(b))
            if bn < 1e-30:
                continue
            b = b / bn
            u = a2 * b
            v = e1 * b
            p = dK @ v                          # true delta profile
            sy2 = max(float(v @ C @ v), 1e-30)
            beta = (C @ v) / sy2
            X2 = np.stack([beta ** 2, beta], 1)
            X4 = np.stack([beta ** 2, beta, v, v * hk3], 1)

            def r2(X):
                c, *_ = np.linalg.lstsq(X, p, rcond=None)
                res = p - X @ c
                return (1 - float(np.sum(res ** 2))
                        / max(float(np.sum(p ** 2)), 1e-300), X @ c)
            q2, fit2 = r2(X2)
            q4, fit4 = r2(X4)
            r2p2.append(q2)
            r2p4.append(q4)
            reads_t.append(float(u @ p))
            reads_m.append(float(u @ fit4))
        rt = np.array(reads_t)
        rm = np.array(reads_m)
        read_r2 = 1 - float(np.sum((rt - rm) ** 2)) \
            / max(float(np.sum(rt ** 2)), 1e-300)
        out[l] = {"profile_r2_2shape_med": round(float(
            np.median(r2p2)), 3),
            "profile_r2_4shape_med": round(float(np.median(r2p4)), 3),
            "read_r2_4shape": round(read_r2, 3)}
        print(rank, f"L{l}", json.dumps(out[l]))
    (HERE / f"RIDGE_PROFILE_rank{rank:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
