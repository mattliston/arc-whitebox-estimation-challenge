#!/usr/bin/env python3
"""TAG h2h3-C1: numerical rank of the t0 endpoint adjoint.
For each source layer l, the t0-chain value of the K21 plane error is
  value(l) = <dK21_l, M_l>,
  M_l = diag(a2_l) [ sum_{L>l} B_{l->L} diag(g_L) B_{l->L}^T ] diag(b1_l)
  B_{l->L} = A_l A_{l+1} ... A_{L-1},  A_j = diag(e1_j) W_{j+1}
  g_L[c] = S_v(L)[c] * (resid back-transported through the phi/W
           consumption chain to layer L)[c]
Gates (prereg, LANE_h2_h3_endpoint_adjoint.md):
  C1: r(95% of |sum_l value|) <= 128
  sanity: sum_l value(l) tracks the banked t0/certified v-channel value.
Usage: run_c1_adjoint_rank.py <rank>"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_tangent_v2 import c1_std_factors


def main():
    rank = int(sys.argv[1])
    d = (np.load(HERE / f"states_rank{rank:04d}_lite.npz")
         if (HERE / f"states_rank{rank:04d}_lite.npz").exists()
         and not (HERE / f"states_rank{rank:04d}.npz").exists()
         else np.load(HERE / f"states_rank{rank:04d}.npz"))
    z = np.load(HERE / f"host_cov_rank{rank:04d}_base.npz")
    tp = np.load(HERE / f"tensor_planes_rank{rank:04d}.npz")
    s = np.load(HERE / f"sidecar_states_rank{rank:04d}.npz")
    tr = np.load(HERE / f"mps_truth_rank{rank:04d}_P8M.npz")
    dep = np.load(HERE / f"deployed_rank{rank:04d}.npy")
    raw20 = s["raw20"]
    P2 = HERE.parent
    STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
             "sealed_target_blind_feature_store_v3")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    W = np.load((STORE / entry["path"]).parent / "weights.npy",
                mmap_mode="r")
    Wd = [np.asarray(W[k], dtype=np.float64) for k in range(16)]
    resid = (tr["mc_post"][15] - dep[15]).astype(np.float64)

    # factors per layer
    e1 = {}
    a2 = {}
    Sv = {}
    for l in range(1, 16):
        i = l - 1
        m_, v_ = d["host_m"][i].astype(np.float64), \
            d["host_v"][i].astype(np.float64)
        c1, e, sd = c1_std_factors(m_, v_)
        e1[l] = c1[1] / sd
        a2[l] = c1[2] / sd ** 2
        al = m_ / sd
        Sv[l] = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi) / (2 * sd)

    # back-transport resid through the consumption chain:
    # forward: r <- phi_l * (r @ Wd[l]) + dds_l  (l = 1..15)
    # adjoint g at layer L: g_L = phi_L * (Wd[L+1] @ (phi_{L+1} * (...)))
    g = {}
    acc = resid.copy()
    for L in range(15, 0, -1):
        # acc currently = d(value)/d(dds_L) BEFORE this layer's transport
        g[L] = acc.copy()
        phi = raw20[L, :, 5].astype(np.float64) if L < 16 else None
        acc = Wd[L] @ (raw20[L, :, 5].astype(np.float64) * acc)
    # weight for dv_hat at layer L: gamma_L = Sv(L) * g[L]
    # (dds_v(L) ~= Sv(L) o dv_hat(L))

    # A_j = diag(e1_j) W_{j+1}; accumulate M_core(l) = sum_{L>l}
    #   B diag(gamma_L) B^T backward: start at l=14
    out = {}
    Mcore = None
    values = {}
    for l in range(14, 0, -1):
        # dv_hat lives at PRE layer l+1: gamma uses layer l+1 factors
        A = e1[l][:, None] * Wd[l + 1]
        if Mcore is None:
            core = np.zeros((1024, 1024))
        else:
            core = Mcore
        gl1 = Sv[l + 1] * g[l + 1]
        core = core + np.diag(gl1)
        Mcore = A @ core @ A.T
        # M_l for source-layer l reads dK21_l:
        M = (a2[l][:, None] * Mcore) * e1[l][None, :]
        if f"pre_{l}" in tp.files:
            dK = (tp[f"pre_{l}"].astype(np.float64)
                  - z["k21"][i := l - 1].astype(np.float64))
            val = float(np.sum(dK * M))
            U, S, Vt = np.linalg.svd(M)
            # value captured by rank-r truncation
            proj = np.array([S[r0] * float(U[:, r0] @ dK @ Vt[r0])
                             for r0 in range(256)])
            cums = np.cumsum(proj)
            def r_at(frac):
                tgt = frac * val
                hits = np.where(np.abs(cums - val) <= (1 - frac)
                                * abs(val))[0]
                return int(hits[0]) + 1 if len(hits) else -1
            values[l] = val
            out[l] = {"value": val,
                      "spec_r95_frob": int(np.searchsorted(
                          np.cumsum(S ** 2) / np.sum(S ** 2), 0.95)) + 1,
                      "r90_value": r_at(0.90), "r95_value": r_at(0.95),
                      "r99_value": r_at(0.99)}
            print(rank, f"L{l}", json.dumps(out[l]))
    tot = sum(values.values())
    noise = float(np.sum(tr["mc_se"][15] ** 2))
    before = float(resid @ resid) - noise
    print(json.dumps({"sum_value": tot,
                      "resid_energy": before,
                      "implied_capture_pct_linear":
                          round(100 * 2 * tot / before, 2)}))
    (HERE / f"C1_ADJOINT_RANK_rank{rank:04d}.json").write_text(
        json.dumps({"layers": out, "sum_value": tot}, indent=1) + "\n")


if __name__ == "__main__":
    main()
