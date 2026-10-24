#!/usr/bin/env python3
"""TAG g29-plane-feed-plane-read-v1: the feed's contribution to the NEXT layer's
pre-plane K21[m,n] = T_mmn (its diagonal = the k3 read), GEMM-class.

Canonical (unsymmetrized) feed at layer l, spectator on the doubled index u
(pattern (tuw, tus, tws) with tws = 0):  G_uws = 1/2 cf K~_uw d_u b_w rho^tus_us a_s,
K~ = K21pre ∘ rho^tuw (K diagonal zeroed).  The symmetrized tensor S = sum over the
6 orientations; next-layer plane read  P[m,n] = sum_ijk S_ijk W_im W_jm W_kn = 2 (A + B + C)
with n placed on s (A), on w (B), on u (C); s != w exclusion handled exactly:
  S_ = rho^t (W∘a),  G_ = K~ (W∘b),  H_ = (K~∘rho^t)^T (W∘d),  J_ = (K~∘rho^t)(W²∘b∘a)
  A = (W∘d∘G_)^T S_  −  (H_∘W∘b∘a)^T W
  B = (W∘d∘S_)^T G_  −  (H_∘W∘b∘a)^T W
  C = (S_∘G_)^T (W∘d) −  J_^T (W∘d)
The spectator-on-w pattern (tus = 0, tws = t) is the same formula with (K~ᵀ, b, d).
Certificates: P's diagonal must equal the k3 read of plane_feed_oracle (same patterns);
dense n=20 brute force in cert_plane_read_small.py.
Usage: plane_feed_plane_read.py <rank> <bank_dir>  -> feed_plane_inject_rank{rank}.npz
(keys dK21_{L} for read layers L = 2..15, float32 1024x1024, = the full feed: plane + diagonal k3)
plus PLANE_READ_rank{rank}.json with the off-diagonal comparison against the 8M truth plane residual.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plane_feed_oracle as pfo

G19 = pfo.G19
PATS = [(0, 1, 0), (0, 0, 1), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1)]   # one-sided, bond <= 2
if os.environ.get("FEED_PATS") == "010":
    PATS = [(0, 1, 0)]                                                        # leading pattern only (bill study)
OUT_TAG = os.environ.get("FEED_OUT_TAG", "")
DEVICE = os.environ.get("FEED_DEVICE", "mps")


def plane_read_first(W, Kt, dcol, bcol, acol, rho_t, t, cf):
    """P for spectator on the FIRST index of Kt (tws = 0, tus = t). W: (in, out). Returns (out, out)."""
    Wd, Wb, Wa = W * dcol[:, None], W * bcol[:, None], W * acol[:, None]
    S_ = rho_t @ Wa                                   # [u, n]
    G_ = Kt @ Wb                                      # [u, m]
    Kr = Kt * rho_t                                   # K~ ∘ rho^t (rho_t already the t-th power)
    H_ = Kr.T @ Wd                                    # [w, m]
    J_ = Kr @ (W * W * (bcol * acol)[:, None])        # [u, m]
    excl = (H_ * W * (bcol * acol)[:, None]).T @ W    # [m, n]
    A = (Wd * G_).T @ S_ - excl
    B = (Wd * S_).T @ G_ - excl
    C = (S_ * G_).T @ Wd - J_.T @ Wd
    return 2.0 * 0.5 * cf * (A + B + C)


def main():
    rank = int(sys.argv[1])
    bank_dir = Path(sys.argv[2])
    dev = torch.device(DEVICE)
    if DEVICE == "cpu":
        torch.set_num_threads(4)
    weights = pfo.load_weights(rank)
    planes_truth = np.load(G19 / f"mps_planes_rank{rank:04d}.npz")
    k3vec = np.load(HERE / f"plane_feed_vectors_rank{rank:04d}.npz")
    f32 = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)
    out = {"tag": "g29-plane-feed-plane-read-v1", "rank": rank, "patterns": PATS, "layers": {}}
    inject = {}
    t0 = time.time()
    for l in range(1, 15):
        fb, fp = bank_dir / f"bank_L{l}.npz", bank_dir / f"plane_L{l}.npz"
        if not (fb.is_file() and fp.is_file()):
            continue
        z = np.load(fb)
        K = np.load(fp)["pre_k21"].astype(np.float64)
        np.fill_diagonal(K, 0.0)
        m_ = z["pre_m"].astype(np.float64)
        v_ = np.maximum(z["pre_v"].astype(np.float64), 1e-30)
        sd, d, b, a = pfo.columns(m_, v_, pfo.TMAX)
        rho = z["pre_cov"].astype(np.float64) / np.outer(sd, sd)
        np.fill_diagonal(rho, 0.0)
        W = f32(np.asarray(weights[l + 1]))
        Kt, rho_t = f32(K), f32(rho)
        rho_pows = {1: rho_t, 2: rho_t * rho_t}
        dT, bT, aT = [f32(x) for x in d], [f32(x) for x in b], [f32(x) for x in a]
        P = torch.zeros((1024, 1024), dtype=torch.float32, device=dev)
        for (tuw, tus, tws) in PATS:
            cf = 1.0 / (np.math.factorial(tuw) * np.math.factorial(tus) * np.math.factorial(tws)) \
                if hasattr(np, "math") else 1.0 / (__import__("math").factorial(tuw) * __import__("math").factorial(tus) * __import__("math").factorial(tws))
            Ktil = Kt * rho_pows[tuw] if tuw else Kt
            if tws == 0:
                P += plane_read_first(W, Ktil, dT[tuw + tus], bT[tuw], aT[tus], rho_pows[tus], tus, cf)
            else:
                P += plane_read_first(W, Ktil.T, bT[tuw + tws], dT[tuw], aT[tws], rho_pows[tws], tws, cf)
        if DEVICE == "mps":
            torch.mps.synchronize()
        Pn = P.cpu().double().numpy()
        L = l + 1
        # certificate 1: diagonal == k3 read (bond<=2 one-sided patterns) from the k3 oracle
        k3ref = k3vec[f"feed_bond2_{L}"].astype(np.float64)
        diag_err = float(np.linalg.norm(np.diag(Pn) - k3ref) / np.linalg.norm(k3ref))
        # truth plane residual at read layer L (8M planes; card pre-plane at layer_index L)
        rec = {"diag_vs_k3read_rel_err": diag_err, "rms_P_offdiag": float(np.sqrt(np.mean(np.square(Pn - np.diag(np.diag(Pn))))))}
        fpL = bank_dir / f"plane_L{L}.npz"
        if f"K21_{L}" in planes_truth.files and fpL.is_file():
            Kt_truth = planes_truth[f"K21_{L}"].astype(np.float64)
            Kc = np.load(fpL)["pre_k21"].astype(np.float64)
            R = Kt_truth - Kc
            off = ~np.eye(1024, dtype=bool)
            p, r = Pn[off], R[off]
            beta = float(p @ r) / float(p @ p)
            corr = float(p @ r) / np.sqrt(float(p @ p) * float(r @ r))
            rec.update({"rms_truth_plane_offdiag": float(np.sqrt(np.mean(Kt_truth[off] ** 2))),
                        "rms_resid_offdiag": float(np.sqrt(np.mean(r ** 2))),
                        "offdiag_beta": beta, "offdiag_corr": corr,
                        "offdiag_energy_explained_at_1": float(1 - np.mean((r - p) ** 2) / np.mean(r ** 2)),
                        "offdiag_energy_explained_at_beta": float(1 - np.mean((r - beta * p) ** 2) / np.mean(r ** 2))})
        out["layers"][str(L)] = rec
        inject[f"dK21_{L}"] = Pn.astype(np.float32)
        print(json.dumps({"read_layer": L, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()},
                          "wall_s": round(time.time() - t0, 1)}), flush=True)
    np.savez(HERE / f"feed_plane_inject_rank{rank:04d}{OUT_TAG}.npz", **inject)
    (HERE / f"PLANE_READ_rank{rank:04d}{OUT_TAG}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"done": True, "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
