#!/usr/bin/env python3
"""TAG g30-cubature-proto (PHASE 2, 1024x16): offline prototype of the spherical-cubature estimator class
(skibidi_toilet's phase-1 method, whest_phase1_writeup_skibidi_toilet.pdf) at the phase-2 shape.
  E_x[M(x)] = E[chi_n] * E_{u~S^{n-1}}[M(u)]   (bias-free ReLU nets are positively homogeneous)
Design: b orthonormal bases of R^1024 with antipodal closure (2048 b points; each +-basis is a tight spherical
3-design; Kerdock real MUBs would make the union a 4-design at b = 513 - not affordable at this shape).
Snaps: exact layer-1 mean (||w||/sqrt(2 pi)) carried as a rank-one shift; exact layer-2 pre-activation variance
(ReLU arc-cosine kernel) imposed by rescaling the centred rows.  Grades the final-layer MSE against the 32M truth
(noise-corrected) and reports V = MSE x k (per-point variance), which with the per-point cost c gives the
adjusted score V c / B (flat in k).  Usage: cubature_proto.py <rank> <bases> [design: ortho|gauss] [snaps 0/1]"""
import json, sys, time
from math import lgamma, exp, sqrt, pi
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
G19 = P2 / "public_g19_attribution_v1"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"

rank = int(sys.argv[1]); B = int(sys.argv[2]); design = sys.argv[3] if len(sys.argv) > 3 else "ortho"
snaps = int(sys.argv[4]) if len(sys.argv) > 4 else 1
n = 1024


def weights_of(rank):
    man = json.loads((STORE / "manifest.json").read_text()); e = man["roles"]["observer_fit"]["row_manifests"][rank]
    W = np.load((STORE / e["path"]).parent / "weights.npy", mmap_mode="r")
    return [np.asarray(W[l], dtype=np.float32) for l in range(16)], int(json.loads((STORE / e["path"]).read_text())["mlp_seed"])


def arccos_kernel(W0):
    """exact Cov-free second moment E[relu(u.x) relu(v.x)] for x ~ N(0, I): (|u||v|/2pi)(sin t + (pi - t) cos t)"""
    G = W0.T.astype(np.float64) @ W0.astype(np.float64)              # <w_a, w_b>
    nrm = np.sqrt(np.diag(G)); cos = np.clip(G / np.outer(nrm, nrm), -1.0, 1.0); t = np.arccos(cos)
    return (np.outer(nrm, nrm) / (2 * pi)) * (np.sin(t) + (pi - t) * cos)


def main():
    Ws, seed = weights_of(rank)
    rng = np.random.default_rng(seed ^ 0x5EED)
    Er = sqrt(2) * exp(lgamma((n + 1) / 2) - lgamma(n / 2))         # E[chi_n]
    k = 2 * B * n
    t0 = time.time()
    if design == "gauss":
        X = rng.standard_normal((k, n)).astype(np.float32)             # plain MC at the same k (reference)
    else:
        blocks = []
        for _ in range(B):
            Q, _r = np.linalg.qr(rng.standard_normal((n, n)))
            Q = Q * np.sign(np.diag(_r))[None, :]
            blocks.append(Q.T); blocks.append(-Q.T)                     # rows = +-basis vectors
        X = (Er * np.concatenate(blocks, 0)).astype(np.float32)
    means = []
    W0 = Ws[0]
    Z = X @ W0                                                          # layer-1 pre-activations (k x n)
    H = np.maximum(Z, 0.0)
    m1_emp = H.mean(0).astype(np.float64)
    if snaps and design != "gauss":
        mu1 = np.linalg.norm(W0.astype(np.float64), axis=0) / sqrt(2 * pi)   # exact layer-1 mean
        H += (mu1 - m1_emp).astype(np.float32)[None, :]                 # rank-one shift so the design mean is exact
        means.append(mu1)
    else:
        means.append(H.mean(0).astype(np.float64))
    for l in range(1, 16):
        Z = H @ Ws[l]
        if l == 1 and snaps and design != "gauss":
            K = arccos_kernel(W0)                                       # E[h1 h1^T] exact
            W1 = Ws[1].astype(np.float64)
            m2 = means[0] @ W1                                          # exact layer-2 pre mean
            v2 = np.einsum("aj,ab,bj->j", W1, K, W1) - m2 ** 2          # exact layer-2 pre variance
            zc = Z.astype(np.float64) - Z.mean(0)[None, :]
            v_emp = (zc ** 2).mean(0)
            Z = (m2[None, :] + zc * np.sqrt(np.maximum(v2, 1e-30) / np.maximum(v_emp, 1e-30))[None, :]).astype(np.float32)
        H = np.maximum(Z, 0.0)
        means.append(H.mean(0).astype(np.float64))
    means = np.stack(means)
    T = np.load(G19 / f"mps_truth_rank{rank:04d}_P32M.npz")
    tr = T["mc_post"].astype(np.float64); se = T["mc_se"].astype(np.float64)
    out = {"rank": rank, "design": design, "bases": B, "k": k, "snaps": snaps, "wall_s": round(time.time() - t0, 1), "layers": {}}
    for l in (0, 1, 2, 4, 8, 12, 15):
        mse = float(np.mean((means[l] - tr[l]) ** 2) - np.mean(se[l] ** 2))
        out["layers"][l] = {"mse_nc": mse, "V_per_point": mse * k}
    fin = out["layers"][15]
    c_plain = 15 * 2 * n * n                                            # per-row propagation flops, no pruning/Winograd
    out["final_mse_nc"] = fin["mse_nc"]; out["V_final"] = fin["V_per_point"]
    out["adj_at_plain_cost"] = fin["V_per_point"] * c_plain / 2 ** 41
    out["adj_at_0p59_cost"] = fin["V_per_point"] * c_plain * 0.59 / 2 ** 41
    print(json.dumps(out, indent=1), flush=True)
    (HERE / f"CUBATURE_PROTO_rank{rank:04d}_{design}_b{B}_s{snaps}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
