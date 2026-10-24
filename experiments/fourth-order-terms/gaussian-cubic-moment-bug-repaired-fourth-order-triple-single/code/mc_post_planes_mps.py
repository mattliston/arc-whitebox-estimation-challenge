#!/usr/bin/env python3
"""TAG g29-mc-post-planes-mps: truth POST-activation fourth-order planes for chosen layers
(same seed law and draw stream as mc_truth_mps.py: PCG64DXSM(seed ^ SEED_XOR), batches of 131072).
For x = relu(h_l) per layer l in LAYERS, accumulates in float64 the pair moments
  S11 = x^T x, S21 = (x^2)^T x, S31 = (x^3)^T x, S22 = (x^2)^T (x^2), and s1..s4 per unit,
and returns the connected cumulant planes (u doubled / w single):
  K21post[u,w] = kappa3(x_u, x_u, x_w),  K31post[u,w] = kappa4(x_u,x_u,x_u,x_w),  K22post[u,w] = kappa4(x_u,x_u,x_w,x_w)
plus post m, v, k3, k4 per unit.  Also the PRE planes of the same layers as a self-check against
mps_planes_rank*.npz (K21/K22/K31 of h_l).  Usage: mc_post_planes_mps.py <rank> <P_millions> <layers...>
Output: post_planes_rank{rank}_P{P}M.npz with keys K21post_l, K22post_l, K31post_l, postm_l, postv_l, postk3_l, postk4_l."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
G19 = HERE.parent / "public_g19_attribution_v1"
STORE = (G19.parent / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")
SEED_XOR = int(os.environ.get("SEED_XOR_OVERRIDE", "0x47414D454F4E5F31"), 16)
BATCH = 131_072


def cumulant_planes(s1, s2, s3, s4, S11, S21, S31, S22, P):
    m1, m2, m3, m4 = s1 / P, s2 / P, s3 / P, s4 / P
    M11, M21, M31, M22 = S11 / P, S21 / P, S31 / P, S22 / P
    mu = m1
    # central pair moments
    C11 = M11 - np.outer(mu, mu)                                   # cov
    C21 = (M21 - 2 * mu[:, None] * M11 - m2[:, None] * mu[None, :]
           + 2 * np.outer(mu ** 2, mu))                             # E[(xu-mu)^2 (xw-mw)]
    C31 = (M31 - 3 * mu[:, None] * M21 + 3 * (mu ** 2)[:, None] * M11
           - m3[:, None] * mu[None, :] + 3 * (mu * m2)[:, None] * mu[None, :]
           - 3 * np.outer(mu ** 3, mu))                             # E[(xu-mu)^3 (xw-mw)]
    C22 = (M22 - 2 * mu[None, :] * M21 - 2 * mu[:, None] * M21.T
           + np.outer(m2, mu ** 2) + np.outer(mu ** 2, m2)
           + 4 * np.outer(mu, mu) * M11 - 3 * np.outer(mu ** 2, mu ** 2))   # E[(xu-mu)^2 (xw-mw)^2]
    v = np.diag(C11).copy()
    K21 = C21
    K31 = C31 - 3 * v[:, None] * C11
    K22 = C22 - np.outer(v, v) - 2 * C11 ** 2
    k3 = m3 - 3 * mu * m2 + 2 * mu ** 3
    k4 = m4 - 4 * mu * m3 - 3 * m2 ** 2 + 12 * mu ** 2 * m2 - 6 * mu ** 4
    return mu, v, k3, k4, K21, K22, K31


def main():
    rank = int(sys.argv[1]); p_samples = int(float(sys.argv[2]) * 1e6)
    layers = [int(a) for a in sys.argv[3:]]
    dev = torch.device("mps")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    rmp = STORE / entry["path"]; row = json.loads(rmp.read_text())
    weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
    w_gpu = [torch.from_numpy(np.ascontiguousarray(weights[l], dtype=np.float32)).to(dev) for l in range(16)]
    rng = np.random.Generator(np.random.PCG64DXSM((int(row["mlp_seed"]) ^ SEED_XOR) & ((1 << 64) - 1)))
    acc = {l: {"post": [np.zeros(1024) for _ in range(4)] + [np.zeros((1024, 1024)) for _ in range(4)],
               "pre": [np.zeros(1024) for _ in range(4)] + [np.zeros((1024, 1024)) for _ in range(4)]} for l in layers}
    done = 0; t0 = time.time(); lmax = max(layers)
    while done < p_samples:
        b = min(BATCH, p_samples - done)
        x = torch.from_numpy(rng.standard_normal(size=(b, 1024), dtype=np.float32)).to(dev)
        for l in range(lmax + 1):
            h = x @ w_gpu[l]
            xp = torch.clamp(h, min=0.0)
            if l in layers:
                for key, z in (("pre", h), ("post", xp)):
                    z2 = z * z; z3 = z2 * z
                    a = acc[l][key]
                    a[0] += z.sum(0).cpu().numpy().astype(np.float64); a[1] += z2.sum(0).cpu().numpy().astype(np.float64)
                    a[2] += z3.sum(0).cpu().numpy().astype(np.float64); a[3] += (z2 * z2).sum(0).cpu().numpy().astype(np.float64)
                    a[4] += (z.T @ z).cpu().numpy().astype(np.float64); a[5] += (z2.T @ z).cpu().numpy().astype(np.float64)
                    a[6] += (z3.T @ z).cpu().numpy().astype(np.float64); a[7] += (z2.T @ z2).cpu().numpy().astype(np.float64)
            x = xp
        done += b
        if done % (BATCH * 8) == 0:
            print(json.dumps({"samples": done, "rate_per_s": int(done / (time.time() - t0))}), flush=True)
    P = float(p_samples); save = {"P": p_samples, "rank": rank}
    for l in layers:
        for key in ("post", "pre"):
            mu, v, k3, k4, K21, K22, K31 = cumulant_planes(*acc[l][key], P)
            tag = "post" if key == "post" else "pre"
            save[f"K21{tag}_{l}"] = K21.astype(np.float32); save[f"K22{tag}_{l}"] = K22.astype(np.float32); save[f"K31{tag}_{l}"] = K31.astype(np.float32)
            save[f"{tag}m_{l}"] = mu; save[f"{tag}v_{l}"] = v; save[f"{tag}k3_{l}"] = k3; save[f"{tag}k4_{l}"] = k4
    out = HERE / f"post_planes_rank{rank:04d}_P{p_samples // 1_000_000}M.npz"
    np.savez(out, **save)
    print(json.dumps({"done": p_samples, "wall_s": round(time.time() - t0, 1), "out": str(out)}))


if __name__ == "__main__":
    main()
