#!/usr/bin/env python3
"""TAG g19g-mc-probe-tensor: PROJECTED truth of the distinct third-moment
tensor at one layer — the affordable form of the one-layer truth-tensor
instrument. For probe pairs (u,v):
    S_true(u,v)[k] = E[(dz^T u)(dz^T v) dz_k]      (dz = z - true mean)
measured by MC at the pre-frame of layer L*, batched on MPS. Probes are
built from the mode analysis (k3^2/v row, m col, deficit SVD u1/v1, k4,
v, g) plus random controls. Output: probes + S_true per pair.
Usage: mc_probe_tensor.py <rank> <layer> <samples_millions>"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_attribution import load_host  # noqa: F401  (path setup)

STORE = (HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")


def build_probes(rank, layer):
    T = np.load(HERE / f"mps_cov2_rank{rank:04d}.npz")
    Pl = np.load(HERE / f"mps_planes_rank{rank:04d}.npz")
    hon = np.load(HERE / f"tensor_planes_rank{rank:04d}_hon.npz")
    m = T["mc_pre_m"][layer].astype(np.float64)
    v = T["mc_pre_v"][layer].astype(np.float64)
    k3 = T["mc_pre_k3"][layer].astype(np.float64)
    k4 = T["mc_pre_k4"][layer].astype(np.float64)
    err = (hon[f"pre_{layer}"].astype(np.float64)
           - Pl[f"K21_{layer}"].astype(np.float64))
    U, S, Vt = np.linalg.svd(err)
    sd = np.sqrt(np.maximum(v, 1e-30))
    al = m / sd
    g = 0.5 * (1 + np.vectorize(__import__("math").erf)(al / np.sqrt(2)))
    probes = {
        "mode_row": k3 * k3 / v, "mode_col": m, "err_u1": U[:, 0],
        "err_v1": Vt[0], "err_u2": U[:, 1], "err_v2": Vt[1],
        "k3": k3, "k4": k4, "v": v, "g": g,
    }
    rng = np.random.default_rng(11)
    for i in range(4):
        probes[f"rand{i}"] = rng.standard_normal(1024)
    for k in probes:
        probes[k] = probes[k] / np.linalg.norm(probes[k])
    return probes, m


def main():
    rank = int(sys.argv[1]) if len(sys.argv) > 1 else 35
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    msamp = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    P = int(msamp * 1e6)
    dev = torch.device("mps" if torch.backends.mps.is_available()
                       else "cpu")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    W = np.load((STORE / entry["path"]).parent / "weights.npy",
                mmap_mode="r")
    w32 = [torch.from_numpy(np.ascontiguousarray(W[l], dtype=np.float32)
                            ).to(dev) for l in range(layer + 1)]
    probes, true_m = build_probes(rank, layer)
    names = sorted(probes)
    Pm = torch.from_numpy(np.stack([probes[k] for k in names]
                                   ).astype(np.float32).T).to(dev)  # N x Q
    mu = torch.from_numpy(true_m.astype(np.float32)).to(dev)
    Q = len(names)
    pairs = [(i, j) for i in range(Q) for j in range(i, Q)]
    pi = torch.tensor([p[0] for p in pairs], device=dev)
    pj = torch.tensor([p[1] for p in pairs], device=dev)
    acc = torch.zeros(len(pairs), 1024, dtype=torch.float32, device=dev)
    BATCH = 65536
    gen = torch.Generator(device="cpu").manual_seed(90210 + rank)
    done = 0
    import time
    t0 = time.time()
    while done < P:
        b = min(BATCH, P - done)
        x = torch.randn(b, 1024, generator=gen).to(dev)
        for l in range(layer):
            x = torch.relu(x @ w32[l])
        h = x @ w32[layer]                       # pre-activation of layer
        d = h - mu[None, :]
        A = d @ Pm                               # [b, Q] probe dots
        wts = A[:, pi] * A[:, pj]                # [b, npairs]
        acc += (wts.T @ d)                       # [npairs, 1024]
        done += b
        if done % (BATCH * 32) == 0:
            print(json.dumps({"samples": done,
                              "wall_s": round(time.time() - t0, 1)}),
                  flush=True)
    S = (acc / float(P)).cpu().numpy().astype(np.float64)
    np.savez(HERE / f"probe_truth_rank{rank:04d}_L{layer}.npz",
             S=S, names=np.array(names),
             pairs=np.array(pairs),
             **{f"probe_{k}": probes[k] for k in names})
    print(json.dumps({"done": P, "pairs": len(pairs),
                      "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
