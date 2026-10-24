#!/usr/bin/env python3
"""TAG g19-mc-truth-mps-v1: MPS-accelerated per-layer truth statistics.
Same estimand as run_attribution.mc_truth (per-layer pre m/v/k3/k4 + post
mean/SE) but on the Metal GPU: RNG on CPU numpy (PCG64DXSM, same seed law,
bit-identical draw stream), GEMMs+moments on MPS in float32 with float64
CPU accumulation per batch. One MPS job at a time; batch sized for <2GB GPU.
Usage: mc_truth_mps.py <rank> <P_millions> [out.npz]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")
SEED_XOR = 0x47414D454F4E5F31
BATCH = 131_072


def main():
    rank = int(sys.argv[1])
    p_samples = int(float(sys.argv[2]) * 1e6)
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else (
        HERE / f"mps_truth_rank{rank:04d}_P{p_samples//1_000_000}M.npz")
    assert torch.backends.mps.is_available(), "MPS not available"
    dev = torch.device("mps")

    if rank >= 3016500:  # virgin fresh instance: weights file by seed
        weights = np.load(HERE / f"fresh_{rank}_weights.npy",
                          mmap_mode="r", allow_pickle=False)
        row = {"mlp_seed": rank}
    else:
        manifest = json.loads((STORE / "manifest.json").read_text())
        entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        weights = np.load(rmp.parent / "weights.npy", mmap_mode="r",
                          allow_pickle=False)
    w_gpu = [torch.from_numpy(
        np.ascontiguousarray(weights[l], dtype=np.float32)).to(dev)
        for l in range(16)]

    rng = np.random.Generator(np.random.PCG64DXSM(
        (int(row["mlp_seed"]) ^ SEED_XOR) & ((1 << 64) - 1)))

    pre_s = np.zeros((16, 6, 1024), dtype=np.float64)
    want_cov = len(sys.argv) > 4 and sys.argv[4] == "cov"
    cov_s = (np.zeros((16, 1024, 1024), dtype=np.float64)
             if want_cov else None)
    pcov_s = (np.zeros((16, 1024, 1024), dtype=np.float64)
              if want_cov else None)
    post_s1 = np.zeros((16, 1024), dtype=np.float64)
    post_s2 = np.zeros((16, 1024), dtype=np.float64)
    done = 0
    t0 = time.time()
    while done < p_samples:
        b = min(BATCH, p_samples - done)
        x_np = rng.standard_normal(size=(b, 1024), dtype=np.float32)
        x = torch.from_numpy(x_np).to(dev)
        for l in range(16):
            h = x @ w_gpu[l]
            h2 = h * h
            s1 = h.sum(dim=0)
            s2 = h2.sum(dim=0)
            s3 = (h2 * h).sum(dim=0)
            s4 = (h2 * h2).sum(dim=0)
            h4 = h2 * h2
            s5 = (h4 * h).sum(dim=0)
            s6 = (h4 * h2).sum(dim=0)
            if want_cov:
                cs = (h.T @ h).cpu().numpy()
                cov_s[l] += cs.astype(np.float64)
            stats = torch.stack((s1, s2, s3, s4, s5, s6)).cpu().numpy()
            pre_s[l] += stats.astype(np.float64)
            x = torch.clamp(h, min=0.0)
            if want_cov:
                pcov_s[l] += (x.T @ x).cpu().numpy().astype(np.float64)
            p1 = x.sum(dim=0)
            p2 = (x * x).sum(dim=0)
            pp = torch.stack((p1, p2)).cpu().numpy().astype(np.float64)
            post_s1[l] += pp[0]
            post_s2[l] += pp[1]
        done += b
        if done % (BATCH * 8) == 0:
            rate = done / (time.time() - t0)
            print(json.dumps({"samples": done,
                              "rate_per_s": int(rate)}), flush=True)
    P = float(p_samples)
    m1 = pre_s[:, 0] / P
    m2 = pre_s[:, 1] / P
    m3 = pre_s[:, 2] / P
    m4 = pre_s[:, 3] / P
    m5r = pre_s[:, 4] / P
    m6r = pre_s[:, 5] / P
    pre_mean = m1
    pre_var = m2 - m1 ** 2
    pre_k3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    pre_k4 = m4 - 4 * m1 * m3 - 3 * m2 ** 2 + 12 * m1 ** 2 * m2 - 6 * m1 ** 4
    post_mean = post_s1 / P
    post_var = post_s2 / P - post_mean ** 2
    post_se = np.sqrt(np.maximum(post_var, 0) / P)
    save = dict(mc_pre_m=pre_mean, mc_pre_v=pre_var,
                mc_pre_k3=pre_k3, mc_pre_k4=pre_k4, pre_m5raw=m5r, pre_m6raw=m6r, mc_post=post_mean,
                mc_post_var=post_var, mc_se=post_se, P=p_samples, rank=rank)
    if want_cov:
        for l in range(16):
            cov_s[l] = cov_s[l] / P - np.outer(pre_mean[l], pre_mean[l])
            pcov_s[l] = pcov_s[l] / P - np.outer(post_mean[l], post_mean[l])
        save["mc_pre_cov"] = cov_s.astype(np.float32)
        save["mc_post_cov"] = pcov_s.astype(np.float32)
    np.savez(out_path, **save)
    dt = time.time() - t0
    print(json.dumps({"done": p_samples, "wall_s": round(dt, 1),
                      "samples_per_s": int(p_samples / dt),
                      "out": str(out_path)}))


if __name__ == "__main__":
    main()
