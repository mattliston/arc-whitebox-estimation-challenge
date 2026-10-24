#!/usr/bin/env python3
"""INDEPENDENT truth draw (different seed XOR) for noise-coherence control: per-layer pre
m/v and post mean/var (no covariances) over P samples on CPU (numpy f32 GEMMs, f64 sums).
Writes public_g19_attribution_v1/mps_truth_rank{rank}_indep{P/1e6}M.npz with mc_post, mc_se,
mc_post_var, mc_pre_m, mc_pre_v (all (16,1024)) and P. Usage: mc_final_indep.py <rank> <P_millions>"""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
G19 = HERE.parent / "public_g19_attribution_v1"
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
SEED_XOR_INDEP = 0x5A5A5A5A1D1D1D1D          # NOT the truth generator's 0x47414D454F4E5F31
rank = int(sys.argv[1]); PM = float(sys.argv[2]); P = int(PM * 1e6); BATCH = 65536
man = json.loads((STORE / "manifest.json").read_text()); entry = man["roles"]["observer_fit"]["row_manifests"][rank]
rmp = STORE / entry["path"]; row = json.loads(rmp.read_text())
W = np.load(rmp.parent / "weights.npy", mmap_mode="r")
w32 = [np.ascontiguousarray(W[l], dtype=np.float32) for l in range(16)]
rng = np.random.Generator(np.random.PCG64DXSM((int(row["mlp_seed"]) ^ SEED_XOR_INDEP) & ((1 << 64) - 1)))
pre_s1 = np.zeros((16, 1024)); pre_s2 = np.zeros((16, 1024)); post_s1 = np.zeros((16, 1024)); post_s2 = np.zeros((16, 1024))
done = 0; t0 = time.time()
while done < P:
    b = min(BATCH, P - done)
    x = rng.standard_normal(size=(b, 1024), dtype=np.float32)
    for l in range(16):
        h = x @ w32[l]
        h64 = h.astype(np.float64)
        pre_s1[l] += h64.sum(0); pre_s2[l] += (h64 ** 2).sum(0)
        x = np.maximum(h, 0.0)
        x64 = x.astype(np.float64)
        post_s1[l] += x64.sum(0); post_s2[l] += (x64 ** 2).sum(0)
    done += b
    if done % (BATCH * 8) == 0:
        print(json.dumps({"samples": done, "s": round(time.time() - t0, 1)}), flush=True)
Pf = float(P)
pre_m = pre_s1 / Pf; pre_v = pre_s2 / Pf - pre_m ** 2
post_m = post_s1 / Pf; post_v = post_s2 / Pf - post_m ** 2
np.savez(G19 / f"mps_truth_rank{rank:04d}_indep{PM:g}M.npz", mc_post=post_m, mc_post_var=post_v,
         mc_se=np.sqrt(np.maximum(post_v, 0) / Pf), mc_pre_m=pre_m, mc_pre_v=pre_v, P=Pf)
print("wrote", f"mps_truth_rank{rank:04d}_indep{PM:g}M.npz", round(time.time() - t0, 1), "s")
