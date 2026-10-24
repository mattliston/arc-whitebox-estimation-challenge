#!/usr/bin/env python3
"""True pre-activation planes (K21, K22c, K31) at chosen layers via MPS.
K21[i,j]=cum(h_i,h_i,h_j); K22c[i,j]=cum(h_i,h_i,h_j,h_j);
K31[i,j]=cum(h_i,h_i,h_i,h_j). Usage: mc_planes_mps.py <rank> <P_millions> <l1,l2,...>"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")
SEED_XOR = 0x47414D454F4E5F31
SEED_EXTRA = 0
BATCH = 65_536

def main():
    rank = int(sys.argv[1])
    p_samples = int(float(sys.argv[2]) * 1e6)
    layers = [int(x) for x in sys.argv[3].split(",")]
    global SEED_EXTRA
    SEED_EXTRA = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    post_mode = len(sys.argv) > 4 and sys.argv[4] == "post"
    dev = torch.device("mps")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    rmp = STORE / entry["path"]
    row = json.loads(rmp.read_text())
    weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
    w_gpu = [torch.from_numpy(np.ascontiguousarray(weights[l], dtype=np.float32)).to(dev)
             for l in range(16)]
    rng = np.random.Generator(np.random.PCG64DXSM(
        (int(row["mlp_seed"]) ^ SEED_XOR ^ SEED_EXTRA) & ((1 << 64) - 1)))
    L = 1024
    acc = {l: {k: np.zeros((L, L)) for k in ("m11","m21","m12","m22","m31")} for l in layers}
    s1 = {l: np.zeros(L) for l in layers}
    s2 = {l: np.zeros(L) for l in layers}
    s3 = {l: np.zeros(L) for l in layers}
    done = 0
    t0 = time.time()
    while done < p_samples:
        b = min(BATCH, p_samples - done)
        x = torch.from_numpy(rng.standard_normal(size=(b, 1024), dtype=np.float32)).to(dev)
        for l in range(16):
            h = x @ w_gpu[l]
            src_t = torch.clamp(h, min=0.0) if post_mode else h
            if l in acc:
                h2 = src_t * src_t
                h3 = h2 * src_t
                a = acc[l]
                a["m11"] += (src_t.T @ src_t).cpu().numpy().astype(np.float64)
                a["m21"] += (h2.T @ src_t).cpu().numpy().astype(np.float64)
                a["m12"] += (src_t.T @ h2).cpu().numpy().astype(np.float64)
                a["m22"] += (h2.T @ h2).cpu().numpy().astype(np.float64)
                a["m31"] += (h3.T @ src_t).cpu().numpy().astype(np.float64)
                s1[l] += src_t.sum(dim=0).cpu().numpy().astype(np.float64)
                s2[l] += h2.sum(dim=0).cpu().numpy().astype(np.float64)
                s3[l] += h3.sum(dim=0).cpu().numpy().astype(np.float64)
            x = torch.clamp(h, min=0.0)
        done += b
        if done % (BATCH*16) == 0:
            print(json.dumps({"samples": done}), flush=True)
    P = float(p_samples)
    out = {}
    for l in layers:
        a = acc[l]
        m1 = s1[l]/P; m2v = s2[l]/P; m3v = s3[l]/P
        M11 = a["m11"]/P; M21 = a["m21"]/P; M12 = a["m12"]/P
        M22 = a["m22"]/P; M31 = a["m31"]/P
        cov = M11 - np.outer(m1, m1)
        var = m2v - m1*m1
        # K21[i,j] = E[hi^2 hj] - 2 m_i E[hi hj] - m_j E[hi^2] + 2 m_i^2 m_j
        K21 = M21 - 2*m1[:,None]*M11 - m1[None,:]*m2v[:,None] + 2*(m1**2)[:,None]*m1[None,:]
        # central moments for k22/k31
        # mu22 = E[(hi-mi)^2 (hj-mj)^2]
        mu22 = (M22 - 2*m1[None,:]*M21 - 2*m1[:,None]*M12
                + (m1**2)[None,:]*m2v[:,None] + (m1**2)[:,None]*m2v[None,:]
                + 4*m1[:,None]*m1[None,:]*M11
                - 3*(m1**2)[:,None]*(m1**2)[None,:])
        K22 = mu22 - np.outer(var, var) - 2*cov*cov
        # mu31 = E[(hi-mi)^3 (hj-mj)]
        mu31 = (M31 - 3*m1[:,None]*M21 - m1[None,:]*m3v[:,None]
                + 3*(m1**2)[:,None]*M11 + 3*m1[:,None]*m1[None,:]*m2v[:,None]
                - 3*(m1**3)[:,None]*m1[None,:])
        K31 = mu31 - 3*var[:,None]*cov
        out[f"K21_{l}"] = K21.astype(np.float32)
        out[f"K22_{l}"] = K22.astype(np.float32)
        out[f"K31_{l}"] = K31.astype(np.float32)
    dest = (HERE / f"mps_planes_rank{rank:04d}{'_post' if post_mode else ''}"
            f"{sys.argv[5] if len(sys.argv) > 5 else ''}.npz")
    if dest.exists():
        dest = dest.with_name(dest.stem + "_new.npz")
    np.savez(dest, **out,
             layers=np.array(layers))
    print(json.dumps({"done": p_samples, "wall_s": round(time.time()-t0, 1)}))

if __name__ == "__main__":
    main()
