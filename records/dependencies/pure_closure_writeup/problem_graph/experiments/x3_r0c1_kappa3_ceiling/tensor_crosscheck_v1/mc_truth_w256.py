#!/usr/bin/env python3
"""TAG r0c1-crosscheck-mc-v1: high-N MC truth for one SCREEN net (w256 d32).
Weights = the public Philox recipe (gate_a0_v3 spelling, verbatim). Tracks
per layer: pre mean/var/k3/k4 (f64 accumulation), pre covariance, pre K21
plane, post mean. Optional seed-offset stream for noise floors.
SCREEN-side diagnostic only (indices 0-999 permitted by the freeze).
Usage: mc_truth_w256.py <screen_seed> <samples_M> [out_suffix] [stream_off]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtri

HERE = Path(__file__).resolve().parent
WIDTH, DEPTH = 256, 32
BATCH = 262_144


def generate_weights(seed: int) -> np.ndarray:
    uniforms = np.random.Generator(np.random.Philox(key=int(seed))).random(
        (DEPTH, WIDTH, WIDTH), dtype=np.float64
    )
    return (
        ndtri(np.maximum(uniforms, 2.0 ** -54)) * np.sqrt(2.0 / WIDTH)
    ).astype(np.float32)


def main():
    seed = int(sys.argv[1])
    n_samples = int(float(sys.argv[2]) * 1e6)
    suffix = sys.argv[3] if len(sys.argv) > 3 else ""
    stream_off = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    assert torch.backends.mps.is_available()
    dev = torch.device("mps")
    W = generate_weights(seed)
    w_gpu = [torch.from_numpy(W[l].copy()).to(dev) for l in range(DEPTH)]
    rng = np.random.Generator(np.random.Philox(key=(seed * 2654435761 + 17
                                                    + stream_off) & (2**63 - 1)))
    pre_s = np.zeros((DEPTH, 6, WIDTH))
    cov_s = np.zeros((DEPTH, WIDTH, WIDTH))
    m21_s = np.zeros((DEPTH, WIDTH, WIDTH))
    post_s = np.zeros((DEPTH, WIDTH))
    post_s2 = np.zeros((DEPTH, WIDTH))
    pm11_s = np.zeros((4, WIDTH, WIDTH))
    pm21_s = np.zeros((4, WIDTH, WIDTH))
    m22_s = np.zeros((3, WIDTH, WIDTH))
    m31_s = np.zeros((3, WIDTH, WIDTH))
    done = 0
    t0 = time.time()
    while done < n_samples:
        b = min(BATCH, n_samples - done)
        x_np = rng.standard_normal(size=(b, WIDTH), dtype=np.float32)
        x = torch.from_numpy(x_np).to(dev)
        for l in range(DEPTH):
            h = x @ w_gpu[l]
            h2 = h * h
            pre_s[l, 0] += h.sum(dim=0).cpu().numpy()
            pre_s[l, 1] += h2.sum(dim=0).cpu().numpy()
            h3 = h2 * h
            h4 = h2 * h2
            pre_s[l, 2] += h3.sum(dim=0).cpu().numpy()
            pre_s[l, 3] += h4.sum(dim=0).cpu().numpy()
            pre_s[l, 4] += (h4 * h).sum(dim=0).cpu().numpy()
            pre_s[l, 5] += (h4 * h2).sum(dim=0).cpu().numpy()
            cov_s[l] += (h.T @ h).cpu().numpy()
            m21_s[l] += (h2.T @ h).cpu().numpy()
            if l < 3:
                m22_s[l] += (h2.T @ h2).cpu().numpy()
                m31_s[l] += ((h2 * h).T @ h).cpu().numpy()
            x = torch.clamp(h, min=0.0)
            post_s[l] += x.sum(dim=0).cpu().numpy()
            post_s2[l] += (x * x).sum(dim=0).cpu().numpy()
            if l < 4:
                x2 = x * x
                pm11_s[l] += (x.T @ x).cpu().numpy()
                pm21_s[l] += (x2.T @ x).cpu().numpy()
        done += b
        if done % (BATCH * 16) == 0:
            print(json.dumps({"samples": done,
                              "rate": int(done / (time.time() - t0))}),
                  flush=True)
    P = float(n_samples)
    m1 = pre_s[:, 0] / P
    m5r = pre_s[:, 4] / P
    m6r = pre_s[:, 5] / P
    m2 = pre_s[:, 1] / P
    m3 = pre_s[:, 2] / P
    m4 = pre_s[:, 3] / P
    v = m2 - m1 ** 2
    k3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    k4 = m4 - 4 * m1 * m3 - 3 * m2 ** 2 + 12 * m1 ** 2 * m2 - 6 * m1 ** 4
    C = cov_s / P - np.einsum("li,lj->lij", m1, m1)
    M21 = m21_s / P
    # connected K21[i,j] = E[h_i^2 h_j] - 2 m_i E[h_i h_j] - m_j E[h_i^2]
    #                      + 2 m_i^2 m_j
    Eij = cov_s / P
    K21 = (M21 - 2 * np.einsum("li,lij->lij", m1, Eij)
           - np.einsum("li,lj->lij", m2, m1)
           + 2 * np.einsum("li,li,lj->lij", m1, m1, m1))
    post_mean = post_s / P
    post_se = np.sqrt(np.maximum(post_s2 / P - post_mean ** 2, 0.0) / P)
    pM11 = pm11_s / P
    pM21 = pm21_s / P
    pmean = post_mean[:4]
    pr2 = post_s2[:4] / P
    postK21 = (pM21 - 2 * np.einsum("li,lij->lij", pmean, pM11)
               - np.einsum("li,lj->lij", pr2, pmean)
               + 2 * np.einsum("li,li,lj->lij", pmean, pmean, pmean))
    M22 = m22_s / P
    M31r = m31_s / P
    m1s = m1[:3]
    RawC = cov_s[:3] / P                      # raw E[hi hj]
    m2s = m2[:3]
    m3s = m3[:3]
    mu11 = RawC - np.einsum("li,lj->lij", m1s, m1s)
    # central mu22 from raws
    M21r = m21_s[:3] / P
    mu22 = (M22 - 2 * np.einsum("lj,lij->lij", m1s, M21r)
            - 2 * np.einsum("li,lji->lij", m1s, M21r)
            + np.einsum("li,li,lj->lij", m1s, m1s, m2s)
            + np.einsum("lj,lj,li->lij", m1s, m1s, m2s)
            + 4 * np.einsum("li,lj,lij->lij", m1s, m1s, RawC)
            - 3 * np.einsum("li,li,lj,lj->lij", m1s, m1s, m1s, m1s))
    vv3 = m2s - m1s ** 2
    K22c = mu22 - np.einsum("li,lj->lij", vv3, vv3) - 2 * mu11 ** 2
    # central mu31 from raws:
    # mu31 = M31 - mj*m3 - 3 mi*M21 + 3 mi mj m2 + 3 mi^2 M11 - 3 mi^3 mj
    mu31 = (M31r - np.einsum("lj,li->lij", m1s, m3s)
            - 3 * np.einsum("li,lij->lij", m1s, M21r)
            + 3 * np.einsum("li,lj,li->lij", m1s, m1s, m2s)
            + 3 * np.einsum("li,li,lij->lij", m1s, m1s, RawC)
            - 3 * np.einsum("li,li,li,lj->lij", m1s, m1s, m1s, m1s))
    K31c = mu31 - 3 * np.einsum("li,lij->lij", vv3, mu11)
    np.savez(HERE / f"mc_w256_seed{seed}{suffix}.npz",
             pre_K22=K22c.astype(np.float32),
             pre_K31=K31c.astype(np.float32),
             post_K21=postK21.astype(np.float32),
             pre_m=m1, pre_v=v, pre_k3=k3, pre_k4=k4,
             pre_m5raw=m5r, pre_m6raw=m6r,
             pre_cov=C.astype(np.float32), pre_K21=K21.astype(np.float32),
             post_mean=post_mean, post_se=post_se, P=P)
    print(json.dumps({"done": n_samples, "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
