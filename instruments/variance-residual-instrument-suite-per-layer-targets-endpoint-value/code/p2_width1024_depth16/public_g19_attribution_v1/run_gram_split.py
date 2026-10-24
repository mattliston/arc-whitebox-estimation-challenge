#!/usr/bin/env python3
"""TAG gram-split: exact chaos-order split of the post-cov at 1024.

Mehler/OU coupling: for x' = rho x + sqrt(1-rho^2) xi,
E[ph_a(z(x))~ ph_b(z(x'))~] = sum_k rho^k G_k[a,b], where G_k is the
order-k chaos Gram of the post-activations in the SHARED input frame.
Sample coupled forward passes at NR rho-nodes, fit per-pair
polynomials in rho (degree <= KD), extract G_3, G_4 (the tracker's
would-be supply) and the tail. Joined with the model-side split
(gated engine evals) this prices the ENTIRE tracker build against
the banked v-residual field BEFORE building it.
Layers: pre-activation z_l pushed through relu at chosen l.
Usage: run_gram_split.py <rank>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_chaos1024_shallow import load_weights

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 35
LAYERS = [5, 8, 11, 13]
RHOS = [-0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0]
KD = 6
N, CHUNK = 200_000, 25_000
SEED = 424242


def forward_collect(Wl, x, layers):
    """One pass; return {l: relu(z_l)} for l in layers."""
    z = x @ Wl[0]
    out = {}
    top = max(layers)
    for j in range(0, top + 1):
        if j > 0:
            z = np.maximum(z, 0.0) @ Wl[j]
        if j in layers:
            out[j] = np.maximum(z, 0.0)
    return out


def main():
    W = load_weights(RANK)
    tr = np.load(HERE / f"mps_cov2_rank{RANK:04d}.npz")
    v0 = tr["mc_pre_v"][0].astype(np.float64)
    W0 = np.asarray(W[0], dtype=np.float32)
    if np.corrcoef(np.sum(W0.astype(np.float64) ** 2, 0), v0)[0, 1] < \
            np.corrcoef(np.sum(W0.astype(np.float64) ** 2, 1), v0)[0, 1]:
        W0 = W0.T
    # z0 = x @ W0; z_{j} = phi @ W[j] (both verified conventions)
    Wl = [np.ascontiguousarray(W0)] + \
        [np.ascontiguousarray(np.asarray(W[j], dtype=np.float32))
         for j in range(1, 15)]
    rng = np.random.default_rng(SEED)
    acc = {(l, r): [np.zeros((1024, 1024)), np.zeros(1024),
                    np.zeros(1024)] for l in LAYERS for r in RHOS}
    for s0 in range(0, N, CHUNK):
        x = rng.standard_normal((CHUNK, 1024)).astype(np.float32)
        xi = rng.standard_normal((CHUNK, 1024)).astype(np.float32)
        phs = forward_collect(Wl, x, set(LAYERS))
        for r in RHOS:
            xr = np.float32(r) * x + np.float32(np.sqrt(1 - r * r)) * xi
            ph2s = forward_collect(Wl, xr, set(LAYERS))
            for l in LAYERS:
                a = acc[(l, r)]
                a[0] += (phs[l].T @ ph2s[l]).astype(np.float64)
                a[1] += phs[l].sum(0).astype(np.float64)
                a[2] += ph2s[l].sum(0).astype(np.float64)
        print(f"chunk {s0 + CHUNK}/{N}", flush=True)
    # fit rho-polynomials per pair
    A = np.vander(np.array(RHOS), KD + 1, increasing=True)  # NR x KD+1
    AtA = A.T @ A
    out = {}
    for l in LAYERS:
        Ms = []
        for r in RHOS:
            a = acc[(l, r)]
            M = a[0] / N - np.outer(a[1] / N, a[2] / N)
            Ms.append(M)
        Y = np.stack([M.ravel() for M in Ms])            # NR x (n^2)
        coefs = np.linalg.solve(AtA, A.T @ Y)            # (KD+1) x n^2
        G = {k: coefs[k].reshape(1024, 1024) for k in range(KD + 1)}
        np.savez_compressed(
            HERE / f"gram_split_rank{RANK:04d}_L{l}.npz",
            **{f"G{k}": G[k].astype(np.float32) for k in range(KD + 1)})
        tot = np.linalg.norm(
            tr["mc_post_cov"][l].astype(np.float64))
        rec = {f"G{k}_rel": round(float(np.linalg.norm(G[k]) / tot), 5)
               for k in range(KD + 1)}
        out[l] = rec
        print(RANK, f"L{l}", json.dumps(rec), flush=True)
    (HERE / f"GRAM_SPLIT_rank{RANK:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
