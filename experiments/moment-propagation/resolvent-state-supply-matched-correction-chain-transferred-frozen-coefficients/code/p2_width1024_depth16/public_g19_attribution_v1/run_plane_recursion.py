#!/usr/bin/env python3
"""TAG plane-recursion: standalone card-class plane recursion + hybrid arms.

Self-contained recursion of (m, v, k3, k4, C, K21, K22, K31) using
the engine's own emission machinery (engine_post_cov, emission_K21,
fourth_emission_1024) and exact GEMM pushes (pair-level truncation,
same information class as the card). Arms: start from banked TRUTH
state at layer l0 (l0=1 -> the recursion's own accumulation, the
card-class baseline; l0=3 -> the 'exact shallow planes' hybrid).
Grades v / k3 / C per layer vs banked truth, alongside the card's
carried errors. This sizes the first deployable increment of the
chaos program (exact identification-free shallow planes feeding the
existing recursion). Usage: run_plane_recursion.py <rank> <l0>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_vresid_field import engine_post_cov
from run_tensor_recursion_v4 import emission_K21, fourth_emission_1024
from run_attribution import edg2_relu_mean
from run_chaos1024_shallow import load_weights

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 35
L0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
LMAX = 14


def push_K21(W, P):
    """K21' from post plane P (diag = post k3), pair-level exact."""
    W2 = W * W
    PW = P @ W
    PtW = P.T @ W
    T1 = W2.T @ P @ W
    T2 = (W * PW).T @ W
    T3 = (W * PtW).T @ W
    D = (W2 * np.diag(P)[:, None]).T @ W
    return T1 + T2 + T3 - 2 * D


def push_K22_K31(W, k4d, K22p, K31p):
    """Fourth-order pushes, LEADING pair-level patterns (numpy).

    Exact for the dominant separable patterns; drops the
    doubly-crossed patterns (1/n-suppressed pair mixings) — a
    truncation of the same class the card makes, and these planes
    only feed subleading emission corrections. Diag conventions:
    K22p/K31p diag = k4d.
    K22'_cd (cc|dd): a=b,e=f leading:  (W2)^T K22p (W2)
    K31'_cd (ccc|d): a=b=e leading:    (W3)^T K31p W
      + 3 x (two-on-c cross with e on c, f on d): K22p pattern
    k4'_c: (W4)^T k4d + 3[diag((W2)^T K22p W2) - (W4)^T k4d]
           + 4[sum_a W3_ac (K31p W)_ac - (W4)^T k4d]."""
    W2 = W * W
    W3 = W2 * W
    W4 = W2 * W2
    K22n = W2.T @ K22p @ W2
    cross = ((W * (K22p.T @ W2)).T @ W)     # 3x pattern, e on c f on d
    K31n = W3.T @ K31p @ W + 3 * cross
    d4 = W4.T @ k4d
    k4n = (d4
           + 3 * (np.diag(W2.T @ K22p @ W2) - d4)
           + 4 * (np.einsum("ac,ac->c", W3, K31p @ W) - d4))
    return k4n, K22n, K31n


def main():
    W = load_weights(RANK)
    st = np.load(HERE / f"states_rank{RANK:04d}.npz")
    tr = np.load(HERE / f"mps_cov2_rank{RANK:04d}.npz")
    hc = np.load(HERE / f"host_cov_rank{RANK:04d}_base.npz")
    # init at truth layer L0
    m = tr["mc_pre_m"][L0].astype(np.float64)
    v = tr["mc_pre_v"][L0].astype(np.float64)
    k3 = tr["mc_pre_k3"][L0].astype(np.float64)
    k4 = tr["mc_pre_k4"][L0].astype(np.float64)
    C = tr["mc_pre_cov"][L0].astype(np.float64)
    i0 = L0 - 1
    K21 = hc["k21"][i0].astype(np.float64) * 0.0
    K22 = np.zeros_like(C)
    K31 = np.zeros_like(C)
    # honest init for pair planes at L0: use the card's carried ones
    # (truth pair planes are not banked); at L0=1 they are near-exact.
    K21 = hc["k21"][i0].astype(np.float64)
    K22 = hc["k22"][i0].astype(np.float64)
    K31 = hc["k31"][i0].astype(np.float64)
    out = {}
    for l in range(L0, LMAX):
        r1 = edg2_relu_mean(m, v, k3, k4)
        mu11 = engine_post_cov(m, v, k3, k4, C, K21, K22, K31)
        P = emission_K21(m, v, k3, k4, C, K21, K22, K31)
        k4d, K22p, K31p = fourth_emission_1024(
            m, v, k3, k4, C, K21, K22, K31)
        Wp = np.asarray(W[l + 1], dtype=np.float64)
        m = Wp.T @ r1
        C = Wp.T @ mu11 @ Wp
        v = np.diag(C).copy()
        K21 = push_K21(Wp, P)
        k3 = np.diag(K21).copy()
        k4, K22, K31 = push_K22_K31(Wp, k4d, K22p, K31p)
        nl = l + 1
        tv = tr["mc_pre_v"][nl].astype(np.float64)
        tk3 = tr["mc_pre_k3"][nl].astype(np.float64)
        tC = tr["mc_pre_cov"][nl].astype(np.float64)
        hv = st["host_v"][nl - 1].astype(np.float64)
        hk3 = st["host_k3"][nl - 1].astype(np.float64)
        rec = {
            "v_rel": round(float(np.sqrt(np.mean((v - tv) ** 2))
                                 / np.sqrt(np.mean(tv ** 2))), 5),
            "card_v_rel": round(float(np.sqrt(np.mean((hv - tv) ** 2))
                                      / np.sqrt(np.mean(tv ** 2))), 5),
            "k3_rel": round(float(np.sqrt(np.mean((k3 - tk3) ** 2))
                                  / np.sqrt(np.mean(tk3 ** 2))), 4),
            "card_k3_rel": round(float(np.sqrt(np.mean(
                (hk3 - tk3) ** 2)) / np.sqrt(np.mean(tk3 ** 2))), 4),
            "C_rel": round(float(np.linalg.norm(C - tC)
                                 / np.linalg.norm(tC)), 5),
        }
        out[nl] = rec
        print(RANK, f"L{nl}", json.dumps(rec), flush=True)
    (HERE / f"PLANE_RECURSION_rank{RANK:04d}_l0{L0}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
