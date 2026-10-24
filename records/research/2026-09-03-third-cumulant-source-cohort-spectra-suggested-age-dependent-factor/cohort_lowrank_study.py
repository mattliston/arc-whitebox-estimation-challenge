#!/usr/bin/env python3
"""TAG g30-cohort-lowrank (T-95): can the CP3 bank's cohorts be stored low-rank without losing the skew plane?
Input: the main-bank dump of the deployed card (RETRO_DUMP_BANK_CP: lam, P, Q after the push at layer l, K21 = the exact
read).  Cohorts = contiguous blocks of 1024 terms in birth order (cap 16384, no drops through L15).  For each cohort b the
factor matrices P_b, Q_b (1024 x 1024) are replaced by their rank-r truncated SVDs; the read
  K21 = (P o P) Lam Q^T + 2 (P o Q) Lam P^T      (estimator_retro._cp_read)
is recomputed and compared with the exact read: off-diagonal relative Frobenius error and diagonal (k3) relative error,
per layer, for r in RANKS; also the energy fractions kept.  Cost model: per cohort-layer, full = 4 GEMM-eq (push p, push q,
read 2); low-rank ~ (2 r_p + 3 r_q) / 1024 GEMM-eq (push U_p, U_q; materialize P; read through R_q^T U_q^T).
Usage: cohort_lowrank_study.py <dump_prefix> [layers]"""
import sys
from pathlib import Path
import numpy as np

prefix = sys.argv[1]
layers = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["4", "8", "12", "14"])]
RANKS = (64, 128, 192, 256, 384, 512)
n = 1024
off = ~np.eye(n, dtype=bool)


def read(lam, P, Q):
    return (P * P) @ (lam[:, None] * Q.T) + 2.0 * ((P * Q) @ (lam[:, None] * P.T))


def trunc(M, r):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r], float(np.sum(s[:r] ** 2) / np.sum(s ** 2))


for L in layers:
    f = Path(f"{prefix}_L{L}.npz")
    if not f.exists():
        print(f"L{L}: no dump"); continue
    z = np.load(f); lam = z["lam"].astype(np.float64); P = z["P"].astype(np.float64); Q = z["Q"].astype(np.float64); K = z["K21"].astype(np.float64)
    R = lam.shape[0]; nb = R // n
    Kex = read(lam, P, Q)
    cert = np.linalg.norm(Kex[off] - K[off]) / np.linalg.norm(K[off])
    print(f"L{L}: bank rank {R} = {nb} cohorts; recomputed exact read vs card read rel diff {cert:.2e}", flush=True)
    for r in RANKS:
        Pt = np.empty_like(P); Qt = np.empty_like(Q); ep = []; eq = []
        for b in range(nb):
            sl = slice(b * n, (b + 1) * n)
            Pt[:, sl], fp = trunc(P[:, sl], r); Qt[:, sl], fq = trunc(Q[:, sl], r); ep.append(fp); eq.append(fq)
        Kt = read(lam, Pt, Qt)
        e_off = np.linalg.norm(Kt[off] - Kex[off]) / np.linalg.norm(Kex[off])
        e_diag = np.linalg.norm(np.diag(Kt) - np.diag(Kex)) / np.linalg.norm(np.diag(Kex))
        cost = nb * (2 * r + 3 * r) / n
        print(f"  rank {r:4d}: K21 off-diag rel err {e_off:.4f}  k3 rel err {e_diag:.4f} | energy kept P {min(ep):.4f}-{max(ep):.4f} Q {min(eq):.4f}-{max(eq):.4f} | cost {cost:.1f} vs full {4 * nb} GEMM-eq", flush=True)
