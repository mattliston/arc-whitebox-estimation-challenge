#!/usr/bin/env python3
"""TAG fable-physicist (CR-23): RANK CENSUS of the unit-specific (2,2) fourth-order plane (necessary condition for ANY small dedicated
fourth-order carrier: a CP4 bank of R columns gives a (2,2) plane of rank <= R, since lam p^(x)4 -> K22_ij = lam p_i^2 p_j^2).
Residual plane R = K22_truth - c*(v v^T + 2 C^2)/3 (isotropic scale-mixture form, c fitted), pre-activation, r35, two independent truth draws
(public_g19 mps_planes_rank0035_new.npz and _noise2.npz; C, v from covhunter's 32M mcall). Noise-free cross estimates: signal energy
<R_A, R_B>; rank-k capture <P_k R_A P_k, R_B> / <R_A, R_B> with P_k = top-|eigenvalue| eigenvectors of R_A; the per-unit k4 residual is
the diagonal: capture of diag = <diag(P_k R_A P_k), diag R_B> / <diag R_A, diag R_B>. usage: k22_rank_census.py <layers, e.g. 12,15>"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[2]; G = P2 / 'public_g19_attribution_v1'
A, B = np.load(G / 'mps_planes_rank0035_new.npz'), np.load(G / 'mps_planes_rank0035_noise2.npz')
MC = np.load(P2 / 'fable_covhunter_v1' / 'mcall_rank0035_P32M.npz')
T = np.load(G / 'mps_truth_rank0035_indep32M.npz')
for L in map(int, sys.argv[1].split(',')):
    KA, KB = [np.asarray(X[f'K22_{L}'], dtype=np.float64) for X in (A, B)]
    KA, KB = (KA + KA.T) / 2, (KB + KB.T) / 2
    C = np.asarray(MC[f'cov_pre_{L}'], dtype=np.float64); v = np.diag(C)
    G22 = (np.outer(v, v) + 2 * C ** 2) / 3
    c = ((KA + KB) / 2 * G22).sum() / (G22 * G22).sum()
    RA, RB = KA - c * G22, KB - c * G22
    dK = np.diag(KA) @ np.diag(KB) / np.sqrt((np.diag(KA) ** 2).sum() * (np.diag(KB) ** 2).sum())
    k4t = T['mc_pre_k4'][L]
    print(f'L{L}: diag(K22) vs 32M per-unit k4: cos {np.diag(KA) @ k4t / np.linalg.norm(np.diag(KA)) / np.linalg.norm(k4t):+.4f}, rel diff A {np.linalg.norm(np.diag(KA) - k4t) / np.linalg.norm(k4t):.3f}; draws A/B diag cos {dK:+.4f}')
    sig = (RA * RB).sum(); tot = (KA * KB).sum(); dsig = np.diag(RA) @ np.diag(RB); dtot = np.diag(KA) @ np.diag(KB)
    print(f'   residual plane: {sig / tot:.3f} of the plane energy; {dsig / dtot:.3f} of the diagonal (per-unit k4) energy (cross-draw, noise-free)')
    ev, U = np.linalg.eigh(RA); order = np.argsort(-np.abs(ev)); ev, U = ev[order], U[:, order]
    row = []
    for k in (1, 4, 16, 64, 256, 512):
        Uk = U[:, :k]; Rk = (Uk * ev[:k]) @ Uk.T
        row.append(f'k={k}: plane {(Rk * RB).sum() / sig:.2f}, diag {np.diag(Rk) @ np.diag(RB) / dsig:.2f}')
    print('   rank-k capture (from draw A, scored on draw B): ' + ' | '.join(row))
