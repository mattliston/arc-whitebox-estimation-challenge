#!/usr/bin/env python3
"""TAG opus-reexam E10 analysis: frame-B truth vs the re-normalisation formula, network <rank>.
Per layer l >= 1 (post state y of layer l under the frame-B measure of layer l-1, then re-normalised):
  s2 = var(t), t = l.y;  C0 = Pi C^y Pi;  c' = Pi C^y l
  residual  D = C^B_l - (1 + s2) C0 - 2 c' c''     ( = -Pi (T.l) Pi + fourth-order terms )
  rigid fit D ~ -eps C0: eps, and the share of |D|^2 (off-diagonal, noise-free via the two halves) that it explains
Also the frame-B per-unit cumulants of zeta = z_l / G_{l-1}: mean and spread of the excess kurtosis (uniform and mean-relevant
weights), the skew spread, and 12 var(t_{l-1}) for comparison.
usage: e10_analyze.py <e10 npz> <rank>"""
import sys, json, numpy as np
from pathlib import Path
d = np.load(sys.argv[1]); rank = int(sys.argv[2]); L = 16; n = 1024
P2 = Path(__file__).resolve().parent.parent
mc = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
W = np.load((STORE / man['path']).parent / 'weights.npy').astype(np.float64)


def cov(key, l, hs=(0, 1)):
    g = sum(float(d[f'{key}{l}_h{h}_g']) for h in hs); s1 = sum(d[f'{key}{l}_h{h}_s1'] for h in hs) / g
    M = sum(d[f'{key}{l}_h{h}_M'].astype(np.float64) for h in hs) / g
    return M - np.outer(s1, s1)


off = lambda A: A - np.diag(np.diag(A))
phi = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
print(f'network {rank}, {float(d["N"]):.0f} samples')
print(" L | var(t) x1e3 | off-diag relative size: C^B - Pi C^y Pi | residual D after second-order terms | noise floor of C^B | rigid eps x1e3 | share of D explained by -eps C0 | eps / var(t)")
s2s = {}
for l in range(1, L):
    m = mc[f'mean_post_{l}'].astype(np.float64); ell = m / (m @ m); mh = m / np.linalg.norm(m); Pi = np.eye(n) - np.outer(mh, mh)
    res = {}
    for hs in ((0,), (1,), (0, 1)):
        CB = cov('B', l, hs); CY = cov('Y', l, hs)
        cbar = CY @ ell; s2 = float(ell @ cbar); cp = Pi @ cbar; C0 = Pi @ CY @ Pi
        res[hs] = (CB, C0, CB - (1 + s2) * C0 - 2 * np.outer(cp, cp), s2)
    CB, C0, D, s2 = res[(0, 1)]; s2s[l] = s2
    DA, DB = off(res[(0,)][2]), off(res[(1,)][2]); C0o = off(C0)
    eps = -(off(D) * C0o).sum() / (C0o * C0o).sum()
    sig = (DA * DB).sum(); expl = eps * eps * (C0o * C0o).sum()
    floor = np.linalg.norm(off(res[(0,)][0] - res[(1,)][0])) / 2 / np.linalg.norm(off(CB))
    print(f'{l:2d} | {s2 * 1e3:6.3f} | {np.linalg.norm(off(CB - C0)) / np.linalg.norm(off(CB)) * 100:6.3f}% | {np.linalg.norm(off(D)) / np.linalg.norm(off(CB)) * 100:6.3f}% | {floor * 100:.3f}% | '
          f'{eps * 1e3:7.3f} | {expl / sig if sig > 0 else float("nan"):6.3f} | {eps / s2:5.2f}')
print()
print(' L | frame-B zeta: excess kurtosis mean (uniform) | mean (mean-relevant weight) | spread | skew spread (mean-relevant weight) | 12 var(t) of the previous layer x1e3')
for l in range(1, L):
    g = sum(float(d[f'Z{l}_h{h}_g']) for h in (0, 1)); m1, m2, m3, m4 = sum(d[f'Z{l}_h{h}_p'] for h in (0, 1)) / g
    v = m2 - m1 ** 2; k3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3; k4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4 - 3 * v ** 2
    g1 = k3 / v ** 1.5; g2 = k4 / v ** 2
    a = (W[l].T @ mc[f'mean_post_{l - 1}'].astype(np.float64) + m1) / np.sqrt(v)
    w4 = ((a * a - 1) * phi(a)) ** 2 + 1e-14; w3 = (a * phi(a)) ** 2 + 1e-14
    print(f'{l:2d} | {g2.mean():+.4f} | {(w4 * g2).sum() / w4.sum():+.4f} | {g2.std():.4f} | {np.sqrt((w3 * g1 * g1).sum() / w3.sum()):.4f} | {12 * s2s.get(l - 1, float("nan")) * 1e3:6.2f}')
