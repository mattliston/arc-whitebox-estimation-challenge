#!/usr/bin/env python3
"""TAG analyst hard networks, Monte Carlo features v1 (lead + physicist 10-13 00:1xZ): per generated network, from a small CPU Monte Carlo (65,536 inputs,
float64 accumulation, single thread): per-unit skewness t3 = k3 / s^3 and excess kurtosis t4 = k4 / s^4 of the PRE-activations at layers 8-15; at kink units
(|alpha| < 1.5, alpha = mean / sd of the pre-activation) their mean |t3|, mean t4 and the spread of |t3| across units; the largest per-unit t4; and the
pre-activation covariance at layers 12 and 15: lambda_1 / tr C and lambda_2 / lambda_1. Default networks: gbeta2's eight 7100584-591 (target c =
gbeta2's change of the final MSE, %), correlated over the eight (|r| > 0.71 ~ p < 0.05). More seeds can be given as arguments (features only).
usage: hard_nets_mc_v1.py [seed ...]   -> table on stdout, hard_nets_mc_v1.json"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[k] = '1'
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE / 'fresh'))
import loader
GB = dict(zip(range(7100584, 7100592), [0.85, -0.94, 0.81, -0.52, 0.08, -0.17, -1.95, 0.04]))
seeds = [int(s) for s in sys.argv[1:]] or list(GB); NS, CH = 65536, 8192; LS = range(8, 16)


def feats(seed):
    w = loader.weights(seed); rng = np.random.default_rng(seed % 100003)
    S = {l: np.zeros((4, 1024)) for l in LS}; C = {l: np.zeros((1024, 1024)) for l in (12, 15)}; M = {l: np.zeros(1024) for l in (12, 15)}
    for _ in range(NS // CH):
        y = rng.standard_normal((CH, 1024)).astype(np.float32)
        for l in range(16):
            z = y @ w[l]
            if l in S:
                z64 = z.astype(np.float64); z2 = z64 * z64; S[l][0] += z64.sum(0); S[l][1] += z2.sum(0); S[l][2] += (z2 * z64).sum(0); S[l][3] += (z2 * z2).sum(0)
                if l in C: C[l] += z64.T @ z64; M[l] += z64.sum(0)
            y = np.maximum(z, 0)
    f = {}; t3k, t4k, t4max = [], [], []
    for l in LS:
        m1, m2, m3, m4 = S[l] / NS; v = m2 - m1 ** 2; c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3; c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
        s = np.sqrt(v); t3 = c3 / s ** 3; t4 = c4 / v ** 2 - 3; kink = np.abs(m1 / s) < 1.5
        t3k.append(np.abs(t3[kink])); t4k.append(t4[kink]); t4max.append(t4.max())
        f[f'mean |t3| at kink units L{l}'] = float(np.abs(t3[kink]).mean()); f[f'mean t4 at kink units L{l}'] = float(t4[kink].mean())
    for band, Ls in (('8-11', range(0, 4)), ('12-15', range(4, 8))):
        f[f'mean |t3| at kink units {band}'] = float(np.mean([t3k[i].mean() for i in Ls])); f[f'sd of |t3| at kink units {band}'] = float(np.mean([t3k[i].std() for i in Ls]))
        f[f'mean t4 at kink units {band}'] = float(np.mean([t4k[i].mean() for i in Ls])); f[f'max per-unit t4 {band}'] = float(np.max([t4max[i] for i in Ls]))
    for l in (12, 15):
        mu = M[l] / NS; cov = C[l] / NS - np.outer(mu, mu); ev = np.linalg.eigvalsh(cov)[::-1]
        f[f'lambda_1 / tr C (pre) L{l}'] = float(ev[0] / ev.sum()); f[f'lambda_2 / lambda_1 (pre) L{l}'] = float(ev[1] / ev[0])
    return f


rows = {}; t0 = time.time()
for s in seeds:
    rows[s] = feats(s); print(json.dumps(dict(seed=s, seconds=round(time.time() - t0))), flush=True)
out = dict(features=rows)
if all(s in GB for s in seeds) and len(seeds) > 3:
    rank = lambda v: np.argsort(np.argsort(v)).astype(float); tc = np.array([GB[s] for s in seeds])
    keys = [k for k in next(iter(rows.values())) if ' L' not in k or k.startswith('lambda')]
    tab = sorted(((k, float(np.corrcoef([rows[s][k] for s in seeds], tc)[0, 1]), float(np.corrcoef(rank(np.array([rows[s][k] for s in seeds])), rank(tc))[0, 1]),
                   float(np.mean([rows[s][k] for s in (7100585, 7100587, 7100590)])), float(np.mean([rows[s][k] for s in (7100584, 7100586)]))) for k in keys), key=lambda z: -abs(z[1]))
    print('gbeta2 eight: feature: r / rho with gbeta2\'s change | helped (585 / 587 / 590) mean vs hurt (584 / 586) mean')
    for k, r_, rho, h, u in tab: print(f"  {k}: r {r_:+.2f} rho {rho:+.2f} | helped {h:.4g} vs hurt {u:.4g}")
    out['table'] = [dict(feature=k, r=r_, rho=rho, helped=h, hurt=u) for k, r_, rho, h, u in tab]
(HERE / ('hard_nets_mc_v1.json' if not sys.argv[1:] else f'hard_nets_mc_v1_{seeds[0]}_{len(seeds)}.json')).write_text(json.dumps(out, indent=1))
