#!/usr/bin/env python3
"""TAG analyst hard networks on FRESH networks v1 (lead 2026-10-12 23:3xZ, second half): the board's public networks cannot be regenerated (hard_nets_v1.py),
so the weight features are taken on generated networks where weights, truths AND our card's predictions exist: the trainer's S52 base arm
(O/trainer/run_RAD_S52_gen/results/a_RS52_base_r<seed>.json, 'final_prediction'; seeds 7100712-7100805 = A43 16M + A44 8M).
Target per network: log of the final-layer MSE with the truth-noise term removed (mean of mc_post_var[15] / N). There is no rival consensus here, so
the residual after a fit on the network's own final-layer variance scale (log mean mc_post_var[15]) stands in for 'our excess'.
Features per network (single thread, light): from the exact truths, per unit and layer, the pre-activation mean mu = m_{l-1} @ W_l and the sd s solved
from the ReLU mean m_l = s phi(mu/s) + mu Phi(mu/s), giving alpha = mu / s: kink fraction (|alpha| < 1.5), dead fraction (alpha < -2), mean |alpha|
over layers 8-15; from 2048 Monte Carlo inputs, the post-activation covariance at layers 8, 12, 15: top-eigenvalue share (scale-mode strength),
participation ratio, share of the trace beyond rank 320; from the weights: top singular value (power iteration) and the column-mean norm, layers 8-15.
usage: hard_nets_fresh_v1.py   -> table on stdout and hard_nets_fresh_v1.json"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[k] = '1'
import json, sys, time, math
from pathlib import Path
import numpy as np
from scipy.special import ndtr
HERE = Path(__file__).resolve().parent; O = HERE.parent; sys.path.insert(0, str(HERE / 'fresh'))
import loader
RES = sorted((O / 'trainer/run_RAD_S52_gen/results').glob('a_RS52_base_r71*.json'))
phi = lambda a: np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)


def solve_s(mu, m):
    lo, hi = np.full_like(mu, 1e-6), np.full_like(mu, 1e3)
    for _ in range(60):                                   # m(s) = s phi(mu/s) + mu Phi(mu/s) increases with s
        s = np.sqrt(lo * hi); a = mu / s; f = s * phi(a) + mu * ndtr(a) - m
        hi = np.where(f > 0, s, hi); lo = np.where(f > 0, lo, s)
    return np.sqrt(lo * hi)


rows = []; t0 = time.time(); rng = np.random.default_rng(1)
for p in RES:
    seed = int(p.name.rsplit('_r', 1)[1].split('.')[0]); d = json.load(open(p)); pred = np.asarray(d['final_prediction'], dtype=np.float64).reshape(-1)[-1024:]
    t = loader.truth(seed, check_weights=False); m, var, N = t['mc_post'], t['mc_post_var'], float(t['N'])
    mse = float(((pred - m[15]) ** 2).mean() - (var[15] / N).mean()); w = loader.weights(seed)
    kink, dead, absa = [], [], []
    for l in range(8, 16):
        mu = m[l - 1] @ w[l]; s = solve_s(mu, m[l]); a = mu / s
        kink.append(np.mean(np.abs(a) < 1.5)); dead.append(np.mean(a < -2)); absa.append(np.mean(np.abs(a)))
    y = rng.standard_normal((2048, 1024)).astype(np.float32); cov = {}
    for l in range(16):
        y = np.maximum(y @ w[l], 0)
        if l in (8, 12, 15):
            c = np.cov(y.T.astype(np.float64)); ev = np.clip(np.linalg.eigvalsh(c)[::-1], 0, None); tr = ev.sum()
            cov[l] = (ev[0] / tr, tr ** 2 / (ev ** 2).sum(), ev[320:].sum() / tr)
    sig, colm = [], []
    for l in range(8, 16):
        v = rng.standard_normal(1024)
        for _ in range(25): v = w[l].T @ (w[l] @ v); v /= np.linalg.norm(v)
        sig.append(float(np.linalg.norm(w[l] @ v))); colm.append(float(np.linalg.norm(w[l].mean(0)) * math.sqrt(1024)))
    rows.append(dict(seed=seed, N=N, mse=mse, var15=float(var[15].mean()), kink=float(np.mean(kink)), dead=float(np.mean(dead)), absa=float(np.mean(absa)),
                     top8=cov[8][0], top12=cov[12][0], top15=cov[15][0], pr12=cov[12][1], pr15=cov[15][1], tail12=cov[12][2], tail15=cov[15][2],
                     sig=float(np.mean(sig)), colm=float(np.mean(colm))))
    if len(rows) % 10 == 0: print(json.dumps(dict(done=len(rows), seconds=round(time.time() - t0))), flush=True)
del w
y = np.log([r['mse'] for r in rows]); x = np.log([r['var15'] for r in rows]); b = np.polyfit(x, y, 1); res = y - np.polyval(b, x)
rank = lambda v: np.argsort(np.argsort(v)).astype(float)
corr = lambda a, c: (float(np.corrcoef(a, c)[0, 1]), float(np.corrcoef(rank(a), rank(c))[0, 1]))
names = dict(kink='kink fraction |alpha| < 1.5, L8-15', dead='dead fraction alpha < -2, L8-15', absa='mean |alpha|, L8-15', top8='top-eigenvalue share L8',
             top12='top-eigenvalue share L12', top15='top-eigenvalue share L15', pr12='participation ratio L12', pr15='participation ratio L15',
             tail12='trace share beyond rank 320, L12', tail15='trace share beyond rank 320, L15', sig='top singular value of W, L8-15', colm='column-mean norm of W, L8-15')
tab = []
for k, nm in names.items():
    f = np.array([r[k] for r in rows]); tab.append((nm, *corr(f, y), *corr(f, res)))
tab.sort(key=lambda z: -abs(z[3]))
print(f"{len(rows)} networks; log MSE (noise-corrected) on log final-layer variance: slope {b[0]:.2f}, the residual's sd {res.std() * 100:.1f}%")
print('feature: r / rho with log MSE | r / rho with the residual after the variance fit (|r| > 0.20 is about p < 0.05 at 94 networks)')
for nm, r1, s1, r2, s2 in tab: print(f"  {nm}: {r1:+.2f} / {s1:+.2f} | {r2:+.2f} / {s2:+.2f}")
(HERE / 'hard_nets_fresh_v1.json').write_text(json.dumps(dict(rows=rows, slope=list(b), table=[dict(feature=t[0], r_mse=t[1], rho_mse=t[2], r_res=t[3], rho_res=t[4]) for t in tab]), indent=1))
