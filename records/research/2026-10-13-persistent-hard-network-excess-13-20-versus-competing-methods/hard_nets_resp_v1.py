#!/usr/bin/env python3
"""TAG analyst hard networks, responsiveness on generated networks v1 (lead 10-13 00:1xZ: 'every physics or refit step hits the same networks less';
the board's target b): the generated-network analogue. S51's 128 networks (O/trainer/run_RAD_S51_gen/results, base + 17 arms, 'final_prediction'):
per network and arm the change of the noise-corrected final-layer log MSE against the base; for each arm that helps on average (mean change < 0) the
change is scaled by the arm's mean |change|; RESPONSIVENESS = the mean of those scaled changes over the helping arms (-1 = helped as much as the average
network; above -1 = helped less, the board's 'least improved'). Also the base's residual after the variance fit (the board's target a analogue).
Reports: corr(responsiveness, residual) (board: corr(a, b) = +0.56), the arms' agreement (mean correlation between arms' per-network changes), and the
mean-field features of hard_nets_v2.py against both.
usage: hard_nets_resp_v1.py   -> table on stdout, hard_nets_resp_v1.json"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[k] = '1'
import json, sys, math
from pathlib import Path
import numpy as np
from scipy.special import ndtr
HERE = Path(__file__).resolve().parent; O = HERE.parent; sys.path.insert(0, str(HERE / 'fresh'))
import loader
R = O / 'trainer/run_RAD_S51_gen/results'
labs = sorted({p.name.rsplit('_r', 1)[0] for p in R.glob('a_RS51_*_r71*.json')})
base = 'a_RS51_base'; arms = [a for a in labs if a != base]
seeds = sorted(int(p.name.rsplit('_r', 1)[1].split('.')[0]) for p in R.glob(f'{base}_r71*.json'))
T = {s: loader.truth(s, check_weights=False) for s in seeds}


def lmse(lab, s):
    p = R / f'{lab}_r{s}.json'
    if not p.exists(): return np.nan
    pred = np.asarray(json.load(open(p))['final_prediction'], dtype=np.float64).reshape(-1)[-1024:]; t = T[s]
    return math.log(((pred - t['mc_post'][15]) ** 2).mean() - (t['mc_post_var'][15] / float(t['N'])).mean())


b = np.array([lmse(base, s) for s in seeds]); E = {}
for a in arms:
    e = np.array([lmse(a, s) for s in seeds]) - b
    if np.isfinite(e).all(): E[a] = e
help_ = {a: e for a, e in E.items() if e.mean() < 0}
resp = np.mean([e / abs(e.mean()) for e in help_.values()], 0)
x = np.log([T[s]['mc_post_var'][15].mean() for s in seeds]); fit = np.polyfit(x, b, 1); res = b - np.polyval(fit, x)
rank = lambda v: np.argsort(np.argsort(v)).astype(float)
corr = lambda a, c: (float(np.corrcoef(a, c)[0, 1]), float(np.corrcoef(rank(a), rank(c))[0, 1]))
names = list(help_); agree = [float(np.corrcoef(help_[i], help_[j])[0, 1]) for n, i in enumerate(names) for j in names[n + 1:]]
print(f"{len(seeds)} networks {seeds[0]}-{seeds[-1]}; {len(E)} arms, {len(help_)} help on average ({', '.join(f'{a[7:]} {help_[a].mean() * 100:+.2f}%' for a in names)})")
print(f"agreement between helping arms (mean pairwise r of per-network changes): {np.mean(agree):+.2f}; responsiveness sd {resp.std():.2f}")
print(f"corr(responsiveness, base residual after the variance fit) r {corr(resp, res)[0]:+.2f} rho {corr(resp, res)[1]:+.2f}   (board: corr(a, b) = +0.56)")
phi = lambda a: np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)


def solve_s(mu, m):
    lo, hi = np.full_like(mu, 1e-6), np.full_like(mu, 1e3)
    for _ in range(60):
        s = np.sqrt(lo * hi); f = s * phi(mu / s) + mu * ndtr(mu / s) - m
        hi = np.where(f > 0, s, hi); lo = np.where(f > 0, lo, s)
    return np.sqrt(lo * hi)


F = {}
for s in seeds:
    m, var = T[s]['mc_post'], T[s]['mc_post_var']; w = loader.weights(s).astype(np.float64); al = {0: np.zeros(1024)}; sd = {0: np.linalg.norm(w[0], axis=0)}
    for l in range(1, 16):
        mu = m[l - 1] @ w[l]; sd[l] = solve_s(mu, m[l]); al[l] = mu / sd[l]
    f = {}
    for band, Ls in (('8-11', range(8, 12)), ('12-15', range(12, 16))):
        A = np.concatenate([al[l] for l in Ls]); f[f'dead {band}'] = np.mean(A < -2.33); f[f'kink {band}'] = np.mean(np.abs(A) < 1.5); f[f'alpha>4 {band}'] = np.mean(A > 4)
        ne = []
        for l in Ls:
            act = ndtr(al[l - 1]) > 0.01; v = np.where(act, var[l - 1], 0.0); W2 = w[l] ** 2; ne.append(np.median((v @ W2) ** 2 / np.maximum((v ** 2) @ (W2 ** 2), 1e-300)))
        f[f'N_eff median {band}'] = float(np.mean(ne))
    f['sigma growth L15 / L0'] = float(np.median(sd[15]) / np.median(sd[0])); f['sigma max / median L12-15'] = float(np.mean([sd[l].max() / np.median(sd[l]) for l in range(12, 16)]))
    F[s] = f
keys = list(F[seeds[0]])
print('feature: r / rho with responsiveness | with the base residual (|r| > 0.17 is about p < 0.05 at 128)')
tab = []
for k in keys:
    v = np.array([F[s][k] for s in seeds]); tab.append((k, *corr(v, resp), *corr(v, res)))
for k, a1, a2, b1, b2 in sorted(tab, key=lambda z: -abs(z[1])): print(f"  {k}: {a1:+.2f} / {a2:+.2f} | {b1:+.2f} / {b2:+.2f}")
(HERE / 'hard_nets_resp_v1.json').write_text(json.dumps(dict(seeds=seeds, responsiveness=resp.tolist(), residual=res.tolist(), arms={a: e.tolist() for a, e in E.items()},
                                                             agreement=float(np.mean(agree)), corr_resp_res=corr(resp, res), table=tab), indent=1))
