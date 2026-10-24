#!/usr/bin/env python3
"""TAG analyst hard networks v2 (lead + physicist 10-13 00:1xZ): the feature table, FIRST CUT (mean-field features; no Monte Carlo yet).
The board's public networks have no weights we can rebuild (board network k is NOT observer store row k; their contest seed is undisclosed), so:
 PART A (board, targets a and b): (a) our final-layer residual vs the four leaders (#335633 against marius #335629, oliver #335467, Luna #335625,
   J2W #335438), (b) our improvement rate since S36 (log final MSE #335633 - #333429, demeaned); features: our billed flops per network (#335633)
   and its change since S36, each leader's flops per network.
 PART B (generated networks with weights and truths, target c and a fresh stand-in): the physicist's mean-field features at layers 8-15 from the
   exact truths (pre-activation mean mu = m_{l-1} @ W_l; sd s from the ReLU mean m_l = s phi(mu/s) + mu Phi(mu/s); alpha = mu / s): dead fraction
   (alpha < -2.33), kink fraction (|alpha| < 1.5), alpha > 4 fraction, threshold densities (|alpha| in [1.4, 1.6] and [2.2, 2.45]), the effective fan-in
   N_eff_i = (sum_j W_ji^2 v_j)^2 / sum_j W_ji^4 v_j^2 over active inputs (Phi(alpha_j) > 0.01, v_j = the truth's post variance; median and 10th
   percentile), sigma growth (median s at L15 over L0), sigma's max / median at L12-15, W column / row means in units of their iid size (rms z).
   Networks: gbeta2's eight 7100584-591 (target c = gbeta2's change of the final MSE, %) and the 94 S52-base networks 7100712-805 (target = the
   noise-corrected final-MSE residual after the variance fit, as hard_nets_fresh_v1.py).
usage: hard_nets_v2.py   -> tables on stdout, hard_nets_v2.json"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[k] = '1'
import json, sys, math
from pathlib import Path
import numpy as np
from scipy.special import ndtr
HERE = Path(__file__).resolve().parent; O = HERE.parent; sys.path.insert(0, str(HERE / 'fresh'))
import loader
rank = lambda v: np.argsort(np.argsort(v)).astype(float)
corr = lambda a, c: (float(np.corrcoef(a, c)[0, 1]), float(np.corrcoef(rank(a), rank(c))[0, 1]))
out = {}
# ---------------- PART A: board
rows = {}
for l in open(HERE / 'scan_merged.jsonl'):
    r = json.loads(l)
    if r.get('http') == 200: rows[r['id']] = r
fin = lambda i: np.array([x[1] for x in sorted(rows[i]['pub'])]); fl = lambda i: np.log(np.array(rows[i]['flops'][:50], dtype=float))
L4 = {'marius': 335629, 'oliver': 335467, 'Luna': 335625, 'J2W': 335438}
cons = np.mean([np.log(fin(i)) for i in L4.values()], 0); ta = np.log(fin(335633)) - cons; ta -= ta.mean()
tb = np.log(fin(335633)) - np.log(fin(333429)); tb -= tb.mean()
feats = {'our flops (log)': fl(335633), 'our flops change since S36 (log)': fl(335633) - fl(333429)}
feats.update({f'{t} flops (log)': fl(i) for t, i in L4.items()})
print(f"PART A (board, 50 public networks; |r| > 0.28 is about p < 0.05). target a = residual vs leaders (sd {ta.std() * 100:.1f}%), b = improvement since S36 (sd {tb.std() * 100:.1f}%); corr(a, b) = {corr(ta, tb)[0]:+.2f}")
weak = [15, 40, 16, 14]
for k, f in feats.items():
    z = (f - f.mean()) / f.std()
    print(f"  {k}: a r {corr(f, ta)[0]:+.2f} rho {corr(f, ta)[1]:+.2f} | b r {corr(f, tb)[0]:+.2f} rho {corr(f, tb)[1]:+.2f} | z on 15/40/16/14: " + ' '.join(f'{z[i]:+.1f}' for i in weak))
out['A'] = {k: dict(a=corr(f, ta), b=corr(f, tb)) for k, f in feats.items()}
# ---------------- PART B: generated networks, mean-field features
phi = lambda a: np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)


def solve_s(mu, m):
    lo, hi = np.full_like(mu, 1e-6), np.full_like(mu, 1e3)
    for _ in range(60):
        s = np.sqrt(lo * hi); f = s * phi(mu / s) + mu * ndtr(mu / s) - m
        hi = np.where(f > 0, s, hi); lo = np.where(f > 0, lo, s)
    return np.sqrt(lo * hi)


def features(seed):
    t = loader.truth(seed, check_weights=False); m, var = t['mc_post'], t['mc_post_var']; w = loader.weights(seed).astype(np.float64)
    al, sd = {}, {}
    sd[0] = np.linalg.norm(w[0], axis=0); al[0] = np.zeros(1024)
    for l in range(1, 16):
        mu = m[l - 1] @ w[l]; sd[l] = solve_s(mu, m[l]); al[l] = mu / sd[l]
    f = {}
    for band, Ls in (('8-11', range(8, 12)), ('12-15', range(12, 16))):
        A = np.concatenate([al[l] for l in Ls])
        f[f'dead {band}'] = np.mean(A < -2.33); f[f'kink {band}'] = np.mean(np.abs(A) < 1.5); f[f'alpha>4 {band}'] = np.mean(A > 4)
        f[f'|alpha| in [1.4,1.6] {band}'] = np.mean((np.abs(A) >= 1.4) & (np.abs(A) <= 1.6)); f[f'|alpha| in [2.2,2.45] {band}'] = np.mean((np.abs(A) >= 2.2) & (np.abs(A) <= 2.45))
        ne_med, ne_p10 = [], []
        for l in Ls:
            act = ndtr(al[l - 1]) > 0.01; v = np.where(act, var[l - 1], 0.0); W2 = w[l] ** 2
            ne = (v @ W2) ** 2 / np.maximum((v ** 2) @ (W2 ** 2), 1e-300); ne_med.append(np.median(ne)); ne_p10.append(np.percentile(ne, 10))
        f[f'N_eff median {band}'] = float(np.mean(ne_med)); f[f'N_eff p10 {band}'] = float(np.mean(ne_p10))
        zc = [np.sqrt(np.mean((w[l].mean(0) * math.sqrt(1024) / w[l].std()) ** 2)) for l in Ls]; zr = [np.sqrt(np.mean((w[l].mean(1) * math.sqrt(1024) / w[l].std()) ** 2)) for l in Ls]
        f[f'W column-mean rms z {band}'] = float(np.mean(zc)); f[f'W row-mean rms z {band}'] = float(np.mean(zr))
    f['sigma growth median L15 / L0'] = float(np.median(sd[15]) / np.median(sd[0]))
    f['sigma max / median L12-15'] = float(np.mean([sd[l].max() / np.median(sd[l]) for l in range(12, 16)]))
    return {k: float(v) for k, v in f.items()}, t


GB = dict(zip(range(7100584, 7100592), [0.85, -0.94, 0.81, -0.52, 0.08, -0.17, -1.95, 0.04]))
fb = {s: features(s)[0] for s in GB}
keys = list(next(iter(fb.values())).keys()); tc = np.array([GB[s] for s in GB])
print(f"PART B1: gbeta2's eight networks 7100584-591, target c = gbeta2's change of the final MSE (%); with 8 networks only |r| > 0.71 is p < 0.05")
helped, hurt = [7100585, 7100587, 7100590], [7100584, 7100586]
tabc = sorted(((k, *corr(np.array([fb[s][k] for s in GB]), tc)) for k in keys), key=lambda z: -abs(z[1]))
for k, r_, rho in tabc:
    print(f"  {k}: r {r_:+.2f} rho {rho:+.2f} | helped (585 / 587 / 590) mean {np.mean([fb[s][k] for s in helped]):.4g} vs hurt (584 / 586) {np.mean([fb[s][k] for s in hurt]):.4g}")
out['B1'] = dict(features=fb, table=[dict(feature=k, r=r_, rho=rho) for k, r_, rho in tabc])
RES = sorted((O / 'trainer/run_RAD_S52_gen/results').glob('a_RS52_base_r71*.json'))
fr, ys, xs = {}, [], []
for p in RES:
    seed = int(p.name.rsplit('_r', 1)[1].split('.')[0]); d = json.load(open(p)); pred = np.asarray(d['final_prediction'], dtype=np.float64).reshape(-1)[-1024:]
    f, t = features(seed); fr[seed] = f
    ys.append(math.log(((pred - t['mc_post'][15]) ** 2).mean() - (t['mc_post_var'][15] / float(t['N'])).mean())); xs.append(math.log(t['mc_post_var'][15].mean()))
ys, xs = np.array(ys), np.array(xs); b = np.polyfit(xs, ys, 1); res = ys - np.polyval(b, xs)
seeds = sorted(fr)
tabf = sorted(((k, *corr(np.array([fr[s][k] for s in seeds]), res)) for k in keys), key=lambda z: -abs(z[1]))
print(f"PART B2: {len(seeds)} S52-base generated networks, target = our noise-corrected final-MSE residual after the variance fit (|r| > 0.20 is about p < 0.05)")
for k, r_, rho in tabf: print(f"  {k}: r {r_:+.2f} rho {rho:+.2f}")
out['B2'] = dict(table=[dict(feature=k, r=r_, rho=rho) for k, r_, rho in tabf])
(HERE / 'hard_nets_v2.json').write_text(json.dumps(out, indent=1))
