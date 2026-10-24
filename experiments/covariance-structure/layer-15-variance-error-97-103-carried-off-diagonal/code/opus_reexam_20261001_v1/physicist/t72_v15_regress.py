#!/usr/bin/env python3
"""TAG opus-physicist T72 (numpy): V15 REGRESSION (lead 10-10 04:40Z). Which pushed term of the previous layer's post covariance explains the card's per-unit VARIANCE error?
Target per unit of layer L (15 on the kept rows |mu| <= 3.5 sigma; 14 on all units): dv / v = (v_card - v_true) / v_card, against the two independent per-unit truths
(analyst A13b halves A / B, 2 x 32M; network 100: the g19 P32M / indep32M pair). Candidates: p_X / v for the terms X of T71 (diag(W' X W) of each term of layer L-1's post
covariance) and resid = the card's total minus the sum of its terms. Both regressed per network on a baseline: the head's 126 inputs at layer L and a constant ('head')
or a constant only ('raw'). Statistic per candidate: partial R^2 of dv / v with the truth noise removed by the halves (cov(x, dA) cov(x, dB) / (var x cov(dA, dB))), the fitted
coefficient theta (dv = theta p_X; 1 would mean the term is missing entirely, -1 that it should not be there), its sign agreement over networks; jointly all candidates,
leave one network out. Implied final-error gain = share explained x the analyst's -9.7% (true per-unit variance at L15 alone; L14 -5.4%).
usage: t72_v15_regress.py"""
import math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent
WORTH = {15: 9.7, 14: 5.4}


def pair(r):
    a, b = O / f'analyst/fresh/pu/mps_truth_rank{r:04d}_A32M.npz', O / f'analyst/fresh/pu/mps_truth_rank{r:04d}_B32M.npz'
    if a.exists() and b.exists(): return np.load(a), np.load(b)
    a, b = P2 / f'public_g19_attribution_v1/mps_truth_rank{r:04d}_P32M.npz', P2 / f'public_g19_attribution_v1/mps_truth_rank{r:04d}_indep32M.npz'
    if a.exists() and b.exists(): return np.load(a), np.load(b)
    return None, None


def resid(Y, B):
    co, *_ = np.linalg.lstsq(B, Y, rcond=None); return Y - B @ co


DATA = {}
for f in sorted(HERE.glob('v15dump_rank*.npz')):
    r = int(f.name[12:16]); TA, TB = pair(r)
    if TA is None: continue
    D = np.load(f)
    for L in (15, 14):
        if f'v_{L}' not in D.files: continue
        v = D[f'v_{L}']; rows = D['kept_15'] > 0.5 if L == 15 else np.ones(len(v), bool)
        dA, dB = (v - TA['mc_pre_v'][L]) / v, (v - TB['mc_pre_v'][L]) / v
        names = sorted(k[2:-3] for k in D.files if k.startswith('p_') and k.endswith(f'_{L}') and k != f'p_total_{L}')
        C = {k: D[f'p_{k}_{L}'] / v for k in names}
        card = [k for k in names if k not in ('trunc', 'r21', 'r34', 's2121', 's33')]
        C['resid'] = D[f'p_total_{L}'] / v - sum(C[k] for k in card)
        C['offdiag'] = sum(C[k] for k in card if k != 'diag_var')      # the whole off-diagonal post covariance pushed
        for base in ('raw', 'head'):
            B = np.ones((rows.sum(), 1)) if base == 'raw' else np.column_stack([np.ones(rows.sum()), D[f'H_{L}'][rows]])
            B = B[:, B.std(0) > 0] if base == 'head' else B
            if base == 'head': B = np.column_stack([np.ones(rows.sum()), B])
            Y = resid(np.column_stack([dA[rows], dB[rows]]), B); Xs = {k: resid(x[rows][:, None], B)[:, 0] for k, x in C.items() if np.isfinite(x).all() and x[rows].std() > 0}
            DATA[(L, base, r)] = (Y, Xs)
nets = sorted({k[2] for k in DATA})
print('T72 V15: networks %s' % nets)
for L in (15, 14):
    for base in ('raw', 'head'):
        keys = [k for k in DATA if k[0] == L and k[1] == base]
        if not keys: continue
        names = sorted(set.intersection(*[set(DATA[k][1]) for k in keys]))
        noise = np.mean([1 - (DATA[k][0][:, 0] @ DATA[k][0][:, 1]) / math.sqrt((DATA[k][0][:, 0] @ DATA[k][0][:, 0]) * (DATA[k][0][:, 1] @ DATA[k][0][:, 1])) for k in keys])
        print('\nlayer %d, baseline %s (%d networks; truth-noise share of dv/v after the baseline %.3f); candidate: partial R^2 (noise-free), theta, sign agreement, implied gain' % (
            L, base, len(keys), noise))
        rows = []
        for nm in names:
            r2s, ths = [], []
            for k in keys:
                (Y, Xs) = DATA[k]; x = Xs[nm]; ya, yb = Y[:, 0], Y[:, 1]; vx = x @ x; cab = ya @ yb
                if vx <= 0 or cab <= 0: continue
                r2s.append((x @ ya) * (x @ yb) / (vx * cab)); ths.append((x @ (ya + yb)) / 2 / vx)
            if not r2s: continue
            sg = np.sign(np.median(ths)); agree = int(sum(1 for z in ths if np.sign(z) == sg))
            rows.append((np.mean(r2s), nm, np.median(ths), agree, len(ths)))
        for r2, nm, th, agree, nn in sorted(rows, reverse=True):
            print('  %-8s R2 %.3f  theta %+.4f  (%d of %d same sign)  implied %.2f%%' % (nm, r2, th, agree, nn, -WORTH[L] * max(r2, 0)))
        # all card terms jointly (plus resid), leave one network out
        J = [nm for nm in names if nm not in ('offdiag',)]
        shares = []
        for k in keys:
            tr = [q for q in keys if q != k]
            XtX = sum(np.column_stack([DATA[q][1][nm] for nm in J]).T @ np.column_stack([DATA[q][1][nm] for nm in J]) for q in tr)
            Xty = sum(np.column_stack([DATA[q][1][nm] for nm in J]).T @ DATA[q][0].mean(1) for q in tr)
            beta = np.linalg.solve(XtX + 1e-9 * np.trace(XtX) / len(J) * np.eye(len(J)), Xty)
            Y, Xs = DATA[k]; p = np.column_stack([Xs[nm] for nm in J]) @ beta; ya, yb = Y[:, 0], Y[:, 1]
            shares.append(((p @ ya) + (p @ yb) - p @ p) / (ya @ yb))
        print('  JOINT (all terms, leave one network out): share %.3f (range %.3f..%.3f) -> implied %.2f%%' % (np.mean(shares), min(shares), max(shares), -WORTH[L] * max(np.mean(shares), 0)))

# DIAGONAL / OFF-DIAGONAL SPLIT of the variance error: dv_L = (W o W)' (pv_card - pv_true)_(L-1) + the error of the pushed OFF-diagonal post covariance.
# The diagonal part is exact per unit from the per-unit truths (post variance mc_post_var of layer L-1); energies with the truth noise removed by the halves.
import sys
sys.path.insert(0, str(O / 'trainer'))
from otr_common import data_paths
print('\nDIAGONAL / OFF-DIAGONAL SPLIT of dv / v (energy shares, truth noise removed): per network')
for f in sorted(HERE.glob('v15dump_rank*.npz')):
    r = int(f.name[12:16]); TA, TB = pair(r); D = np.load(f)
    if TA is None or 'pv_14' not in D.files or 'mc_post_var' not in TA.files: continue
    Wt = np.load(data_paths(r)[0], mmap_mode='r'); line = []
    for L in (15, 14):
        W2 = np.asarray(Wt[L], dtype=np.float64) ** 2; pv = D[f'pv_{L - 1}']
        if L == 15 and np.abs(W2.T @ pv - D['p_diag_var_15']).max() > 1e-6 * np.abs(D['p_diag_var_15']).max(): W2 = W2.T      # orientation check
        v = D[f'v_{L}']; rows = D['kept_15'] > 0.5 if L == 15 else np.ones(len(v), bool)
        dA, dB = (v - TA['mc_pre_v'][L]) / v, (v - TB['mc_pre_v'][L]) / v
        gA, gB = (W2.T @ (pv - TA['mc_post_var'][L - 1])) / v, (W2.T @ (pv - TB['mc_post_var'][L - 1])) / v
        oA, oB = dA - gA, dB - gB
        E = lambda a, b: float((a[rows] * b[rows]).mean())
        line.append('L%d: dv rms %.2e | diagonal part %.0f%% of dv energy, off-diagonal %.0f%%, cross %.0f%% | corr(diag part, dv) %.2f' % (
            L, math.sqrt(max(E(dA, dB), 0)), 100 * E(gA, gB) / E(dA, dB), 100 * E(oA, oB) / E(dA, dB), 100 * (E(gA, oB) + E(gB, oA)) / E(dA, dB),
            (E(gA, dB) + E(gB, dA)) / 2 / math.sqrt(max(E(gA, gB), 1e-300) * E(dA, dB))))
    print('  %d  %s' % (r, ' ; '.join(line)))
