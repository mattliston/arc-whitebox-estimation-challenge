#!/usr/bin/env python3
"""TAG opus-reexam E3: where does the torch S6b card's per-unit skew (k3) and kurtosis (k4) error sit: in the collective-gain part
(between-norm-bin part of the law of total cumulance, E1) or in the bulk part (within-bin)?
Per layer: card error rms (standardised by truth v), truth spreads of the gain part A and bulk part W, least-squares coefficients of the
error on (A, W) with R^2, and how the card's own carriers line up: gain carrier 3 v b vs A; bank + star vs W.
All statistics also for the mean-relevant weighting w = (a phi(a))^2 for k3 and ((a^2-1) phi(a))^2 for k4, a = mu/sigma.
usage: e3_card_vs_truth_split.py <e1 npz> <e2 npz>"""
import sys, numpy as np
d = np.load(sys.argv[1]); c = np.load(sys.argv[2]); N = float(d['N']); L = 16
def cum_from_raw(m1, m2, m3, m4):
    v = m2 - m1 ** 2; k3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    return v, k3, mu4 - 3 * v ** 2
def wrms(x, w): return np.sqrt((w * x * x).sum() / w.sum())
def fit(y, X, w):
    Xw = X * np.sqrt(w)[:, None]; yw = y * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    res = yw - Xw @ coef
    return coef, 1 - (res ** 2).sum() / (yw ** 2).sum()
phi = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
print(f'N={N:.0f}: per-unit noise g1 {np.sqrt(6 / N):.4f} g2 {np.sqrt(24 / N):.4f}')
for label in ('uniform weight', 'mean-relevant weight'):
    print(f'\n##### SKEW (k3 / v^1.5), {label} #####')
    print(' L | truth rms: total  gainA  bulkW | card err rms (rel to total) | err ~ a*A + b*W: a      b     R2 | card b-term vs A: slope corr | bank+star vs W: slope corr | bank+star vs (W + A - bterm_fit): corr')
    for l in range(1, L):
        cnt = d[f'norm_{l}_cnt']; p = cnt / cnt.sum()
        m = [d[f'norm_{l}_{i}'] / cnt[:, None] for i in range(4)]
        vb, k3b, k4b = cum_from_raw(*m)
        M = [(p[:, None] * x).sum(0) for x in m]
        v, k3, k4 = cum_from_raw(*M)
        s3 = v ** 1.5
        W3 = (p[:, None] * k3b).sum(0) / s3; T3 = k3 / s3; A3 = T3 - W3
        mu = c[f'mu_{l}']; vc = c[f'v_{l}']; a = mu / np.sqrt(vc)
        w = np.ones_like(a) if label.startswith('uniform') else (a * phi(a)) ** 2 + 1e-12
        kc = c[f'k3raw_{l}'] if l == 15 else c[f'k3_{l}']
        e = kc / s3 - T3
        bt = c[f'k3b_{l}'] / s3; bs = (c[f'k3bank_{l}'] + c[f'k3star_{l}']) / s3
        coef, r2 = fit(e, np.stack([A3, W3], 1), w)
        sl_b = (w * bt * A3).sum() / (w * A3 * A3).sum(); co_b = (w * bt * A3).sum() / np.sqrt((w * bt * bt).sum() * (w * A3 * A3).sum())
        sl_s = (w * bs * W3).sum() / (w * W3 * W3).sum(); co_s = (w * bs * W3).sum() / np.sqrt((w * bs * bs).sum() * (w * W3 * W3).sum())
        tgt = T3 - bt; co_t = (w * bs * tgt).sum() / np.sqrt((w * bs * bs).sum() * (w * tgt * tgt).sum())
        print(f'{l:2d} | {wrms(T3, w):.4f} {wrms(A3, w):.4f} {wrms(W3, w):.4f} | {wrms(e, w):.4f} ({wrms(e, w) / wrms(T3, w):.3f}) | {coef[0]:+.3f} {coef[1]:+.3f} {r2:.2f} | '
              f'{sl_b:.3f} {co_b:.3f} | {sl_s:.3f} {co_s:.3f} | {co_t:.3f}')
    print(f'\n##### KURTOSIS (k4 / v^2), {label} #####')
    print(' L | truth: mean  spread | gainA: mean spread | bulkW: mean spread | card k4/v2 (radial) | card err rms | err(centred) ~ a*A + b*W: a b R2')
    for l in range(1, L):
        cnt = d[f'norm_{l}_cnt']; p = cnt / cnt.sum()
        m = [d[f'norm_{l}_{i}'] / cnt[:, None] for i in range(4)]
        vb, k3b, k4b = cum_from_raw(*m)
        M = [(p[:, None] * x).sum(0) for x in m]
        v, k3, k4 = cum_from_raw(*M)
        T4 = k4 / v ** 2; W4 = (p[:, None] * k4b).sum(0) / v ** 2; A4 = T4 - W4
        mu = c[f'mu_{l}']; vc = c[f'v_{l}']; a = mu / np.sqrt(vc)
        w = np.ones_like(a) if label.startswith('uniform') else ((a * a - 1) * phi(a)) ** 2 + 1e-12
        rad = float(c[f'radial_{l}'])
        e = rad * vc ** 2 / v ** 2 - T4
        wm = lambda x: (w * x).sum() / w.sum()
        ws = lambda x: np.sqrt(max(wm((x - wm(x)) ** 2) - 24 / N, 0))
        coef, r2 = fit(e - wm(e), np.stack([A4 - wm(A4), W4 - wm(W4)], 1), w)
        print(f'{l:2d} | {wm(T4):+.4f} {ws(T4):.4f} | {wm(A4):+.4f} {ws(A4):.4f} | {wm(W4):+.4f} {ws(W4):.4f} | {rad:.4f} | {wrms(e, w):.4f} | {coef[0]:+.3f} {coef[1]:+.3f} {r2:.2f}')
