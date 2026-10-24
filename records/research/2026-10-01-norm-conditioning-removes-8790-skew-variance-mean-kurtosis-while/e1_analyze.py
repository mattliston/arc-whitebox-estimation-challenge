#!/usr/bin/env python3
"""TAG opus-reexam E1 analysis: law of total cumulance per unit, per layer, w.r.t. a collective variable (see e1_gain_conditioning_mc.py).
Prints, per layer and collective variable: mean and across-unit spread (noise-corrected) of the standardised skew g1 = k3/v^1.5 and
excess kurtosis g2 = k4/v^2 of the pre-activation, for (a) the full law, (b) the within-bin (conditional) part, and the correlation
between them; plus the same for the gain-normalised variable u = z/|y_prev| under the plain and the gain-tilted measure.
usage: e1_analyze.py <npz>"""
import sys, numpy as np
d = np.load(sys.argv[1]); N = float(d['N']); B = int(d['B']); L = 16
def cum_from_raw(m1, m2, m3, m4):
    v = m2 - m1 ** 2
    k3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    return v, k3, mu4 - 3 * v ** 2
def sig(x, noise_var):
    s2 = x.var() - noise_var
    return np.sqrt(max(s2, 0.0))
def wc(p, *xs):
    """joint cumulant across bins (weights p[:,None]) of 2..4 per-bin arrays, per unit."""
    mean = [(p[:, None] * x).sum(0) for x in xs]
    c = [x - m for x, m in zip(xs, mean)]
    if len(xs) == 2: return (p[:, None] * c[0] * c[1]).sum(0)
    if len(xs) == 3: return (p[:, None] * c[0] * c[1] * c[2]).sum(0)
    if len(xs) == 4:
        e = lambda a, b: (p[:, None] * a * b).sum(0)
        return ((p[:, None] * c[0] * c[1] * c[2] * c[3]).sum(0) - e(c[0], c[1]) * e(c[2], c[3]) - e(c[0], c[2]) * e(c[1], c[3])
                - e(c[0], c[3]) * e(c[1], c[2]))
print(f'N={N:.0f} bins={B}; noise sd per unit: g1 {np.sqrt(6 / N):.4f}, g2 {np.sqrt(24 / N):.4f}')
for kind in ('norm', 'pc1'):
    print(f'\n=== collective variable: {kind} ===')
    print(' L |  g1: mean  spread | within: mean spread  corr | g2: mean  spread | within: mean  spread  corr | 3Var(var)/v2 mean spread | share of g2 spread^2 left within')
    for l in range(L):
        cnt = d[f'{kind}_{l}_cnt']; p = cnt / cnt.sum()
        S = [d[f'{kind}_{l}_{i}'] for i in range(4)]
        m = [s / cnt[:, None] for s in S]
        vb, k3b, k4b = cum_from_raw(*m)
        M = [(p[:, None] * x).sum(0) for x in m]
        v, k3, k4 = cum_from_raw(*M)
        g1 = k3 / v ** 1.5; g2 = k4 / v ** 2
        w3 = (p[:, None] * k3b).sum(0) / v ** 1.5; w4 = (p[:, None] * k4b).sum(0) / v ** 2
        vv = 3 * wc(p, vb, vb) / v ** 2
        # consistency check of the total-cumulance identity
        k4_id = (p[:, None] * k4b).sum(0) + 4 * wc(p, m[0], k3b) + 3 * wc(p, vb, vb) + 6 * wc(p, vb, m[0], m[0]) + wc(p, m[0], m[0], m[0], m[0])
        assert np.allclose(k4_id, k4, rtol=1e-6, atol=1e-9 * abs(k4).max() + 1e-12), abs(k4_id - k4).max()
        n1, n2 = 6 / N, 24 / N
        s_g1, s_w3 = sig(g1, n1), sig(w3, n1); s_g2, s_w4 = sig(g2, n2), sig(w4, n2)
        print(f'{l:2d} | {g1.mean():+.4f} {s_g1:.4f} | {w3.mean():+.4f} {s_w3:.4f} {np.corrcoef(g1, w3)[0, 1]:+.2f} | '
              f'{g2.mean():+.4f} {s_g2:.4f} | {w4.mean():+.4f} {s_w4:.4f} {np.corrcoef(g2, w4)[0, 1]:+.2f} | {vv.mean():+.4f} {sig(vv, 0):.4f} | '
              f'{(s_w4 / s_g2) ** 2 if s_g2 > 0 else float("nan"):.2f}')
print('\n=== gain-normalised variable u = z/|y_prev| ===')
print(' L | plain g1: mean spread | plain g2: mean spread | tilted g1: mean spread | tilted g2: mean spread | raw g2 mean spread (norm bins total)')
for l in range(L):
    cntN, sumn = d[f'usum_{l}']
    out = []
    for key, den in (('u', cntN), ('ut', sumn)):
        m = [d[f'{key}_{l}_{i}'] / den for i in range(4)]
        v, k3, k4 = cum_from_raw(*m)
        g1 = k3 / v ** 1.5; g2 = k4 / v ** 2
        out += [g1.mean(), sig(g1, 6 / N), g2.mean(), sig(g2, 24 / N)]
    cnt = d[f'norm_{l}_cnt']; p = cnt / cnt.sum()
    M = [(p[:, None] * (d[f'norm_{l}_{i}'] / cnt[:, None])).sum(0) for i in range(4)]
    v, k3, k4 = cum_from_raw(*M); g2 = k4 / v ** 2
    print(f'{l:2d} | {out[0]:+.4f} {out[1]:.4f} | {out[2]:+.4f} {out[3]:.4f} | {out[4]:+.4f} {out[5]:.4f} | {out[6]:+.4f} {out[7]:.4f} | {g2.mean():+.4f} {sig(g2, 24 / N):.4f}')
