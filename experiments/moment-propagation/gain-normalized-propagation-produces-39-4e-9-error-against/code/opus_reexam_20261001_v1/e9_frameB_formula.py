#!/usr/bin/env python3
"""TAG opus-reexam E9: how large are the non-Gaussian corrections to the post-activation MEAN in the gain-normalised frame?
A ReLU network without biases is homogeneous: y_l = rho * relu(W' u) with rho = |y_{l-1}|, u = y_{l-1}/rho, so
  E[y_l] = E[rho] * E_tilted[relu(zeta)],   zeta = W' u = z_l / rho,   tilted = weights rho / E[rho].
The e1 Monte Carlo files hold per-unit power sums of zeta (plain and tilted) and of z (binned; summed over bins = plain lab frame).
Per layer, with EXACT per-unit cumulants from the same samples, the mean formula (Gaussian; + third cumulant; + fourth) is evaluated
 lab frame : m = F(mu, v, k3, k4) of z
 normalised: m = E[rho] * F(mu, v, k3, k4) of zeta under the tilted measure
and compared with the organizers' 1e9-sample truth. Shows how much of the correction beyond Gaussian each frame needs.
usage: e9_frameB_formula.py <e1 npz> <rank>"""
import sys, numpy as np
from math import sqrt, pi
from pathlib import Path
from scipy.special import ndtr
d = np.load(sys.argv[1]); rank = int(sys.argv[2]); L = 16
P2 = Path(__file__).resolve().parent.parent
tp = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
N = float(d['N'])
phi = lambda a: np.exp(-a * a / 2) / sqrt(2 * pi)
def cum(s1, s2, s3, s4, wsum, ref):
    m1, m2, m3, m4 = s1 / wsum, s2 / wsum, s3 / wsum, s4 / wsum      # raw moments of the centred variable
    v = m2 - m1 ** 2; k3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    k4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4 - 3 * v ** 2
    return ref + m1, v, k3, k4
def F(mu, v, k3=None, k4=None):
    s = np.sqrt(np.maximum(v, 1e-30)); a = mu / s
    out = s * (phi(a) + a * ndtr(a))
    if k3 is not None: out = out - k3 * a * phi(a) / (6 * v)            # (k3/6) E[relu'''] , relu''' = delta'
    if k4 is not None: out = out + k4 * (a * a - 1) * phi(a) / (24 * s ** 3)   # (k4/24) E[relu''''], relu'''' = delta''
    return out
def wstat(x, w):
    mean = (w * x).sum() / w.sum(); return mean, np.sqrt(max((w * (x - mean) ** 2).sum() / w.sum(), 0))
print(f'network {rank}, {N:.0f} samples; MSE of the post mean vs organizer truth, x1e-9 (truth noise floor about 0.07-0.7)')
print(' L | lab: Gauss   +k3    +k3+k4 | normalised: Gauss   +k3    +k3+k4 | E[rho] | skew spread (mean-relevant weight): lab  norm | kurtosis mean, spread (weighted): lab | norm')
for l in range(1, L):
    cnt = d[f'norm_{l}_cnt']; tot = cnt.sum()
    S = [d[f'norm_{l}_{i}'].sum(0) for i in range(4)]
    mu, v, k3, k4 = cum(*S, tot, d[f'ref_{l}'])
    us = d[f'usum_{l}']; Erho = us[1] / us[0]
    T = [d[f'ut_{l}_{i}'] for i in range(4)]
    mub, vb, k3b, k4b = cum(*T, us[1], d[f'uref_{l}'])
    t = tp[l]
    e = lambda m: ((m - t) ** 2).mean() * 1e9
    a = mu / np.sqrt(v); ab = mub / np.sqrt(vb)
    w3 = (a * phi(a)) ** 2 + 1e-14; w4 = ((a * a - 1) * phi(a)) ** 2 + 1e-14
    g1, g1b = k3 / v ** 1.5, k3b / vb ** 1.5; g2, g2b = k4 / v ** 2, k4b / vb ** 2
    print(f'{l:2d} | {e(F(mu, v)):9.2f} {e(F(mu, v, k3)):8.2f} {e(F(mu, v, k3, k4)):7.2f} | {e(Erho * F(mub, vb)):9.2f} {e(Erho * F(mub, vb, k3b)):8.2f} {e(Erho * F(mub, vb, k3b, k4b)):7.2f} | {Erho:7.3f} | '
          f'{np.sqrt((w3 * g1 ** 2).sum() / w3.sum()):.4f} {np.sqrt((w3 * g1b ** 2).sum() / w3.sum()):.4f} | {wstat(g2, w4)[0]:+.4f} {wstat(g2, w4)[1]:.4f} | {wstat(g2b, w4)[0]:+.4f} {wstat(g2b, w4)[1]:.4f}')
