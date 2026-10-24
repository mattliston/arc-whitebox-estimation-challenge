#!/usr/bin/env python3
"""TAG opus-physicist T16 (open loop, truth inputs): how much of the per-unit mean is carried by each ORDER of the cumulant expansion.
For each layer the mean formula is evaluated with the TRUE per-unit pre-activation mean, variance and cumulants (32M-sample truth,
public_g19 P32M; the 5th and 6th cumulants from the stored raw moments) and compared with the true post mean (organizers' 1e9 truth at
every layer). Orders (eps = one power of 1/sqrt(n)):  1: k3;  2: k4, k3^2;  3: k5, k3 k4, k3^3;  4: k6, k4^2, k3 k5, k3^2 k4, k3^4.
   E relu = sum over terms of coefficient x E_G relu^(n),  E_G relu^(n) = (-1)^n He_(n-2)(alpha) phi / sigma^(n-1)  (n >= 2).
The card's formula is 'k3 + k4' (order 2 without k3^2). Reported: mean squared error over the 1,024 units, x 1e9, and the same when the
noise of the 32M inputs is removed by the split of a second independent 32M draw (cross product of the two errors).
usage: t16_mean_orders.py <rank>"""
import sys, math, json
import numpy as np
import pcard_light as pl
rank = int(sys.argv[1]); A, B = pl.pu(rank, 'P32M'), pl.pu(rank, 'indep32M'); tp = pl.truth(rank)


def cumulants(z, l):
    m, v, k3, k4 = (z[k][l].astype(np.float64) for k in ('mc_pre_m', 'mc_pre_v', 'mc_pre_k3', 'mc_pre_k4'))
    r5, r6 = z['pre_m5raw'][l].astype(np.float64), z['pre_m6raw'][l].astype(np.float64)
    raw = [np.ones_like(m), m, v + m * m, k3 + 3 * m * v + m ** 3, k4 + 3 * v * v + 4 * m * k3 + 6 * m * m * v + m ** 4, r5, r6]
    cen = lambda n: sum(math.comb(n, k) * raw[k] * (-m) ** (n - k) for k in range(n + 1))
    mu5, mu6 = cen(5), cen(6); mu4 = k4 + 3 * v * v
    return m, v, k3, k4, mu5 - 10 * k3 * v, mu6 - 15 * mu4 * v - 10 * k3 * k3 + 30 * v ** 3


def terms(c):
    m, v, k3, k4, k5, k6 = c; s = np.sqrt(v); a = m / s; ph = np.exp(-a * a / 2) / math.sqrt(2 * math.pi); g = np.array([0.5 * math.erfc(-x / math.sqrt(2)) for x in a])
    He = [np.ones_like(a), a]
    for n in range(2, 12): He.append(a * He[-1] - (n - 1) * He[-2])
    G = lambda n: (-1) ** n * He[n - 2] * ph / s ** (n - 1)
    return dict(gauss=s * ph + m * g, k3=k3 * G(3) / 6, k4=k4 * G(4) / 24, k3k3=k3 * k3 * G(6) / 72, k5=k5 * G(5) / 120, k3k4=k3 * k4 * G(7) / 144, k3k3k3=k3 ** 3 * G(9) / 1296,
                k6=k6 * G(6) / 720, k4k4=k4 * k4 * G(8) / 1152, k3k5=k3 * k5 * G(8) / 720, k3k3k4=k3 * k3 * k4 * G(10) / 1728, k3k3k3k3=k3 ** 4 * G(12) / 31104)


SETS = (('gauss', ['gauss']), ('+ k3', ['k3']), ('+ k4 (the card form)', ['k4']), ('+ k3^2', ['k3k3']), ('+ k5', ['k5']), ('+ k3 k4', ['k3k4']), ('+ k3^3', ['k3k3k3']),
        ('+ k6', ['k6']), ('+ k4^2 + k3 k5 + k3^2 k4 + k3^4', ['k4k4', 'k3k5', 'k3k3k4', 'k3k3k3k3']))
for l in range(1, 16):
    ta, tb = terms(cumulants(A, l)), terms(cumulants(B, l)); row = dict(rank=rank, layer=l); fa = fb = 0.0
    for name, keys in SETS:
        fa = fa + sum(ta[k] for k in keys); fb = fb + sum(tb[k] for k in keys)
        row[name] = (round(1e9 * float(((fa - tp[l]) ** 2).mean()), 3), round(1e9 * float(((fa - tp[l]) * (fb - tp[l])).mean()), 3))
    row['rms of single terms x1e6'] = {k: round(1e6 * float(np.sqrt((ta[k] ** 2).mean())), 2) for k in ('k3', 'k4', 'k3k3', 'k5', 'k3k4', 'k3k3k3', 'k6', 'k4k4')}
    row['t5_rms, t6_rms (A; noise by the A-B split)'] = [round(float(np.sqrt(((cumulants(A, l)[i] / cumulants(A, l)[1] ** (n / 2)) ** 2).mean())), 4) for i, n in ((4, 5), (5, 6))] + \
        [round(float(np.sqrt((((cumulants(A, l)[i] - cumulants(B, l)[i]) / cumulants(A, l)[1] ** (n / 2)) ** 2).mean() / 2)), 4) for i, n in ((4, 5), (5, 6))]
    print(json.dumps(row), flush=True)
