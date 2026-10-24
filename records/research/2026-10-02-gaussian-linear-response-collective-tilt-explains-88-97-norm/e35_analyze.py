#!/usr/bin/env python3
"""TAG opus-reexam E35 analysis (network 35, layer 12 -> 13): at which step does the norm-coupling matrix leave the simple law?
 A  pre pairs, post norm against pre pairs, pre norm :  Mmix = E[z~ z~' R~]  on  Mpre = E[z~ z~' S~]   (is the gamma-weighted contraction the uniform one?)
 B  the ReLU                                         :  Mpost = E[x~ x~' R~] on  G Mmix G  (+ the row forms d t g C, (d/2) diag(Mmix) g C)
 C  the push                                         :  Mnext = E[z'~ z'~' S'~] on 2 W' Mpost W
Each: cross-half share of the target's NETWORK-SPECIFIC off-diagonal energy explained (isotropic and mean forms of the target's own space projected
out of target and predictors), and the fitted coefficients. The Gaussian pairings are left inside the M's (they are forms or transported exactly)."""
import sys, json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(HERE / 'physicist'))
import pcard as pc
rank = 35; l = 12; n = 1024; off = ~np.eye(n, dtype=bool)
D = np.load(HERE / f'e35_rank{rank:04d}_L{l}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
f = lambda k: MC[k].astype(np.float64); sym = lambda x: (x + x.T) / 2
C, mu, Cp, mp, Cn, mun = f(f'cov_pre_{l}'), f(f'mean_pre_{l}'), f(f'cov_post_{l}'), f(f'mean_post_{l}'), f(f'cov_pre_{l + 1}'), f(f'mean_pre_{l + 1}')
W = pc.weights(rank)[l + 1].astype(np.float64)
v = np.diag(C); sd = np.sqrt(v); al = mu / sd; pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = np.array([math.erfc(-a / math.sqrt(2)) / 2 for a in al]); d = pdf / sd; gam = g - mp * d
def forms(Cx, mx):
    Cm = Cx @ mx; return np.stack([(np.trace(Cx) * Cx)[off], (Cx @ Cx)[off], np.outer(mx, mx)[off], (np.outer(mx, Cm) + np.outer(Cm, mx))[off]], 1)
Fpre, Fpost, Fnext = forms(C, mu), forms(Cp, mp), forms(Cn, mun)
def ns(A, Fm): return A[off] - Fm @ np.linalg.lstsq(Fm, A[off], rcond=None)[0]
M = {k: [sym(D[f'{k}_{h}']) for h in (0, 1)] for k in ('Mpre', 'Mpost', 'Mmix', 'Mnext')}; tz = [D[f'tz_{h}'] for h in (0, 1)]
def score(target, Ft, preds, names):
    out = {}
    for h, hp in ((0, 1), (1, 0)):
        y, y2 = ns(target[h], Ft), ns(target[hp], Ft); X = np.stack([ns(p[hp], Ft) for p in preds], 1)
        for k in range(1, len(preds) + 1):
            co = np.linalg.lstsq(X[:, :k], y, rcond=None)[0]; fit = X[:, :k] @ co
            out.setdefault('+'.join(names[:k]), []).append((round(float((2 * fit @ y2 - fit @ fit) / (y @ y2)), 4), [float('%.3g' % c) for c in co]))
    return {k: dict(share=[x[0] for x in v_], coef=v_[0][1]) for k, v_ in out.items()}
G = lambda A: (g[:, None] * A) * g[None, :]
rowform = lambda phi: (lambda A: A + A.T)((phi[:, None] * C) * g[None, :])
print(json.dumps(dict(gamma_bar=round(float(gam.mean()), 4), share_of_offdiag_energy_that_is_network_specific={k: round(float((ns(M[k][0], F_) @ ns(M[k][1], F_)) / (M[k][0][off] @ M[k][1][off])), 4)
                                                                                 for k, F_ in (('Mpre', Fpre), ('Mmix', Fpre), ('Mpost', Fpost), ('Mnext', Fnext))})))
print('A', json.dumps(score(M['Mmix'], Fpre, [M['Mpre']], ['Mpre'])))
print('B', json.dumps(score(M['Mpost'], Fpost, [[G(x) for x in M['Mmix']], [rowform(d * t_) for t_ in tz], [rowform(d / 2 * np.diag(x)) for x in M['Mmix']]], ['G Mmix G', 'd t g C', '(d/2) diag g C'])))
r2 = v * ((1 + al * al) * g + al * pdf); S3 = 3 * (1 - g) * (r2 - 2 * mp * mp)
hub3 = rowform(S3)                       # each unit's own (3,1) cumulant with a partner: kappa4(x_a, x_a, x_a, x_b) = S3_a g_b C_ab
print('B3', json.dumps(score(M['Mpost'], Fpost, [[hub3, hub3], [G(x) for x in M['Mmix']], [rowform(d * t_) for t_ in tz], [rowform(d / 2 * np.diag(x)) for x in M['Mmix']]], ['S3 g C', 'G Mmix G', 'd t g C', '(d/2) diag g C'])))
# the same question one level up: the pushed post matrix with the hub form taken out first
hubn = 2 * W.T @ hub3 @ W
print('C3', json.dumps(score(M['Mnext'], Fnext, [[hubn, hubn], [W.T @ G(x) @ W for x in M['Mmix']], [W.T @ G(x) @ W for x in M['Mpre']]], ['pushed S3 g C', 'W G Mmix G W', 'W G Mpre G W'])))
print('B2', json.dumps(score(M['Mpost'], Fpost, [[G(x) for x in M['Mpre']]], ['G Mpre G'])))
print('C', json.dumps(score(M['Mnext'], Fnext, [[2 * W.T @ x @ W for x in M['Mpost']]], ['2 W Mpost W'])))
print('C2', json.dumps(score(M['Mnext'], Fnext, [[W.T @ G(x) @ W for x in M['Mpre']]], ['W G Mpre G W'])))

# ---- with the Gaussian pairings taken out in every space (the cumulant matrices themselves)
S2 = 2 * (g - g * g - mp * d); CGC = (C * gam[None, :]) @ C; GC = g[:, None] * C
Npre = [x - 2 * C @ C for x in M['Mpre']]; Nmix = [x - 2 * CGC for x in M['Mmix']]; Npost = [x - 2 * Cp @ Cp for x in M['Mpost']]; Nnext = [x - 2 * Cn @ Cn for x in M['Mnext']]
gauss = (GC * S2[None, :]) @ GC.T
print('A2', json.dumps(score(Nmix, Fpre, [Npre], ['gammabar Npre'])))
print('B4', json.dumps(score(Npost, Fpost, [[G(x) for x in Nmix], [gauss, gauss], [hub3, hub3], [rowform(d * t_) for t_ in tz]], ['G Nmix G', 'G C S2 C G', 'S3 g C', 'd t g C'])))
print('B5', json.dumps(score(Npost, Fpost, [[G(x) for x in Npre], [gauss, gauss], [hub3, hub3], [rowform(d * t_) for t_ in tz]], ['G Npre G', 'G C S2 C G', 'S3 g C', 'd t g C'])))
print('C4', json.dumps(score(Nnext, Fnext, [[2 * W.T @ x @ W for x in Npost]], ['2 W Npost W'])))
print('C5', json.dumps(score(Nnext, Fnext, [[W.T @ G(x) @ W for x in Npre], [2 * W.T @ gauss @ W] * 2, [2 * W.T @ hub3 @ W] * 2], ['W G Npre G W', 'pushed gauss', 'pushed S3 g C'])))

# ---- E35c: the ReLU step with COLLECTIVE tilts (the unit's own part of the norm taken out of the tilt and treated exactly)
r3 = v * sd * ((al ** 3 + 3 * al) * g + (al * al + 2) * pdf); r4 = v * v * ((al ** 4 + 6 * al * al + 3) * g + (al ** 3 + 5 * al) * pdf); mG = sd * (al * g + pdf)
beta = 2 * (mG - mp * g)                                         # Cov(z~, h) / v,  h = (f - m)^2
Ez2f = r3 - 2 * mu * r2 + mu * mu * mG; Ez2f2 = r4 - 2 * mu * r3 + mu * mu * r2
Pself = Ez2f2 - 2 * mp * Ez2f + mp * mp * v - v * (r2 - mp * mp)          # Cov(z~^2, h(z)) for a Gaussian unit
tself = beta * v
print('self parts: t_self / t rms', round(float(np.sqrt((tself ** 2).mean() / (tz[0] ** 2).mean())), 3), ' P_self / diag(Mmix) rms', round(float(np.sqrt((Pself ** 2).mean() / (np.diag(M['Mmix'][0]) ** 2).mean())), 3))
tc = [t_ - tself for t_ in tz]; Pc = [np.diag(x) - Pself for x in M['Mmix']]
feat = lambda: None
B6 = score(Npost, Fpost, [[G(x) for x in Nmix], [gauss, gauss], [hub3, hub3], [rowform(d * t_) for t_ in tc], [rowform(-(d * al / (2 * sd)) * p_) for p_ in Pc]],
           ['G Nmix G', 'G C S2 C G', 'S3 g C', 'd tc g C', '-(d a / 2 s) Pc g C'])
print('B6', json.dumps(B6))
# all coefficients at their predicted value 1
for h, hp in ((0, 1), (1, 0)):
    y, y2 = ns(Npost[h], Fpost), ns(Npost[hp], Fpost)
    for name, parts in (('gate only', [G(Nmix[hp])]), ('gate + births', [G(Nmix[hp]), gauss]), ('gate + births + own(3,1)', [G(Nmix[hp]), gauss, hub3]),
                        ('+ mean tilt', [G(Nmix[hp]), gauss, hub3, rowform(d * tc[hp])]), ('+ variance tilt', [G(Nmix[hp]), gauss, hub3, rowform(d * tc[hp]), rowform(-(d * al / (2 * sd)) * Pc[hp])])):
        x = ns(sum(parts), Fpost); print('   target half', h, name, 'share at coefficient 1:', round(float((2 * x @ y2 - x @ x) / (y @ y2)), 4))
