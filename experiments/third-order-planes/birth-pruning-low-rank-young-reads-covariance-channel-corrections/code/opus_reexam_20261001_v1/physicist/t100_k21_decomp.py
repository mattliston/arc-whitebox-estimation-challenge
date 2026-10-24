#!/usr/bin/env python3
"""TAG opus-physicist T100: WHAT IN THE K21 ERROR DOES THE COVARIANCE CHANNEL WANT? (after gate #1's scale-mode arm failed: gamma 1 +263%)
Card: S47 / Nd on the v11 simulator; the pair map's K21 input at layers 10-14 on public networks 35 and 100 (t99_worker_pindump_v2.py).
Truth: the 8M-sample K21 planes (public_g19_attribution_v1/mps_planes_rank0035 / 0100). Error dK = K21_true - K21_card off the diagonal, weighted as
the covariance output weighs it (first-order Edgeworth: Cov(y_a, y_b) gains K21[a, b] pdf_a / (2 s_a) g_b, so w_ab = (pdf_a g_b / s_a)^2).
Candidate corrections (linear models of dK, fitted on 35, judged on 100: % of the weighted error energy removed):
  S   the scale-mode form v_a mu_b + 2 C_ab mu_a (gate #1's direction)
  G   one global factor on the card's own K21
  R   per-row factors on the card's K21, linear in row features (1, alpha, alpha^2, gate, pdf)
  RC  R plus per-column factors (the same features of b)
  ALL S + RC
usage: t100_k21_decomp.py"""
import math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
D = {r: np.load(HERE / 't99_pindump_v2' / 'dumps' / f'P_{r}.npz') for r in (35, 100)}
T = {r: np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{r:04d}.npz') for r in (35, 100)}
erf = np.vectorize(math.erf)


def layer(r, l):
    d = D[r]; C = d[f'C_{l}'].astype(np.float64); v = np.diag(C).clip(1e-12); s = np.sqrt(v); mu = d[f'mu_{l}']
    K = d[f'K21_{l}'].astype(np.float64); Kt = T[r][f'K21_{l}'].astype(np.float64)
    a = mu / s; pdf = np.exp(-a * a / 2) / math.sqrt(2 * math.pi); g = 0.5 * (1 + erf(a / math.sqrt(2)))
    w = (pdf / s)[:, None] * g[None, :]; w = w * w; np.fill_diagonal(w, 0.0)
    dK = Kt - K; np.fill_diagonal(dK, 0.0); Kc = K.copy(); np.fill_diagonal(Kc, 0.0)
    F = v[:, None] * mu[None, :] + 2 * C * mu[:, None]; np.fill_diagonal(F, 0.0)
    feats = np.stack([np.ones_like(a), a, a * a, g, pdf], 1)
    return dict(w=w, dK=dK, Kc=Kc, F=F, feats=feats, Kt=Kt)


def cols(L, model):
    out = []
    if model in ('S', 'ALL'): out.append(L['F'])
    if model == 'G': out.append(L['Kc'])
    if model in ('R', 'RC', 'ALL'): out += [L['feats'][:, k][:, None] * L['Kc'] for k in range(L['feats'].shape[1])]
    if model in ('RC', 'ALL'): out += [L['Kc'] * L['feats'][:, k][None, :] for k in range(1, L['feats'].shape[1])]
    return out


print('T100: % of the covariance-weighted K21 error energy removed on network 100 by a model fitted on 35 (and the card error energy share of the truth)')
for l in range(10, 15):
    A, B = layer(35, l), layer(100, l)
    sw = np.sqrt(A['w']).ravel(); row = []
    for model in ('S', 'G', 'R', 'RC', 'ALL'):
        X = np.stack([c.ravel() * sw for c in cols(A, model)], 1); y = A['dK'].ravel() * sw
        th = np.linalg.lstsq(X, y, rcond=None)[0]
        E = B['dK'] - sum(t_ * c for t_, c in zip(th, cols(B, model)))
        row.append('%s %+.1f%%' % (model, 100 * (1 - (E * E * B['w']).sum() / (B['dK'] ** 2 * B['w']).sum())))
    share = (B['dK'] ** 2 * B['w']).sum() / ((B['Kt'] - np.diag(np.diag(B['Kt']))) ** 2 * B['w']).sum()
    print('  layer %d (weighted card error = %.0f%% of the true plane energy): %s' % (l, 100 * share, '; '.join(row)), flush=True)
