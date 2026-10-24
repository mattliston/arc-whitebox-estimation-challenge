#!/usr/bin/env python3
"""TAG opus-physicist T99 gate #2 (offline): can a per-row model of the (3,1) plane recover the K31 oracle (T78: true K31 at 13-14 = -2.4 / -2.1%)?
Card (S47 / Nd on the v11 simulator, t99_worker_pindump.py dumps of the pair map's inputs at 10-14 on public networks 35 and 100): its normalised
plane T31_card = t4_i rowf_i rho_ij + rem_ij (t4 = k4 / v^2, rowf = the card's row factor, rem = the modules' remainder). Truth: the 8M-sample
K31 planes (public_g19_attribution_v1/mps_planes_rank0035 / 0100) normalised the card's way, T31_true = K31 / (v_i s_i s_j).
Models fitted on 35, judged on 100 (fraction of the off-diagonal error energy |T31_true - T31_model|^2 removed vs the card), per layer, unweighted and
weighted by pdf_i g_j (the rows / columns the covariance's (3,1) term weighs): M1 one global factor on the card's row model; M2 a per-row factor
linear in per-unit features (1, alpha, alpha^2, skew, t4, mu k3 / v^2, gate, pdf) times t4_i rho_ij; M3 = M2 + c rho_ij (a t4-free row form).
usage: t99_k31_gate.py"""
import math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
D = {r: np.load(HERE / 't99_pindump' / 'dumps' / f'P_{r}.npz') for r in (35, 100)}
T = {r: np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{r:04d}.npz') for r in (35, 100)}
erf = np.vectorize(math.erf)


def layer_data(r, l):
    d = D[r]; C = d[f'C_{l}'].astype(np.float64); v = np.diag(C).clip(1e-12); s = np.sqrt(v); mu = d[f'mu_{l}']; k3, k4 = d[f'k3_{l}'], d[f'k4_{l}']
    rho = C / np.outer(s, s); np.fill_diagonal(rho, 0.0); t4 = k4 / v ** 2
    rowf = d[f'rowf_{l}'] if f'rowf_{l}' in d.files else np.ones_like(v)
    rem = d[f'rem_{l}'].astype(np.float64) if f'rem_{l}' in d.files else np.zeros_like(C)
    card = t4[:, None] * rowf[:, None] * rho + rem; np.fill_diagonal(card, 0.0)
    true = T[r][f'K31_{l}'].astype(np.float64) / (v[:, None] * s[:, None] * s[None, :]); np.fill_diagonal(true, 0.0)
    a = mu / s; pdf = np.exp(-a * a / 2) / math.sqrt(2 * math.pi); g = 0.5 * (1 + erf(a / math.sqrt(2)))
    feats = np.stack([np.ones_like(a), a, a * a, k3 / (v * s), t4, mu * k3 / v ** 2, g, pdf], 1)
    W = pdf[:, None] * g[None, :]; np.fill_diagonal(W, 0.0)
    return dict(rho=rho, t4=t4, rowf=rowf, rem=rem, card=card, true=true, feats=feats, W=W)


def design(L, model):
    """columns of the linear model for E = true - card (each column an n x n plane); the model adds sum_k theta_k col_k to the card"""
    base = L['t4'][:, None] * L['rowf'][:, None] * L['rho']
    if model == 'M1': return [base]
    cols = [L['feats'][:, k][:, None] * L['t4'][:, None] * L['rho'] for k in range(L['feats'].shape[1])]
    if model == 'M3': cols.append(L['rho'])
    return cols


print('T99 gate #2: (3,1) plane per-row models, fit on 35 -> judged on 100; % of the off-diagonal error energy removed (unweighted | weighted pdf_i g_j)')
for l in range(10, 15):
    A, B = layer_data(35, l), layer_data(100, l)
    out = []
    for model in ('M1', 'M2', 'M3'):
        res = []
        for wkey in (None, 'W'):
            wa = np.ones_like(A['rho']) if wkey is None else A['W']; wb = np.ones_like(B['rho']) if wkey is None else B['W']
            Xa = np.stack([c[wa > 0].ravel() * np.sqrt(wa[wa > 0]) for c in design(A, model)], 1)
            ya = ((A['true'] - A['card'])[wa > 0] * np.sqrt(wa[wa > 0])).ravel()
            theta = np.linalg.lstsq(Xa, ya, rcond=None)[0]
            Eb = B['true'] - B['card'] - sum(th * c for th, c in zip(theta, design(B, model)))
            e0 = ((B['true'] - B['card']) ** 2 * wb).sum(); e1 = (Eb ** 2 * wb).sum()
            res.append(100 * (1 - e1 / e0))
        out.append('%s %+.1f%% | %+.1f%%' % (model, *res))
    cardshare = 100 * (((B['true'] - B['card']) ** 2).sum() / (B['true'] ** 2).sum())
    print('  layer %d (card error energy on 100 = %.0f%% of the true plane): %s' % (l, cardshare, '; '.join(out)))
