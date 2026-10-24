#!/usr/bin/env python3
"""TAG opus-physicist T67: OFFLINE FIT OF NEW FINAL-LAYER HEAD COLUMNS on the layer-15 dumps of T66 (lead 10-10 03:05Z / 03:22Z: the shared encoder frozen, new columns only).
Model, per unit i of layer 15 (S42's head, parameters c3q4zd6_optimum.pt; nothing else changes):
  h1 = silu(H_i W0' + b0 + z_i U), h2 = silu(h1 W2' + b2), state = h2 W4' + b4, delta_i = sigma_i rms15 (1 + 0.25 tanh gain15) (state . head15 + hb15)
  final prediction = m_i + kappa (delta_i(U) - delta_i(0)), m = S42's final prediction (dumped), kappa = (m - host) / delta(0) (1 if the card applies delta as is).
  z_i = the new columns, each standardised over the network's KEPT rows (|mu / sigma| <= 3.5 at layer 15), clipped at 8, 0 on the other rows. U (k x 128) starts at 0.
Fit: Gauss-Newton on the training networks (the change of delta linearised in U: d delta_i = z_i (x) g_i, g_i = d delta_i / d h1pre_i at U = 0, exact by autograd) with
weight decay lam |U|^2, then the EXACT nonlinear model along the Gauss-Newton step, step size t chosen on the training networks; JUDGED once on the held-out networks:
pooled final-layer MSE (mean over networks of the unit mean) against S42's, per-network ratios, networks improved. Certificate: delta(0) recomputed from H = the dumped delta.
usage: t67_l15_fit.py [mode: singles | sets | both]"""
import sys, math, re, time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2); t = torch; f64 = t.float64
HERE = Path(__file__).resolve().parent; O = HERE.parent
MODE = sys.argv[1] if len(sys.argv) > 1 else 'both'
S = torch.load(O / 'trainer/cert/fold/c3q4zd6_optimum.pt', map_location='cpu', weights_only=True)['arrays']
W0, b0, W2, b2, W4, b4 = (S['p.' + k].to(f64) for k in ('enc0_w', 'enc0_b', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b'))
HD, HB, RMS, GN = S['p.head'][15].to(f64), S['p.head_b'][15].to(f64), S['p.rms'][15].to(f64), S['p.gain'][15].to(f64)
silu = lambda z: z / (1 + t.exp(-z))
rd = lambda f: [int(x) for x in re.findall(r'\d+', (HERE / f).read_text())]
TRAIN = [r for r in rd('t66_train200.txt') if (HERE / f'l15dump_rank{r:04d}.npz').exists()]
VAL = [r for r in rd('t66_val50.txt') if (HERE / f'l15dump_rank{r:04d}.npz').exists()]


def he(a): return a * a - 1, a ** 3 - 3 * a, a ** 4 - 6 * a * a + 3, a ** 5 - 10 * a ** 3 + 15 * a, a ** 6 - 15 * a ** 4 + 45 * a * a - 15


def candidates(D):
    mu, v = D['mu'], D['v'].clip(1e-12); s = np.sqrt(v); a = mu / s; ph = np.exp(-a * a / 2) / math.sqrt(2 * math.pi); t3, t4 = D['k3'] / (v * s), D['k4'] / (v * v)
    h2, h3, h4, h5, h6 = he(a)
    c = dict(t3=t3, t4=t4, e34=-t3 * t4 * h5 * ph / 144, e44=t4 * t4 * h6 * ph / 1152, m1k3=-t3 * a * ph / 6, m1k4=t4 * h2 * ph / 24, ls=np.log(s), alpha=a)
    for k in ('c17', 'c18', 'c19', 'b6_f15', 'b6_qold', 'b6_eold', 'pv', 'pk3', 'pk4', 'pd', 'pdd'):
        if k in D.files: c[k.replace('b6_', '')] = D[k]
    for pre in ('q', 'f', 'c', 'e', 's'):
        for ag in ('1', '2', '3', '4', '56', '7'):
            if pre + ag in D.files: c[pre + ag] = D[pre + ag]
    if 'mshare' in D.files:
        for j in range(D['mshare'].shape[1]): c[f'msh{j + 1}'] = D['mshare'][:, j]; c[f'mld{j + 1}'] = D['mload'][:, j]
        c['mtop'] = D['mshare'].sum(1)
    return {k: x for k, x in c.items() if np.isfinite(x).all()}


def standardise(x, kept):
    y = np.zeros_like(x); z = x[kept]; sd = z.std(ddof=1)
    if np.isfinite(sd) and sd > 0: y[kept] = np.clip((z - z.mean()) / sd, -8, 8)
    return y


NET = {}
cert = []
for r in TRAIN + VAL:
    D = np.load(HERE / f'l15dump_rank{r:04d}.npz')
    H = t.tensor(D['H'], dtype=f64); sig = t.tensor(D['sigma'], dtype=f64); kept = D['kept'] > 0.5
    pre = (H @ W0.T + b0).requires_grad_(True)
    scale = sig * RMS * (1 + 0.25 * t.tanh(GN))
    d0 = scale * ((silu(silu(pre) @ W2.T + b2) @ W4.T + b4) @ HD + HB)
    g, = t.autograd.grad(d0.sum(), pre)
    dd = D['delta']; cert.append(float(np.abs(d0.detach().numpy() - dd).max() / np.abs(dd).max()))
    kappa = float(np.median((D['m'] - D['host'])[np.abs(dd) > 1e-3 * np.abs(dd).max()] / dd[np.abs(dd) > 1e-3 * np.abs(dd).max()])) if 'host' in D.files else 1.0
    cand = candidates(D)
    NET[r] = dict(pre=pre.detach(), scale=scale, d0=d0.detach(), g=g.detach(), m=t.tensor(D['pred'], dtype=f64), y=t.tensor(D['truth'], dtype=f64), kappa=kappa,
                  Z={k: t.tensor(standardise(x, kept), dtype=f64) for k, x in cand.items()}, kept=t.tensor(kept))
NAMES = sorted(set.intersection(*[set(NET[r]['Z']) for r in NET]))
G = t.cat([NET[r]['g'] for r in TRAIN]); sv = t.linalg.svdvals(G)
print('T67 offline layer-15 column fit: %d training networks, %d held-out networks; certificate delta(0) recomputed vs dumped: max rel %.2e; kappa median %.4f (range %.4f..%.4f);'
      ' rank-one share of g over the training rows %.3f; S42 pooled final-layer MSE: train %.4e, held-out %.4e' % (
          len(TRAIN), len(VAL), max(cert), np.median([NET[r]['kappa'] for r in NET]), min(NET[r]['kappa'] for r in NET), max(NET[r]['kappa'] for r in NET),
          float(sv[0] ** 2 / (sv ** 2).sum()), np.mean([float(((NET[r]['m'] - NET[r]['y']) ** 2).mean()) for r in TRAIN]),
          np.mean([float(((NET[r]['m'] - NET[r]['y']) ** 2).mean()) for r in VAL]) if VAL else float('nan')), flush=True)


def pred(r, cols, U):
    N = NET[r]; Z = t.stack([N['Z'][c] for c in cols], 1)
    k = N['kept']; d = N['d0'].clone()
    d[k] = N['scale'][k] * ((silu(silu(N['pre'][k] + Z[k] @ U) @ W2.T + b2) @ W4.T + b4) @ HD + HB)      # rows outside the kept set have z = 0: unchanged
    return N['m'] + N['kappa'] * (d - N['d0'])


def mse(nets, cols, U):
    return np.array([float(((pred(r, cols, U) - NET[r]['y']) ** 2).mean()) for r in nets])


def per_net(cols):
    """per training network: A_r = Phi' Phi / 1024, b_r = Phi' res / 1024 (Gauss-Newton pieces of that network's unit-mean squared error), base MSE"""
    out = {}
    for r in TRAIN:
        N = NET[r]; k = N['kept']; Z = t.stack([N['Z'][c] for c in cols], 1)[k]; g = N['g'][k] * N['kappa']; res = (N['m'] - N['y'])[k]
        Phi = (Z[:, :, None] * g[:, None, :]).reshape(len(res), -1)
        out[r] = (Phi.T @ Phi / 1024, Phi.T @ res / 1024, float(((N['m'] - N['y']) ** 2).mean()))
    return out


LAMS = (1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def solve(P, nets, lam):
    A = sum(P[r][0] for r in nets) / len(nets); b = sum(P[r][1] for r in nets) / len(nets)
    return -t.linalg.solve(A + lam * A.diagonal().mean() * t.eye(len(b), dtype=f64), b)


def quad(P, nets, U):
    A = sum(P[r][0] for r in nets) / len(nets); b = sum(P[r][1] for r in nets) / len(nets); base = np.mean([P[r][2] for r in nets])
    return float(2 * b @ U + U @ A @ U) / base


def fit(cols, steps=None, folds=5):
    """lam by 5-fold cross-validation over the training networks (quadratic model on the left-out fold); weights on all training networks; optional step multiple
    chosen on the training networks with the exact model. Returns (cv change, lam, step, held-out change, networks lower, se of the mean log ratio, U)."""
    P = per_net(cols); idx = list(TRAIN); F = [idx[i::folds] for i in range(folds)]
    cv = {lam: np.mean([quad(P, f, solve(P, [r for r in idx if r not in set(f)], lam)) for f in F]) for lam in LAMS}
    lam = min(cv, key=cv.get); U = solve(P, idx, lam).reshape(len(cols), 128); s_best = 1.0
    if steps:
        base_tr = mse(TRAIN, cols, t.zeros(len(cols), 128, dtype=f64)).mean()
        s_best = min(steps, key=lambda s_: mse(TRAIN, cols, s_ * U).mean())
    if not VAL: return cv[lam], lam, s_best, None, None, None, U
    b0v = mse(VAL, cols, t.zeros(len(cols), 128, dtype=f64)); nv = mse(VAL, cols, s_best * U)
    return cv[lam], lam, s_best, nv.mean() / b0v.mean() - 1, int((nv < b0v).sum()), float(np.std(np.log(nv / b0v), ddof=1) / math.sqrt(len(nv))), U


if MODE in ('singles', 'both'):
    print('\nSINGLE COLUMNS (train change in-sample | held-out change, networks lower, se of the mean log ratio):', flush=True)
    rows = []
    for c in NAMES:
        v_cv, lam, s_, v_va, lower, se, U = fit([c]); rows.append((v_va if v_va is not None else v_cv, c, v_cv, lam, v_va, lower, se))
    for key, c, v_cv, lam, v_va, lower, se in sorted(rows):
        print('  %-7s 5-fold cv on training (quadratic) %+7.3f%% (lam %g)  held-out exact %s' % (c, 100 * v_cv, lam, ('%+7.3f%% (%d of %d lower, se %.2f%%)' % (100 * v_va, lower, len(VAL), 100 * se)) if v_va is not None else '-'), flush=True)
if MODE in ('sets', 'both'):
    SETS = [('17', ['c17']), ('17 qold', ['c17', 'qold']), ('17 qold f56 f3', ['c17', 'qold', 'f56', 'f3']), ('17 qold f56 f3 q56 e44', ['c17', 'qold', 'f56', 'f3', 'q56', 'e44']),
            ('17-19', ['c17', 'c18', 'c19']), ('17 f15 qold eold', ['c17', 'f15', 'qold', 'eold']),
            ('17 qold f-ages', ['c17', 'qold', 'f1', 'f2', 'f3', 'f4', 'f56', 'f7']), ('17 qold q-ages', ['c17', 'qold', 'q1', 'q2', 'q3', 'q4', 'q56', 'q7'])]
    print('\nCOLUMN SETS:', flush=True)
    for name, cols in SETS:
        if not all(c in NAMES for c in cols): continue
        v_cv, lam, s_, v_va, lower, se, U = fit(cols, steps=(0.5, 0.75, 1.0, 1.25))
        print('  %-22s 5-fold cv (quadratic) %+7.3f%% (lam %g, step %.2f)  held-out exact %s' % (name, 100 * v_cv, lam, s_, ('%+7.3f%% (%d of %d lower, se %.2f%%)' % (100 * v_va, lower, len(VAL), 100 * se)) if v_va is not None else '-'), flush=True)
        np.save(HERE / ('t67_U_%s.npy' % name.replace(' ', '_')), (s_ * U).numpy())
