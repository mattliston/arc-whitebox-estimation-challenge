#!/usr/bin/env python3
"""TAG opus-physicist T64 (numpy): ZERO-WEIGHT GRADIENT OF EVERY CANDIDATE HEAD INPUT AT ONCE, from the adjoint dumps of T63 (S42 torch card, trainer's gradient mode).
For a candidate column x (per layer, standardised as the head would get it) entering the head's first layer with 128 weights u at zero, dL/du = sum_l x_l' G_l
(G_l = the gradient of 1e8 x final-layer MSE against the per-unit bias of that first layer, T63). Two weight sets, as the lead decided (10-10 02:25Z):
u for layers 1-14 (shared) and u15 for layer 15 alone. Conventions: layers 1-14 'a' = standardised over all units, 'r' = over the read's response rows (0 elsewhere,
the native convention of column 6); layer 15 = over the kept rows of the final row cut |mu / sigma| <= 3.5 (0 elsewhere).
RAW = the trainer's test (his statistics: weights of 128 beyond 3 standard errors over networks, mean t^2 (1.0 = no signal), cosine of the two halves' mean gradients,
first-order change of the pooled final-layer MSE for a step of 0.05 along minus the mean gradient with the noise share out).
PARTIAL = the same for the part of x that the head does NOT already see: x minus its least-squares fit, per network and layer, on the head's 126 inputs (30 features
+ 96 messages), column 6 and a constant. A column whose raw gradient is large but partial gradient small asks for a move the existing weights could make.
Also: cosine of each mean gradient with column 6's remaining gradient (the trainer's 'used up' check), and the rank-one test of G.
usage: t64_adjoint_screen.py [ranks, default = every adump file]"""
import sys, math, glob
from pathlib import Path
import numpy as np
from scipy import stats as _st
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'trainer'))
from otr_common import data_paths
args = [a for a in sys.argv[1:] if a != 'full']; FULL = 'full' in sys.argv[1:]
files = sorted(HERE.glob('adump_rank*.npz')) if not args else [HERE / f'adump_rank{int(r):04d}.npz' for r in args[0].split(',')]
n = 1024


def std(x, mask):
    y = np.zeros(n); z = x[mask]; sd = z.std(ddof=1) if mask.sum() > 2 else 0.0
    if np.isfinite(sd) and sd > 0: y[mask] = np.clip((z - z.mean()) / sd, -8, 8)
    return y


def he(a): return a * a - 1, a ** 3 - 3 * a, a ** 4 - 6 * a * a + 3, a ** 5 - 10 * a ** 3 + 15 * a, a ** 6 - 15 * a ** 4 + 45 * a * a - 15


def candidates(D, l, W=None):
    """raw per-unit candidate vectors of layer l (W = this layer's weights, (units of l-1) x (units of l), for the pushed candidates)"""
    k = lambda name: f'{name}_{l}'
    c = {}
    if k('v') not in D.files: return c
    if W is not None and f'host_{l - 1}' in D.files and f'delta_{l - 1}' in D.files and l >= 2:
        h, d = W.T @ D[f'host_{l - 1}'], W.T @ D[f'delta_{l - 1}']; vv = D[k('v')].clip(1e-12)
        c['hc'] = -(2 * h * d + d * d) / vv          # the head's mean correction of layer l-1 moved into C (step line 307), as it reaches v_l, over v_l
        c['pdel'] = d / np.sqrt(vv)                   # the head's correction of layer l-1 pushed into layer l's pre-activation mean, over sigma_l
        c['hc_abs'] = np.abs(c['hc'])
    v = D[k('v')].astype(np.float64).clip(1e-12); mu = D[k('mu')].astype(np.float64); s = np.sqrt(v); a = mu / s; ph = np.exp(-a * a / 2) / math.sqrt(2 * math.pi)
    k3 = D[k('k3')]; k4 = D[k('k4')] if k('k4') in D.files else np.zeros(n); t3, t4 = k3 / (v * s), k4 / (v * v); h2, h3, h4, h5, h6 = he(a)
    c.update(t3=t3, t4=t4, m2=t3 * t3 * h4 * ph / 72, e34=-t3 * t4 * h5 * ph / 144, e44=t4 * t4 * h6 * ph / 1152, m1k3=-t3 * a * ph / 6, m1k4=t4 * h2 * ph / 24, alpha=a)
    for name in ('ls', 'qb', 'qk', 'kmu', 'kg', 'pv', 'pk3', 'pk4', 'pd', 'pdd'):
        if k(name) in D.files: c[name] = D[k(name)].astype(np.float64)
    for pre in ('q', 'f', 'c', 'e', 's'):
        for ag in ('1', '2', '3', '4', '56', '7'):
            if k(pre + ag) in D.files: c[pre + ag] = D[k(pre + ag)].astype(np.float64)
        parts = [c[pre + ag] for ag in ('3', '4', '56', '7') if pre + ag in c]      # pairs aged 3 and more
        if parts: c[pre + 'old'] = np.sum(parts, 0)
    for name in ('m2', 'k3d2', 'gk', 'gb', 'xk4', 'rg', 'f15', 'qold', 'eold', 'qb'):
        if f'b6_{name}_{l}' in D.files: c['b6' + name] = D[f'b6_{name}_{l}'].astype(np.float64)
    if k('mshare') in D.files:
        ms, ml = D[k('mshare')], D[k('mload')]
        for j in range(ms.shape[1]): c[f'msh{j + 1}'] = ms[:, j]; c[f'mld{j + 1}'] = ml[:, j]
        c['mtop'] = ms.sum(1); c['m2to8'] = ms[:, 1:].sum(1)
    return {nm: x for nm, x in c.items() if np.isfinite(x).all() and x.std() > 0}


ROWS = {}      # (candidate, set, kind) -> list over networks of the 128-gradient
MSE = []; SV = []
for f in files:
    if not f.exists(): print('missing', f.name); continue
    D = np.load(f); sc = float(D['scale']); MSE.append(float(D['final_x1e9']) * 1e-9)
    WT = np.load(data_paths(int(f.name[10:14]))[0], mmap_mode='r')
    hu = D['hu_grads'] / sc
    for j in range(hu.shape[0]): ROWS.setdefault((f'col{j}', '1-15', 'module'), []).append(hu[j])
    for l in range(1, 16):
        if f'G_{l}' not in D.files: continue
        G = D[f'G_{l}'].astype(np.float64) / sc; u_, s_, vt_ = np.linalg.svd(G, full_matrices=False); SV.append((l, s_[0] ** 2 / (s_ ** 2).sum()))
        H = np.column_stack([D[f'xn_{l}'], D[f'msgs_{l}'], D[f'cols_{l}'][:, 6]]).astype(np.float64); H = np.column_stack([np.ones(n), H[:, H.std(0) > 0]])
        Q, _ = np.linalg.qr(H)
        Wl = np.asarray(WT[l], dtype=np.float64) if WT is not None else None
        cand = candidates(D, l, Wl)
        if l <= 14:
            resp = D[f'resp_{l}'] > 0.5 if f'resp_{l}' in D.files else np.ones(n, bool)
            sets = (('a', np.ones(n, bool)), ('r', resp)); band = '1-14'
        else:
            mu, v = D[f'mu_{l}'], D[f'v_{l}'].clip(1e-12); sets = (('k', np.abs(mu / np.sqrt(v)) <= 3.5),); band = '15'
        for conv, mask in sets:
            names = list(cand); X = np.column_stack([std(cand[nm], mask) for nm in names])
            keep = X.std(0) > 0; names = [nm for nm, kp in zip(names, keep) if kp]; X = X[:, keep]
            Xp = X - Q @ (Q.T @ X)
            gr, gp = X.T @ G, Xp.T @ G
            for i, nm in enumerate(names):
                for kind, g in (('raw', gr[i]), ('partial', gp[i])):
                    key = (nm + '@' + conv, band, kind); L = ROWS.setdefault(key, [])
                    if len(L) < len(MSE): L.append(np.zeros(128))
                    L[-1] = L[-1] + g
N = len(MSE); mse = float(np.mean(MSE))
print('T64 adjoint screen: %d networks (%s); pooled final-layer MSE %.4e; top singular share of G per layer (rank-one test): %s' % (
    N, ','.join(f.name[10:14] for f in files if f.exists()), mse,
    ' '.join('L%d %.2f' % (l, np.mean([s for ll, s in SV if ll == l])) for l in range(1, 16) if any(ll == l for ll, _ in SV))))


CHANCE = None


def stats(Gs):
    Gs = np.array(Gs); m, se = Gs.mean(0), Gs.std(0, ddof=1) / math.sqrt(len(Gs)); t = m / np.maximum(se, 1e-300)
    a, b = Gs[0::2].mean(0), Gs[1::2].mean(0); left = max(float((m * m).sum() - (se * se).sum()), 0.0) ** .5
    return int((np.abs(t) > 2.5).sum()), float((t * t).mean()), float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum() + 1e-300)), -100 * 0.05 * left / mse, m


ref6 = np.mean(ROWS[('col6', '1-15', 'module')], 0)
CHANCE = 128 * 2 * _st.t.sf(2.5, N - 1); print('chance count of weights beyond 2.5 se: %.2f of 128 (t with %d dof); the trainer\'s rule: at least %.1f' % (CHANCE, N - 1, 3 * CHANCE))
res = []
for key, L in ROWS.items():
    if len(L) != N: continue
    b3, mt2, hc, fo, m = stats(L); res.append((key, b3, mt2, hc, fo, float(m @ ref6 / math.sqrt((m @ m) * (ref6 @ ref6) + 1e-300))))
for band in ('1-15', '1-14', '15'):
    for kind in (('module',) if band == '1-15' else ('raw', 'partial')):
        R = sorted([r for r in res if r[0][1] == band and r[0][2] == kind], key=lambda r: -r[2])
        print('\n%s, layers %s: candidate | weights beyond 2.5 se (chance %.1f; PASS = at least %.1f) | mean t^2 (1.0 if none) | halves cosine | first order, step 0.05 (%%) | cosine to column 6' % (kind.upper(), band, CHANCE, 3 * CHANCE))
        for key, b3, mt2, hc, fo, c6 in (R if FULL else R[:(40 if kind != 'module' else 21)]):
            print('  %-12s %4d%s %7.2f %+6.2f %+8.3f%% %+6.2f' % (key[0], b3, ' PASS' if b3 >= 3 * CHANCE else '     ', mt2, hc, fo, c6))
