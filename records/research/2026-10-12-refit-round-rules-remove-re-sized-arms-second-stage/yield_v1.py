"""radlab yield_v1 (10-12): the arm-yield table of one round from radlab/cache/<round>.npz (build_v1.py), numpy only.
Reproduces rad_v6.py's freeze (ridge relative to each arm's energy on the grid, leave-one-network-out, candidates all / no dials / no fixed /
fixed only) and adds per arm: trial size, weight and sized step, single-arm LOO, in-sample share of the frozen move, drop-one LOO (what the
out-of-sample gain loses without the arm), single-arm t (mean / se over networks of the arm's gradient B_k), sign of the weight on the two
halves of the networks and its bootstrap stability, and the noise cost per arm (sandwich: diag of A^-1 S A^-1 x diag A / n / E0).
usage: python -B radlab/yield_v1.py S50 [first n networks]"""
import json
import sys
from pathlib import Path

import numpy as np

O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
read = lambda p: json.loads(Path(p).read_text())


def load(name, n=None):
    z = np.load(O / 'radlab' / 'cache' / ('%s.npz' % name))
    D, r = z['D'].astype(np.float64), z['r']
    if n:
        D, r = D[:n], r[:n]
    A = np.einsum('nup,nuq->npq', D, D); B = np.einsum('nup,nu->np', D, r); E = (r ** 2).sum(1)
    return dict(arms=[str(a) for a in z['arms']], seeds=z['seeds'][:len(D)], A=A, B=B, E=E, noise=z['noise'][:len(D)], D=D, r=r,
                BA=np.einsum('nup,nu->np', D, z['rA'][:len(D)]), BB=np.einsum('nup,nu->np', D, z['rB'][:len(D)]))


def fit(A, B, cols, lam):
    Am, Bm = A[np.ix_(cols, cols)], B[cols]
    M = Am + lam * np.diag(np.diag(Am))
    return np.linalg.lstsq(M, Bm, rcond=None)[0] if np.linalg.matrix_rank(M) < len(M) else np.linalg.solve(M, Bm)      # an arm that moved nothing on a subset


def loss(Ai, Bi, Ei, cols, w):      # sum over the given networks of ||r - D w||^2
    return Ei.sum() - 2 * (Bi[:, cols] @ w).sum() + np.einsum('p,npq,q->', w, Ai[np.ix_(range(len(Ai)), cols, cols)], w)


def loo(d, cols, lam, idx=None):
    idx = np.arange(len(d['A'])) if idx is None else idx
    A, B, E = d['A'][idx], d['B'][idx], d['E'][idx]
    As, Bs = A.sum(0), B.sum(0); tot = 0.0
    for i in range(len(idx)):
        w = fit(As - A[i], Bs - B[i], cols, lam)
        tot += E[i] - 2 * B[i, cols] @ w + w @ A[i][np.ix_(cols, cols)] @ w
    return tot / E.sum() - 1


def best_loo(d, cols, idx=None, grid=GRID):
    t = {lam: loo(d, cols, lam, idx) for lam in grid}
    lam = min(grid, key=lambda x: t[x])
    return 100 * t[lam], lam


def candidates(arms, moved):
    c = dict(all=moved, no_dials=[a for a in moved if not a.startswith('d')], no_fixed=[a for a in moved if not a.startswith('f')],
             fixed_only=[a for a in moved if a.startswith('f')])
    out = {}
    for k, v in c.items():
        cols = [arms.index(a) for a in v]
        if cols and cols not in out.values():
            out[k] = cols
    return out


def solve(d, idx=None):
    idx = np.arange(len(d['A'])) if idx is None else idx
    arms = d['arms']; A = d['A'][idx].sum(0)
    moved = [a for k, a in enumerate(arms) if A[k, k] > 0]
    cand = candidates(arms, moved)
    tab = {k: best_loo(d, cols, idx) for k, cols in cand.items()}
    chosen = min(cand, key=lambda k: (tab[k][0], k != 'all'))
    l, lam = tab[chosen]
    w = np.zeros(len(arms))
    if l < 0:
        w[cand[chosen]] = fit(A, d['B'][idx].sum(0), cand[chosen], lam)
    return dict(chosen=chosen, cols=cand[chosen], lam=lam, loo=l, w=w, tab=tab)


def sizes(name):
    R = O / 'trainer' / 'cert' / 'RAD'
    out = {}
    if name == 'S50':
        for d in ('S48/arms_fixed_v1', 'S48/arms_fixed_v2', 'S48/arms_fixed_v3'):
            out.update(read(R / d / 'arms.json')['size'])
    else:
        for d in (name + '/arms_fixed_v1',):
            if (R / d / 'arms.json').exists():
                out.update(read(R / d / 'arms.json')['size'])
        txt = (R / name / 'arms_v1.out').read_text() if (R / name / 'arms_v1.out').exists() else ''
        if 'trial sizes' in txt:
            out.update({k: float(v) for k, v in eval(txt.split('trial sizes ')[1].split(';')[0]).items()})
    return out


if __name__ == '__main__':
    name = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    d = load(name, n); arms = d['arms']; N = len(d['A']); E0 = d['E'].sum(); A, B = d['A'].sum(0), d['B'].sum(0)
    s = solve(d); w, cols, lam = s['w'], s['cols'], s['lam']
    model = 100 * (loss(d['A'], d['B'], d['E'], cols, w[cols]) / E0 - 1)
    print('%s: %d networks, %d arms; candidates %s -> %s ridge %s LOO %+.3f%%, model %+.3f%%, noise share %.3f' % (
        name, N, len(arms), {k: '%+.3f@%s' % v for k, v in s['tab'].items()}, s['chosen'], lam, s['loo'], model, d['noise'].sum() / E0))
    Aw = A @ w; share = w * (2 * B - Aw) / E0 * 100      # in-sample reduction per arm (sums to -model)
    alone = {a: best_loo(d, [k]) if A[k, k] > 0 else (0.0, 0.0) for k, a in enumerate(arms)}
    drop = {}
    for k, a in enumerate(arms):
        c = [j for j in cols if j != k]
        drop[a] = (best_loo(d, c)[0] - s['loo']) if k in cols else 0.0
    g = d['B'] / np.sqrt(np.maximum(np.einsum('npp->np', d['A']).mean(0), 1e-300))      # per-network gradient along each arm, normalised
    tval = np.where(g.std(0) > 0, g.mean(0) / np.maximum(g.std(0, ddof=1) / np.sqrt(N), 1e-300), 0.0)
    h1, h2 = np.arange(N // 2), np.arange(N // 2, N)
    w1 = np.zeros(len(arms)); w2 = np.zeros(len(arms))
    w1[cols] = fit(d['A'][h1].sum(0), d['B'][h1].sum(0), cols, lam); w2[cols] = fit(d['A'][h2].sum(0), d['B'][h2].sum(0), cols, lam)
    rng = np.random.default_rng(0); boot = []
    for _ in range(400):
        b = rng.integers(0, N, N); wb = np.zeros(len(arms)); wb[cols] = fit(d['A'][b].sum(0), d['B'][b].sum(0), cols, lam); boot.append(wb)
    boot = np.array(boot); same = (np.sign(boot) == np.sign(w)).mean(0)
    # sandwich noise cost per arm at the frozen solution (unridged scores): S = cov over networks of (B_i - A_i w)
    sc = d['B'][:, cols] - np.einsum('npq,q->np', d['A'][np.ix_(range(N), cols, cols)], w[cols])
    Am = A[np.ix_(cols, cols)] / N; S = np.cov(sc.T) if len(cols) > 1 else np.array([[sc.var(ddof=1)]])
    reg = Am + lam * np.diag(np.diag(Am)); V = np.linalg.solve(reg, np.linalg.solve(reg, S).T)      # covariance of w x N
    cost = np.zeros(len(arms)); cost[cols] = 100 * np.diag(V) * np.diag(Am) / (E0 / N)      # % excess per arm at n = 1 network (divide by n)
    tot_cost = 100 * np.trace(V @ Am) / (E0 / N)
    sz = sizes(name)
    print('%-12s %8s %8s %9s %8s %8s %8s %7s %6s %6s %6s %8s' % ('arm', 'trial', 'weight', 'step', 'aloneLOO', 'share%', 'dropLOO', 't', 'h1', 'h2', 'boot', 'cost/n'))
    rows = []
    for k, a in enumerate(arms):
        tr = sz.get(a, float('nan'))
        rows.append(dict(arm=a, trial=tr, weight=float(w[k]), step=float(w[k] * tr) if tr == tr else None, alone_loo=alone[a][0], alone_ridge=alone[a][1],
                         share=float(share[k]), drop_loo=float(drop[a]), t=float(tval[k]), w_h1=float(w1[k]), w_h2=float(w2[k]), boot_same=float(same[k]),
                         cost_n1=float(cost[k])))
        print('%-12s %8.4g %+8.3f %+9.4f %+8.3f %+8.3f %+8.3f %+7.2f %+6.2f %+6.2f %6.2f %8.2f' % (
            a, tr, w[k], w[k] * tr if tr == tr else float('nan'), alone[a][0], share[k], drop[a], tval[k], w1[k], w2[k], same[k], cost[k] / N))
    print('sum of shares %+.3f%% (= -model %+.3f%%); total noise cost (sandwich, unshrunk) %.3f%% at n=%d' % (share.sum(), -model, tot_cost / N, N))
    out = O / 'radlab' / ('yield_%s%s_v1.json' % (name, '_n%d' % n if n else ''))
    out.write_text(json.dumps(dict(round=name, networks=N, chosen=s['chosen'], ridge=lam, loo_pct=s['loo'], model_pct=model, cand={k: v for k, v in s['tab'].items()},
                                   noise_share=float(d['noise'].sum() / E0), total_cost_pct_at_n=float(tot_cost / N), rows=rows), indent=1))
