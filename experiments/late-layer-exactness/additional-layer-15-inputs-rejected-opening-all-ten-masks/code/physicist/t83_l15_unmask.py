#!/usr/bin/env python3
"""TAG opus-physicist T83: THE TEN MASKED LAYER-15 INPUTS (13-15, 22-28), PRICED EXACTLY OFFLINE from the layer-15 dumps of t80_shadow.py arm dump15 (lead 10-10 20:49Z).
Model per unit i of layer 15 (everything of the card fixed except what is named): pre_i = the dumped first-layer pre-activation of the learned correction,
  delta_i(pre) = sigma_i rms15 (1 + 0.25 tanh gain15) ((silu(silu(pre_i) W2' + b2) W4' + b4) . head15 + hb15),
  final_i = m_i + kappa (delta_i(pre + z_i U) - delta_i(pre)), m = the card's final prediction, kappa = median (state_out's m - host) / delta (0.7844 at jmIv).
(1) FIXED PARTS: open the mask for a group: U = the existing first-layer weights of those inputs (W0[:, k]), z_k = the head's own normalisation clamp((f_k - c_k) / s_k, -8, 8).
(2) REFIT (the zero-weight route): NEW weights U (k x 128) for the group at layer 15 only, from 0: Gauss-Newton with weight decay chosen by 5-fold cross-validation over the
    training networks, step multiple chosen on the training networks with the exact model; judged once on the held-out networks. z = the head's normalisation ('head')
    or standardised over the network's rows ('std').
Certificates: delta(0) recomputed from pre = the dumped delta; the card's final MSE from the dump; fixed-parts groups vs the direct replay rows of t80_unmask.jsonl.
usage: t83_l15_unmask.py <label> <training networks csv> <held-out networks csv>"""
import sys, json, math
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2); t = torch; f64 = t.float64
HERE = Path(__file__).resolve().parent
LABEL, TRAIN0, VAL0 = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], [int(x) for x in sys.argv[3].split(',')]
have = lambda r: (HERE / 'l15j_dump' / f'{LABEL}_{r}.npz').exists()
TRAIN, VAL = [r for r in TRAIN0 if have(r)], [r for r in VAL0 if have(r)]
FEATS = (13, 14, 15, 22, 23, 24, 25, 26, 27, 28)
GROUPS = dict(a=(13, 14, 15, 26), r=(13, 14, 15), q=(26,), f=(13, 24, 25, 26), p=(22, 23, 24, 25), o=(27, 28), b=(13, 14, 15, 22, 23, 24, 25, 26), c=FEATS)
silu = lambda z: z / (1 + t.exp(-z))
NET, cert = {}, []
for r in TRAIN + VAL:
    D = np.load(HERE / 'l15j_dump' / f'{LABEL}_{r}.npz'); T = lambda k: t.tensor(D[k], dtype=f64)
    W0, W2, b2, W4, b4, HD, HB, RMS, GN = (T(k) for k in ('enc0_w', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b', 'head', 'head_b', 'rms', 'gain'))
    scale = T('sigma').reshape(-1) * RMS * (1 + 0.25 * t.tanh(GN.reshape(-1)[0]))
    head = lambda pre, W2=W2, b2=b2, W4=W4, b4=b4, HD=HD, HB=HB, scale=scale: scale * ((silu(silu(pre) @ W2.T + b2) @ W4.T + b4) @ HD.reshape(-1) + HB.reshape(-1)[0])
    pre = T('pre').requires_grad_(True); d0 = head(pre); g, = t.autograd.grad(d0.sum(), pre); d0 = d0.detach(); pre = pre.detach()
    dd = T('delta'); m, host, y = T('m'), T('host'), T('truth'); big = dd.abs() > 1e-3 * dd.abs().max()
    kappa = float(((m - host)[big] / dd[big]).median())
    cert.append((float((d0 - dd).abs().max() / dd.abs().max()), abs(float(((T('final') - y) ** 2).mean()) / float(D['mse']) - 1), kappa))
    c, s = T('centers'), T('scales')
    zh = {k: ((T('f%d' % k) - c[k]) / s[k]).clamp(-8, 8) for k in FEATS}
    zs = {k: (lambda v: (v - v.mean()) / v.std().clamp_min(1e-300))(T('f%d' % k)).clamp(-8, 8) for k in FEATS}
    NET[r] = dict(pre=pre, d0=d0, g=g, head=head, m=T('final'), y=y, kappa=kappa,      # the base = the engine's final prediction (state_out's m differs by rms 2e-6)
                  W0=W0, zh=zh, zs=zs, clip={k: float((((T('f%d' % k) - c[k]) / s[k]).abs() >= 8).double().mean()) for k in FEATS})
base = {r: float(((N['m'] - N['y']) ** 2).mean()) for r, N in NET.items()}
print('T83 %s: %d training + %d held-out networks; certificates: delta(0) from pre vs dumped max rel %.2e; final MSE from the dump vs the run max rel %.2e; kappa %.4f..%.4f' % (
    LABEL, len(TRAIN), len(VAL), max(x[0] for x in cert), max(x[1] for x in cert), min(x[2] for x in cert), max(x[2] for x in cert)), flush=True)
print('head normalisation at layer 15 (all networks): per input mean / sd of z and share of rows clipped at 8: ' + '; '.join(
    '%d %+.2f / %.2f / %.1f%%' % (k, np.mean([float(N['zh'][k].mean()) for N in NET.values()]), np.mean([float(N['zh'][k].std()) for N in NET.values()]),
                                  100 * np.mean([N['clip'][k] for N in NET.values()])) for k in FEATS), flush=True)


def final(r, z, ks, U):
    N = NET[r]; Z = t.stack([N[z][k] for k in ks], 1)
    return N['m'] + N['kappa'] * (N['head'](N['pre'] + Z @ U) - N['d0'])


def mse(r, z, ks, U): return float(((final(r, z, ks, U) - NET[r]['y']) ** 2).mean())


def table(nets, new):
    a = np.array([new[r] for r in nets]); b = np.array([base[r] for r in nets]); lr = np.log(a / b)
    return '%+.3f%% pooled, mean log %+.3f%% (se %.3f%%), lower on %d of %d' % (100 * (a.mean() / b.mean() - 1), 100 * lr.mean(), 100 * lr.std(ddof=1) / math.sqrt(len(lr)), int((a < b).sum()), len(a))


ALL = TRAIN + VAL
print('\n(1) FIXED LEARNED PARTS, the mask opened (exact; all %d networks; final-layer error vs the card):' % len(ALL), flush=True)
FIX = {}
for gname, ks in GROUPS.items():
    FIX[gname] = {r: mse(r, 'zh', ks, NET[r]['W0'][:, list(ks)].T) for r in ALL}
    print('  %-2s %-38s %s' % (gname, ','.join(map(str, ks)), table(ALL, FIX[gname])), flush=True)
rows = {}
if (HERE / 't80_unmask.jsonl').exists():
    for line in (HERE / 't80_unmask.jsonl').read_text().splitlines():
        d = json.loads(line); rows[(d['net'], d['arm'])] = d['mse']
chk = [(r, gn, FIX[gn][r], rows[(r, 'unmask15:' + gn)] / rows[(r, 'base')] * base[r]) for r in ALL for gn in ('a', 'r', 'q') if (r, 'unmask15:' + gn) in rows and (r, 'base') in rows]
if chk: print('  certificate vs the direct replay (t80_unmask): %d rows, max rel difference of the arm\'s error %.2e; base rows max rel %.2e' % (
    len(chk), max(abs(a / b - 1) for _, _, a, b in chk), max(abs(rows[(r, 'base')] / base[r] - 1) for r, _, _, _ in chk)), flush=True)

LAMS = (1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def pieces(z, ks):
    out = {}
    for r in TRAIN:
        N = NET[r]; Z = t.stack([N[z][k] for k in ks], 1); g = N['g'] * N['kappa']; res = N['m'] - N['y']
        Phi = (Z[:, :, None] * g[:, None, :]).reshape(len(res), -1)
        out[r] = (Phi.T @ Phi / len(res), Phi.T @ res / len(res), base[r])
    return out


def solve(P, nets, lam):
    A = sum(P[r][0] for r in nets) / len(nets); b = sum(P[r][1] for r in nets) / len(nets)
    return -t.linalg.solve(A + lam * A.diagonal().mean() * t.eye(len(b), dtype=f64), b)


def quad(P, nets, U):
    A = sum(P[r][0] for r in nets) / len(nets); b = sum(P[r][1] for r in nets) / len(nets)
    return float(2 * b @ U + U @ A @ U) / np.mean([P[r][2] for r in nets])


def fit(z, ks, folds=5):
    P = pieces(z, ks); F = [TRAIN[i::folds] for i in range(folds)]
    cv = {lam: np.mean([quad(P, f, solve(P, [r for r in TRAIN if r not in set(f)], lam)) for f in F]) for lam in LAMS}
    lam = min(cv, key=cv.get); U = solve(P, TRAIN, lam).reshape(len(ks), 128)
    step = min((0.5, 0.75, 1.0, 1.25), key=lambda s_: np.mean([mse(r, z, ks, s_ * U) for r in TRAIN]))
    return cv[lam], lam, step, step * U


print('\n(2) REFIT: new layer-15 weights from 0 (cross-validated on the %d training networks; held-out = %d networks never used in the fit):' % (len(TRAIN), len(VAL)), flush=True)
SAVE = {}
for z in ('head', 'std'):
    for name, ks in [(str(k), (k,)) for k in FEATS] + [(g, ks) for g, ks in GROUPS.items() if len(ks) > 1]:
        if z == 'std' and len(ks) == 1: continue
        zz = 'zh' if z == 'head' else 'zs'
        v_cv, lam, step, U = fit(zz, ks)
        tr = table(TRAIN, {r: mse(r, zz, ks, U) for r in TRAIN}); va = table(VAL, {r: mse(r, zz, ks, U) for r in VAL}) if VAL else '-'
        print('  %-4s %-5s cv %+.3f%% (lam %g, step %.2f) | train %s | HELD-OUT %s' % (z, name, 100 * v_cv, lam, step, tr, va), flush=True)
        SAVE['%s_%s' % (z, name)] = U.numpy()
np.savez(HERE / f't83_U_{LABEL}.npz', **SAVE)
