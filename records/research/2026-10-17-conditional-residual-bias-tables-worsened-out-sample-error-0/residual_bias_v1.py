"""[opus-lead] residual analysis on the store networks: is the final-layer error e = yhat - y conditionally biased given per-unit
statistics (a = mu/s, s, yhat) or network-level hardness? Truth = mean of the two 32M halves (noise averages out in bin means);
noise-free gain estimates use the cross-half U. Leave-one-network-out (LORO) for every correction. usage: residual_bias_v1.py <ctrl dir>"""
import json, sys, numpy as np
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
PU = O / 'analyst/fresh/pu'; D = Path(sys.argv[1])
rows = {r['rank']: r for r in map(json.loads, (D / 'rows.jsonl').read_text().splitlines()) if 'rank' in r}
nets = []
for rank in sorted(rows):
    ta, tb = PU / f'mps_truth_rank{rank:04d}_A32M.npz', PU / f'mps_truth_rank{rank:04d}_B32M.npz'
    if not (ta.exists() and tb.exists()): continue
    p = np.fromfile(D / f'pred_{rank}.bin', dtype='<f4').reshape(16, 1024).astype('d')
    A, B = np.load(ta), np.load(tb)
    yA, yB = A['mc_post'][15], B['mc_post'][15]
    mu = 0.5 * (A['mc_pre_m'][15] + B['mc_pre_m'][15]); v = 0.5 * (A['mc_pre_v'][15] + B['mc_pre_v'][15])
    k3 = 0.5 * (A['mc_pre_k3'][15] + B['mc_pre_k3'][15]); k4 = 0.5 * (A['mc_pre_k4'][15] + B['mc_pre_k4'][15])
    s = np.sqrt(np.maximum(v, 1e-12)); a = mu / s
    # L14 kink fraction of the network (hardness proxy) from true L14 pre-moments
    a14 = 0.5 * (A['mc_pre_m'][14] + B['mc_pre_m'][14]) / np.sqrt(np.maximum(0.5 * (A['mc_pre_v'][14] + B['mc_pre_v'][14]), 1e-12))
    nets.append(dict(rank=rank, yhat=p[15], yA=yA, yB=yB, a=a, s=s, k3=k3 / s**3, k4=k4 / v**2, kink14=float(np.mean(np.abs(a14) < 1.0))))
print('networks', len(nets))
def U(yhat, yA, yB): return float(np.mean((yhat - yA) * (yhat - yB)))
base = np.array([U(n['yhat'], n['yA'], n['yB']) for n in nets]); print('base mean U %.4e' % base.mean())
# per-network scale bias: e = yhat - y vs y
print('\n== per-network multiplicative calibration (LORO): fit yhat*(1+c) on others, apply')
Y = np.concatenate([n['yhat'] for n in nets]); T = np.concatenate([0.5 * (n['yA'] + n['yB']) for n in nets])
for i, n in enumerate(nets):
    pass
def loro_gain(correction_fn, label):
    gains = []
    for i, n in enumerate(nets):
        train = [m for j, m in enumerate(nets) if j != i]
        corr = correction_fn(train, n)
        u1 = U(n['yhat'] - corr, n['yA'], n['yB']); gains.append(u1 / base[i] - 1)
    g = np.array(gains); print('%-48s LORO mean %+.3f%% (se %.3f) lower on %d/%d' % (label, 100 * g.mean(), 100 * g.std(ddof=1) / np.sqrt(len(g)), int((g < 0).sum()), len(g)))
def const_scale(train, n):
    num = sum(float(np.dot(m['yhat'], 0.5 * (m['yA'] + m['yB']) - m['yhat'])) for m in train); den = sum(float(np.dot(m['yhat'], m['yhat'])) for m in train)
    c = num / den; return -c * n['yhat']
loro_gain(const_scale, 'global scale yhat*(1+c)')
def binned(feature, nb):
    edges = np.quantile(np.concatenate([n[feature] for n in nets]), np.linspace(0, 1, nb + 1)); edges[0] -= 1; edges[-1] += 1
    def fn(train, n):
        idx = np.digitize(n[feature], edges) - 1
        means = np.zeros(nb)
        for b in range(nb):
            e = np.concatenate([(m['yhat'] - 0.5 * (m['yA'] + m['yB']))[np.digitize(m[feature], edges) - 1 == b] for m in train])
            means[b] = e.mean() if len(e) else 0.0
        return means[idx]
    return fn
for feat, nb in (('a', 16), ('a', 32), ('s', 16), ('yhat', 16), ('k3', 16), ('k4', 16)):
    loro_gain(binned(feat, nb), 'bias table by %s (%d bins)' % (feat, nb))
def binned2(nb):
    ea = np.quantile(np.concatenate([n['a'] for n in nets]), np.linspace(0, 1, nb + 1)); es = np.quantile(np.concatenate([n['s'] for n in nets]), np.linspace(0, 1, nb + 1))
    ea[0] -= 1; ea[-1] += 1; es[0] -= 1; es[-1] += 1
    def cell(m): return (np.digitize(m['a'], ea) - 1) * nb + (np.digitize(m['s'], es) - 1)
    def fn(train, n):
        means = np.zeros(nb * nb)
        allc = np.concatenate([cell(m) for m in train]); alle = np.concatenate([(m['yhat'] - 0.5 * (m['yA'] + m['yB'])) for m in train])
        for c in range(nb * nb):
            sel = allc == c
            if sel.any(): means[c] = alle[sel].mean()
        return means[cell(n)]
    return fn
loro_gain(binned2(8), 'bias table by (a, s) 8x8')
# in-sample ceiling: how much MSE would a perfect per-bin bias removal give (a 32 bins)?
edges = np.quantile(np.concatenate([n['a'] for n in nets]), np.linspace(0, 1, 33)); edges[0] -= 1; edges[-1] += 1
allc = np.concatenate([np.digitize(n['a'], edges) - 1 for n in nets]); alle = np.concatenate([(n['yhat'] - 0.5 * (n['yA'] + n['yB'])) for n in nets])
tot = np.mean(alle**2); expl = sum(((alle[allc == b]).mean())**2 * (allc == b).sum() for b in range(32)) / len(alle)
print('\nin-sample: mean(e^2) %.4e (incl. truth noise), bias-explained share by 32 a-bins %.2f%%' % (tot, 100 * expl / tot))
# error energy by |a| band (where does the MSE live?)
for lo, hi in ((-9, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 9)):
    sel = (np.concatenate([n['a'] for n in nets]) >= lo) & (np.concatenate([n['a'] for n in nets]) < hi)
    print('  a in [%g,%g): share of units %.1f%%, share of e^2 %.1f%%, mean e %+.2e (se %.1e)' % (lo, hi, 100 * sel.mean(), 100 * (alle[sel]**2).sum() / (alle**2).sum(), alle[sel].mean(), alle[sel].std() / np.sqrt(sel.sum())))
# network hardness vs error
k14 = np.array([n['kink14'] for n in nets]); print('\ncorr(kink14 fraction, base U) = %.2f' % np.corrcoef(k14, base)[0, 1])
