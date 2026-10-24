"""[fable-lead 2026-10-19] L15 k3 PAIR concentration probe (v2 = v1 + the cohort pair structure). Captures the layer-15 push (lam, young
roster p on the previous rows, old frame factors, pushed rows pp, state['cohorts']); checks that roster columns come in equal-and-opposite
PAIRS (cohort block (start, width): pair i = columns start+i and start+width/2+i); prices a PAIR cut of the cubic contraction
k3 = sum_j lam_j pp_j^3 with the pair-wise oracle (|net pair contribution| summed over rows) and pre-push proxies (old: |lam_t| |f_t|^3 +
|lam_s| |f_s|^3; young: the net cube of the pre-push columns on the previous rows, and the |lam| |p|_3^3 sum): relative k3 error when
only the top fraction of pairs is pushed. usage: probe_l15k3_v2.py <candidate.py> <seed> <out.json>"""
import json, sys, types
from pathlib import Path
import numpy as np
import flopscope as flops
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O / 'analyst/fresh')); import loader
candidate, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3]).resolve(); assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9; mod._FAST_RESCUE_CAP_S = 1e9
CAP = {}
_push0 = mod._shared_old384_push
def push(layer, w32, p, previous, newborn, lam, state, *a, **k):
    pp = _push0(layer, w32, p, previous, newborn, lam, state, *a, **k)
    if int(layer) == 15 and 'pp' not in CAP:
        F = state.get('F'); g = state.get('lazy')
        if F is None and g is not None:
            F = np.concatenate([np.asarray(g['map']) @ np.asarray(g['base']), np.asarray(g['join'])], axis=1)
        CAP.update(pp=np.asarray(pp, dtype=np.float64), lam=np.asarray(lam, dtype=np.float64), p=np.asarray(p, dtype=np.float64),
                   F=None if F is None else np.asarray(F, dtype=np.float64), old=int(state.get('columns') or 0),
                   cohorts=[(int(s), int(w)) for s, w in state['cohorts']], newborn_cols=int(newborn[0]) if newborn is not None else None)
    return pp
mod._shared_old384_push = push
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
w = loader.weights(seed)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=seed, name=f'probe-{seed}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = est.predict(mlp, 2**41); total = int(ctx.summary_dict()['flops_used'])
assert 'pp' in CAP
pp, lam, p, F, old, cohorts = CAP['pp'], CAP['lam'], CAP['p'], CAP['F'], CAP['old'], CAP['cohorts']
m, n = pp.shape
print(json.dumps(dict(seed=seed, rows=m, columns=n, old=old, young=n - old, p_shape=list(p.shape), F_shape=None if F is None else list(F.shape), cohorts=cohorts, newborn_cols=CAP['newborn_cols'])))
contrib = lam[None, :] * pp ** 3
k3 = contrib.sum(axis=1); k3n = np.linalg.norm(k3)
# pairs from the cohort blocks
pairs = []
for s, wd in cohorts:
    h = wd // 2
    for i in range(h):
        if s + h + i < n: pairs.append((s + i, s + h + i))
pairs = np.array(pairs)
covered = set(pairs.ravel().tolist())
print('pairs %d covering %d of %d columns; uncovered columns %d' % (len(pairs), len(covered), n, n - len(covered)))
t, s = pairs[:, 0], pairs[:, 1]
# equal-and-opposite check
num = (pp[:, t] * pp[:, s]).sum(axis=0); den = np.sqrt((pp[:, t] ** 2).sum(axis=0) * (pp[:, s] ** 2).sum(axis=0)) + 1e-30
corr = num / den
print('pair correlation of pushed columns: median %.3f, 10th pct %.3f, 90th pct %.3f; |lam_t|/|lam_s| median %.3f' % (np.median(corr), np.percentile(corr, 10), np.percentile(corr, 90), np.median(np.abs(lam[t]) / (np.abs(lam[s]) + 1e-30))))
net = contrib[:, t] + contrib[:, s]                      # rows x pairs: the net pair contribution
oracle_pair = np.abs(net).sum(axis=0)
col_sum = np.abs(contrib[:, t]).sum(axis=0) + np.abs(contrib[:, s]).sum(axis=0)
print('cancellation: sum|net| / sum(|t|+|s|) = %.4f' % (oracle_pair.sum() / col_sum.sum()))
is_old = t < old
def curve(order, sel, label):
    idx = np.where(sel)[0]; o = order[idx]; srt = idx[np.argsort(-o, kind='stable')]
    res = {}
    for frac in (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
        keep = srt[:int(round(frac * len(srt)))]; drop = np.setdiff1d(idx, keep)
        err = np.linalg.norm(net[:, drop].sum(axis=1)) / k3n
        res[str(frac)] = round(float(err), 5)
    print('%-26s ' % label + ' '.join('%s:%.4f' % (f, res[f]) for f in res))
    return res
rep = dict(seed=seed, rows=m, columns=n, old=old, pairs=int(len(pairs)), k3_norm=float(k3n), cancellation=float(oracle_pair.sum() / col_sum.sum()))
rep['oracle_old'] = curve(oracle_pair, is_old, 'ORACLE pair old')
rep['oracle_young'] = curve(oracle_pair, ~is_old, 'ORACLE pair young')
rep['oracle_all'] = curve(oracle_pair, np.ones(len(t), bool), 'ORACLE pair all')
# old proxy from frame coordinates
if F is not None and old:
    ft, fs = F[:, t[is_old]], F[:, s[is_old]]
    prox_old = np.abs(lam[t[is_old]]) * np.linalg.norm(ft, axis=0) ** 3 + np.abs(lam[s[is_old]]) * np.linalg.norm(fs, axis=0) ** 3
    full = np.zeros(len(t)); full[is_old] = prox_old
    rep['proxy_old'] = curve(full, is_old, 'PROXY old |lam||f|^3')
    rep['sp_old'] = float(np.corrcoef(np.argsort(np.argsort(prox_old)), np.argsort(np.argsort(oracle_pair[is_old])))[0, 1]); print('  spearman old proxy vs pair oracle %.3f' % rep['sp_old'])
# young proxies from the pre-push roster p (previous rows x young columns): columns of p align with young columns old..n-1?
if p.shape[1] == n - old:
    ty, sy = t[~is_old] - old, s[~is_old] - old
    netp = (lam[t[~is_old]][None, :] * p[:, ty] ** 3 + lam[s[~is_old]][None, :] * p[:, sy] ** 3)
    prox_net = np.abs(netp).sum(axis=0)
    prox_abs = np.abs(lam[t[~is_old]]) * (np.abs(p[:, ty]) ** 3).sum(axis=0) + np.abs(lam[s[~is_old]]) * (np.abs(p[:, sy]) ** 3).sum(axis=0)
    for name, pr in (('PROXY young net cube(p)', prox_net), ('PROXY young |lam||p|_3^3', prox_abs)):
        full = np.zeros(len(t)); full[~is_old] = pr
        rep['proxy_young_' + name.split()[2]] = curve(full, ~is_old, name)
        print('  spearman %s vs pair oracle %.3f' % (name, np.corrcoef(np.argsort(np.argsort(pr)), np.argsort(np.argsort(oracle_pair[~is_old])))[0, 1]))
else:
    print('young roster p has %d columns vs %d young columns: proxy skipped' % (p.shape[1], n - old))
json.dump(rep, out.open('x'), indent=1)
