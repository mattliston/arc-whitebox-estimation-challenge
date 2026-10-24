"""[fable-lead 2026-10-19] L15 k3 COLUMN CONCENTRATION probe: exec a candidate in-process (0.12.3 flopscope package), run setup + one
predict on one seed, capture the final-layer k3 push (_l15_rowcut_k3 -> _shared_old384_push at layer 15: lam, young roster p, old
factors, pushed rows pp), and price a COLUMN CUT of that cubic contraction: k3_rows = sum_j lam_j pp[:, j]^3. Reports, for the oracle
order (|lam_j| sum_r |pp_rj|^3) and for pre-push proxies (young: |lam_j| ||p_j||_3^3; old: |lam_j| ||f_j||^3), the relative k3 error
||k3 - k3_K|| / ||k3|| when only the top K columns are pushed, K = 95/90/80/70/60/50/40/30% of the columns, young and old separately.
usage: probe_l15k3_v1.py <candidate.py> <seed> <out.json>   env: BLAS 1."""
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
mod._PB_WALL_S = 1e9
mod._FAST_RESCUE_CAP_S = 1e9
CAP = {}
_push0 = mod._shared_old384_push
def push(layer, w32, p, previous, newborn, lam, state, *a, **k):
    pp = _push0(layer, w32, p, previous, newborn, lam, state, *a, **k)
    if int(layer) == 15 and 'pp' not in CAP:
        F = state.get('F'); g = state.get('lazy')
        if F is None and g is not None:
            F = np.concatenate([np.asarray(g['map']) @ np.asarray(g['base']), np.asarray(g['join'])], axis=1)
        CAP.update(pp=np.asarray(pp, dtype=np.float64), lam=np.asarray(lam, dtype=np.float64), p=np.asarray(p, dtype=np.float64),
                   F=None if F is None else np.asarray(F, dtype=np.float64), old=int(state.get('columns') or 0), rows=int(w32.shape[1]))
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
assert 'pp' in CAP, 'no layer-15 push captured'
pp, lam, p, F, old = CAP['pp'], CAP['lam'], CAP['p'], CAP['F'], CAP['old']
m, n = pp.shape
contrib = lam[None, :] * pp ** 3                      # rows x columns
k3 = contrib.sum(axis=1)
oracle = np.abs(lam) * (np.abs(pp) ** 3).sum(axis=0)
proxy = np.full(n, np.nan)
young = slice(old, n)
proxy[young] = np.abs(lam[young]) * (np.abs(p) ** 3).sum(axis=0)[:n - old] if p.shape[1] >= n - old else np.nan
if F is not None and old:
    proxy[:old] = np.abs(lam[:old]) * (np.linalg.norm(F, axis=0) ** 3)[:old]
def curve(order, cols, label):
    res = {}
    w_ = order[cols]; idx = cols[np.argsort(-w_, kind='stable')]
    for frac in (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
        keep = idx[:int(round(frac * len(idx)))]
        drop = np.setdiff1d(cols, keep)
        k3_cut = k3 - contrib[:, drop].sum(axis=1)
        res[str(frac)] = dict(err_rel=float(np.linalg.norm(k3 - k3_cut) / np.linalg.norm(k3)), kept=int(len(keep)),
                             share=float(order[keep].sum() / order[cols].sum()))
    return res
cols_all, cols_old, cols_young = np.arange(n), np.arange(old), np.arange(old, n)
rep = dict(candidate=str(candidate), seed=seed, total_flops=total, rows=m, columns=n, old=old, young=n - old,
           k3_norm=float(np.linalg.norm(k3)), k3_abs_mean=float(np.abs(k3).mean()),
           oracle=dict(all=curve(oracle, cols_all, 'all'), old=curve(oracle, cols_old, 'old') if old else None, young=curve(oracle, cols_young, 'young')),
           proxy=dict(young=curve(np.nan_to_num(proxy), cols_young, 'young'), old=curve(np.nan_to_num(proxy), cols_old, 'old') if (old and F is not None) else None),
           proxy_vs_oracle_spearman=dict(young=float(np.corrcoef(np.argsort(np.argsort(proxy[young])), np.argsort(np.argsort(oracle[young])))[0, 1]),
                                         old=float(np.corrcoef(np.argsort(np.argsort(proxy[:old])), np.argsort(np.argsort(oracle[:old])))[0, 1]) if (old and F is not None) else None))
json.dump(rep, out.open('x'), indent=1)
print(json.dumps({k: rep[k] for k in ('seed', 'rows', 'columns', 'old', 'young', 'k3_norm')}))
for side in ('all', 'old', 'young'):
    c = rep['oracle'][side]
    if c: print('ORACLE %-5s ' % side + ' '.join('%s:%.4f' % (f, c[f]['err_rel']) for f in c))
for side in ('young', 'old'):
    c = rep['proxy'][side]
    if c: print('PROXY  %-5s ' % side + ' '.join('%s:%.4f' % (f, c[f]['err_rel']) for f in c) + '  spearman %.3f' % (rep['proxy_vs_oracle_spearman'][side] or 0))
