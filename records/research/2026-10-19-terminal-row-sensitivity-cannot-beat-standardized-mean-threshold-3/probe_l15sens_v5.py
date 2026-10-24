"""[fable-lead 2026-10-19] L15 k3 SENSITIVITY price list. One in-process predict (0.12.3 package); captures at the terminal layer the
k3 push (pp rows x columns, lam), the row-cut rows and the (mu, v) of all units, the k3 the physics used, and the predict output. Identifies
which per-unit quantity (post mean m or post variance var of the terminal map) the output row is, then prices
  (a) every KEPT row's k3 by its effect on the output: dy_a = (dy/dk3)_a * k3_a (the terminal map is per unit: numerical derivative of
      the identified quantity w.r.t. k3 with the row's own mu, v, k4), and the cumulative MSE added when the cheapest rows drop their k3
      (set to 0), vs the fraction of rows dropped (each row ~ an equal share of the L15 push bill);
  (b) every PAIR (t, s) of columns by the sum over rows of (dy/dk3)_a^2 * net_aj^2, the cumulative EXACT MSE added when the cheapest pairs
      are dropped (dy_a = S_a * sum of dropped net_aj), vs the fraction of pairs dropped, old and young separately.
MSE scale: the public final MSE is ~1.3e-8 per unit; added MSE is reported relative to that. v5 (+ two-pass estimate ranking). usage: probe_l15sens_v5.py <cand> <seed> <out>"""
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
        CAP.update(pp=np.asarray(pp, dtype=np.float64), lam=np.asarray(lam, dtype=np.float64), old=int(state.get('columns') or 0),
                   cohorts=[(int(s), int(w)) for s, w in state['cohorts']])
    return pp
mod._shared_old384_push = push
_sel0 = mod.select
FLAG = [False]
def sel(score, floor):
    r = _sel0(score, floor)
    if FLAG[0] and 'sel_rows' not in CAP:
        CAP['sel_rows'] = np.asarray(r[0]).astype(int).copy()
    return r
mod.select = sel
_rc0 = mod._l15_rowcut_k3
def rowcut(layer, mu, v, *a, **k):
    FLAG[0] = True
    try:
        k3 = _rc0(layer, mu, v, *a, **k)
    finally:
        FLAG[0] = False
    if 'mu' not in CAP:
        z = np.abs(np.asarray(mu, dtype=np.float64) / np.sqrt(np.maximum(np.asarray(v, dtype=np.float64), 1e-10)))
        CAP.update(mu=np.asarray(mu, dtype=np.float64), v=np.asarray(v, dtype=np.float64), k3=np.asarray(k3, dtype=np.float64), rows=np.where(z <= mod._L15_ROWCUT_Z)[0])
    return k3
mod._l15_rowcut_k3 = rowcut
_term0 = mod._nonlinear_terminal_live
def term(mu, C, K21, k4, *a, **k):
    res = _term0(mu, C, K21, k4, *a, **k)
    if 'k4' not in CAP:
        CAP.update(k4=np.array(np.broadcast_to(np.asarray(k4, dtype=np.float64), (len(mu),))), m_out=np.asarray(res[0], dtype=np.float64),
                   target=np.asarray(res[2], dtype=np.float64) if res[2] is not None and hasattr(res[2], 'shape') else None)
    return res
mod._nonlinear_terminal_live = term
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
w = loader.weights(seed)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=seed, name=f'probe-{seed}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = est.predict(mlp, 2**41); total = int(ctx.summary_dict()['flops_used'])
y = np.asarray(y, dtype=np.float64)
print('output shape', y.shape, 'captured:', sorted(CAP.keys()))
mu, v, k3, rows, pp, lam, old = CAP['mu'], CAP['v'], CAP['k3'], CAP['rows'], CAP['pp'], CAP['lam'], CAP['old']
k4 = CAP.get('k4', np.zeros_like(mu))   # v2: the terminal-map wrapper may not fire (the loop binds the map by another route); k4 does not enter dm/dk3
n = len(mu)
# the terminal per-unit formulas (same as _nonlinear_terminal_live)
def formulas(mu, v, k3, k4):
    s, a, pdf, g, mg, r2g, r3g, r4g = mod.gaussian_raw(mu, v)
    s, a, pdf, g, mg, r2g, r3g, r4g = [np.asarray(x, dtype=np.float64) for x in (s, a, pdf, g, mg, r2g, r3g, r4g)]
    d = pdf / s; e = -a * pdf / v; f = (a * a - 1) * pdf / (v * s)
    m = mg + k3 * e / 6 + k4 * f / 24
    r2 = r2g + k3 * d / 3 + k4 * e / 12
    return m, r2 - m * m
m0, var0 = formulas(mu, v, k3, k4)
yl = y[-1] if y.ndim == 2 else y
cand = {'m': m0, 'var': var0}
if 'm_out' in CAP: cand['m_out'] = CAP['m_out']
if CAP.get('target') is not None and CAP['target'].shape == yl.shape: cand['target'] = CAP['target']
for name, arr in cand.items():
    if arr.shape == yl.shape: print('output row vs %-6s: max|diff|/max|y| = %.3e' % (name, np.max(np.abs(arr - yl)) / max(1e-30, np.max(np.abs(yl)))))
which = min((name for name in ('m', 'var') if cand[name].shape == yl.shape), key=lambda nm: np.max(np.abs(cand[nm] - yl)))
print('output quantity =', which)
eps = 1e-3 * (np.abs(k3) + 1e-12)
mp, vp = formulas(mu, v, k3 + eps, k4); mm, vm = formulas(mu, v, k3 - eps, k4)
S = ((mp - mm) if which == 'm' else (vp - vm)) / (2 * eps)                    # dy/dk3 per unit
base_mse = 1.3e-8
dy_row = S * k3                                                                # effect of dropping the row's k3 entirely
imp_row = dy_row ** 2 / n
# v3: align the pushed rows (pp, padded to a multiple of 32 by select) with the units by matching each pushed row's k3 to the physics' k3 vector
unit_of_pp = CAP['sel_rows']
assert len(unit_of_pp) == pp.shape[0], (len(unit_of_pp), pp.shape)
k3_pp = (lam[None, :] * pp ** 3).sum(axis=1)       # the cubic part of the physics' k3 on the pushed rows (the star diagonal is added separately)
kept = unit_of_pp; print('rows pushed by the physics (padded): %d of %d; recomputed |z|<=%.1f set: %d' % (len(kept), n, mod._L15_ROWCUT_Z, len(rows)))
order = kept[np.argsort(imp_row[kept])]
cum = np.cumsum(imp_row[order])
coef = np.abs(S)
order_pre = kept[np.argsort(coef[kept])]
cum_pre = np.cumsum(imp_row[order_pre])
print('ROW PRICE LIST, PRE-PUSH ranking by the coefficient |dy/dk3| alone (no k3 knowledge; cumulative added MSE / 1.3e-8):')
for frac in (0.05, 0.1, 0.15, 0.2, 0.3):
    k = int(frac * len(kept)); zz = np.abs(mu / np.sqrt(np.maximum(v, 1e-10)))[order_pre[:k]]
    print('  drop %3d%% of pushed rows (%3d): +%.3f%% of MSE; their |z|: min %.2f median %.2f max %.2f' % (int(frac * 100), k, 100 * cum_pre[k - 1] / base_mse, zz.min(), np.median(zz), zz.max()))
# v5: TWO-PASS estimate: rank rows by |coef * k3_est| where k3_est comes from the heaviest X% of pairs (pre-push weight: old |lam||f|^3-like
# via the pair's gross |contrib| sum is post-push, so use the pair's lam-weighted column energy proxy = |lam_t| * sum_rows pp_t^2 ... not
# pre-push either; here the ORACLE pair weight is used only to pick the heavy pairs (optimistic), giving an upper bound on what a two-pass
# scheme could achieve).
pairs_tmp = []
for s_, wd in CAP['cohorts']:
    h = wd // 2
    for i in range(h):
        if s_ + h + i < pp.shape[1]: pairs_tmp.append((s_ + i, s_ + h + i))
pairs_tmp = np.array(pairs_tmp); tt, ss = pairs_tmp[:, 0], pairs_tmp[:, 1]
contrib_tmp = lam[None, :] * pp ** 3
net_tmp = contrib_tmp[:, tt] + contrib_tmp[:, ss]
gross = np.abs(net_tmp).sum(axis=0)
heavy_order = np.argsort(-gross)
print('TWO-PASS ROW PRICE LIST: rows ranked by |coef * k3_est|, k3_est from the heaviest X% of pairs (oracle-picked); added MSE of dropping the cheapest rows:')
for px in (0.05, 0.1, 0.2):
    hp = heavy_order[:int(px * len(heavy_order))]
    k3_est = net_tmp[:, hp].sum(axis=1)
    imp_est = (S[kept] * k3_est) ** 2
    order_est = np.argsort(imp_est)
    cum_est = np.cumsum(imp_row[kept][order_est])
    print('  pairs %3d%%: ' % int(px * 100) + ' '.join('%d%%rows:+%.3f%%' % (int(f * 100), 100 * cum_est[int(f * len(kept)) - 1] / base_mse) for f in (0.1, 0.15, 0.2, 0.3)))
print('ROW PRICE LIST, ORACLE ranking by |dy/dk3 * k3| (cumulative added MSE / 1.3e-8):')
for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
    k = int(frac * len(kept)); print('  drop %3d%% of kept rows (%3d): +%.3f%% of MSE' % (int(frac * 100), k, 100 * cum[k - 1] / base_mse))
z = np.abs(mu / np.sqrt(np.maximum(v, 1e-10)))
print('  reference: the rows with 3 < |z| <= 3.5 (already dropped by the fired cut):', int(np.sum((z > 3) & (z <= 3.5))), 'rows, would add +%.3f%% of MSE' % (100 * np.sum(imp_row[(z > 3) & (z <= 3.5)]) / base_mse))
# pairs
pairs = []
for s_, wd in CAP['cohorts']:
    h = wd // 2
    for i in range(h):
        if s_ + h + i < pp.shape[1]: pairs.append((s_ + i, s_ + h + i))
pairs = np.array(pairs); t, s_ = pairs[:, 0], pairs[:, 1]
contrib = lam[None, :] * pp ** 3                       # rows(kept) x columns
net = contrib[:, t] + contrib[:, s_]                    # pushed rows x pairs (aligned to units via unit_of_pp)
Sk = S[kept][:, None]
imp_pair = np.sum((Sk * net) ** 2, axis=0) / n
is_old = t < old
rep = dict(seed=seed, output=which, n=n, kept=int(len(kept)), pairs=int(len(pairs)), old_pairs=int(is_old.sum()), rows_curve={}, pairs_curve={})
for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7): rep['rows_curve'][str(frac)] = float(100 * cum[int(frac * len(kept)) - 1] / base_mse)
print('PAIR PRICE LIST (drop the cheapest pairs by sensitivity-weighted contribution; exact cumulative added MSE / 1.3e-8):')
for side, sel in (('old', is_old), ('young', ~is_old), ('all', np.ones(len(t), bool))):
    idx = np.where(sel)[0]; o = idx[np.argsort(imp_pair[idx])]
    line = []
    for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        k = int(frac * len(o)); drop = o[:k]
        dy = S[kept] * net[:, drop].sum(axis=1)
        added = 100 * np.sum(dy ** 2) / n / base_mse
        line.append('%d%%:+%.3f%%' % (int(frac * 100), added)); rep['pairs_curve']['%s_%s' % (side, frac)] = float(added)
    print('  %-5s ' % side + ' '.join(line))
json.dump(rep, out.open('x'), indent=1)
