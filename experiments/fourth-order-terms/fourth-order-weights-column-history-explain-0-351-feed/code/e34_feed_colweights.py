#!/usr/bin/env python3
"""TAG opus-reexam E34: can the per-column fourth-order weights (the physicist's free fit holds 0.49 / 0.60 of the true feed at layers 9 / 13,
against 0.34 for isotropic + exact hub-local terms) be built from how a column EVOLVED? Open loop, network 35, the physicist's truth and operators.
K4pre(l) ~ isotropic + sum_r lam_r A_r pp_r^4; its feed into the births of layer l is the column shift (d/2) o pp_r^2 with weight lam_r A_r.
Candidate A_r, each with one fitted coefficient (cross-block score = share of the true feed's energy on the kink block of layer l+1):
  ruleA   S1 / (coef gap) of the birth hub, ages <= 2 (F4's rule)                     const_<age group>: one constant per age group
  convD   12 sum_a g^3 d v pp'^3 / sum_a (g pp')^4   (conversion of the column's own third-order content at layer l-1, diagonal of C)   [free]
  convF   12 (g d pp'^2)' C (g^2 pp') / |g pp'|^4     (the same with the full covariance)                                              [a push]
  convT   the top covariance mode's part of convF                                                                                      [free]
pp' = the column one layer earlier (pre-activation space of layer l-1). usage: e34_feed_colweights.py <rank> <layer> [threads=3]"""
import sys, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; PH = HERE / 'physicist'; P2 = HERE.parent
sys.path.insert(0, str(PH))
import numpy as np, torch
import t2_lib as L
from t4_lib import Op, T
torch.set_grad_enabled(False)
rank = int(sys.argv[1]); l = int(sys.argv[2]); torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 3)
BP = np.load(PH / f't2_budget_rank{rank:04d}_planes.npz'); BI = np.load(PH / f't4births_rank{rank:04d}.npz')
MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
WALL = np.load((STORE / man['path']).parent / 'weights.npy', mmap_mode='r')
W = T(WALL[l + 1])
COL = np.load(PH / f't4cols_rank{rank:04d}_L{l}.npz'); pp = T(COL['pp']).to(torch.float64); lam = T(COL['lam'].astype(np.float64)); alive = COL['alive'].astype(np.float64)
PRE = np.load(HERE / f'e15cols_rank{rank:04d}_L{l - 1}.npz'); ppm = T(PRE['pp']).to(torch.float64); nprev = ppm.shape[1]
stops = [0] + [int(x) for x in COL['stops']]; assert stops[l - 1] == nprev, (stops, nprev)
mu = MC[f'mean_pre_{l}'].astype(np.float64); C = MC[f'cov_pre_{l}'].astype(np.float64); v = np.diag(C).copy(); sd = np.sqrt(v); al = mu / sd
pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = np.array([math.erfc(-a / math.sqrt(2)) / 2 for a in al]); d = pdf / sd
Gm = T(alive * g)[:, None]; dd = T(d / 2)[:, None]
blk = BP[f'blk_L{l}']; F = [BP[f't_F4_L{l}_b{b}'].astype(np.float64)[blk] for b in (0, 1)]; FF = float(F[0] @ F[1])
Fiso = (L.feed_push_model(W.numpy(), d, g, C, 1.0, 0.0) - L.Pi(W.numpy(), L.feed_plane_model(d, g, C, 1.0, 0.0)))[blk]
t0 = time.time(); n = 1024; ncol = pp.shape[1]
ages = np.zeros(ncol, dtype=int)
for j in range(l): ages[stops[j]:stops[j + 1]] = l - j
# features
A = {}
ruleA = np.zeros(ncol)
for j in range(l):
    b = lambda k: BI[f'{k}_{j}']
    s_ = b('sel'); a, s, ph = (b(k)[s_].astype(np.float64) for k in ('alpha', 'sigma', 'pdf')); tp, tm, coef = (b(k)[s_].astype(np.float64) for k in ('plus', 'minus', 'coef'))
    S1 = -a * ph / (s * s); k0 = stops[j]; K = (stops[j + 1] - k0) // 2; val = S1 / (coef * (tp - tm)); ruleA[k0:k0 + K] = val; ruleA[k0 + K:k0 + 2 * K] = val
A['ruleA_le2'] = ruleA * (ages <= 2); A['ruleA_3to4'] = ruleA * ((ages >= 3) & (ages <= 4)); A['ruleA_5p'] = ruleA * (ages >= 5)
for name, m_ in (('const_1', ages == 1), ('const_2', ages == 2), ('const_34', (ages >= 3) & (ages <= 4)), ('const_5p', ages >= 5)): A[name] = m_.astype(np.float64)
# conversion features from the previous layer (columns that existed at l-1)
Cm = T(MC[f'cov_pre_{l - 1}']); am, sm, gm, pm = (T(BI[f'{k}_{l - 1}'].astype(np.float64)) for k in ('alpha', 'sigma', 'gate', 'pdf')); dm = pm / sm; vm = sm * sm
gp = gm[:, None] * ppm; den = (gp ** 4).sum(0).clamp_min(1e-300)
convD = 12 * ((gm ** 3 * dm * vm)[:, None] * ppm ** 3).sum(0) / den
left = (gm * dm)[:, None] * ppm ** 2; right = (gm * gm)[:, None] * ppm
convF = 12 * (left * (Cm @ right)).sum(0) / den
ev, evec = torch.linalg.eigh(Cm); top = evec[:, -1]; lam1 = ev[-1]
convT = 12 * lam1 * (top @ left) * (top @ right) / den
for name, x in (('convD', convD), ('convF', convF), ('convT', convT)):
    full = np.zeros(ncol); full[:nprev] = x.numpy(); A[name] = full
    for gname, m_ in (('2', ages == 2), ('34', (ages >= 3) & (ages <= 4)), ('5p', ages >= 5)): A[f'{name}_{gname}'] = full * m_
u = Gm * pp; op = Op(W, u, dd * pp * pp); planes = {}
for k, a_ in A.items():
    planes[k] = op.fwd(lam * T(a_)).numpy()[blk]
    print(json.dumps(dict(feature=k, weighted_rms=float('%.3g' % np.sqrt((a_ ** 2 * (lam.numpy() ** 2) * ((pp ** 2).sum(0).numpy() ** 3)).sum() / ((lam.numpy() ** 2) * ((pp ** 2).sum(0).numpy() ** 3)).sum())), wall=round(time.time() - t0, 1))), flush=True)
def cv(keys):
    Xm = np.stack([Fiso] + [planes[k] for k in keys], 1); co = [np.linalg.lstsq(Xm, F[b_], rcond=None)[0] for b_ in (0, 1)]; fit = [Xm @ co[b_] for b_ in (0, 1)]
    sc = np.mean([(2 * float(fit[b_] @ F[1 - b_]) - float(fit[b_] @ fit[b_])) / FF for b_ in (0, 1)])
    return round(float(sc), 4), [float('%.3g' % x) for x in (co[0] + co[1]) / 2]
consts = ['const_1', 'const_2', 'const_34', 'const_5p']
res = {'iso': cv([]), 'iso + ruleA(<=2)': cv(['ruleA_le2']), 'iso + ruleA(<=2) + consts by age': cv(['ruleA_le2'] + consts),
       'iso + ruleA all ages': cv(['ruleA_le2', 'ruleA_3to4', 'ruleA_5p']), 'iso + ruleA all ages + consts': cv(['ruleA_le2', 'ruleA_3to4', 'ruleA_5p'] + consts),
       'iso + ruleA(<=2) + convD': cv(['ruleA_le2', 'convD']), 'iso + ruleA(<=2) + convD by age': cv(['ruleA_le2', 'convD_2', 'convD_34', 'convD_5p']),
       'iso + ruleA(<=2) + convF': cv(['ruleA_le2', 'convF']), 'iso + ruleA(<=2) + convF by age': cv(['ruleA_le2', 'convF_2', 'convF_34', 'convF_5p']),
       'iso + ruleA(<=2) + convT': cv(['ruleA_le2', 'convT']),
       'everything free (ruleA all ages, consts, convD by age, convT by age)': cv(['ruleA_le2', 'ruleA_3to4', 'ruleA_5p'] + consts + ['convD_2', 'convD_34', 'convD_5p', 'convT_2', 'convT_34', 'convT_5p']),
       'everything incl. convF by age': cv(['ruleA_le2', 'ruleA_3to4', 'ruleA_5p'] + consts + ['convD_2', 'convD_34', 'convD_5p', 'convF_2', 'convF_34', 'convF_5p'])}
print(json.dumps(dict(TAG='opus-reexam E34', rank=rank, layer=l, columns=int(ncol), by_age={str(a_): int((ages == a_).sum()) for a_ in range(1, 6)}, results=res)))
