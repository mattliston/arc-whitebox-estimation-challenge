#!/usr/bin/env python3
"""TAG opus-reexam E15c (E15b + exchange terms TB1, TB2; see the block marked E15c): the COLLECTIVE part of the fourth-cumulant feed (open loop, network 35, physicist's truth and operators).
The feed into the births of layer l with hub k is (d_k/2) g_i g_j kappa(z_i, z_j, z_k, z_k). With z_k = sum_a W_ak x_a the same-unit
part of z_k^2 is D_k = sum_a W_ak^2 (x_a - m_a)^2, and (x_a - m_a)^2 responds linearly to the previous layer's pre-activation,
d Var(x_a)/d mu_a = V21_a = 2 m_a (1 - g_a). So
     kappa(z_i, z_j, z_k, z_k) ~ sum_a W_ak^2 V21_a kappa(z_i, z_j, zprev_a)
and kappa(z_i, z_j, zprev_a) is the third cumulant with one leg left UNGATED at the previous layer: for a carried bank column r it is
lam_r pp_r(l)_i pp_r(l)_j pp_r(l-1)_a, for a pair born at the previous layer at hub b the ungated leg is the covariance column C e_b.
   COLL   W_ak^2 -> 2/n: every column shifts along the SAME vector d/2 by its own scalar c_r = (2/n) V21 . (ungated leg)   [free]
   COLLX  exact W_ak^2: third leg (d/2) o [(W o W)' (V21 o ungated leg)]                                                   [one more push]
Scored like E15 (share of the true feed's energy on the kink block of layer l+1, cross-block), alone and with the local terms.
usage: e15b_collective_feed.py <rank> <layer> [threads=2]"""
import sys, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; PH = HERE / 'physicist'; P2 = HERE.parent
sys.path.insert(0, str(PH))
import numpy as np, torch
import t2_lib as L
from t4_lib import Op, T
torch.set_grad_enabled(False)
rank = int(sys.argv[1]); l = int(sys.argv[2]); torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 2)
BP = np.load(PH / f't2_budget_rank{rank:04d}_planes.npz'); BI = np.load(PH / f't4births_rank{rank:04d}.npz')
MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
WALL = np.load((STORE / man['path']).parent / 'weights.npy', mmap_mode='r')
W = T(WALL[l + 1]); Wl = T(WALL[l])
COL = np.load(PH / f't4cols_rank{rank:04d}_L{l}.npz'); pp = T(COL['pp']); lam3 = COL['lam'].astype(np.float64); alive = COL['alive'].astype(np.float64)
PRE = np.load(HERE / f'e15cols_rank{rank:04d}_L{l - 1}.npz'); ppm = T(PRE['pp']); nprev = ppm.shape[1]
stops = [0] + [int(x) for x in COL['stops']]; assert stops[l - 1] == nprev, (stops, nprev)
mu = MC[f'mean_pre_{l}'].astype(np.float64); C = MC[f'cov_pre_{l}'].astype(np.float64); v = np.diag(C).copy(); sd = np.sqrt(v); al = mu / sd
pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = np.array([math.erfc(-a / math.sqrt(2)) / 2 for a in al]); d = pdf / sd
Gm = T(alive * g)[:, None]; dd = T(d / 2)[:, None]
blk = BP[f'blk_L{l}']; F = [BP[f't_F4_L{l}_b{b}'].astype(np.float64)[blk] for b in (0, 1)]; FF = float(F[0] @ F[1])
Fiso = (L.feed_push_model(W.numpy(), d, g, C, 1.0, 0.0) - L.Pi(W.numpy(), L.feed_plane_model(d, g, C, 1.0, 0.0)))[blk]
t0 = time.time(); n = 1024
# previous layer: V21 = 2 m (1 - g) and its covariance
Cm = T(MC[f'cov_pre_{l - 1}']); mm = BI[f'host_{l - 1}'].astype(np.float64); gm = BI[f'gate_{l - 1}'].astype(np.float64); V21 = T(2 * mm * (1 - gm))
sel = BI[f'sel_{l - 1}']; K = len(sel); assert stops[l] - stops[l - 1] == 2 * K
ung = torch.cat((ppm.to(torch.float64), Cm[:, sel], Cm[:, sel]), 1)            # ungated legs: carried columns, then the newborn pairs (plus block, minus block)
c = (2.0 / n) * (V21[:, None] * ung).sum(0)                                     # collective scalar per column
X = (Wl * Wl).T @ (V21[:, None] * ung)                                          # exact k-dependent third leg, before d/2
u = Gm * pp.to(torch.float64); lam = T(lam3); planes = {}
ages = np.zeros(pp.shape[1], dtype=int)
for j in range(l): ages[stops[j]:stops[j + 1]] = l - j
grp = {'age1': ages == 1, 'age2': ages == 2, 'age34': (ages >= 3) & (ages <= 4), 'age5p': ages >= 5}
for name, m_ in grp.items():
    if not m_.any(): continue
    mt = torch.from_numpy(m_)
    planes[f'COLL_{name}'] = Op(W, u[:, mt], dd * torch.ones((n, int(m_.sum())), dtype=torch.float64)).fwd(lam[mt] * c[mt]).numpy()[blk]
    planes[f'COLLX_{name}'] = Op(W, u[:, mt], dd * X[:, mt]).fwd(lam[mt]).numpy()[blk]
    print(json.dumps({name: dict(cols=int(m_.sum()), c_med_abs=float(c[mt].abs().median()), c_mean=float(c[mt].mean()),
                                 shift_over_column=float(((c[mt].abs()[None, :] * dd).norm(dim=0) / u[:, mt].norm(dim=0)).median()), wall=round(time.time() - t0, 1))}), flush=True)
names = [k for k in grp if f'COLL_{k}' in planes]
planes['COLL'] = sum(planes[f'COLL_{k}'] for k in names); planes['COLLX'] = sum(planes[f'COLLX_{k}'] for k in names)
planes['COLLX_fluct'] = planes['COLLX'] - planes['COLL']
# local terms of E15 (all four coefficients at their physical values), all ages together
def hub_data(j):
    b = lambda k: BI[f'{k}_{j}']
    s_ = b('sel'); a, s, gt, ph, m = (b(k)[s_].astype(np.float64) for k in ('alpha', 'sigma', 'gate', 'pdf', 'host'))
    tp, tm = (b(k)[s_].astype(np.float64) for k in ('plus', 'minus')); vv = s * s; dj = ph / s
    r2 = vv * ((1 + a * a) * gt + a * ph); r3 = vv * s * ((a ** 3 + 3 * a) * gt + (a * a + 2) * ph); r4 = vv * vv * ((a ** 4 + 6 * a * a + 3) * gt + (a ** 3 + 5 * a) * ph)
    return tp, tm, (-a * ph / vv, 2 * (gt - gt * gt - m * dj), 3 * (1 - gt) * (r2 - 2 * m * m), r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2)
loc = 0.0
for j in range(l):
    tp, tm, S = hub_data(j); k0 = stops[j]; Kj = (stops[j + 1] - k0) // 2
    pplus = pp[:, k0:k0 + Kj].to(torch.float64); pminus = pp[:, k0 + Kj:k0 + 2 * Kj].to(torch.float64); gap = T(tp - tm)[None, :]
    Wt = (pplus - pminus) / gap; Y = (T(tp)[None, :] * pminus - T(tm)[None, :] * pplus) / gap; S1, S2, S3, S4 = (T(x)[None, :] for x in S)
    a = Gm * Y; b = Gm * Wt; one = torch.ones(Kj, dtype=torch.float64); yy, yw, ww = dd * Y * Y, dd * Y * Wt, dd * Wt * Wt
    h1 = 2 * S1 * yw + S2 * ww; h2 = S1 * yy + 2 * S2 * yw + S3 * ww; h3 = S2 * yy + 2 * S3 * yw + S4 * ww
    loc = loc + Op(W, torch.cat((a, a + b, a - b, b), 1), torch.cat((h1, h2, h2, h3), 1)).fwd(torch.cat((one, one / 2, -one / 2, one))).numpy()[blk]
planes['LOCAL'] = loc

# ---- E15c: s-channel exchange between a bank vertex and a Gaussian vertex of the previous layer (the fourth cumulant born when a leg of
# the third cumulant sits on a kinked unit h of layer l-1 that also has a covariance pendant; E14's "T-born" term, in the feed channel)
#   TB1: (i,j) on the bank column r, (k,k) = hub h and its pendant: third leg (d/2) 2 theta_r, theta_r(k) = sum_h P_hr d_h W_hk y_h(k)
#   TB2: (i,j) = hub h and its pendant (the age-1 pair's cross term, with the V21 w w companion), (k,k) on the bank:
#        both columns of the pair of hub h shift by (d/2) Pi[., h],  Pi[k, h] = sum_r lam_r pp_r(l)_k^2 pp_r(l-1)_h
dm = T(BI[f'pdf_{l - 1}'].astype(np.float64) / BI[f'sigma_{l - 1}'].astype(np.float64)); gmT = T(gm)
Coff = Cm - torch.diag(torch.diagonal(Cm)); KW = (dm[:, None] * Coff * gmT[None, :]) @ Wl            # KW[h,k] = d_h y_h(k)
ung0 = torch.cat((ppm.to(torch.float64), Coff[:, sel], Coff[:, sel]), 1)
theta = (Wl * KW).T @ ung0
for name, m_ in grp.items():
    if not m_.any(): continue
    mt = torch.from_numpy(m_); planes[f'TB1_{name}'] = Op(W, u[:, mt], dd * 2 * theta[:, mt]).fwd(lam[mt]).numpy()[blk]
planes['TB1'] = sum(planes[f'TB1_{k}'] for k in names)
Pi_ = ((pp[:, :nprev].to(torch.float64) ** 2) * lam[:nprev][None, :]) @ ppm.to(torch.float64)[sel, :].T        # [k, hub]
tp, tm, _S = hub_data(l - 1); k0 = stops[l - 1]
pplus = pp[:, k0:k0 + K].to(torch.float64); pminus = pp[:, k0 + K:k0 + 2 * K].to(torch.float64); gap = T(tp - tm)[None, :]
Wt = (pplus - pminus) / gap; Y = (T(tp)[None, :] * pminus - T(tm)[None, :] * pplus) / gap; a = Gm * Y; b = Gm * Wt; one = torch.ones(K, dtype=torch.float64)
h2 = dd * Pi_ * dm[sel][None, :]; h3 = dd * Pi_ * V21[sel][None, :]
planes['TB2'] = Op(W, torch.cat((a + b, a - b, b), 1), torch.cat((h2, h2, h3), 1)).fwd(torch.cat((one / 2, -one / 2, one))).numpy()[blk]
planes['TB2_cross_only'] = Op(W, torch.cat((a + b, a - b), 1), torch.cat((h2, h2), 1)).fwd(torch.cat((one / 2, -one / 2))).numpy()[blk]
print(json.dumps(dict(E15c=dict(theta_med_abs=float(theta.abs().median()), Pi_med_abs=float(Pi_.abs().median()), wall=round(time.time() - t0, 1)))), flush=True)


def cv(keys, fixed=(), iso=True):
    base = sum(planes[k] for k in fixed) if fixed else 0.0
    cols = ([Fiso] if iso else []) + [planes[k] for k in keys]
    if not cols:
        fit = [base, base]; co = [np.zeros(0)] * 2
    else:
        Xm = np.stack(cols, 1); co = [np.linalg.lstsq(Xm, F[b_] - base, rcond=None)[0] for b_ in (0, 1)]; fit = [Xm @ co[b_] + base for b_ in (0, 1)]
    sc = np.mean([(2 * float(fit[b_] @ F[1 - b_]) - float(fit[b_] @ fit[b_])) / FF for b_ in (0, 1)])
    return round(float(sc), 4), [round(float(x), 4) for x in (co[0] + co[1]) / 2]


print(json.dumps(dict(TAG='opus-reexam E15b collective feed', rank=rank, layer=l,
                      norms_pct_of_feed={k: round(100 * float(np.linalg.norm(p_)) / math.sqrt(FF), 1) for k, p_ in planes.items()},
                      iso_norm_at_fit=round(100 * abs(cv([])[1][0]) * float(np.linalg.norm(Fiso)) / math.sqrt(FF), 1),
                      cos_with_feed={k: round((float(p_ @ F[0]) + float(p_ @ F[1])) / 2 / max(float(np.linalg.norm(p_)) * math.sqrt(FF), 1e-300), 3) for k, p_ in planes.items()},
                      cos_with_iso={k: round(float(p_ @ Fiso) / max(float(np.linalg.norm(p_)) * float(np.linalg.norm(Fiso)), 1e-300), 3) for k, p_ in planes.items()})), flush=True)
res = {}
res['iso'] = cv([])
res['COLL at 1, no iso'] = cv([], fixed=['COLL'], iso=False)
res['COLL fitted, no iso'] = cv(['COLL'], iso=False)
res['iso + COLL fitted'] = cv(['COLL'])
res['iso + COLL fitted per age'] = cv([f'COLL_{k}' for k in names])
res['COLLX at 1, no iso'] = cv([], fixed=['COLLX'], iso=False)
res['COLLX fitted, no iso'] = cv(['COLLX'], iso=False)
res['iso + COLLX fitted'] = cv(['COLLX'])
res['iso + COLLX fitted per age'] = cv([f'COLLX_{k}' for k in names])
res['iso + COLL + COLLX_fluct fitted'] = cv(['COLL', 'COLLX_fluct'])
res['iso + LOCAL fitted'] = cv(['LOCAL'])
res['iso + LOCAL + COLL fitted'] = cv(['LOCAL', 'COLL'])
res['iso + LOCAL + COLLX fitted'] = cv(['LOCAL', 'COLLX'])
res['LOCAL + COLLX at 1, no iso'] = cv([], fixed=['LOCAL', 'COLLX'], iso=False)
res['LOCAL + COLL at 1, no iso'] = cv([], fixed=['LOCAL', 'COLL'], iso=False)
res['iso + LOCAL + COLLX per age fitted'] = cv(['LOCAL'] + [f'COLLX_{k}' for k in names])
res['iso + TB1 fitted'] = cv(['TB1'])
res['iso + TB1 fitted per age'] = cv([f'TB1_{k}' for k in names])
res['iso + TB2 fitted'] = cv(['TB2'])
res['iso + TB2 cross only fitted'] = cv(['TB2_cross_only'])
res['iso + TB1 + TB2 fitted'] = cv(['TB1', 'TB2'])
res['iso + LOCAL + TB1 + TB2 fitted'] = cv(['LOCAL', 'TB1', 'TB2'])
res['iso + LOCAL + COLLX + TB1 + TB2 fitted'] = cv(['LOCAL', 'COLLX', 'TB1', 'TB2'])
res['iso + (LOCAL + COLLX + TB1 + TB2 at 1)'] = cv([], fixed=['LOCAL', 'COLLX', 'TB1', 'TB2'])
res['LOCAL + COLLX + TB1 + TB2 at 1, no iso'] = cv([], fixed=['LOCAL', 'COLLX', 'TB1', 'TB2'], iso=False)
for k_, v_ in res.items(): print(json.dumps({k_: v_}), flush=True)
print(json.dumps(dict(done=True, wall=round(time.time() - t0, 1))), flush=True)
