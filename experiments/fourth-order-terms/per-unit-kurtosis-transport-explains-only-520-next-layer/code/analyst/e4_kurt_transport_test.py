#!/usr/bin/env python3
"""TAG opus-reexam E4: can a PER-UNIT carrier transport the network-specific kurtosis (and skew) from layer l to l+1?
Oracle test with MC truth only (net 35): y_l = per-unit standardised excess kurtosis of the pre-activation at layer l (32M truth).
Predict y_{l+1} from y_l pushed through candidate linear operators built from W_{l+1} and the true post covariance of layer l:
  Tvol[a,i] = W[i,a] (C W)[i,a] / v'_a   (share of unit a's variance that flows through upstream unit i; rows sum to 1)
  Tcoin[a,i] = W[i,a]^2 C[i,i] / sum_i(...) (coincident-index share)
on the POST side of layer l the pre kurtosis is first dressed by the unit's variance elasticity (dlog postvar / dlog prevar).
Reports corr / R2 of each predictor against the truth at l+1, plus a ridge on all of them. If even the truth at layer l cannot
predict layer l+1, no per-unit fourth-order carrier can work."""
import sys, json
from pathlib import Path
import numpy as np
from math import erf
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent   # analyst copy
rank = int(sys.argv[1]) if len(sys.argv) > 1 else 35
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
W = np.load((STORE / man['path']).parent / 'weights.npy').astype(np.float64)
t = np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_P32M.npz')
_ti = P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_indep32M.npz'; ti = np.load(_ti) if _ti.exists() else None   # analyst copy: any rank with a second draw
mc = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / np.sqrt(2)))); phi = lambda a: np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
def g2(tt, l): return tt['mc_pre_k4'][l].astype(np.float64) / tt['mc_pre_v'][l].astype(np.float64) ** 2
def g1(tt, l): return tt['mc_pre_k3'][l].astype(np.float64) / tt['mc_pre_v'][l].astype(np.float64) ** 1.5
def r2(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X)); c, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ c
    return 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(' l->l+1 | spread y_l  y_l+1 | reliability(y_l+1, two draws) | corr: raw y_l(same idx)  Tvol*y  Tvol*(e*y)  Tcoin*y  Tabs*y | R2 all | same for skew g1: Tvol*g1 corr, R2 | R2 of y_l+1 on [alpha^2, |alpha|] of l+1')
for l in range(2, 15):
    y0, y1 = g2(t, l), g2(t, l + 1)
    C = mc[f'cov_post_{l}']; Wn = W[l + 1]                 # z' = y @ Wn ; unit a of layer l+1 = column a
    CW = C @ Wn; vprime = (Wn * CW).sum(0)
    Tvol = (Wn * CW).T / vprime[:, None]                    # [a, i]
    co = (Wn ** 2 * np.diag(C)[:, None]).T; Tcoin = co / co.sum(1, keepdims=True)
    Tabs = np.abs(Wn * CW).T; Tabs = Tabs / Tabs.sum(1, keepdims=True)
    # elasticity of post variance w.r.t. pre variance at fixed mean, Gaussian formula: Var(relu) = v[(1+a^2)Phi + a phi] - v (a Phi + phi)^2
    mu = t['mc_pre_m'][l].astype(np.float64); v = t['mc_pre_v'][l].astype(np.float64); a = mu / np.sqrt(v)
    def pv(vv):
        aa = mu / np.sqrt(vv); P_ = Phi(aa); p_ = phi(aa)
        return vv * ((1 + aa * aa) * P_ + aa * p_) - vv * (aa * P_ + p_) ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        el = (np.log(pv(v * 1.001)) - np.log(pv(v / 1.001))) / (2 * np.log(1.001))
    nbad = int((~np.isfinite(el)).sum()); el = np.where(np.isfinite(el), el, 0.0)   # analyst copy: dead units (post variance underflows) pass nothing on
    f = [Tvol @ y0, Tvol @ (el * y0), Tcoin @ y0, Tabs @ y0, Tvol @ (el * el * y0)]
    rel = np.corrcoef(g2(ti, l + 1), y1)[0, 1] if ti is not None else float('nan')
    s0, s1 = g1(t, l), g1(t, l + 1)
    fs = [Tvol @ s0, Tvol @ (el * s0), Tcoin @ s0]
    mu1 = t['mc_pre_m'][l + 1].astype(np.float64); v1 = t['mc_pre_v'][l + 1].astype(np.float64); a1 = mu1 / np.sqrt(v1)
    print(f'{l:2d}->{l + 1:2d} | {y0.std():.4f} {y1.std():.4f} | {rel:.3f} | '
          + ' '.join(f'{np.corrcoef(x, y1)[0, 1]:+.3f}' for x in [y0] + f[:4]) + f' | {r2(y1, f):.3f} | '
          f'{np.corrcoef(fs[0], s1)[0, 1]:+.3f} {r2(s1, fs):.3f} | {r2(y1, [a1 ** 2, np.abs(a1)]):.3f} | dead units zeroed: {nbad}')
