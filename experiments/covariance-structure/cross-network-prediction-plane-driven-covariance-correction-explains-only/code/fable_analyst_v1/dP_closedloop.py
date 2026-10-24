#!/usr/bin/env python3
"""TAG fable-analyst (AN-25): closed-loop value of the LINEAR prediction of the plane-driven covariance correction (dP_fit.py pooled ridge
coefficients over layers 2-14). At every pair-map call l = 2..14 the candidate planes of dP_data.py are combined with the coefficients
(one extra pair-map call: the map is linear in each plane) plus the two direct covariance features; the projected prediction
dP_hat = P [...] P is added at strength s to the card's covariance output (all other outputs unchanged). Coefficients are cross-fitted:
net 35 uses the fit on 100, net 100 the fit on 35, other nets the mean of both. Learned parts ON, no refit.
--k31row=c0,c1: baseline card carries the physicist's K31 row factor (L2-14) and the coefficients come from dP_data_r*_k31row.npz.
usage: dP_closedloop.py <ranks> <strengths e.g. 0.25,0.5,1> <threads> [--k31row=0.85,0.5]"""
import sys, json, time
from pathlib import Path
HERE0 = Path(__file__).resolve().parent; P2 = HERE0.parent; sys.path.insert(0, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
from common import *
import model_bank_research as model
HERE = HERE0
ranks = [int(x) for x in sys.argv[1].split(',')]; S = [float(x) for x in sys.argv[2].split(',')]; torch.set_num_threads(int(sys.argv[3]))
K31ROW = [tuple(float(x) for x in a_.split('=')[1].split(',')) for a_ in sys.argv if a_.startswith('--k31row=')]
K31ROW = K31ROW[0] if K31ROW else None
D = {r: np.load(HERE / (f'dP_data_r{r}' + ('_k31row' if K31ROW else '') + '.npz')) for r in (35, 100)}; names = list(D[35]['names']); L = range(2, 15)
def fit(r, lam=1e-6):
    G = sum(D[r][f'G_{l}'] for l in L); b = sum(D[r][f'b_{l}'] for l in L)
    s = np.sqrt(np.diag(G)); return np.linalg.solve(G / np.outer(s, s) + lam * np.eye(len(b)), b / s) / s
B = {35: fit(100), 100: fit(35)}; Bmean = (fit(35) + fit(100)) / 2
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
p, a, c, e = model.torch_payload(torch, np, False)
def offz(X): return X - torch.diag(torch.diagonal(X))
def run(rank, strength):
    beta = dict(zip(names, B.get(rank, Bmean)))
    w = np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy')
    truth = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
    card = model.Card(torch, w, p, a, c, e); orig = card.env['_nonlinear_pair_batch']
    def wpair(mu, C, K21, k4, k22_override=None, metric='identity', remainder31=None):
        l = card.call_layer
        R0 = None
        if K31ROW and 2 <= l <= 14:
            C64_ = C.double(); v_ = torch.diagonal(C64_); s_ = v_.sqrt(); q_ = mu.double() * torch.diagonal(K21.double()) / (v_ * v_); rad_ = float(k4.double()[0] / (v_[0] * v_[0]))
            R0 = offz((rad_ * ((K31ROW[0] - 1) + K31ROW[1] * q_))[:, None] * (C64_ / (s_[:, None] * s_[None, :]))).to(torch.float32)
        out = orig(mu, C, K21, k4, k22_override, metric, remainder31=R0) if R0 is not None else orig(mu, C, K21, k4, k22_override, metric)
        if strength == 0 or not (2 <= l <= 14): return out
        mu64, C64, K = mu.double(), C.double(), K21.double(); k22 = k22_override.double(); k4v = k4.double()
        v = torch.diagonal(C64); s = v.sqrt(); k3 = torch.diagonal(K); rad = float(k4v[0] / (v[0] * v[0]))
        cov0 = out[1].double(); hh = out[0].double() / out[0].double().norm()
        def proj(Dm):
            Dh = Dm @ hh; hD = hh @ Dm; hDh = hh @ Dh
            return offz(Dm - torch.outer(hh, hD) - torch.outer(Dh, hh) + hDh * torch.outer(hh, hh))
        muC = C64 * mu64[:, None]
        d21 = (beta['21_card'] * offz(K) + beta['21_cardT'] * offz(K.T) + beta['21_vmu'] * offz(torch.outer(v, mu64)) + beta['21_Cmu'] * offz(muC)
               + beta['21_k3reg'] * offz(k3[:, None] * C64 / v[:, None]) + beta['21_k3regT'] * offz(k3[None, :] * C64 / v[None, :]))
        d22 = (beta['22_card'] * offz(k22) + beta['22_vv'] * offz(rad * torch.outer(v, v)) + beta['22_CC'] * offz(rad * C64 * C64)
               + beta['22_gs'] * offz((rad / 2) * (mu64[:, None] * K.T + mu64[None, :] * K)) + beta['22_k3k3'] * offz(torch.outer(k3, k3) * C64 / torch.outer(v, v))
               + beta['22_mumuC'] * offz(rad * torch.outer(mu64, mu64) * C64))
        d31 = (beta['31_card'] * offz(rad * v[:, None] * C64) + beta['31_q'] * offz(rad * (mu64 * k3 / v)[:, None] * C64)
               + beta['31_gs'] * offz((rad / 4) * (3 * mu64[:, None] * K + torch.outer(k3, mu64)))
               + beta['31_P1'] * offz(3 * (k3 / v)[:, None] * (K - C64 * (k3 / v)[:, None])) + beta['31_cardT'] * offz(rad * v[None, :] * C64))
        r31 = offz(d31 / (v[:, None] * s[:, None] * s[None, :])).to(torch.float32)
        if R0 is not None: r31 = r31 + R0
        covp = orig(mu, C, (K + d21).to(K21.dtype), k4, (k22 + d22).to(k22_override.dtype), metric, remainder31=r31)[1].double()
        g = torch.erfc(-(mu64 / s) / np.sqrt(2)) / 2
        Dhat = proj(covp - cov0) + beta['cov_C'] * proj(cov0) + beta['cov_ggC'] * proj(torch.outer(g, g) * C64)
        return (out[0], (cov0 + strength * Dhat).to(out[1].dtype)) + tuple(out[2:])
    card.env['_nonlinear_pair_batch'] = wpair
    with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
    return float(((pred[15] - truth) ** 2).mean())
for rk in ranks:
    base = run(rk, 0.0)
    for s_ in S:
        t0 = time.time(); m = run(rk, s_)
        print(json.dumps(dict(rank=rk, strength=s_, mse=m, ratio=round(m / base, 4), wall=round(time.time() - t0, 1))), flush=True)
