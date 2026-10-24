#!/usr/bin/env python3
"""TAG fable-analyst (AN-22): is the plane-driven covariance correction PREDICTABLE? Torch research card (unmodified trajectory), CPU.
At every pair-map call l = 2..14: cov_card = the pair map's covariance output with the card's planes; cov_truth = the same with the 8M MC truth
planes (public_g19 mps_planes: K21 off-diagonal truth + card diagonal k3; K22 off-diagonal truth; K31 truth via remainder31), exactly the
lead's planes_covonly_oracle construction. Target dP = offdiag(P (cov_truth - cov_card) P), P = I - hh^T/|h|^2, h = the pair map's host.
Features = the pair map's EXACT covariance read (it is linear in each plane) of candidate planes built from the card's own state, projected
the same way: K21 slot {card K21, card K21^T, Sym(C,mu) split into v_a mu_c and C_ac mu_a, k3 regression forms k3_a C_ac/v_a and
k3_c C_ac/v_c}; K22 slot {card K22, radial v_a v_c, radial C_ac^2, gain-skew (radial/2)(mu_a K21_ca + mu_c K21_ac), k3_a k3_c C_ac/(v_a v_c),
radial mu_a mu_c C_ac}; K31 slot {card t4*rho form radial v_a C_ac, q-row radial mu_a k3_a C_ac/v_a, gain-skew (radial/4)(3 mu_a K21_ac +
mu_c k3_a), P1 = 3(k3_a/v_a)(K21_ac - C_ac k3_a/v_a), transposed card form radial v_c C_ac}; plus direct covariance features {card post C,
g_a g_c Cpre_ac}. Saves per layer the sufficient statistics (X^T X, X^T y, y^T y) and, at L4/7/10/13, dP itself (float32).
--k31row=c0,c1: the card's pair map carries the physicist's K31 row factor T31 = radial (c0 + c1 q) rho at L2-14 (package baseline);
the truth call replaces T31 entirely, candidate K31 reads are added on top. Output dP_data_r<rank>[_k31row].npz.
usage: dP_data.py <rank> <threads> [--k31row=0.85,0.5]"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
from common import *
import model_bank_research as model
HERE = Path(__file__).resolve().parent   # common's star import rebinds HERE to fable_refit_v1
rank = int(sys.argv[1]); torch.set_num_threads(int(sys.argv[2]))
K31ROW = [tuple(float(x) for x in a_.split('=')[1].split(',')) for a_ in sys.argv if a_.startswith('--k31row=')]
K31ROW = K31ROW[0] if K31ROW else None
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
planes = np.load(P2 / 'public_g19_attribution_v1' / f'mps_planes_rank{rank:04d}.npz')
p, a, c, e = model.torch_payload(torch, np, False)
w = np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy')
card = model.Card(torch, w, p, a, c, e); orig = card.env['_nonlinear_pair_batch']
NAMES = ['21_card', '21_cardT', '21_vmu', '21_Cmu', '21_k3reg', '21_k3regT',
         '22_card', '22_vv', '22_CC', '22_gs', '22_k3k3', '22_mumuC',
         '31_card', '31_q', '31_gs', '31_P1', '31_cardT', 'cov_C', 'cov_ggC']
stats = {}; keep = {}
def offz(X): return X - torch.diag(torch.diagonal(X))
def pair(l, mu, C, K21, k4, k22, r31=None, base31=True):
    if base31 and K31ROW and 2 <= l <= 14:
        R0 = rowr31(mu, C, K21, k4); r31 = R0 if r31 is None else R0 + r31
    return orig(mu, C, K21, k4, k22, 'covariance', remainder31=r31) if r31 is not None else orig(mu, C, K21, k4, k22, 'covariance')
def rowr31(mu, C, K21, k4):
    C64 = C.double(); v = torch.diagonal(C64); s = v.sqrt(); q = mu.double() * torch.diagonal(K21.double()) / (v * v); rad = float(k4.double()[0] / (v[0] * v[0]))
    return offz((rad * ((K31ROW[0] - 1) + K31ROW[1] * q))[:, None] * (C64 / (s[:, None] * s[None, :]))).to(torch.float32)
def wpair(mu, C, K21, k4, k22_override=None, metric='identity', remainder31=None):
    l = card.call_layer
    R0 = rowr31(mu, C, K21, k4) if (K31ROW and 2 <= l <= 14) else None
    out = orig(mu, C, K21, k4, k22_override, metric, remainder31=R0) if R0 is not None else orig(mu, C, K21, k4, k22_override, metric)
    if not (2 <= l <= 14) or f'K21_{l}' not in planes.files:
        return out
    t0 = time.time()
    with torch.no_grad():
        mu64, C64, K = mu.double(), C.double(), K21.double(); k22 = k22_override.double(); k4v = k4.double()
        v = torch.diagonal(C64); s = v.sqrt(); k3 = torch.diagonal(K); rad = float(k4v[0] / (v[0] * v[0]))
        cov0 = out[1].double(); h = out[0].double(); hh = h / h.norm()
        def proj(D):
            Dh = D @ hh; hD = hh @ D; hDh = hh @ Dh
            return offz(D - torch.outer(hh, hD) - torch.outer(Dh, hh) + hDh * torch.outer(hh, hh))
        def r31_of(X): return offz(X / (v[:, None] * s[:, None] * s[None, :])).to(torch.float32)
        # truth planes (lead's construction)
        T21 = torch.as_tensor(planes[f'K21_{l}'], dtype=torch.float64); K21t = offz(T21) + torch.diag(k3)
        T22 = torch.as_tensor(planes[f'K22_{l}'], dtype=torch.float64); K22t = offz(T22) + torch.diag(torch.diagonal(k22))
        K31t = torch.as_tensor(planes[f'K31_{l}'], dtype=torch.float64)
        r31t = r31_of(K31t - k4v[:, None] * C64 / v[:, None])
        covt = pair(l, mu, C, K21t.to(K21.dtype), k4, K22t.to(k22_override.dtype), r31t, base31=False)[1].double()
        y = proj(covt - cov0)
        # candidate planes from the card state
        muC = C64 * mu64[:, None]
        X21 = {'21_card': offz(K), '21_cardT': offz(K.T), '21_vmu': offz(torch.outer(v, mu64)), '21_Cmu': offz(muC),
               '21_k3reg': offz(k3[:, None] * C64 / v[:, None]), '21_k3regT': offz(k3[None, :] * C64 / v[None, :])}
        X22 = {'22_card': offz(k22), '22_vv': offz(rad * torch.outer(v, v)), '22_CC': offz(rad * C64 * C64),
               '22_gs': offz((rad / 2) * (mu64[:, None] * K.T + mu64[None, :] * K)), '22_k3k3': offz(torch.outer(k3, k3) * C64 / torch.outer(v, v)),
               '22_mumuC': offz(rad * torch.outer(mu64, mu64) * C64)}
        X31 = {'31_card': offz(rad * v[:, None] * C64), '31_q': offz(rad * (mu64 * k3 / v)[:, None] * C64),
               '31_gs': offz((rad / 4) * (3 * mu64[:, None] * K + torch.outer(k3, mu64))),
               '31_P1': offz(3 * (k3 / v)[:, None] * (K - C64 * (k3 / v)[:, None])), '31_cardT': offz(rad * v[None, :] * C64)}
        feats = {}
        for k_, X in X21.items(): feats[k_] = proj(pair(l, mu, C, (K + X).to(K21.dtype), k4, k22_override)[1].double() - cov0)
        for k_, X in X22.items(): feats[k_] = proj(pair(l, mu, C, K21, k4, (k22 + X).to(k22_override.dtype))[1].double() - cov0)
        for k_, X in X31.items(): feats[k_] = proj(pair(l, mu, C, K21, k4, k22_override, r31_of(X))[1].double() - cov0)
        g = torch.erfc(-(mu64 / s) / np.sqrt(2)) / 2
        feats['cov_C'] = proj(cov0); feats['cov_ggC'] = proj(torch.outer(g, g) * C64)
        mask = ~torch.eye(v.shape[0], dtype=torch.bool)
        Xm = torch.stack([feats[k_][mask] for k_ in NAMES], 1); ym = y[mask]
        stats[l] = dict(G=(Xm.T @ Xm).numpy(), b=(Xm.T @ ym).numpy(), yy=float(ym @ ym),
                        y_norm=float(ym.norm()), cov_off_norm=float(cov0[mask].norm()), feat_norms=Xm.norm(dim=0).numpy())
        if l in (4, 7, 10, 13):
            keep[f'dP_{l}'] = y.to(torch.float32).numpy(); keep[f'h_{l}'] = h.numpy()
        print(json.dumps(dict(rank=rank, layer=l, dP_rel_to_cov=round(float(ym.norm() / cov0[mask].norm()), 6), wall=round(time.time() - t0, 1))), flush=True)
    return out
card.env['_nonlinear_pair_batch'] = wpair
with torch.no_grad(): pred = card.run(False)
out = dict(names=np.array(NAMES))
for l, d in stats.items():
    for k_, v_ in d.items(): out[f'{k_}_{l}'] = np.asarray(v_)
out.update(keep)
np.savez(HERE / (f'dP_data_r{rank}' + ('_k31row' if K31ROW else '') + '.npz'), **out); print('saved', flush=True)
