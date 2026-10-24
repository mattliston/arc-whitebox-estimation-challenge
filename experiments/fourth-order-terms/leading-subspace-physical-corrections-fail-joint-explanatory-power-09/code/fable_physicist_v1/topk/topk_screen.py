#!/usr/bin/env python3
"""TAG fable-physicist (TK-1): oracle-grade screen of PHYSICAL third/fourth-order rules in the one-sided top-k covariance block.
At every pair-map call l = 2..14 of the unmodified torch research card (learned ON): cov0 = card pair covariance, P = I - hh^T/|h|^2
(h = pair host), U = top-k eigvecs of P cov0 P (k = 30), target R = P (cov_pair(8M truth K21 off + card k3, truth K22 off, truth K31)
- cov0) P U (the analyst's Rp_data construction). Candidates (each a plane supply; D = P (cov_pair(card + supply) - cov0) P U):
  tree31  : tree-level Gaussian-born K31 (star + path) of the ReLU at l-1, pushed by W_l (card state at l-1)
  P1      : 0.45 * 3 (k3_a/v_a)(K21_ac - C_ac k3_a/v_a) as a K31 supply (card K21)
  gsk     : gain-on-skewed-base K31 (radial/4)(3 mu_a K21_ac + mu_c k3_a)
  gskK22  : its K22 partner (radial/2)(mu_a K21_ca + mu_c K21_ac)
  sym21   : kappa Sym(C, mu) off-diagonal K21 with kappa = radial/6 - bpre.mu/mu.mu (clamped >= 0)
Per candidate: correlation rho = <D, R>/(|D||R|) over the n x k block and the optimal scale beta. Noise: independent draw (L2/5/9/12).
usage: topk_screen.py <rank> <threads> [k]"""
import sys, json, time, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; PH = HERE.parent; P2 = PH.parent
sys.path.insert(0, str(PH)); sys.path.insert(1, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
import model_bank_audit as model
rank = int(sys.argv[1]); torch.set_num_threads(int(sys.argv[2])); KTOP = int(sys.argv[3]) if len(sys.argv) > 3 else 30
G19 = P2 / 'public_g19_attribution_v1'
PL = np.load(G19 / f'mps_planes_rank{rank:04d}.npz'); PN = np.load(G19 / 'mps_planes_rank0035_noise2.npz') if rank == 35 else None
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
W = np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy')
p, a, c, e = model.torch_payload(torch, np, False)
card = model.Card(torch, W, p, a, c, e); orig = card.env['_nonlinear_pair_batch']
def offz(X): return X - torch.diag(torch.diagonal(X))
prev = {}
def tree31(mu, C, Wl):
    v = torch.diagonal(C); s = v.sqrt(); al = mu / s
    pdf = torch.exp(-al * al / 2) / math.sqrt(2 * math.pi); e1 = torch.erfc(-al / math.sqrt(2)) / 2; e2 = pdf / s; e3 = -al * pdf / v
    Co = offz(C); X = Co @ (e1[:, None] * Wl); Q = Co @ (e2[:, None] * Wl * X)
    return (3 * ((e3[:, None] * Wl * X * X).T @ X) + (X * X * X).T @ (e3[:, None] * Wl) + 6 * ((e2[:, None] * Wl * Q).T @ X) + 6 * ((e2[:, None] * X * Q).T @ Wl))
res = {}
def wpair(mu, C, K21, k4, k22_override=None, metric='identity', remainder31=None):
    out = orig(mu, C, K21, k4, k22_override, metric); l = card.call_layer
    mu64, C64, K = mu.double(), C.double(), K21.double()
    if 2 <= l <= 14:
        t0 = time.time()
        with torch.no_grad():
            v = torch.diagonal(C64); s = v.sqrt(); k3 = torch.diagonal(K); k4v = k4.double(); rad = float(k4v[0] / (v[0] * v[0]))
            cov0 = out[1].double(); hh = out[0].double() / out[0].double().norm()
            def PDP(X): Y = X - torch.outer(hh, hh @ X); return Y - torch.outer(Y @ hh, hh)
            lam, V = torch.linalg.eigh(PDP(cov0)); U = V[:, -KTOP:].flip(1)
            def covd(K21m=None, K22m=None, K31add=None, PLf=None):
                K21u = K21 if K21m is None else K21m.to(K21.dtype); K22u = k22_override if K22m is None else K22m.to(k22_override.dtype)
                r31 = None
                if K31add is not None:
                    r31 = offz(K31add / (v[:, None] * s[:, None] * s[None, :])).to(torch.float32)
                o = orig(mu, C, K21u, k4, K22u, metric, remainder31=r31) if r31 is not None else orig(mu, C, K21u, k4, K22u, metric)
                return PDP(o[1].double() - cov0) @ U
            def truthD(PLf):
                K21t = offz(torch.as_tensor(PLf[f'K21_{l}'], dtype=torch.float64)) + torch.diag(k3)
                K22t = offz(torch.as_tensor(PLf[f'K22_{l}'], dtype=torch.float64)) + torch.diag(torch.diagonal(k22_override.double()))
                K31a = torch.as_tensor(PLf[f'K31_{l}'], dtype=torch.float64) - k4v[:, None] * C64 / v[:, None]
                return covd(K21t, K22t, K31a)
            R = truthD(PL)
            q = k3 / v; P1 = 0.45 * 3 * q[:, None] * (K - C64 * q[:, None])
            Kt21 = offz(torch.as_tensor(PL[f'K21_{l}'], dtype=torch.float64)) + torch.diag(k3)
            P1t = 0.45 * 3 * q[:, None] * (Kt21 - C64 * q[:, None])
            cands = {'P1': covd(K31add=P1), 'P1t': covd(K31add=P1t),
                     'gskt': covd(K31add=(rad / 4) * (3 * mu64[:, None] * Kt21 + torch.outer(k3, mu64))),
                     'gsk': covd(K31add=(rad / 4) * (3 * mu64[:, None] * K + torch.outer(k3, mu64))),
                     'gskK22': covd(K22m=k22_override.double() + offz((rad / 2) * (mu64[:, None] * K.T + mu64[None, :] * K)))}
            if l - 1 in prev:
                pm, pC = prev[l - 1]; cands['tree31'] = covd(K31add=tree31(pm, pC, torch.as_tensor(W[l], dtype=torch.float64)))
            bpre = card._au_bpre if hasattr(card, '_au_bpre') else None
            mu2 = float(mu64 @ mu64)
            rec = dict(rank=rank, layer=l, R_norm=float(R.norm()))
            for nm, D in cands.items():
                dn = float(D.norm()); ip = float((D * R).sum())
                rec[nm] = dict(rho=round(ip / (dn * float(R.norm()) + 1e-300), 4), beta=round(ip / (dn * dn + 1e-300), 3))
            def joint(names):
                X = torch.stack([cands[n_].reshape(-1) for n_ in names if n_ in cands], 1); y = R.reshape(-1)
                cf = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]; r2 = 1 - float(((y - X @ cf) ** 2).sum() / (y @ y))
                return dict(R2=round(r2, 4), coef=[round(float(x), 3) for x in cf])
            rec['joint_P1_gsk'] = joint(['P1', 'gsk']); rec['joint_P1_gsk_tree'] = joint(['P1', 'gsk', 'tree31'])
            rec['joint_truthK21_P1_gsk'] = joint(['P1t', 'gskt'])
            if PN is not None and f'K21_{l}' in PN.files:
                RB = truthD(PN); rec['noise_share'] = round(float(((R - RB).norm() / math.sqrt(2) / R.norm()) ** 2), 4)
        print(json.dumps(rec), flush=True)
    prev[l] = (mu64.detach().clone(), C64.detach().clone())
    return out
card.env['_nonlinear_pair_batch'] = wpair
with torch.no_grad(): card.run(False)
