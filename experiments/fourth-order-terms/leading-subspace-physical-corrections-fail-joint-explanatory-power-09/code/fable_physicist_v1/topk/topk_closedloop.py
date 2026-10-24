#!/usr/bin/env python3
"""TAG fable-physicist (TK-2): closed-loop value of physical plane rules applied ONLY in the one-sided top-k covariance block.
At every pair-map call l = 2..14 (torch card, learned ON, no refit): cov0 = card pair covariance, P = I - hh^T/|h|^2, U = top-k eigvecs of
P cov0 P, D = P (cov_pair(card + supply) - cov0) P U (exact pair-map response), block X = D U^T + U D^T - U sym(U^T D) U^T,
cov <- cov0 + eps X. Supplies (K31 via remainder31): P1:<c> = c * 3 (k3_a/v_a)(K21_ac - C_ac k3_a/v_a); gsk:<f> = f * (radial/4)
(3 mu_a K21_ac + mu_c k3_a); 'full' suffix = the supply's whole covariance change (no block restriction) instead of X.
usage: topk_closedloop.py <ranks> <specs e.g. P1:0.9@0.5,gsk:2@0.25,P1:0.9:full@0.5> <k> <threads>"""
import sys, json, time, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; PH = HERE.parent; P2 = PH.parent
sys.path.insert(0, str(PH)); sys.path.insert(1, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
import model_bank_audit as model
ranks = [int(x) for x in sys.argv[1].split(',')]; specs = sys.argv[2].split(','); KTOP = int(sys.argv[3]); torch.set_num_threads(int(sys.argv[4]))
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
p, a, c, e = model.torch_payload(torch, np, False)
def offz(X): return X - torch.diag(torch.diagonal(X))
for rank in ranks:
    W = np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy')
    TP = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
    base = None
    for spec in ['base'] + specs:
        card = model.Card(torch, W, p, a, c, e); orig = card.env['_nonlinear_pair_batch']
        if spec != 'base':
            sup, _, eps = spec.partition('@'); eps = float(eps); parts = sup.split(':'); kind, coef = parts[0], float(parts[1]); full = len(parts) > 2
            def wpair(mu, C, K21, k4, k22_override=None, metric='identity', remainder31=None, kind=kind, coef=coef, full=full, eps=eps, card=card, orig=orig):
                out = orig(mu, C, K21, k4, k22_override, metric); l = card.call_layer
                if not (2 <= l <= 14): return out
                mu64, C64, K = mu.double(), C.double(), K21.double(); v = torch.diagonal(C64); s = v.sqrt(); k3 = torch.diagonal(K)
                rad = float(k4.double()[0] / (v[0] * v[0])); q = k3 / v
                if kind == 'P1': X31 = coef * 3 * q[:, None] * (K - C64 * q[:, None])
                else: X31 = coef * (rad / 4) * (3 * mu64[:, None] * K + torch.outer(k3, mu64))
                r31 = offz(X31 / (v[:, None] * s[:, None] * s[None, :])).to(torch.float32)
                o1 = orig(mu, C, K21, k4, k22_override, metric, remainder31=r31)
                cov0 = out[1].double(); dcov = o1[1].double() - cov0
                if full:
                    newc = cov0 + eps * dcov
                else:
                    hh = out[0].double() / out[0].double().norm()
                    def PDP(Xm): Y = Xm - torch.outer(hh, hh @ Xm); return Y - torch.outer(Y @ hh, hh)
                    lam, V = torch.linalg.eigh(PDP(cov0)); U = V[:, -KTOP:]
                    D = PDP(dcov) @ U; S = U.T @ D; S = (S + S.T) / 2
                    newc = cov0 + eps * (D @ U.T + U @ D.T - U @ S @ U.T)
                return (out[0], newc.to(out[1].dtype)) + tuple(out[2:])
            card.env['_nonlinear_pair_batch'] = wpair
        t0 = time.time()
        with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
        mse = float(((pred[15] - TP) ** 2).mean()); base = mse if spec == 'base' else base
        print(json.dumps(dict(rank=rank, spec=spec, k=KTOP, mse=mse, rel=round(mse / base - 1, 5), wall=round(time.time() - t0, 1))), flush=True)
