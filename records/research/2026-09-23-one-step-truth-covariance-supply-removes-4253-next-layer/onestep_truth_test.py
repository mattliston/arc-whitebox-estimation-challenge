#!/usr/bin/env python3
"""TAG fable-mc (MC-2): ONE-STEP decomposition of the next-layer variance error. At layer l, run the card's pair map (torch translation of
_nonlinear_pair_batch, with an optional explicit K31 input) on (a) the card's own captured state, (b) ALL-truth inputs (MC mean, full
covariance, K21, K22, K31, per-unit k3/k4), (c) truth second order + card planes, (d) card second order + truth planes; report the
post-covariance error vs the MC truth (off-diagonal rel, diagonal rel), the post-mean error, and v_{l+1} = diag(W^T Cpost W) vs the
32M truth. Separates closure-intrinsic error (b) from inherited state error (a vs b).
usage: onestep_truth_test.py <rank> <layers e.g. 4,7,10,13> <mc2.npz>"""
import sys, json, ast
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
from common import *
import model_bank_research as model
torch.set_num_threads(3)
rank = int(sys.argv[1]); layers = [int(x) for x in sys.argv[2].split(',')]; mc2 = np.load(sys.argv[3])
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
tr32 = np.load(P2 / "public_g19_attribution_v1" / f"mps_truth_rank{rank:04d}_P32M.npz")
planes = np.load(P2 / "public_g19_attribution_v1" / f"mps_planes_rank{rank:04d}.npz")
p, a, c, e = model.torch_payload(torch, np, False)
card = model.Card(torch, W, p, a, c, e)
# pair map variant with explicit K31: patch 'T31 = t4[:, None] * rho if k31 else None' to use a module global when set
text = model.SOURCE.read_text(); tree = ast.parse(text)
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_nonlinear_pair_batch'][-1]
src = ast.get_source_segment(text, node)
old = "    T31 = t4[:, None] * rho if k31 else None\n"
assert src.count(old) == 1
src = src.replace(old, "    T31 = (t4[:, None] * rho) if _K31_OVERRIDE[0] is None else (_K31_OVERRIDE[0] / (v[:, None] * s[:, None] * s[None, :]))\n    T31 = T31 - diag(diag(T31)) if _K31_OVERRIDE[0] is not None else T31\n")
card.env['_K31_OVERRIDE'] = [None]
card.env['diag'] = torch.diag
t_ = model.Translate().visit(ast.parse(src)); ast.fix_missing_locations(t_); exec(compile(t_, 'pair31', 'exec'), card.env)
pair = card.env['_nonlinear_pair_batch']
# capture the card's own state at each layer (baseline run)
cap = {}
orig = pair
def capt(mu, C, K21, k4, k22_override=None, metric='identity'):
    cap[card.call_layer] = (mu.detach().clone(), C.detach().clone(), K21.detach().clone(), k4.detach().clone(), k22_override.detach().clone())
    return orig(mu, C, K21, k4, k22_override, metric)
card.env['_nonlinear_pair_batch'] = capt
with torch.no_grad(): card.run(False)
card.env['_nonlinear_pair_batch'] = orig
T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float64)
off = ~np.eye(1024, dtype=bool)
for l in layers:
    mu_c, C_c, K21_c, k4_c, K22_c = cap[l]
    mu_t = T(mc2[f'mean_pre_{l}']); C_t = T(mc2[f'cov_pre_{l}'])
    K21_t = T(planes[f'K21_{l}']); K22_t = T(planes[f'K22_{l}']); K31_t = T(planes[f'K31_{l}'])
    k3_t = T(tr32['mc_pre_k3'][l]); k4_t = T(tr32['mc_pre_k4'][l])
    K21_t = torch.diagonal_scatter(K21_t, k3_t); K22_t = torch.diagonal_scatter(K22_t, k4_t)
    Cp_true = mc2[f'cov_post_{l}']; mp_true = mc2[f'mean_post_{l}']
    noiseAB = np.linalg.norm((mc2[f'cov_post_A_{l}'] - mc2[f'cov_post_B_{l}'])[off]) / 2 / np.linalg.norm(Cp_true[off])
    v_next_true = tr32['mc_pre_v'][l + 1].astype(np.float64); Wn = W[l + 1].astype(np.float64)
    def run(mu, C, K21, k4, K22, K31=None):
        card.env['_K31_OVERRIDE'][0] = K31
        with torch.no_grad(): m, cov, post21, radial, aux = pair(mu, C, K21, k4, K22, 'covariance')
        card.env['_K31_OVERRIDE'][0] = None
        cov = cov.numpy(); m = m.numpy()
        vn = np.einsum('ij,ik,kj->j', Wn, cov, Wn)
        return dict(cov_off_rel=float(np.linalg.norm((cov - Cp_true)[off]) / np.linalg.norm(Cp_true[off])),
                    cov_diag_rel=float(np.linalg.norm(np.diag(cov) - np.diag(Cp_true)) / np.linalg.norm(np.diag(Cp_true))),
                    mean_rms=float(np.sqrt(np.mean((m - mp_true) ** 2))),
                    vnext_rel=float(np.sqrt(np.mean(((vn - v_next_true) / v_next_true) ** 2))),
                    vnext_bias=float(np.mean((vn - v_next_true) / v_next_true)))
    rows = {'card_state': run(mu_c, C_c, K21_c, k4_c, K22_c),
            'truth_all': run(mu_t, C_t, K21_t, k4_t, K22_t, K31_t),
            'truth_all_cardK31': run(mu_t, C_t, K21_t, k4_t, K22_t),
            'truth_2nd_card_planes': run(mu_t, C_t, K21_c, k4_c, K22_c),
            'card_2nd_truth_planes': run(mu_c, C_c, K21_t, k4_t, K22_t, K31_t)}
    print(json.dumps(dict(layer=l, mc_noise_cov_off=round(noiseAB, 5), **{k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in rows.items()})), flush=True)
