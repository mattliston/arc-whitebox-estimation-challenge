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
    def runcov(mu, C, K21, k4, K22, K31=None):
        card.env['_K31_OVERRIDE'][0] = K31
        with torch.no_grad(): m, cov, post21, radial, aux = pair(mu, C, K21, k4, K22, 'covariance')
        card.env['_K31_OVERRIDE'][0] = None
        return cov.numpy(), m.numpy(), {k: (v.numpy() if hasattr(v, 'numpy') else v) for k, v in aux.items()}
    cov_T, m_T, aux_T = runcov(mu_t, C_t, K21_t, k4_t, K22_t, K31_t)
    offm = torch.tensor(off)
    cases = {'planes_only (true mu,C)': runcov(mu_t, C_t, K21_c, k4_c, K22_c, None)[0],
             'K21off_only': runcov(mu_t, C_t, torch.where(offm, K21_c, K21_t), k4_t, K22_t, K31_t)[0],
             'card_state (full one-step)': runcov(mu_c, C_c, K21_c, k4_c, K22_c, None)[0]}
    g = aux_T['gate']; sg = aux_T['sigma']; pdf = aux_T['pdf']; mu_np = mu_t.numpy()
    vecs = {'m': m_T, 'sigma*phi': sg * pdf, 'g*mu': g * mu_np, 'sd_post': np.sqrt(np.maximum(np.diag(cov_T), 1e-12)), 'g': g, 'ones': np.ones(1024)}
    Wn = W[l + 1].astype(np.float64)
    for name, covc in cases.items():
        D = covc - cov_T; D[~off] = 0
        U_, S_, Vt_ = np.linalg.svd(D); en = S_ ** 2 / (S_ ** 2).sum()
        dv = np.einsum('ij,ik,kj->j', Wn, D, Wn)
        # v-impact captured by rank-k truncations
        def vimp(k):
            Dk = (U_[:, :k] * S_[:k]) @ Vt_[:k]; dvk = np.einsum('ij,ik,kj->j', Wn, Dk, Wn); return 1 - np.sum((dv - dvk) ** 2) / np.sum(dv ** 2)
        al = {vn: round(float(abs(U_[:, 0] @ vv) / np.linalg.norm(vv)), 3) for vn, vv in vecs.items()}
        # regression of D on symmetric rank-2 forms
        forms = {'sp.gmu': np.outer(vecs['sigma*phi'], vecs['g*mu']) + np.outer(vecs['g*mu'], vecs['sigma*phi']),
                 'm.m': np.outer(m_T, m_T), 'sp.sp': np.outer(vecs['sigma*phi'], vecs['sigma*phi']), 'gmu.gmu': np.outer(vecs['g*mu'], vecs['g*mu'])}
        y = D[off]; X = np.stack([f[off] for f in forms.values()], 1); b, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = 1 - np.sum((y - X @ b) ** 2) / np.sum(y ** 2)
        dvr = np.einsum('ij,ik,kj->j', Wn, sum(bb * f for bb, f in zip(b, forms.values())) * off, Wn)
        vr2 = 1 - np.sum((dv - dvr) ** 2) / np.sum(dv ** 2)
        print(f"L{l:02d} {name:28s} |D|/|C| {np.linalg.norm(D)/np.linalg.norm(cov_T[off]):.4f}  energy top1 {en[0]:.3f} top8 {en[:8].sum():.3f}  v-impact rank1 {vimp(1):.3f} rank8 {vimp(8):.3f}  sv1~{al}  rank2-forms R2 {r2:.3f} (v-impact {vr2:.3f}) coefs {np.round(b,5)}", flush=True)
