#!/usr/bin/env python3
"""TAG fable-physicist (CR-7): which part of the card's post DISTINCT content carries the distinct error? By linearity, with no new
dense run: D_card(L) = D_tr (the carried bank, transported) + D_new (the newborn CP columns) + D_b (the b channel's distinct part), each
the push-plane at L minus the push of its own post plane at L-1 (Pi = _flms_star). Regress e_D = D_true - D_card on the three parts
(kink block at L, off-diagonal). Check: D_tr + D_new + D_b must reproduce decompose.py's D_card. usage: distinct_regress.py"""
import json, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[2]; G = P2 / 'public_g19_attribution_v1'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
WTS = np.load(STORE / Path(man[35]['path']).parent / 'weights.npy', mmap_mode='r')
cap, tgt, c0 = np.load(HERE / 'capture_r35.npz'), np.load(HERE / 'capture_r35_target.npz'), np.load(HERE / 'loc_C0_all.npz')
PRE = np.load(G / 'mps_planes_rank0035.npz')
POST = {}
for f in ('mps_planes_rank0035_post.npz', 'mps_planes_rank0035_post2.npz', 'mps_planes_rank0035_post3.npz', 'mps_planes_rank0035_post_p0.npz'):
    z = np.load(G / f)
    for k in z.files:
        if k.startswith('K21_'): POST[int(k[4:])] = z[k].astype(np.float64)
T = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
def Wd(L): return T(np.asarray(WTS[L], dtype=np.float32))
def Pi(W, P):
    A = 3 * P.T - 2 * torch.diag(torch.diagonal(P)); wt = W.T; pushed = wt @ A
    res = (2 * ((pushed * wt) @ W) + (wt * wt) @ pushed.T) / 3; res.diagonal().copy_((pushed * wt * wt).sum(1)); return res
def cp_plane(P, lam): return ((P * P) * lam[None, :]) @ P.T      # plane of sum_c lam_c p_c^(x)3: (i,i,j) = sum lam p_i^2 p_j
nrm = lambda x: float(torch.linalg.norm(x))
print('L  | |e_D|  | |D_tr| |D_new| |D_b| | cos(e_D, D_tr / D_new / D_b) | LS fit coefs (tr, new, b) -> |e_D| left | check |D_card - sum|')
for L in (6, 8, 10, 12, 14):
    l = L - 1; W = Wd(L)
    resp = cap[f'resp_{L}'] > .5; alive = cap[f'alive_{L}'] > .5; blk = np.ix_(resp, alive); off = torch.from_numpy(~np.eye(1024, dtype=bool)[blk])
    B = lambda M: M[torch.from_numpy(np.where(resp)[0])][:, torch.from_numpy(np.where(alive)[0])] * off
    Kt, pre = T(PRE[f'K21_{L}']), T(c0[f'pre_{L}'])
    live = T(np.outer(cap[f'alive_{l}'], cap[f'alive_{l}'])); cpm = T(np.outer(cap[f'resp_{l}'], cap[f'alive_{l}'])); cpm.fill_diagonal_(1.0)
    tr = T(c0[f'tr_{l}']); Pn, lam = T(cap[f'P_{l}']), T(cap[f'lam_{l}'])
    born = cp_plane(Pn, lam) * live
    rest = T(tgt[f'target_{l}'] - tgt[f'trace_{l}']) - live * (tr * cpm + born)
    sp = Pi(W, rest)                                            # C0's own star push (the star is plane-supported with values rest)
    bterm = T(cap[f'bterm_{L}'])
    Q = W.T @ Pn; new_full = cp_plane(Q, lam)                    # push-plane of the newborn columns
    cpd = pre - sp - bterm                                       # plane(push(T_post)) of the dense bank
    D_new = new_full - Pi(W, cp_plane(Pn, lam))
    D_tr = (cpd - new_full) - Pi(W, tr)
    D_b = bterm - Pi(W, T(tgt[f'trace_{l}']))
    Pc = T(tgt[f'target_{l}']) + live * (1 - cpm) * tr
    D_card = pre - Pi(W, Pc); D_true = Kt - Pi(W, T(POST[l]))
    eD = B(D_true - D_card); comps = [B(D_tr), B(D_new), B(D_b)]
    X = torch.stack([c.flatten() for c in comps], 1); coef = torch.linalg.lstsq(X, eD.flatten()[:, None]).solution.flatten()
    left = nrm(eD.flatten() - X @ coef); nt = nrm(B(Kt))
    cos = [float((eD * c).sum() / (nrm(eD) * nrm(c))) for c in comps]
    chk = nrm(B(D_card - (D_tr + D_new + D_b))) / nt
    print(f'{L:<2} | {nrm(eD) / nt:.4f} | {nrm(comps[0]) / nt:.3f} {nrm(comps[1]) / nt:.3f} {nrm(comps[2]) / nt:.3f} | '
          f'{cos[0]:+.3f} / {cos[1]:+.3f} / {cos[2]:+.3f} | {np.round(coef.numpy(), 3)} -> {left / nt:.4f} | {chk:.1e}')
