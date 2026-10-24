#!/usr/bin/env python3
"""TAG fable-physicist (CR-25, lead GO 09-25 01:37, H2 falsifier of leadergap section O): the second-order born diagram our card lacks,
T3_abc = f''_a f''_b g_c C_ab K3_abc (+ the two other bond placements): the carried FULL third cumulant dressed by one covariance bond
through the ReLU curvature. (The plane diagrams f'''_a g g C K21 and f'' f'' g C K21 are the card's j1 / j0 legs; the (2,1,1) fourth-order
feed exists only isotropically, via b and the birth weight radial/3.) K3 = the card's CP bank at pre-activation l (pp = W_l^T p_{l-1},
lam), f'' = d = pdf/sigma, g = gate, C = Cpre at l (capture_r35.npz), all from the B2 ep1 card on r35 (capture_bank.py).
Screen exactly as leg_screen.py: the candidate's DISTINCT push to the pre K21 plane at L = l+1 (full push minus the push of its own
(2,1)-plane pattern, Pi), on the kink block, against the card's distinct-content error e_D(L). Coefficient 1 = the physical diagram.
PASS (lead): cosine >= .3 with a one-layer cut of e_D >= 5%.  A tiny brute-force certificate of the algebra runs first.
usage: t3_screen.py cert | t3_screen.py screen"""
import sys, os, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[2]; G = P2 / 'public_g19_attribution_v1'
import numpy as np, torch
torch.set_grad_enabled(False)
def Pi(W, P):      # push of a (2,1)-plane-pattern post tensor (entries (a,a,b) = P_ab, diag P_aa) to the pre (i,i,j) plane (leg_screen.py)
    A = 3 * P.T - 2 * torch.diag(torch.diagonal(P)); wt = W.T; pushed = wt @ A
    res = (2 * ((pushed * wt) @ W) + (wt * wt) @ pushed.T) / 3; res.diagonal().copy_((pushed * wt * wt).sum(1)); return res
def t3_push(W, C, d, g, pp, lam, batch=32, dev='cpu'):
    """M_ij = sum_abc W_ai W_bi W_cj T3_abc for T3 = sum_r lam_r [d_a d_b g_c C_ab + d_a g_b d_c C_ac + g_a d_b d_c C_bc] p_ra p_rb p_rc"""
    n = W.shape[0]; Wd, Cd = W.to(dev), C.to(dev); M = torch.zeros((n, n), dtype=W.dtype, device=dev)
    U = (W.T @ (g[:, None] * pp)).to(dev)                                   # u'_r = W^T (g o p_r), columns
    for s in range(0, pp.shape[1], batch):
        P_ = (d[:, None] * pp[:, s:s + batch]).to(dev)                      # (n, B): d o p_r
        A = Wd[None, :, :] * P_.T[:, :, None]                               # (B, n, n): A_r[a, i] = W_ai d_a p_ra
        Y = A.transpose(1, 2) @ (Cd[None] @ A)                              # (B, n, n): Y_r = A_r^T C A_r
        lb = lam[s:s + batch].to(dev); ub = U[:, s:s + batch]
        M += (torch.diagonal(Y, dim1=1, dim2=2) * lb[:, None]).T @ ub.T     # sum_r lam_r diag(Y_r)_i u'_rj
        M += 2 * ((lb[:, None] * ub.T)[:, :, None] * Y).sum(0)             # sum_r lam_r 2 u'_ri Y_r[i, j]
    return M.cpu()
def t3_plane(C, d, g, pp, lam):
    """post (2,1)-plane of T3: P3_ab = T3_aab, P3_aa = T3_aaa"""
    v = torch.diagonal(C); S2 = ((pp * pp) * lam[None, :]) @ pp.T; S2g = ((pp * pp) * lam[None, :]) @ (g[:, None] * pp).T
    P3 = (d * d * v)[:, None] * S2g + 2 * (d * g)[:, None] * d[None, :] * C * S2
    P3.diagonal().copy_(3 * d * d * g * v * torch.diagonal(S2)); return P3
if sys.argv[1] == 'cert':
    torch.manual_seed(0); n, R = 6, 3; f = torch.float64
    W = torch.randn(n, n, dtype=f); X = torch.randn(n, n, dtype=f); C = X @ X.T / n + torch.eye(n, dtype=f)
    d, g, lam = torch.rand(n, dtype=f) + .2, torch.rand(n, dtype=f), torch.randn(R, dtype=f); pp = torch.randn(n, R, dtype=f)
    T = torch.zeros(n, n, n, dtype=f)
    for r in range(R):
        p = pp[:, r]
        T += lam[r] * (torch.einsum('a,b,c,ab->abc', d * p, d * p, g * p, C) + torch.einsum('a,b,c,ac->abc', d * p, g * p, d * p, C)
                       + torch.einsum('a,b,c,bc->abc', g * p, d * p, d * p, C))
    Mbf = torch.einsum('ai,bi,cj,abc->ij', W, W, W, T); Pbf = torch.einsum('aab->ab', T)
    mask = torch.ones(n, n, n, dtype=f)
    for a in range(n):
        for b in range(n):
            mask[a, a, b] = mask[a, b, a] = mask[b, a, a] = 0.0
    Dbf = torch.einsum('ai,bi,cj,abc->ij', W, W, W, T * mask)                  # push of the distinct-index entries only
    M = t3_push(W, C, d, g, pp, lam, batch=2); P3 = t3_plane(C, d, g, pp, lam); D = M - Pi(W, P3)
    Tp = torch.zeros(n, n, n, dtype=f)                                         # Pi convention check: a pure (2,1)-pattern tensor
    for a in range(n):
        for b in range(n):
            if a == b: Tp[a, a, a] = P3[a, a]
            else: Tp[a, a, b] = Tp[a, b, a] = Tp[b, a, a] = P3[a, b]
    r = dict(push=float((M - Mbf).abs().max() / Mbf.abs().max()), plane=float((P3 - Pbf).abs().max() / Pbf.abs().max()),
             Pi=float((Pi(W, P3) - torch.einsum('ai,bi,cj,abc->ij', W, W, W, Tp)).abs().max() / Mbf.abs().max()),
             distinct=float((D - Dbf).abs().max() / Dbf.abs().max()))
    print(json.dumps(dict(certificate=r, PASS=all(x < 1e-10 for x in r.values())))); sys.exit(0)
# ---------------- screen (B2 ep1 card, r35) ----------------
sys.path.insert(0, str(P2 / 'fable_engineer_v1')); import mps_lock
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
WTS = np.load(STORE / Path(man[35]['path']).parent / 'weights.npy', mmap_mode='r')
cap, bank = np.load(HERE / 'capture_r35.npz'), np.load(HERE / 'capture_r35_bank.npz')
tgt, c0 = np.load(HERE / 'capture_r35_target.npz'), np.load(HERE / 'loc_C0_all.npz')
PRE = np.load(G / 'mps_planes_rank0035.npz'); POST = {}
for fn in ('mps_planes_rank0035_post.npz', 'mps_planes_rank0035_post2.npz', 'mps_planes_rank0035_post3.npz', 'mps_planes_rank0035_post_p0.npz'):
    z = np.load(G / fn)
    for k in z.files:
        if k.startswith('K21_'): POST[int(k[4:])] = z[k].astype(np.float64)
T64 = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
nrm = lambda x: float(torch.linalg.norm(x))
LEG = json.loads((HERE / 'leg_screen.json').read_text()) if (HERE / 'leg_screen.json').exists() else {}
out = {}; t0 = time.time()
with mps_lock.hold(f't3_screen r35 (pid {os.getpid()})'):
    for L in (6, 8, 10, 12, 14):
        l = L - 1; W = T64(np.asarray(WTS[L], dtype=np.float32)); Wl = T64(np.asarray(WTS[l], dtype=np.float32))
        resp = cap[f'resp_{L}'] > .5; alive = cap[f'alive_{L}'] > .5; ri, ci = torch.from_numpy(np.where(resp)[0]), torch.from_numpy(np.where(alive)[0])
        off = torch.from_numpy(~np.eye(1024, dtype=bool)[np.ix_(resp, alive)]); B = lambda M: M[ri][:, ci] * off
        live = T64(np.outer(cap[f'alive_{l}'], cap[f'alive_{l}'])); cpm = T64(np.outer(cap[f'resp_{l}'], cap[f'alive_{l}'])); cpm.fill_diagonal_(1.0)
        Pc = T64(tgt[f'target_{l}']) + live * (1 - cpm) * T64(c0[f'tr_{l}'])
        eD = B((T64(PRE[f'K21_{L}']) - Pi(W, T64(POST[l]))) - (T64(c0[f'pre_{L}']) - Pi(W, Pc))); nt = nrm(B(T64(PRE[f'K21_{L}'])))
        C, g, d = T64(cap[f'C_{l}']), T64(cap[f'g_{l}']), T64(cap[f'd_{l}'])
        al = T64(cap[f'alive_{l}']); g, d = g * al, d * al
        pp = Wl.T @ T64(bank[f'p_{l - 1}']); lam = T64(bank[f'lam_{l - 1}'])        # the card's pre-activation CP bank at layer l
        M = t3_push(W.float(), C.float(), d.float(), g.float(), pp.float(), lam.float(), batch=32, dev='mps').double()
        P3 = t3_plane(C, d, g, pp, lam); D3 = M - Pi(W, P3)
        dX = B(D3); cos = float((dX * eD).sum() / max(nrm(dX) * nrm(eD), 1e-300)); x = float((dX * eD).sum() / max(nrm(dX) ** 2, 1e-300))
        j1 = LEG.get(str(L), {}).get('j1_phys', {})
        res = dict(ncols=int(pp.shape[1]), eD=round(nrm(eD) / nt, 5), size=round(nrm(dX) / nt, 5), cos=round(cos, 3), best_scale=round(x, 3),
                   eD_at_1=round(nrm(eD - dX) / nt, 5), eD_at_best=round(math.sqrt(max(nrm(eD) ** 2 - (cos * nrm(eD)) ** 2, 0)) / nt, 5),
                   cut_at_1=round(1 - nrm(eD - dX) / nrm(eD), 4), plane_part=round(nrm(B(Pi(W, P3))) / nt, 5), full_push=round(nrm(B(M)) / nt, 5),
                   j1_phys_for_scale=j1, wall=round(time.time() - t0))
        out[L] = res; print(json.dumps({L: res}), flush=True)
dst = HERE / 't3_screen.json'; assert not dst.exists(), dst
dst.write_text(json.dumps(out, indent=1))
