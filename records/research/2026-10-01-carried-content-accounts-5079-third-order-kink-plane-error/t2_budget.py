#!/usr/bin/env python3
"""TAG opus-physicist T2 budget: WHICH mechanism of the post-activation third cumulant does the card get wrong? (network <rank>)
For each birth layer l (next layer L = l+1) the pre (2,1) plane at L is  K21_L = Pi(P_l) + D_l : the push of the post PLANE at l plus
the push of the post DISTINCT-index content. Truth (t2_mc.py, two independent 8M blocks) splits D_l by mechanism:
   D_true = D_lin + D_G + D_K3 + D_K4 + rest
   D_lin  gate-linear transport of the true pre-activation third cumulant, g_i g_j g_k K3(i,j,k)          (Monte Carlo, exact)
   D_G    births from a Gaussian with the true mean and covariance, all orders                             (Monte Carlo, exact)
   D_K3   third-cumulant insertions on the legs / vertex of a tree birth at physical strength, from the true pre plane:
          j0 (d_i/2) K21[i,c], j1 -(g_i/2) eta_c K21[c,i], j3 gate skew, vertex skew (k3_c/6) E f^(5)      (closed form, hub algebra)
   D_K4   fourth-cumulant feed (d_c/2) g_i g_j Q_ijcc (Monte Carlo, exact) + K4 legs: -(d_i/4) eta_c K22[i,c],
          (g_i/6)(alpha_c^2-1)/v_c K31[c,i], -(eta_i d_i/6) K31[i,c] and the vertex (k4_c/24) E f^(6)      (closed form)
Card (t2_capture.py, closed loop, learned ON):  K21_card,L = Pi(P_card,l) + D_tr + D_new + D_b + E_star
   D_tr  carried bank columns (transported by the linear gate) ; D_new newborn columns ; D_b the Sym(C x b) carrier (parts: transported
   g*bpre, source radial*d*v/6, b-fix) ; E_star = the card's star read minus the exact plane push of its star.
Matched errors (card minus truth):  transport = D_tr + D_b[transported] - D_lin ; gauss = D_new[bp + j2 legs, weight d] - D_G ;
   k3ins = D_new[j0, j1 legs] - D_K3 ; k4 = D_b[source + fix] + D_new[weight d*radial/3] - D_K4 ; rest = -(truth rest) ;
   plane = Pi(P_card - P_true) ; star = E_star.  They sum to the card's total plane error at L.
Metrics, per piece X (all noise-free: products of the two independent truth blocks):
   blk  = ||X|| on the kink block (response rows x alive columns of L, off-diagonal) / ||K21_true|| there                     [%]
   shr  = <X, e_total> / <e_total, e_total> on the kink block: the piece's signed share of the squared plane error (shares sum to 1)
   read = the same two numbers in the covariance-read metric: the next layer's per-unit variance change caused by the plane change,
          dv_b = sum_ac W'_ab W'_cb d_a g_c X[a,c] / v_b  (d, g of layer L; W' = W_{L+1}), relative rms [%] and signed share
usage: t2_budget.py <rank> [layers]  -> t2_budget_rank{rank}.json, t2_budget_rank{rank}_planes.npz"""
import sys, json, math, time
from pathlib import Path
import numpy as np
import t2_lib as L
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent
rank = int(sys.argv[1])
MCF = np.load(HERE / f't2mc_rank{rank:04d}.npz'); CAP = np.load(HERE / f't2cap_rank{rank:04d}.npz')
MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz'); PL = np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}.npz')
layers = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [int(x) for x in MCF['layers']]
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
WALL = np.load((STORE / man['path']).parent / 'weights.npy', mmap_mode='r')
DIALS = json.loads((P2 / 'fable_physicist_v1/refit6b/s6b_loopR8s_joint_3ep/dials_epoch1.json').read_text())
N = 1024; eye = np.eye(N, dtype=bool); nrm = np.linalg.norm
f8 = lambda x: np.asarray(x, dtype=np.float64)
erfc = np.vectorize(math.erfc)


def gauss(mu, v):
    s = np.sqrt(v); al = mu / s; pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = erfc(-al / math.sqrt(2)) / 2
    return s, al, pdf, g, pdf / s


def offd(X):
    Y = X.copy(); np.fill_diagonal(Y, 0.0); return Y


out = dict(TAG='opus-physicist T2 budget', rank=rank, counts=[int(x) for x in MCF['counts']], layers={}); SAVE = {}
for l in layers:
    t0 = time.time(); Lx = l + 1; W = f8(WALL[Lx]); W2 = f8(WALL[Lx + 1]) if Lx + 1 <= 15 else None
    mu = f8(MC[f'mean_pre_{l}']); C = f8(MC[f'cov_pre_{l}']); v = np.diag(C).copy(); s, al, pdf, g, d = gauss(mu, v); eta = al / s
    Cpost = f8(MC[f'cov_post_{l}']); CL = f8(MC[f'cov_pre_{Lx}']); muL = f8(MC[f'mean_pre_{Lx}']); vL = np.diag(CL).copy()
    sL, alL, pdfL, gL, dL = gauss(muL, vL)
    G = g[:, None] * W; covy = G.T @ C @ G
    resp = CAP[f'resp_{Lx}'] > .5; alive = CAP[f'alive_{Lx}'] > .5; blk = np.outer(resp, alive) & ~eye
    v2 = np.diag(f8(MC[f'cov_pre_{Lx + 1}'])) if W2 is not None else None

    def read(X):          # next-layer per-unit variance change caused by a change X of the pre plane at L (covariance read d_a g_c)
        R = offd(dL[:, None] * X * gL[None, :]); R = (R + R.T) / 2
        return (W2 * (R @ W2)).sum(0) / v2

    # ---------------- truth, per block
    TR = []
    for b in (0, 1):
        q = lambda k: f8(MCF[f'{k}_L{l}_b{b}'])
        Kz = L.k21_from_sums(q('M21_z'), q('S1_z'), q('S2_z'), C); Px = L.k21_from_sums(q('M21_x'), q('S1_x'), q('S2_x'), Cpost)
        Kzn = L.k21_from_sums(q('M21_zn'), q('S1_zn'), q('S2_zn'), CL); Kyl = L.k21_from_sums(q('M21_yl'), q('S1_yl'), q('S2_yl'), covy)
        PxG = L.k21_from_sums(q('M21_xG'), q('S1_xG'), q('S2_xG'), Cpost); KyG = L.k21_from_sums(q('M21_yG'), q('S1_yG'), q('S2_yG'), CL)
        m1 = q('S1_z'); Cs = q('M11_z') - np.outer(m1, m1); vs = np.diag(Cs).copy()
        K22 = q('M22') - np.outer(q('S2_z'), q('S2_z')) - 2 * Cs * Cs; K31 = q('M31') - 3 * q('S2_z')[:, None] * Cs
        k3 = np.diag(Kz).copy(); k4 = np.diag(q('M22')) - 3 * q('S2_z') ** 2
        t = dict(K=Kzn, PiP=L.Pi(W, Px))
        t['D'] = Kzn - t['PiP']
        t['lin'] = Kyl - L.Pi(W, (g * g)[:, None] * Kz * g[None, :])
        t['G'] = KyG - L.Pi(W, PxG)
        # closed forms on the true planes: hub tensors, legs p[:, c] of hub c
        bp = offd(g[:, None] * C)
        j0 = offd((d / 2)[:, None] * Kz); j1 = offd((-g / 2)[:, None] * Kz.T * eta[None, :]); j2 = offd((-d / 2)[:, None] * (C * C) * eta[None, :])
        skew = k3 / s ** 3; j3 = offd(C * (skew * (al * al - 1) * pdf / 6)[:, None])
        a5 = -(al ** 3 - 3 * al) * pdf / s ** 4; a6 = (al ** 4 - 6 * al * al + 3) * pdf / s ** 5
        h0 = L.hub_D(W, bp, d); hG = L.hub_D(W, bp + j2, d)
        t['tree'] = h0; t['G2'] = hG
        t['j0'] = L.hub_D(W, bp + j0, d) - h0; t['j1'] = L.hub_D(W, bp + j1, d) - h0; t['j3'] = L.hub_D(W, bp + j3, d) - h0
        t['vsk'] = L.hub_D(W, bp, k3 / 6 * a5)
        t['K3'] = L.hub_D(W, bp + j2 + j0 + j1 + j3, d + k3 / 6 * a5) - hG
        full = L.feed_push(W, d, q('A1'), q('A2'), q('A3'), q('S2_yl'), q('S2_z'), q('Sq'), covy)
        t['F4'] = full - L.Pi(W, L.feed_plane(d, g, K22, K31, k4))
        l22 = offd((-d / 4)[:, None] * K22 * eta[None, :]); l31a = offd((g / 6)[:, None] * K31.T * ((al * al - 1) / v)[None, :])
        l31b = offd((-eta * d / 6)[:, None] * K31)
        t['l22'] = L.hub_D(W, bp + l22, d) - h0; t['l31'] = L.hub_D(W, bp + l31a + l31b, d) - h0; t['v4'] = L.hub_D(W, bp, k4 / 24 * a6)
        t['K4'] = t['F4'] + t['l22'] + t['l31'] + t['v4']
        t['rest'] = t['D'] - t['lin'] - t['G'] - t['K3'] - t['K4']
        t.update(Px=Px, Kz=Kz, K22=K22, K31=K31, k3=k3, k4=k4)
        TR.append(t)
    radial_true = float(np.mean((TR[0]['k4'] + TR[1]['k4']) / 2 / v ** 2))
    Fiso22 = L.feed_push_model(W, d, g, C, 1.0, 0.0) - L.Pi(W, L.feed_plane_model(d, g, C, 1.0, 0.0))
    Fiso11 = L.feed_push_model(W, d, g, C, 0.0, 1.0) - L.Pi(W, L.feed_plane_model(d, g, C, 0.0, 1.0))
    cert = dict(K21_L_vs_file=float(nrm((TR[0]['K'] - f8(PL[f'K21_{Lx}']))[blk]) / nrm(f8(PL[f'K21_{Lx}'])[blk])),
                K21_l_vs_file=float(nrm(offd(TR[0]['Kz'] - f8(PL[f'K21_{l}']))) / nrm(offd(f8(PL[f'K21_{l}'])))),
                K31_l_vs_file=float(nrm(offd(TR[0]['K31'] - f8(PL[f'K31_{l}']))) / nrm(offd(f8(PL[f'K31_{l}'])))),
                K22_l_vs_file=float(nrm(offd(TR[0]['K22'] - f8(PL[f'K22_{l}']))) / nrm(offd(f8(PL[f'K22_{l}'])))))

    # ---------------- card
    c = lambda k, ll=l: f8(CAP[f'{k}_{ll}'])
    gc, dc, al_c = c('g'), c('d'), c('alive'); live = np.outer(al_c, al_c)
    s_carry, s_new = DIALS['s_carry'], DIALS['s_new']
    Pcar = s_carry * live * (gc * gc)[:, None] * c('cpfull') * gc[None, :]
    Pn, lam = c('P'), c('lam'); Pnew = L.cp_plane(Pn, lam)
    star = c('star'); rest = offd(star.T / 3) + np.diag(np.diag(star))
    Cc, bb = c('C'), c('b'); trace = lambda b_: np.diag(Cc)[:, None] * b_[None, :] + 2 * Cc * b_[:, None]
    Pcard = Pcar + Pnew + rest + trace(bb)
    cpL = c('cpfull', Lx); spL = c('sp', Lx); KcardL = c('k21', Lx); CpreL = c('Cpre', Lx)
    btermL = lambda b_: np.diag(CpreL)[:, None] * (W.T @ b_)[None, :] + 2 * CpreL * (W.T @ b_)[:, None]
    Kfull = cpL + spL + btermL(bb)                                  # the card's pre plane at L with the bank plane on ALL rows
    newfull = L.cp_plane(W.T @ Pn, lam)
    cd_ = dict(K=Kfull, PiP=L.Pi(W, Pcard), Estar=spL - L.Pi(W, rest))
    cd_['Dnew'] = newfull - L.Pi(W, Pnew); cd_['Dtr'] = (cpL - newfull) - L.Pi(W, Pcar)
    for part in ('b_tr', 'b_src', 'b_fix'): cd_['D' + part] = btermL(c(part)) - L.Pi(W, trace(c(part)))
    # the card's legs, rebuilt (bp + learned / physical leg corrections), hub algebra
    Cp, k21c, muc = c('Cpre'), c('k21'), c('mu'); sc = np.sqrt(np.diag(Cp)); alc = muc / sc; etac = alc / sc
    bpc = offd(gc[:, None] * Cp); cj = c('c')
    dirs = [offd((dc / 2)[:, None] * k21c), offd((-gc / 2)[:, None] * k21c.T * etac[None, :]), offd((-dc / 2)[:, None] * (Cp * Cp) * etac[None, :])]
    den = np.maximum((bpc * bpc).sum(0), 1e-20); orth = lambda D_: D_ - bpc * ((bpc * D_).sum(0) / den)[None, :]
    legG = (bpc + cj[2][None, :] * orth(dirs[2]) + DIALS['X2'] * dirs[2]) * al_c[:, None]
    legK3 = (cj[0][None, :] * orth(dirs[0]) + DIALS['X0'] * dirs[0] + DIALS['X1'] * dirs[1]) * al_c[:, None]
    births = c('births'); radial = float(c('radial')); wG = s_new * dc * births; birthw = s_new * c('coef') * (c('plus') - c('minus')) * births
    cd_['newG'] = L.hub_D(W, legG, wG); cd_['newK3'] = L.hub_D(W, legG + legK3, wG) - cd_['newG']
    cd_['newK4w'] = L.hub_D(W, legG + legK3, birthw - wG)
    cert['card_hub_vs_columns'] = float(nrm((cd_['newG'] + cd_['newK3'] + cd_['newK4w'] - cd_['Dnew'])[blk]) / nrm(cd_['Dnew'][blk]))
    cert['card_identity'] = float(nrm((Kfull - cd_['PiP'] - cd_['Dtr'] - cd_['Dnew'] - cd_['Db_tr'] - cd_['Db_src'] - cd_['Db_fix'] - cd_['Estar'])[blk]) / nrm(Kfull[blk]))
    cert['card_k21_vs_cpfull'] = float(nrm((Kfull - KcardL)[blk]) / nrm(KcardL[blk]))
    cert['birth_weight_vs_d'] = float(np.median((birthw / np.maximum(wG, 1e-30))[births > .5]))

    # ---------------- matched errors, per truth block
    def pieces(t):
        e = dict(total=cd_['K'] - t['K'], plane=cd_['PiP'] - t['PiP'], star=cd_['Estar'],
                 transport=cd_['Dtr'] + cd_['Db_tr'] - t['lin'], gauss=cd_['newG'] - t['G'], k3ins=cd_['newK3'] - t['K3'],
                 k4=cd_['Db_src'] + cd_['Db_fix'] + cd_['newK4w'] - t['K4'], rest=-t['rest'])
        e['distinct'] = e['transport'] + e['gauss'] + e['k3ins'] + e['k4'] + e['rest']
        return e
    E = [pieces(TR[0]), pieces(TR[1])]
    nK = math.sqrt(float((TR[0]['K'][blk] * TR[1]['K'][blk]).sum()))
    rK = [read(t['K']) for t in TR] if W2 is not None else None

    def x2(A, B):          # noise-free product of two block-dependent arrays (A, B: lists over blocks)
        return (float((A[0][blk] * B[1][blk]).sum()) + float((A[1][blk] * B[0][blk]).sum())) / 2

    def xr(A, B):
        return (float((A[0] * B[1]).mean()) + float((A[1] * B[0]).mean())) / 2

    rec = dict(cert=cert, n_resp=int(resp.sum()), n_alive=int(alive.sum()), radial_card=radial, radial_true=radial_true, err={}, truth={}, card={})
    et = [E[0]['total'], E[1]['total']]; ett = x2(et, et)
    if W2 is not None: rt = [read(e) for e in et]; rtt = xr(rt, rt)
    for k in ('total', 'plane', 'star', 'distinct', 'transport', 'gauss', 'k3ins', 'k4', 'rest'):
        X = [E[0][k], E[1][k]]
        r = dict(blk=100 * math.sqrt(max(x2(X, X), 0.0)) / nK, shr=x2(X, et) / ett)
        if W2 is not None:
            rX = [read(x) for x in X]; r.update(read=100 * math.sqrt(max(xr(rX, rX), 0.0)), read_shr=xr(rX, rt) / rtt)
        rec['err'][k] = r
    for k in ('PiP', 'D', 'lin', 'G', 'G2', 'tree', 'K3', 'j0', 'j1', 'j3', 'vsk', 'K4', 'F4', 'l22', 'l31', 'v4', 'rest'):
        X = [TR[0][k], TR[1][k]]
        rec['truth'][k] = dict(blk=100 * math.sqrt(max(x2(X, X), 0.0)) / nK, noise=100 * float(nrm((X[0] - X[1])[blk])) / math.sqrt(2) / nK)
        if W2 is not None: rX = [read(x) for x in X]; rec['truth'][k]['read'] = 100 * math.sqrt(max(xr(rX, rX), 0.0))
    for k in ('PiP', 'Estar', 'Dtr', 'Dnew', 'Db_tr', 'Db_src', 'Db_fix', 'newG', 'newK3', 'newK4w'):
        rec['card'][k] = dict(blk=100 * float(nrm(cd_[k][blk])) / nK)
        if W2 is not None: rec['card'][k]['read'] = 100 * float(np.sqrt((read(cd_[k]) ** 2).mean()))
    # the feed against the isotropic form (least squares on the kink block; each block's fit applied to the other block)
    F = [TR[0]['F4'], TR[1]['F4']]; A = np.stack([Fiso22[blk], Fiso11[blk]], 1)
    co = [np.linalg.lstsq(A, F[b][blk], rcond=None)[0] for b in (0, 1)]
    res = [F[b][blk] - A @ co[1 - b] for b in (0, 1)]
    rec['feed'] = dict(c22_fit=[float(co[0][0]), float(co[1][0])], c11_fit=[float(co[0][1]), float(co[1][1])], radial3_card=radial / 3, radial3_true=radial_true / 3,
                       explained=1 - float((res[0] * res[1]).sum()) / x2(F, F), iso22_blk=100 * float(nrm(Fiso22[blk])) * radial_true / 3 / nK,
                       iso11_blk=100 * float(nrm(Fiso11[blk])) * 2 * radial_true / 3 / nK)
    # Gaussian births: MC vs the closed forms
    for k in ('tree', 'G2'):
        X = [TR[0][k], TR[1][k]]; Gm = [TR[0]['G'], TR[1]['G']]
        rec['truth'][k]['vs_G_resid'] = 100 * math.sqrt(max(x2([Gm[0] - X[0], Gm[1] - X[1]], [Gm[0] - X[0], Gm[1] - X[1]]), 0.0)) / nK
    out['layers'][l] = rec
    for b in (0, 1):
        for k in ('total', 'plane', 'distinct', 'transport', 'gauss', 'k3ins', 'k4', 'rest'): SAVE[f'e_{k}_L{l}_b{b}'] = E[b][k].astype(np.float32)
        for k in ('K', 'D', 'lin', 'G', 'K3', 'K4', 'F4', 'rest', 'j0', 'j1', 'l22', 'l31'): SAVE[f't_{k}_L{l}_b{b}'] = TR[b][k].astype(np.float32)
    for k in ('K', 'Dtr', 'Dnew', 'Db_tr', 'Db_src', 'Db_fix', 'newG', 'newK3', 'newK4w', 'Estar'): SAVE[f'c_{k}_L{l}'] = cd_[k].astype(np.float32)
    SAVE[f'blk_L{l}'] = blk
    e = rec['err']; f = lambda k: f"{e[k]['blk']:.2f}%/{e[k]['shr']:+.2f}" + (f" [{e[k]['read']:.4f}%/{e[k]['read_shr']:+.2f}]" if 'read' in e[k] else '')
    print(f"l={l}->L={Lx} resp {rec['n_resp']} alive {rec['n_alive']} | cert {json.dumps({k: round(v_, 5) for k, v_ in cert.items()})}", flush=True)
    print('   card error (kink-block % of |K21 true| / share of squared error [read metric % / share]): ' + ' | '.join(f'{k} {f(k)}' for k in e), flush=True)
    print('   truth pieces (kink-block %, noise %): ' + ' | '.join(f"{k} {r['blk']:.2f} ({r['noise']:.2f})" for k, r in rec['truth'].items()), flush=True)
    print('   card pieces (kink-block %): ' + ' | '.join(f"{k} {r['blk']:.2f}" for k, r in rec['card'].items()) + f" | feed fit {json.dumps({k: (round(v_, 4) if not isinstance(v_, list) else [round(x, 4) for x in v_]) for k, v_ in rec['feed'].items()})} | {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(HERE / f't2_budget_rank{rank:04d}.json', 'w'), indent=1)
np.savez(HERE / f't2_budget_rank{rank:04d}_planes.npz', **SAVE)
print(json.dumps(dict(done=True)), flush=True)
