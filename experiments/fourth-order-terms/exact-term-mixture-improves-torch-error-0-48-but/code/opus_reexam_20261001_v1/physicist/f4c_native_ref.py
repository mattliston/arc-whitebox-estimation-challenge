#!/usr/bin/env python3
"""TAG opus-physicist F4c native reference: the held-out MIXTURE (F4's rule at theta + the exact S1, S2 of every young hub at c1, c2)
written the way the native step would do it: F4's own step unchanged at strength theta, PLUS an exact part whose shifts are quadratic
forms of the pair's two pushed columns with SIX CONSTANTS PER PAIR FIXED AT ITS BIRTH (as in the engineer's K4L fragment).
Plain array operations only (numpy here); harness against f4b.py (mode 'mix', metric 'birth').

At the birth of a pair (hub h of layer j; all (K,) arrays):
    tp, tm   the two hub entries t+, t-            s2   |s|^2 = sum_i (plus column)_i^2 - tp^2   (the pair's common leg)
    S1 = -alpha pdf / v,   S2 = 2 (g - g^2 - m pdf / sigma)     (alpha, sigma, pdf, g of the hub; v = sigma^2; m = the pair map's mean of the hub)
    exact_constants(xp, tp, tm, S1, S2, s2, c1, c2) -> n_p, n_m, each (3, K):
        x+ = (d/2) (n_p[0] p^2 + n_p[1] p D + n_p[2] D^2),   x- = (d/2) (n_m[0] p^2 + n_m[1] p D + n_m[2] D^2),   D = p - q,
        p, q = the pushed + and - columns at a later layer; x+- = (third-order weight) x (shift of the gated column).
At the end of step l, for every cohort born 1 or 2 layers earlier (f4c_step):
    shift_p = (d/2) [theta A p^2] + x+ / lam_p,     shift_q = (d/2) [theta A q^2] + x- / lam_q        (A = F4's birth scalar)
    anchor sums beta, diag over the shifted columns exactly as in F4 (X = lam o shift; u = gated column):
        beta_k = [ X @ (v @ (u o u)) + 2 u @ (v @ (u o X)) ]_k / sum(v^2),   diag_i = 3 sum_r u_ri^2 X_ri
    star: trace term built with b + beta, diagonal of rest lowered by diag - 3 v beta; the carried b stays b.
Billed work per pair element, my count: D, p D, D^2 (3; p^2 and q^2 exist in F4's step), two quadratic forms (10), u o X twice (2),
four more matrix-vector passes over X (8): about 23 on top of F4's 32 per pair element if nothing is shared; about 8 if the anchor's
passes are taken once over the summed shift (F4's five passes become passes over the total X).
usage: f4c_native_ref.py <rank> <threads>"""
import sys


def exact_constants(xp, tp, tm, S1, S2, s2, c1, c2):
    z = xp.zeros_like(S1); gap = tp - tm; ok = xp.abs(gap) > 1e-12; rg = xp.where(ok, 1.0 / xp.where(ok, gap, 1.0), 0.0)
    M = xp.array([[z, 2 * c1 * S1, c2 * S2], [c1 * S1, 2 * c2 * S2, z], [c2 * S2, z, z]])            # acts on (Y^2, Y Wt, Wt^2)
    G = xp.array([s2 * s2, 2 * s2, xp.ones_like(s2)])                                                 # metric: |gY|^2 : |gWt|^2 = |s|^2 : 1, no cross term
    Bm = xp.array([[xp.ones_like(tp), xp.ones_like(tp)], [tp, tm], [tp * tp, tm * tm]])               # 3 x 2 x K
    BtG = xp.einsum('qsk,qk->sqk', Bm, G)                                                             # 2 x 3 x K
    G2 = xp.einsum('sqk,qtk->stk', BtG, Bm)
    det = G2[0, 0] * G2[1, 1] - G2[0, 1] * G2[1, 0]; ok = ok & (G2[0, 0] > 0) & (G2[1, 1] > 0) & (det > 1e-8 * G2[0, 0] * G2[1, 1])
    rdet = xp.where(ok, 1.0 / xp.where(ok, det, 1.0), 0.0)
    inv = xp.array([[G2[1, 1], -G2[0, 1]], [-G2[1, 0], G2[0, 0]]]) * rdet[None, None, :]
    N = xp.einsum('stk,tqk,qrk->srk', inv, BtG, M)                                                    # 2 x 3 x K on (Y^2, Y Wt, Wt^2)
    tau = tp * rg                                                                                     # Y = p - tau D, Wt = D / gap
    n = [xp.array([N[s, 0], N[s, 1] * rg - 2 * tau * N[s, 0], tau * tau * N[s, 0] - tau * rg * N[s, 1] + rg * rg * N[s, 2]]) for s in (0, 1)]
    return n[0], n[1]


def f4c_step(xp, p, q, u_p, u_q, lam_p, lam_q, A, n_p, n_m, theta, alive, d, v):
    f8 = xp.float64; p8, q8, up, uq = (x.astype(f8) for x in (p, q, u_p, u_q)); hl = (0.5 * alive * d)[:, None]
    D = p8 - q8; P2 = p8 * p8; PD = p8 * D; D2 = D * D
    lp = xp.where(lam_p != 0, lam_p, 1.0).astype(f8); lq = xp.where(lam_q != 0, lam_q, 1.0).astype(f8)
    sh_p = hl * (theta * A[None, :] * P2 + (n_p[0][None, :] * P2 + n_p[1][None, :] * PD + n_p[2][None, :] * D2) / lp[None, :])
    sh_q = hl * (theta * A[None, :] * (q8 * q8) + (n_m[0][None, :] * P2 + n_m[1][None, :] * PD + n_m[2][None, :] * D2) / lq[None, :])
    X_p = sh_p * lam_p.astype(f8)[None, :]; X_q = sh_q * lam_q.astype(f8)[None, :]
    beta = (X_p @ (v @ (up * up)) + X_q @ (v @ (uq * uq)) + 2.0 * (up @ (v @ (up * X_p)) + uq @ (v @ (uq * X_q)))) / xp.sum(v * v)
    diag = 3.0 * ((up * up * X_p).sum(1) + (uq * uq * X_q).sum(1))
    return (up + sh_p).astype(xp.float32), (uq + sh_q).astype(xp.float32), beta, diag


if __name__ == '__main__':
    import json
    import numpy as np, torch
    import pcard as pc, f4b
    rank = int(sys.argv[1]); pc.set_threads(sys.argv[2]); W = pc.weights(rank); tp_ = pc.truth(rank); AGE = 2; C1, C2, TH = 0.48, 0.48, 0.378

    def install_ref(cd):
        side = dict(B={})
        push0 = cd.shared['push']
        def push(tt, layer, w, p, lam, previous, Q, F, core, stops, rank=384):
            r = push0(tt, layer, w, p, lam, previous, Q, F, core, stops, rank=rank); side['pp'] = r[0]; return r
        cd.shared['push'] = push
        bd0 = cd.env['_bd_update']
        def bd(bp, Cpre, K21, aux, d, alive, births, a, tplus, tminus, coefficients):
            side['tpm'] = (tplus.detach().numpy(), tminus.detach().numpy()); return bd0(bp, Cpre, K21, aux, d, alive, births, a, tplus, tminus, coefficients)
        cd.env['_bd_update'] = bd
        sel0 = cd.select
        def select(score):
            r = sel0(score); side['sel'] = r[0].numpy(); return r
        cd.select = select
        prev = cd.hooks.get('state_out')
        def state_out(l, out, ctx):
            if prev is not None: out = prev(l, out, ctx)
            if l > 14: return out
            aux = ctx['aux']; C, p, lam, star, alive, stops = out[1], out[2], out[3], out[4], out[6], out[12]
            pp = side['pp'].numpy(); Cn = C.numpy(); pn = p.numpy().copy(); lamn = lam.numpy(); d = (aux['pdf'] / aux['sigma']).numpy(); v = np.diag(Cn)
            beta = np.zeros(1024); diag = np.zeros(1024); young = list(range(max(l - AGE, 0), l))
            for j in young:
                B = side['B'][j]; k0 = int(stops[j - 1]) if j else 0; K = (int(stops[j]) - k0) // 2; P, Mi = slice(k0, k0 + K), slice(k0 + K, k0 + 2 * K)
                new_p, new_q, be, dg = f4c_step(np, pp[:, P], pp[:, Mi], pn[:, P], pn[:, Mi], lamn[P], lamn[Mi], B['A'], B['n_p'], B['n_m'], TH, alive.numpy(), d, v)
                pn[:, P], pn[:, Mi] = new_p, new_q; beta += be; diag += dg
            if young:
                dP = v[:, None] * beta[None, :] + 2 * Cn * beta[:, None]; np.fill_diagonal(dP, diag)
                star = star - torch.from_numpy((3 * dP.T - 2 * np.diag(np.diag(dP))).astype(np.float32)); p = torch.from_numpy(pn)
            sel = side['sel']; a_, s_, ph_, g_, m_ = (x.numpy()[sel].astype(np.float64) for x in (aux['alpha'], aux['sigma'], aux['pdf'], aux['gate'], ctx['host']))
            S1 = -a_ * ph_ / (s_ * s_); S2 = 2.0 * (g_ - g_ * g_ - m_ * ph_ / s_)
            tp = side['tpm'][0][sel].astype(np.float64); tm = side['tpm'][1][sel].astype(np.float64)
            k0n = int(stops[l - 1]) if l else 0; Kn = (int(stops[l]) - k0n) // 2
            s2 = np.maximum((out[2].numpy()[:, k0n:k0n + Kn].astype(np.float64) ** 2).sum(0) - tp * tp, 1e-30)
            n_p, n_m = exact_constants(np, tp, tm, S1, S2, s2, C1, C2)
            A_ = (-a_ / (s_ * (1 + float(ctx['radial']) / 3))).astype(np.float32).astype(np.float64)
            side['B'][l] = dict(n_p=n_p, n_m=n_m, A=A_)
            return out[:2] + (p, lam, star) + out[5:]
        cd.hooks['state_out'] = state_out
        return cd
    a = pc.run(f4b.install(pc.build(W), torch, 1.0, age=AGE, mode='mix', cs=(C1, C2, 0.0, 0.0), ths=(TH, TH, TH, TH), metric='birth')); r = pc.run(install_ref(pc.build(W))); base = pc.run(pc.build(W))
    print(json.dumps(dict(TAG='opus-physicist F4c native reference (rule theta + exact part with birth constants) against f4b.py mix / birth metric', rank=rank,
                          base_final_x1e9=round(float(pc.mse(base, tp_)[15]) * 1e9, 4), f4b_final_x1e9=round(float(pc.mse(a, tp_)[15]) * 1e9, 4),
                          ref_final_x1e9=round(float(pc.mse(r, tp_)[15]) * 1e9, 4), max_abs_diff_ref_vs_f4b=float(np.abs(a - r).max()),
                          rms_diff_final=float(np.sqrt(((a - r)[15] ** 2).mean())), effect_rms_final=float(np.sqrt(((a - base)[15] ** 2).mean()))), indent=1))
