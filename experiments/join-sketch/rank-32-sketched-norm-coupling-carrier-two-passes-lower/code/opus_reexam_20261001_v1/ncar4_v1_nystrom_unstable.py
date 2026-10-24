"""TAG opus-reexam NCAR4 (lead, 2026-10-02): a SKETCHED rank-r carrier of the second-chaos matrix: the physicist's law (nlaw.py, lean or complete, as in
ncar3.py) with a generic low-rank state, for the part of the matrix that is not 'collective mode x unit vector' (oracle split E41: on network 35 the
complement of the top-16 covariance eigenvectors carries most of the K31 gain).

State of layer l (pre-activation space):   Nt = a C + Z M Z'      (Z: n x r, M: r x r symmetric)
One layer:
    exact transport of the low-rank part:   Z+ = W' (g o Z),  M+ = 2 gbar M                       (r products with W, nothing sketched)
    fresh part F = 2 W' S W + 2 kappa C_next - (multiple of C_next),   S = the law with the low-rank part's gate term taken out
                   (its diagonal and its share of kappa stay in: they are per-unit vectors and one scalar)
    Nystrom sketch with k probes Omega = [top-K eigenvectors of C_next | Gaussian]:   Y = F Omega,   F ~ Y (Omega' Y)^+ Y'
    re-cut: [Z+ | Y] -> QR -> eigen-decomposition of the (r + k) core -> the r directions of largest |eigenvalue|
    a_next = sum_m lambda_m u_m' Nt_next u_m / sum_m lambda_m^2 on the K eigenvector probes (as in ncar3)
Plane: R = Z M Z' with its two-sided block on the top-K eigenvectors taken out (P R P: where the C^2-like and mean-form content sits),
remainder31_ij = k31 (3/n) R_ij / (sigma_i^3 sigma_j), layers lo..hi.
Per layer: r + 2 k products with W, 3 k (lean) products with C, 2 k bank passes when the conversion is on.

install(cd, torch, st, k31=1.0, r=32, k=24, K=8, conv=1, k21=True, lean=True, lo=1, hi=14, seed=0, dump=None)"""
import math


def install(cd, torch, st, k31=1.0, r=32, k=24, K=8, conv=1, k21=True, lean=True, lo=1, hi=14, seed=0, dump=None, cleanup=True):
    t = torch; f64 = t.float64; n = 1024; S = dict(prev=None); info = {}; cd.ncar_info = info
    gen = t.Generator().manual_seed(40000 + int(seed)); old_pin = cd.hooks.get('pair_in')
    W = lambda l: cd.ws[l].to(f64)

    def S_apply(P, y):          # the law's matrix of the previous layer WITHOUT the gate term of the low-rank part, applied to y (post space)
        g, d, S2, e3, gb, Co, kap = (P[k_] for k_ in ('g', 'd', 'S2', 'e3', 'gb', 'Co', 'kap'))
        yg = g[:, None] * y; Coyg = Co @ yg; aCyg = P['a'] * (Coyg + P['v'][:, None] * yg)
        conv_y = P['ppc'] @ (P['c'][:, None] * (P['ppc'].T @ yg)) if P['c'] is not None else 0.0
        inner_co = S2[:, None] * Coyg + P['rv'][:, None] * y
        out = g[:, None] * (gb * aCyg + conv_y) + P['rv'][:, None] * Coyg - kap * g[:, None] * Coyg
        if P['Ko'] is not None and not lean:
            Ko, q1, CV, KoV, KoTg = P['Ko'], P['q1'], P['CV'], P['KoV'], P['KoTgam']
            Koyg = Ko @ yg; KoTdy = Ko.T @ (d[:, None] * y)
            inner_co = inner_co + q1[:, None] * Koyg + (S2 / 2)[:, None] * KoTdy + (d * KoTg)[:, None] * y + ((e3 / 2) * KoV)[:, None] * y
            out = out + g[:, None] * (Ko.T @ (q1[:, None] * Coyg + ((e3 / 2) * CV)[:, None] * y)) + d[:, None] * (Ko @ ((S2 / 2)[:, None] * Coyg + (0.5 * CV * d)[:, None] * y))
            out = out + (d * KoTg)[:, None] * Coyg + (0.5 * d * CV)[:, None] * KoTdy + ((e3 / 2) * CV)[:, None] * Koyg + ((e3 / 2) * KoV)[:, None] * Coyg
        out = out + g[:, None] * (Co @ inner_co)
        return out + (P['dgn'] - P['diag_sub'])[:, None] * y

    def pin_hook(l, pin, ctx):
        if old_pin is not None: pin = old_pin(l, pin, ctx)
        if l > 14: return pin
        mu, Cpre, k21_, k4v, k22, rem = pin
        C = Cpre.to(f64); P = S['prev']; lam, V = t.linalg.eigh(C); U = V[:, -K:].flip(1); lamK = lam[-K:].flip(0)
        if P is None:
            a = 0.0; Z = t.zeros((n, 0), dtype=f64); M = t.zeros((0, 0), dtype=f64)
        else:
            w = W(l); Zp = w.T @ (P['g'][:, None] * P['Z']); Mp = 2 * P['gb'] * P['M']                      # exact transport
            Om = t.cat((U, t.randn((n, max(k - K, 0)), generator=gen, dtype=f64)), 1)
            Y = 2 * (w.T @ S_apply(P, w @ Om)) + 2 * P['kap'] * (C @ Om)                                        # fresh part applied to the probes (with its multiple of C)
            NU = Y[:, :K] + Zp @ (Mp @ (Zp.T @ U))                                                               # the whole next state applied to the eigenvector probes
            a = float((lamK * (U * NU).sum(0)).sum() / (lamK * lamK).sum())
            Y = Y - a * (C @ Om)                                                                                 # fresh network-specific part
            G = Om.T @ Y; G = (G + G.T) / 2; le, Ve = t.linalg.eigh(G); keep = le.abs() > 1e-5 * le.abs().max()
            Gi = (Ve[:, keep] / le[keep][None, :]) @ Ve[:, keep].T                                               # pseudo-inverse of the probe core
            B = t.cat((Zp, Y), 1); Qb, Rb = t.linalg.qr(B); core = t.zeros((B.shape[1], B.shape[1]), dtype=f64)
            core[:Zp.shape[1], :Zp.shape[1]] = Mp; core[Zp.shape[1]:, Zp.shape[1]:] = Gi
            core = Rb @ core @ Rb.T; core = (core + core.T) / 2; lc, Vc = t.linalg.eigh(core); o_ = lc.abs().argsort(descending=True)[:r]
            Z = Qb @ Vc[:, o_]; M = t.diag(lc[o_])
            info[l] = dict(a_over_trC=a / float(t.trace(C)), kept=float((lc[o_] ** 2).sum() / (lc ** 2).sum()), fresh_over_carried=float(Y.norm() / max(float(Zp.norm()), 1e-30)))
        S['cur'] = dict(l=l, C=C, U=U, lamK=lamK, a=a, Z=Z, M=M)
        if k31 and lo <= l <= hi and P is not None and Z.shape[1]:
            Rm = Z @ M @ Z.T
            if cleanup: PR = U @ (U.T @ Rm); Rm = Rm - (PR @ U) @ U.T                                             # two-sided block on the top-K eigenvectors out
            if dump is not None: dump[l] = dict(R=Rm.to(t.float32).clone())
            s = t.diagonal(C).clamp_min(1e-10).sqrt()
            rm = k31 * (3.0 / n) * (Rm - t.diag(t.diagonal(Rm))) / (s ** 3)[:, None] / s[None, :]
            rem = rm if rem is None else rem + rm
        return (mu, Cpre, k21_, k4v, k22, rem)
    cd.hooks['pair_in'] = pin_hook

    old_out = cd.hooks.get('state_out')

    def out_hook(l, out, ctx):
        if old_out is not None: out = old_out(l, out, ctx)
        if not (0 <= l <= 14 and 'cur' in S): return out
        cur = S.pop('cur'); C = cur['C']; U, lamK, a, Z, M = cur['U'], cur['lamK'], cur['a'], cur['Z'], cur['M']
        mu = ctx['mu'].detach().to(f64); m = out[0].detach().to(f64)
        v = t.diagonal(C).clamp_min(1e-12); s = v.sqrt(); al = mu / s; ph = t.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = t.erfc(-al / math.sqrt(2)) / 2; d = ph / s
        r2 = v * ((1 + al * al) * g + al * ph); r3 = v * s * ((al ** 3 + 3 * al) * g + (al * al + 2) * ph); r4 = v * v * ((al ** 4 + 6 * al * al + 3) * g + (al ** 3 + 5 * al) * ph)
        gam = g - m * d; V = 2 * m * (1 - g); S2 = 2 * (gam - g * g); S3 = 3 * (1 - g) * (r2 - 2 * m * m)
        S4 = r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2; e3 = -al * ph / v; varG = r2 - (s * ph + mu * g) ** 2
        gb = float(gam.mean()); Co = C - t.diag(v); Co2 = Co * Co; CV = Co @ V + v * V
        ZM = Z @ M; Ntd = a * v + (ZM * Z).sum(1)                                                               # diagonal of the state
        Pv = gb * Ntd + 2 * (Co2 @ gam)
        c = ppc = None; K3Vd = t.zeros(n, dtype=f64)
        if conv and st.get('pp') is not None and st['pp'].shape[1]:
            first = int(st.get('old', 0)) if conv == 2 else 0
            ppc = st['pp'].to(f64)[:, first:]; lam_in = st['lam_in'].detach().to(f64)[first:first + ppc.shape[1]]
            if ppc.shape[1]: c = lam_in * (V @ ppc); K3Vd = (ppc * ppc) @ c
            else: ppc = None
        prev = dict(l=l, C=C, Co=Co, v=v, a=a, Z=Z, M=M, g=g, d=d, S2=S2, e3=e3, gb=gb, CV=CV, c=c, ppc=ppc, Ko=None)
        rv = d * CV + (e3 / 2) * Pv + S3
        dgn = S4 + V * (CV - V * v) + 2 * gam * (Co2 @ gam) - 2 * g * g * (Co2 @ (g * g)) + gam * gb * Ntd
        diag_sub = g * g * (gb * a * v + K3Vd + Co2 @ S2)                    # diagonal entries of the off-diagonal operators applied in S_apply (the low-rank gate term is not in S)
        dgn = dgn - g * g * gb * (ZM * Z).sum(1)                             # ... and the law's own diagonal replaces the transported low-rank part's diagonal too
        if k21 and l >= 1:
            Kf = ctx['k21'].detach().to(f64); Ko = Kf - t.diag(t.diagonal(Kf)); KoV = Ko @ V; KoTg = Ko.T @ gam; KC = Ko * Co
            rv = rv + d * KoTg + (e3 / 2) * KoV                               # the K21 contractions inside the row factor (kept by the lean law)
            dgn = dgn + V * KoTg + gam * KoV - 2 * g * d * (KC @ (g * g)) - 2 * g * g * (KC.T @ (g * d))
            if not lean:
                q1 = d * (1 - g + m * al / s); prev.update(Ko=Ko, q1=q1, KoV=KoV, KoTgam=KoTg)
                rv = rv - d * KoTg - (e3 / 2) * KoV                           # complete law: those two enter through S_apply's own terms
                diag_sub = diag_sub + 2 * g * g * (KC.T @ q1) + 2 * d * g * (KC @ (S2 / 2))
        # kappa inside the top-K subspace
        PU = gb * (a * U * lamK[None, :] + ZM @ (Z.T @ U)) + 2 * (C @ (gam[:, None] * (U * lamK[None, :])))
        if c is not None: PU = PU + ppc @ (c[:, None] * (ppc.T @ U))
        kap = float((lamK * (U * PU).sum(0)).sum() / (lamK * lamK).sum())
        prev.update(rv=rv, kap=kap, dgn=dgn - kap * varG, diag_sub=diag_sub)
        S['prev'] = prev
        return out
    cd.hooks['state_out'] = out_hook
    return cd
