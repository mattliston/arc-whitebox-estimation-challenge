"""TAG opus-reexam NCAR3 (lead, 2026-10-02): the physicist's complete one-layer law of the second-chaos matrix (O/physicist/nlaw.py, ncarry.py)
applied to K VECTORS with matrix-vector products only: the native-ready form of the carrier.

State of layer l (pre-activation space), the 'partner' structure of ncarry.py:
    Nt = a C + U Qf' + Qf U' - U A U',    U = top-K eigenvectors of the layer's covariance C (n x K), Qf = (Nt - a C) U (n x K), A = U' Qf (K x K),
    a  = sum_m lambda_m u_m' Nt u_m / sum_m lambda_m^2      (the multiple of C, projected inside the top-K subspace; ncarry uses the full Frobenius projection)
Law (nlaw.law, every term with its predicted coefficient), applied to y = W U_next (post space of layer l), yg = g o y:
    N y = g o [gb Nt yg + K3V yg + Co (S2 o Co yg)]                                      transport, conversion, Gaussian births
        + rv o Co yg + g o Co (rv o y),   rv = d CV + (e3 / 2) Pv + S3                    mean tilt, variance tilt, own (3,1)
        + K21 terms (Ko = the card's K21 plane without its diagonal, q1 = d (1 - g + m alpha / sigma)):
            g o Ko' (q1 o Co yg) + g o Co (q1 o Ko yg)
            d o Ko ((S2/2) o Co yg) + g o Co ((S2/2) o Ko' (d o y))
            d (Ko' gamma) o Co yg + g o Co (d (Ko' gamma) o y)
            (d/2) o Ko (CV d o y) + (d CV / 2) o Ko' (d o y)                               [the term d d' o Co o K3V is left out: a product over all bank columns]
            (e3/2) CV o Ko yg + (e3/2)(Ko V) o Co yg + g o Ko' ((e3/2) CV o y) + g o Co ((e3/2)(Ko V) o y)
        - kappa g o Co yg                                                                  homogeneity (the gain part is carried by 2 kappa C_next)
        - (diagonals of the product terms) o y + (dgn - kappa varG) o y                   the diagonal is the law's own
    kappa = sum_m lambda_m u_m' P u_m / sum_m lambda_m^2,  P u = gb Nt u + 2 C (gamma o C u) + K3V u      (subspace form of <P, C> / <C, C>)
    Nt_next U_next = 2 W' (N y) + 2 kappa C_next U_next
with CV = C V, V = 2 m (1 - g), Pv = gb diag(Nt) + 2 (Co o Co) gamma, K3V = sum_r c_r pp_r pp_r', c_r = lam_r (V . pp_r) on the pushed bank columns
(conv = 1: all columns, 2: the young tier only, 0: none), m = the card's post mean of the layer.
Use: K31 extra plane (3/n) R_ij / (sigma_i^3 sigma_j), R = U Qp' + Qp U' with Qp = Qf - U A (plane='one': the one-sided block) or
R = U Qf' + Qf U' - U A U' (plane='full': the whole network-specific part of the state), layers lo..hi.
Per mode: 2 products with W, 3 with Co, 2 with Ko, 2 with Ko', and the bank pass when conv is on.

install(cd, torch, st, k31=1.0, K=16, conv=1, k21=True, plane='one', lo=1, hi=14, sweeps=0, skip=(), dump=None)
  st: harness dict with st['pp'] (pushed columns of the current layer), st['lam_in'] (their weights as given to the push), st['old'] (number of old-tier columns)."""
import math


def install(cd, torch, st, k31=1.0, K=16, conv=1, k21=True, plane='one', lo=1, hi=14, sweeps=0, skip=(), dump=None):
    t = torch; f64 = t.float64; n = 1024; S = dict(prev=None, NU=None); info = {}; cd.ncar_info = info
    old_pin = cd.hooks.get('pair_in')

    def subspace(C, P):
        if sweeps and P is not None:
            U = t.linalg.qr(cd_w(P['l'] + 1).T @ (P['g'][:, None] * P['U']))[0]
            for _ in range(int(sweeps)): U = t.linalg.qr(C @ U)[0]
            lamK = ((C @ U) * U).sum(0)
        else:
            lam, V = t.linalg.eigh(C); U = V[:, -K:].flip(1); lamK = lam[-K:].flip(0)
        return U, lamK

    def cd_w(l): return cd.ws[l].to(f64)

    def Nt_apply(P, y):         # the partner-cut state of the previous layer applied to y (n x k)
        out = P['a'] * (P['C'] @ y)
        if P['Qf'] is not None: out = out + P['U'] @ (P['Qf'].T @ y) + P['Qf'] @ (P['U'].T @ y) - P['U'] @ (P['A'] @ (P['U'].T @ y))
        return out

    def K3V_apply(P, y):
        if P['c'] is None: return t.zeros_like(y)
        return P['ppc'] @ (P['c'][:, None] * (P['ppc'].T @ y))

    def N_apply(P, y):          # the law's matrix of the previous layer applied to y (post space)
        g, d, gam, V, S2, S3, e3, gb, Co, kap = (P[k] for k in ('g', 'd', 'gam', 'V', 'S2', 'S3', 'e3', 'gb', 'Co', 'kap'))
        yg = g[:, None] * y; Coyg = Co @ yg
        inner_co = S2[:, None] * Coyg + P['rv'][:, None] * y           # everything that goes through one more product with Co under an outer factor g
        out = g[:, None] * (gb * Nt_apply(P, yg) + K3V_apply(P, yg)) + P['rv'][:, None] * Coyg - kap * g[:, None] * Coyg
        if P['Ko'] is not None:
            Ko, q1, CV, KoV, KoTg = P['Ko'], P['q1'], P['CV'], P['KoV'], P['KoTgam']
            Koyg = Ko @ yg; KoTdy = Ko.T @ (d[:, None] * y)
            inner_co = inner_co + q1[:, None] * Koyg + (S2 / 2)[:, None] * KoTdy + (d * KoTg)[:, None] * y + ((e3 / 2) * KoV)[:, None] * y
            out = out + g[:, None] * (Ko.T @ (q1[:, None] * Coyg + ((e3 / 2) * CV)[:, None] * y))
            out = out + d[:, None] * (Ko @ ((S2 / 2)[:, None] * Coyg + (0.5 * CV * d)[:, None] * y))
            out = out + (d * KoTg)[:, None] * Coyg + (0.5 * d * CV)[:, None] * KoTdy + ((e3 / 2) * CV)[:, None] * Koyg + ((e3 / 2) * KoV)[:, None] * Coyg
        out = out + g[:, None] * (Co @ inner_co)
        return out + (P['dgn'] - P['diag_sub'])[:, None] * y

    def pin_hook(l, pin, ctx):
        if old_pin is not None: pin = old_pin(l, pin, ctx)
        if l > 14: return pin
        mu, Cpre, k21_, k4v, k22, rem = pin
        C = Cpre.to(f64); P = S['prev']; U, lamK = subspace(C, P)
        if P is None:
            NU = t.zeros((n, K), dtype=f64)
        else:
            y = cd_w(l) @ U
            NU = 2 * (cd_w(l).T @ N_apply(P, y)) + 2 * P['kap'] * (U * lamK[None, :])
        a = float((lamK * (U * NU).sum(0)).sum() / (lamK * lamK).sum())
        Qf = NU - a * U * lamK[None, :]; A = U.T @ Qf; A = (A + A.T) / 2
        S['cur'] = dict(l=l, C=C, U=U, lamK=lamK, a=a, Qf=Qf, A=A)
        if dump is not None: dump[l] = dict(U=U.clone(), Q=(Qf - U @ A).clone())
        if k31 and lo <= l <= hi and P is not None:
            Qp = Qf - U @ A
            M = U @ Qp.T; M = M + M.T
            if plane == 'full': M = M + U @ (A @ U.T)
            s = t.diagonal(C).clamp_min(1e-10).sqrt()
            rm = k31 * (3.0 / n) * (M - t.diag(t.diagonal(M))) / (s ** 3)[:, None] / s[None, :]
            info[l] = dict(a_over_trC=a / float(t.trace(C)), kap_prev_over_trC=float(P['kap']) / float(t.trace(P['C'])), q_norm=float(Qp.norm()))
            rem = rm if rem is None else rem + rm
        return (mu, Cpre, k21_, k4v, k22, rem)
    cd.hooks['pair_in'] = pin_hook

    old_out = cd.hooks.get('state_out')

    def out_hook(l, out, ctx):
        if old_out is not None: out = old_out(l, out, ctx)
        if not (0 <= l <= 14 and 'cur' in S): return out
        cur = S.pop('cur'); C = cur['C']; U, lamK, a, Qf, A = cur['U'], cur['lamK'], cur['a'], cur['Qf'], cur['A']
        mu = ctx['mu'].detach().to(f64); m = out[0].detach().to(f64)
        v = t.diagonal(C).clamp_min(1e-12); s = v.sqrt(); al = mu / s; ph = t.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = t.erfc(-al / math.sqrt(2)) / 2; d = ph / s
        r2 = v * ((1 + al * al) * g + al * ph); r3 = v * s * ((al ** 3 + 3 * al) * g + (al * al + 2) * ph); r4 = v * v * ((al ** 4 + 6 * al * al + 3) * g + (al ** 3 + 5 * al) * ph)
        gam = g - m * d; V = 2 * m * (1 - g); S2 = 2 * (gam - g * g); S3 = 3 * (1 - g) * (r2 - 2 * m * m)
        S4 = r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2; e3 = -al * ph / v; varG = r2 - (s * ph + mu * g) ** 2
        gb = float(gam.mean()); Co = C - t.diag(v); Co2 = Co * Co; CV = Co @ V + v * V
        Ntd = a * v + 2 * (U * Qf).sum(1) - ((U @ A) * U).sum(1)                                  # diagonal of the partner-cut state
        Pv = gb * Ntd + 2 * (Co2 @ gam)
        rv = t.zeros(n, dtype=f64)
        if 'rows' not in skip: rv = rv + d * CV
        if 'vtilt' not in skip: rv = rv + (e3 / 2) * Pv
        if 'own31' not in skip: rv = rv + S3
        # conversion from the bank: columns pushed into this layer, weights as given to the push
        c = ppc = None; K3Vd = t.zeros(n, dtype=f64)
        if conv and st.get('pp') is not None and st['pp'].shape[1]:
            first = int(st.get('old', 0)) if conv == 2 else 0
            ppc = st['pp'].to(f64)[:, first:]; lam_in = st['lam_in'].detach().to(f64)[first:first + ppc.shape[1]]
            if ppc.shape[1]:
                c = lam_in * (V @ ppc); K3Vd = (ppc * ppc) @ c
            else: ppc = None
        dgn = S4 + V * (CV - V * v) + 2 * gam * (Co2 @ gam) - 2 * g * g * (Co2 @ (g * g)) + gam * gb * Ntd
        diag_sub = g * g * (gb * Ntd + K3Vd + Co2 @ S2)
        prev = dict(l=l, C=C, Co=Co, U=U, Qf=Qf, A=A, a=a, g=g, d=d, gam=gam, V=V, S2=S2, S3=S3, e3=e3, gb=gb, rv=rv, CV=CV, c=c, ppc=ppc, Ko=None)
        if k21 and l >= 1 and 'k3c' not in skip:
            Kf = ctx['k21'].detach().to(f64); Ko = Kf - t.diag(t.diagonal(Kf)); q1 = d * (1 - g + m * al / s); KC = Ko * Co
            prev.update(Ko=Ko, q1=q1, KoV=Ko @ V, KoTgam=Ko.T @ gam)
            dgn = dgn + V * prev['KoTgam'] + gam * prev['KoV'] - 2 * g * d * (KC @ (g * g)) - 2 * g * g * (KC.T @ (g * d))
            diag_sub = diag_sub + 2 * g * g * (KC.T @ q1) + 2 * d * g * (KC @ (S2 / 2))
        # kappa inside the top-K subspace: P u = gb Nt u + 2 C (gamma o C u) + K3V u
        if 'gain' in skip: kap = 0.0
        else:
            NtU = a * U * lamK[None, :] + U @ A + (Qf - U @ A)                                    # the state applied to its own vectors = a C U + Qf
            PU = gb * NtU + 2 * (C @ (gam[:, None] * (U * lamK[None, :])))
            if c is not None: PU = PU + ppc @ (c[:, None] * (ppc.T @ U))
            kap = float((lamK * (U * PU).sum(0)).sum() / (lamK * lamK).sum())
        prev.update(kap=kap, dgn=dgn - kap * varG, diag_sub=diag_sub)
        S['prev'] = prev
        return out
    cd.hooks['state_out'] = out_hook
    return cd
