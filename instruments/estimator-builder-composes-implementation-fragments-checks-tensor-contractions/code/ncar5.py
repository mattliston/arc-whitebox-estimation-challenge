"""TAG opus-reexam NCAR5 (lead, 2026-10-02): the STRUCTURED low-rank carrier of the second-chaos matrix: no probes, no applications of the law to vectors.

Why: the sketched carrier (ncar4.py) finds the fresh part of the matrix by applying the law to 24-40 probe vectors twice; each application is 2 products
with W, 3-4 with the covariance and 2 passes over the bank, which natively costs several times what the plane earns (engineer's price list 11:22Z).
The fresh sources of the lean certified law (physicist's nlaw.py, corrected 11:05Z) are covariance-shaped,
    off the diagonal   N_ab = g_a g_b [gbar Nt + K3V + Co S2 Co]_ab + Co_ab (rv_a g_b + g_a rv_b) - kappa g_a g_b C_ab,
    rv = d (Co V) + (e/2) Pv + S3 + d (K21' gamma) + (e/2) (K21 V),    Pv = gbar diag(Nt) + 2 (Co o Co) gamma,
so with the covariance of the layer replaced by its top-K eigenpairs, C ~ U Lam U', they are EXPLICIT low-rank matrices on pushed columns:
    births     2 W' G Co S2 Co G W        ~ 2 GU [Lam (U' S2 U) Lam] GU'            GU = W' (g o U)
    row forms  2 W' (R Co G + G Co R) W   ~ 2 (RU Lam GU' + GU Lam RU')              RU = W' (rv o U)
    carried    2 gbar W' G (Z M Z') G W   = Zp (2 gbar M) Zp'                       Zp = W' (g o Z)     (exact)
Dropped: every diagonal sandwich W' D W (full rank; the law's diagonal, the diagonal completions of C), the gain term (a multiple of the next covariance plus a
diagonal sandwich), and the conversion from the third-order bank (that is the BANK CHANNEL, carried for nothing as a second weight per bank column: physicist's
ncar_bank.py / ncar_fold.py; here only its diagonal enters, through Pv, when bankdiag is on).
State: Nt = a C + (bank channel) + Z M Z',  a = ca x (radial / 3) x tr C  (the isotropic part; Monte Carlo: 0.75 of the card's radial / 3, notes section 20).
One layer: the basis [Zp | GU | RU] (n x (r + 2K)), QR, eigen-decomposition of the small core, the r directions of largest |eigenvalue| kept.
Plane: R = Z M Z' with its two-sided block on the layer's top-K covariance eigenvectors taken out; remainder31_ij = k31 (3/n) R_ij / (sigma_i^3 sigma_j), layers lo..hi.
Native count per layer: r + 2K products with W, the top-K eigenpairs of the covariance (warm start = GU, which is already in the basis), the rank-r plane.

install(cd, torch, st, k31=1.0, r=32, K=16, lo=1, hi=14, births=1.0, rows=1.0, k21=True, cleanup=True, ca=0.75, sweeps=0, extra=0, bankdiag=True, dump=None)
  sweeps > 0: the eigenpairs are Ritz pairs of a block of K + extra columns: the previous block pushed through the gates and W, `sweeps` power sweeps with the
  covariance, one Rayleigh-Ritz rotation (the first carried layer takes exact eigenvectors); 0 = exact eigenvectors at every layer.
  births, rows: strengths of the two fresh sources;  bankdiag: the diagonal of the bank channel (second weights w_r <- 2 (gbar w_r + lam_r V.pp_r)) enters Pv.
cd.ncar_info[l] = dict(kept, fresh_over_carried, a_over_trC, ritz) for the log."""
import math


def install(cd, torch, st, k31=1.0, r=32, K=16, lo=1, hi=14, births=1.0, rows=1.0, k21=True, cleanup=True, ca=0.75, sweeps=0, extra=0, bankdiag=True, dump=None):
    t = torch; f64 = t.float64; n = 1024; S = dict(prev=None, w=None); info = {}; cd.ncar_info = info
    old_pin = cd.hooks.get('pair_in'); Kb = K + int(extra)

    def pin_hook(l, pin, ctx):
        if old_pin is not None: pin = old_pin(l, pin, ctx)
        if l > 14: return pin
        mu, Cpre, k21_, k4v, k22, rem = pin
        C = Cpre.to(f64); P = S['prev']; a = ca * float(ctx['radial']) / 3 * float(t.trace(C))
        Z = t.zeros((n, 0), dtype=f64); M = t.zeros((0, 0), dtype=f64); Ub = None
        if P is not None:
            w = cd.ws[l].to(f64); g = P['g']
            GUb = w.T @ (g[:, None] * P['Ub'])                                                       # the whole Ritz block pushed (its first K columns are the basis GU)
            GU = GUb[:, :K]; RU = w.T @ (P['rv'][:, None] * P['Ub'][:, :K]); Zp = w.T @ (g[:, None] * P['Z'])
            L = P['lamK']; S2U = P['Ub'][:, :K].T @ (P['S2'][:, None] * P['Ub'][:, :K])
            cf = t.zeros((2 * K, 2 * K), dtype=f64)
            cf[:K, :K] = 2 * births * (L[:, None] * S2U * L[None, :]); cf[:K, K:] = 2 * rows * t.diag(L); cf[K:, :K] = 2 * rows * t.diag(L)
            B = t.cat((Zp, GU, RU), 1); Qb, Rb = t.linalg.qr(B); core = t.zeros((B.shape[1], B.shape[1]), dtype=f64); rz = Zp.shape[1]
            core[:rz, :rz] = 2 * P['gb'] * P['M']; core[rz:, rz:] = cf
            core = Rb @ core @ Rb.T; core = (core + core.T) / 2; lc, Vc = t.linalg.eigh(core); o_ = lc.abs().argsort(descending=True)[:r]
            Z = Qb @ Vc[:, o_]; M = t.diag(lc[o_])
            fresh = (GU @ cf[:K, :K] @ GU.T + RU @ cf[K:, :K] @ GU.T + GU @ cf[:K, K:] @ RU.T).norm() if dump is not None else t.tensor(0.0)
            info[l] = dict(kept=float((lc[o_] ** 2).sum() / (lc ** 2).sum()), a_over_trC=a / float(t.trace(C)), fresh=float(fresh))
            if sweeps:
                X = t.linalg.qr(GUb)[0]
                for _ in range(int(sweeps)): X = t.linalg.qr(C @ X)[0]
                Tm = X.T @ (C @ X); Tm = (Tm + Tm.T) / 2; lt, Vt = t.linalg.eigh(Tm); Ub = X @ Vt.flip(1); lamb = lt.flip(0)
        if Ub is None:
            lam, V = t.linalg.eigh(C); Ub = V[:, -Kb:].flip(1); lamb = lam[-Kb:].flip(0)
        U = Ub[:, :K]; S['cur'] = dict(l=l, C=C, Ub=Ub, lamK=lamb[:K], a=a, Z=Z, M=M)
        if k31 and lo <= l <= hi and Z.shape[1]:
            Rm = Z @ M @ Z.T
            if cleanup: PR = U @ (U.T @ Rm); Rm = Rm - (PR @ U) @ U.T                                 # two-sided block on the top-K eigenvectors out
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
        cur = S.pop('cur'); C = cur['C']; a, Z, M = cur['a'], cur['Z'], cur['M']
        mu = ctx['mu'].detach().to(f64); m = out[0].detach().to(f64)
        v = t.diagonal(C).clamp_min(1e-12); s = v.sqrt(); al = mu / s; ph = t.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = t.erfc(-al / math.sqrt(2)) / 2; d = ph / s
        r2 = v * ((1 + al * al) * g + al * ph); gam = g - m * d; V = 2 * m * (1 - g); S2 = 2 * (gam - g * g); S3 = 3 * (1 - g) * (r2 - 2 * m * m); e3 = -al * ph / v
        gb = float(gam.mean()); Co = C - t.diag(v)
        Ntd = a * v + ((Z @ M) * Z).sum(1)                                                            # diagonal of the state
        if st.get('pp') is not None and st['pp'].shape[1]:                                            # the bank channel's second weights (only their diagonal is used here)
            pp = st['pp'].detach().to(f64); ncol = pp.shape[1]; lam_in = st['lam_in'].detach().to(f64)[:ncol]
            wb = t.zeros(ncol, dtype=f64) if S['w'] is None else S['w'][:ncol]
            if bankdiag: Ntd = Ntd + (pp * pp) @ wb
            S['w'] = t.cat((2 * (gb * wb + lam_in * (V @ pp)), t.zeros(max(int(out[3].numel()) - ncol, 0), dtype=f64)))
        Pv = gb * Ntd + 2 * ((Co * Co) @ gam)
        rv = d * (Co @ V) + (e3 / 2) * Pv + S3
        if k21 and l >= 1:
            Kf = ctx['k21'].detach().to(f64); Ko = Kf - t.diag(t.diagonal(Kf)); rv = rv + d * (Ko.T @ gam) + (e3 / 2) * (Ko @ V)
        S['prev'] = dict(l=l, g=g, gb=gb, S2=S2, rv=rv, Ub=cur['Ub'], lamK=cur['lamK'], Z=Z, M=M)
        return out
    cd.hooks['state_out'] = out_hook
    return cd
