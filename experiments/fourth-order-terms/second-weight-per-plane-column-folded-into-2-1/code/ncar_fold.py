"""TAG opus-physicist NCAR_FOLD (2026-10-02): the BANK CHANNEL folded into the K21 plane, so that a native card needs NO extra pass over the bank columns.

The bank channel (ncar_bank.py, option columns): a second weight per bank column, w_r <- 2 (gbar w_r + lam_r (V . pp_r)), new columns at 0; the matrix of a layer is
B_ij = sum_r w_r pp_ir pp_jr on the layer's pushed columns, and its use is the K31 extra plane (3/n) B_ij (forms removed).
To first order in the pair correlation the pair map's K31 term is (1/6) K31_ij e_i g_j with e_i = E relu'''(z_i) = - alpha_i pdf_i / v_i, and its K21 term is
(1/2) K21_ij d_i g_j with d_i = pdf_i / sigma_i. So the plane (3/n) B_ij acts as an EFFECTIVE K21:
        K21_eff[i, j] = K21[i, j] - (mu_i / (n v_i)) B_ij         (K21[i, j] = cum(z_i, z_i, z_j), row index doubled)
and inside the card's bank read (plane_ij = sum_r lam_r pp_ir^2 pp_jr on the response rows) this is an in-place change of the left factor,
        pp_ir^2 lam_r  ->  pp_ir (pp_ir lam_r - (mu_i / (n v_i)) w_r),
with no further product over the columns. This module is the torch emulation of that:

install(cd, torch, st, strength=1.0, lo=1, hi=14, forms=1, level='pair', rows='resp', top=0, check=None, dfit=0, dscale=1.0, k31=None)
   vbeta   True: the conversion weights use the card's own beta = 2 (gaussian_mean - host gate) alive in place of V = 2 m (1 - g) (natively pp_r . beta is already computed)
   dresp   True (level 'read'): the diagonal regression uses the response rows only (the rows on which the native left factor exists); dfit = 1: regressor tr C v_i alone
   transport, source   factors on the carried weights and on the fresh conversion (dials; 1 = the law)
   k22, k4 strengths of the per-unit uses of the SAME matrix's diagonal (minus its fitted forms): K22_ij += k22 (B_ii + B_jj) / n, k4_i += k4 (6 / n) B_ii (need dfit)
   dscale  with dfit: factor on the diagonal-fitted coefficient of tr C x C before it is removed;  k31: alias of strength
   level   'pair': the fold is added to the K21 the PAIR MAP receives (hook pair_in); the card's other uses of K21 keep the plain plane.
           'read': the fold is added inside the bank read (cd.shared['read'] is wrapped), on the response rows and alive columns, so EVERY consumer of the plane gets it
                   (pair map, the post K21 target and the star behind it, the births, the learned features, the edge source): what a single native read would do.
           'r31' : no fold, the plane goes through remainder31 (reference: equals ncar_bank.py with columns=True)
   rows    'resp': only the card's response rows (where the bank read exists); 'all': every row (level 'pair' and 'r31' only)
   forms   least-squares removal on the off-diagonal before use: 4 = tr C x C, C^2, mu mu', mu (C mu)' + sym; 3 = without C^2; 2 = tr C x C, mu mu'; 1 = tr C x C;
           0 = nothing removed (the plane as the read would produce it)
   dfit    0: the form coefficients by least squares on the off-diagonal of B (needs the whole matrix); 3 / 4: by a regression of the DIAGONAL B_ii = sum_r w_r pp_ir^2 on
           (tr C v_i, mu_i^2, 2 mu_i (C mu)_i [, (C^2)_ii]) over the units, which a native card can do without any product over pairs; `forms` then says which of
           tr C x C, mu mu', mu (C mu)' + sym are removed with those coefficients (the C^2 form is never removed in this mode)
   top     m > 0: only the m columns with the largest |w_r| |pp_r|^2
   check   a dict: at layers 5, 9, 13 (level 'pair') the pair map is evaluated with the plane as remainder31 and as the fold; check[l] = (cosine, size ratio) of the two
           changes of the post covariance (certificate of the first-order identity above)
st = the harness side table: st['pp'] pushed columns of the layer, st['lam_in'] their weights, st['old'] the old-tier column count (as for the lead's ncar modules)."""
import math
from ncar_bank import specific


def install(cd, torch, st, strength=1.0, lo=1, hi=14, forms=1, level='pair', rows='resp', top=0, check=None, dfit=0, dscale=1.0, k31=None,
            vbeta=False, dresp=False, transport=1.0, source=1.0, k22=0.0, k4=0.0):
    if k31 is not None: strength = k31          # the keyword the lead's and the trainer's harnesses pass
    t = torch; f64 = t.float64; n = 1024; S = dict(w=None, mu=None, C=None, l=None); info = {}; cd.ncar_info = info; k22s, k4s = k22, k4

    def plane_of(pp32, C, mu, sub=None):          # the bank-channel matrix of the layer (forms removed as asked), off-diagonal; sub = the rows the diagonal regression may use
        pp = pp32.detach().to(f64); w = S['w'][:pp.shape[1]]
        if top and top < w.numel():
            sel = (w.abs() * (pp * pp).sum(0)).argsort(descending=True)[:top]; pp, w = pp[:, sel], w[sel]
        M = (pp * w[None, :]) @ pp.T
        if dfit and forms:      # the form coefficients from the DIAGONAL alone (B_ii = sum_r w_r pp_ir^2: no product over pairs): regression over the units
            v = t.diagonal(C); Cmu = C @ mu; trC = t.trace(C)
            X = t.stack([trC * v] + ([mu * mu, 2 * mu * Cmu] if dfit >= 3 else []) + ([(C * C).sum(1)] if dfit == 4 else []), 1); dM = t.diagonal(M)
            pick = sub if (dresp and sub is not None) else slice(None)
            co = t.linalg.lstsq(X[pick], dM[pick][:, None]).solution[:, 0]; S['dres'], S['dres_l'] = dM - X @ co, S.get('l_now')      # the diagonal minus its forms: the per-unit part
            if dresp and sub is not None:       # natively B_ii exists on the response rows only: the per-unit part is zero elsewhere
                keep = t.zeros(n, dtype=f64); keep[sub] = 1.0; S['dres'] = S['dres'] * keep
            R = M - dscale * co[0] * trC * C          # dscale: the off-diagonal carries about 1.4 times the isotropic coefficient of the diagonal (ora23 check)
            if forms >= 2: R = R - co[1] * t.outer(mu, mu)
            if forms >= 3: R = R - co[2] * (t.outer(mu, Cmu) + t.outer(Cmu, mu))
            if check is not None and S.get('l_now') in (5, 9, 13):      # the same coefficients from the off-diagonal least squares (what the exact removal uses)
                off = ~t.eye(n, dtype=t.bool); Xo = t.stack([(trC * C)[off], t.outer(mu, mu)[off], (t.outer(mu, Cmu) + t.outer(Cmu, mu))[off]] + ([(C @ C)[off]] if dfit == 4 else []), 1)
                ce = t.linalg.lstsq(Xo, M[off][:, None]).solution[:, 0]; check[S['l_now']] = dict(**{'diag_fit_%d' % i: float(x) for i, x in enumerate(co)}, **{'offdiag_fit_%d' % i: float(x) for i, x in enumerate(ce)})
            return R - t.diag(t.diagonal(R))
        if forms: return specific(t, M, C, mu, forms)[0]
        return M - t.diag(t.diagonal(M))

    def fold_of(R, C, mu):
        return -(mu / (n * t.diagonal(C).clamp_min(1e-12)))[:, None] * R

    old_pin = cd.hooks.get('pair_in')

    def pin_hook(l, pin, ctx):
        if old_pin is not None: pin = old_pin(l, pin, ctx)
        mu, Cpre, k21_, k4v, k22, rem = pin
        if level != 'read' and S['w'] is not None and lo <= l <= hi and strength and st.get('pp') is not None:
            C = Cpre.detach().to(f64); m64 = mu.detach().to(f64); S['l_now'] = l; R = plane_of(st['pp'], C, m64)
            if rows == 'resp': R = R * ctx['response_mask'].detach().to(f64)[:, None]
            if level == 'r31':
                s = t.diagonal(C).clamp_min(1e-10).sqrt(); rm = strength * (3.0 / n) * R / (s ** 3)[:, None] / s[None, :]; rem = rm if rem is None else rem + rm
            else:
                F = fold_of(R, C, m64)
                if check is not None and l in (5, 9, 13):
                    s = t.diagonal(C).clamp_min(1e-10).sqrt(); rm = (3.0 / n) * R / (s ** 3)[:, None] / s[None, :]; pair = ctx['pair']
                    c0 = pair(mu, Cpre, k21_, k4v, k22, 'covariance')[1]; c1 = pair(mu, Cpre, k21_, k4v, k22, 'covariance', remainder31=rm)[1]; c2 = pair(mu, Cpre, k21_ + F, k4v, k22, 'covariance')[1]
                    a, b = (c1 - c0).to(f64), (c2 - c0).to(f64); off = ~t.eye(n, dtype=t.bool)
                    check[l] = dict(cosine_offdiag=float((a[off] * b[off]).sum() / (a[off].norm() * b[off].norm())), size_ratio_offdiag=float(b[off].norm() / a[off].norm()),
                                    cosine_diag=float((t.diagonal(a) * t.diagonal(b)).sum() / (t.diagonal(a).norm() * t.diagonal(b).norm()).clamp_min(1e-300)),
                                    size_ratio_diag=float(t.diagonal(b).norm() / t.diagonal(a).norm().clamp_min(1e-300)))
                k21_ = k21_ + strength * F
            info[l] = dict(plane_rms=float((R * R).mean().sqrt()))
        if (k22s or k4s) and S.get('dres') is not None and S.get('dres_l') == l and lo <= l <= hi:      # the same matrix's DIAGONAL in the per-unit slots of the pair map:
            dr = S['dres']                                    # second chaos of K22_ij = (1/n)(Ntilde_ii + Ntilde_jj), of k4_i = (6/n) Ntilde_ii (diagonal minus its forms)
            if k22s: k22 = k22 + (k22s * (dr[:, None] + dr[None, :]) / n).to(k22.dtype)
            if k4s: k4v = k4v + (k4s * 6.0 / n) * dr.to(k4v.dtype)
        return (mu, Cpre, k21_, k4v, k22, rem)
    cd.hooks['pair_in'] = pin_hook

    if level == 'read':     # the layer's mean and covariance are known before the read (hooks 'mu' and 'cpre' run first in the step)
        old_mu, old_c = cd.hooks.get('mu'), cd.hooks.get('cpre')

        def mu_hook(l, mu, ctx):
            if old_mu is not None: mu = old_mu(l, mu, ctx)
            S['mu'], S['l'] = mu.detach().to(f64), l; return mu

        def c_hook(l, C, ctx):
            if old_c is not None: C = old_c(l, C, ctx)
            S['C'] = C.detach().to(f64); return C
        cd.hooks['mu'], cd.hooks['cpre'] = mu_hook, c_hook
        read0 = cd.shared['read']

        def read(tt, pp, lam, response_indices, indices, U, F, old_columns):
            plane = read0(tt, pp, lam, response_indices, indices, U, F, old_columns)
            if S['w'] is not None and S['l'] is not None and lo <= S['l'] <= hi and strength:
                S['l_now'] = S['l']; Fd = fold_of(plane_of(pp, S['C'], S['mu'], response_indices), S['C'], S['mu']); info[S['l']] = dict(fold_rms=float((Fd * Fd).mean().sqrt()))
                plane = plane + strength * Fd[response_indices][:, indices].to(plane.dtype)
            return plane
        cd.shared['read'] = read

    old_out = cd.hooks.get('state_out')

    def out_hook(l, out, ctx):
        if old_out is not None: out = old_out(l, out, ctx)
        if 1 <= l <= 14 and st.get('pp') is not None and st['pp'].shape[1]:
            C = ctx['Cpre'].detach().to(f64); mu = ctx['mu'].detach().to(f64); m = out[0].detach().to(f64)
            v = t.diagonal(C).clamp_min(1e-12); s = v.sqrt(); al = mu / s; ph = t.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = t.erfc(-al / math.sqrt(2)) / 2
            gb = (g - m * ph / s).mean(); V = 2 * m * (1 - g)
            if vbeta:        # the card's own vector of its omitted-radial source (natively the dot products pp_r . beta already exist)
                aux = ctx['aux']; V = (2 * (aux['gaussian_mean'] - ctx['host'] * aux['gate']) * ctx['alive']).detach().to(f64)
            pp = st['pp'].detach().to(f64); ncol = pp.shape[1]; c = source * st['lam_in'].detach().to(f64)[:ncol] * (V @ pp)
            w = c if S['w'] is None else c + transport * gb * S['w'][:ncol]
            S['w'] = t.cat((2 * w, t.zeros(max(int(out[3].numel()) - ncol, 0), dtype=f64)))
        return out
    cd.hooks['state_out'] = out_hook
    return cd
