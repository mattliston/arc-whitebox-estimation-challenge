"""TAG opus-physicist NCARRY: a CARRIER of the second-chaos matrix inside the torch card (no Monte Carlo): the complete one-layer law of
nlaw.py (certified: t19k_law_cert.py at layer 0 -> 1, t19n0_law_depth_partial.py at depth) run on the card's OWN state, layer after layer,
and used through the K31 extra plane (3/n) R of the pair map (the use that pays in the oracle arms: -8% / -16% on networks 35 / 651).
   state       Nt(l) = the estimate of Ntilde of layer l's pre-activations (n x n; Nt(0) = 0: Gaussian inputs)
   at layer l  pair_in:   Nt(l) = pushed part + 2 kappa Cpre(l);  R = Nt(l) minus its forms (tr C x C, C^2, mu mu', mu (C mu)' + sym off the diagonal);
                          remainder31 += strength (3/n) R_ij / (sigma_i^3 sigma_j)    [rank r > 0: only the r largest |eigenvalue| directions of R]
               state_out: N = law(Cpre, mu, m = the card's post mean, Nt(l), K21 = the card's plane), pushed part of layer l+1 = 2 W' N W, kappa kept
install(cd, torch, strength=1.0, rank=0, k21=True, start=1, stop=14, skip=(), conv='', k31v=False)      cd.ncarry_info[l] = sizes for the log
   conv   which parts of the card's third-order state enter the conversion sum_c V_c kappa3(a,b,c): 'b' bank columns, 's' star, 'm' the b mode ('' = no conversion)
   k31v   the V-contraction of K31 from the isotropic model, (radial/3)(v_a (C V)_b + 2 C_ab (C V)_a)
   conv_age    bank columns of the conversion: 0 all, a > 0 the cohorts born at most a layers ago, -1 the young tier (not yet joined), -2 the old tier only (the joined,
               compressed columns: natively a second r x r core on the pushed basis); cd.ncarry_info['cols'][l] = (pushed columns, of which old tier) at layer l
   state_rank  r > 0: the carried state is cut to (multiple of C) + rank r at every layer (what a native carrier would hold)
   law_start   the matrix is 0 up to this layer and the law runs from there
   partner     m > 0: the carried state is cut at every layer to (multiple of C) + sum_k (u_k p_k' + p_k u_k') on the top m covariance eigenvectors
   forms       4 (default): the K31 plane uses the matrix minus its four forms; 3: without the C^2 form; 2: tr C x C and mu mu' only; 1: minus the multiple of C only
   basis       which vectors span the subspace of the oneside cut: 'T' the previous state's r columns pushed through the gates and W, 'C' the top-K eigenvectors of the layer's
               covariance, 'G' / 'R' the top-K eigenvectors of the PREVIOUS layer's covariance times the gate / times the law's row factor, pushed through W (default 'TC')
   ritz        True: the cut is the two-sided projection Q (Q' D Q) Q' (Rayleigh-Ritz) instead of the one-sided form; both need one application of the law per basis vector
   law_ck      K > 0: inside the law the covariance is its diagonal plus its top-K eigenpairs (prices a law application that needs no product with C)
   passive     True: with strength 0 the matrix is still carried and its network-specific part recorded in cd.ncarry_R (for audits of other carriers on the same card)
   bank_exact  True: the bank channel g g' o (gbar B + K3V) is carried exactly (natively: second weights on the bank columns) and the state cuts (state_rank, oneside,
               partner) act only on the rest of the matrix
   ownrow      0 (default, the certified law): the mean-tilt row form sums over c != a; 1: with the unit's own term, as every version up to frozen s2q ran (a double count of
               the own (3,1) term, found 11:05Z: t19u_cvown.py); values between scale that extra term
   oneside     (r, K), r > 0: the carried state is cut at every layer to (multiple of C) + rank r, the rank-r part taken from the one-sided form on the span of the previous
               state's pushed columns and the top-K covariance eigenvectors (r + K applications of the law per layer natively; state_rank is the same cut from the full matrix)
   skip        'gain' 'rows' 'vtilt' 'own31' 'k3c' (all K21 terms) 'k3off' (their off-diagonal parts) 'k3prod' (only the two that need K21 x C products) 'k3el' (K21 x per-unit-vector products) 'k3x' ((d d') o Co o K3V);
               the LEAN law of the cost study = skip ('k3prod', 'k3el', 'k3x'): no block product with K21
               'convonly': N = g g' o (gbar Nt + K3V) and nothing else (the bank channel: transport + conversion); 'nobirth': the Gaussian births C S2 C dropped;
               'notrans' (with 'convonly'): no transport either, the matrix of a layer is the one-layer conversion of the layer before (a second plane read, no state)
The module is a research form (full n x n matrices, about ten n^3 products per layer); a native form would sketch a rank-32 piece."""
import math


def unit_stats(t, C, mu, m):
    v = t.diagonal(C).clamp_min(1e-12); s = v.sqrt(); al = mu / s; ph = t.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = t.erfc(-al / math.sqrt(2)) / 2; d = ph / s
    r2 = v * ((1 + al * al) * g + al * ph); r3 = v * s * ((al ** 3 + 3 * al) * g + (al * al + 2) * ph); r4 = v * v * ((al ** 4 + 6 * al * al + 3) * g + (al ** 3 + 5 * al) * ph)
    gam = g - m * d
    return dict(v=v, s=s, al=al, ph=ph, g=g, d=d, gam=gam, V=2 * m * (1 - g), S2=2 * (gam - g * g), S3=3 * (1 - g) * (r2 - 2 * m * m),
                S4=r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2, e3=-al * ph / v, varG=r2 - (s * ph + mu * g) ** 2)


def law(t, C, mu, m, Nt, K21=None, K3V=None, K31V=None, skip=(), ownrow=0.0):
    """torch port of nlaw.law. K3V_ab = sum_c V_c kappa3(a,b,c) (conversion from the third cumulant), K31V_ab = sum_c V_c kappa4(a,a,b,c)."""
    u = unit_stats(t, C, mu, m); v, s, al, g, d, gam, V, S2, S3, S4, e3 = (u[k] for k in ('v', 's', 'al', 'g', 'd', 'gam', 'V', 'S2', 'S3', 'S4', 'e3'))
    gg = t.outer(g, g); Co = C - t.diag(v); CV = Co @ V + v * V; gb = gam.mean()
    K3 = t.zeros_like(C) if K3V is None else K3V
    if 'convonly' in skip:        # the bank channel alone: transport + conversion (natively a second weight per bank column, no product at all)
        return gg * ((0.0 if 'notrans' in skip else gb) * Nt + K3), t.zeros((), dtype=C.dtype), u
    P = gb * Nt + 2 * (C * gam[None, :]) @ C + K3; kap = t.zeros((), dtype=C.dtype) if 'gain' in skip else (P * C).sum() / (C * C).sum()
    N = gg * (gb * Nt + K3) if 'nobirth' in skip else gg * (gb * Nt + K3 + (Co * S2[None, :]) @ Co)
    CVr = Co @ V + ownrow * v * V       # the mean-tilt row form runs over c != a (c = a is the own (3,1) term); ownrow = 1: the double count of the versions before s2r
    if 'rows' not in skip: N = N + Co * (d * CVr)[:, None] * g[None, :] + Co * g[:, None] * (d * CVr)[None, :]
    if 'vtilt' not in skip:
        Pv = gb * t.diagonal(Nt) + 2 * ((Co * Co) @ gam); x = Co * ((e3 / 2) * Pv)[:, None] * g[None, :]; N = N + x + x.T
    if 'own31' not in skip: N = N + Co * (S3[:, None] * g[None, :] + g[:, None] * S3[None, :])
    if K31V is not None:
        x = (d / 2)[:, None] * K31V * g[None, :]; N = N + x + x.T
    dgn = S4 + V * (CV - V * v) + ((2 * gam[:, None] * gam[None, :] - 2 * (g * g)[:, None] * (g * g)[None, :]) * Co * Co).sum(1) + gam * gb * t.diagonal(Nt)
    if K21 is not None and 'k3c' not in skip:
        Ko = K21 - t.diag(t.diagonal(K21)); q1 = d * (1 - g + m * al / s)
        if 'k3off' not in skip:
            if 'k3prod' not in skip:      # the two terms that need products of the K21 plane with the covariance
                N = N + gg * ((Ko.T * q1[None, :]) @ Co + (Co * q1[None, :]) @ Ko)
                x = d[:, None] * ((Ko * (S2 / 2)[None, :]) @ Co) * g[None, :]; N = N + x + x.T
            x = d[:, None] * (Co * (Ko.T @ gam)[:, None]) * g[None, :]; N = N + x + x.T          # K21 inside the row factor (per-unit contraction K21' gamma)
            if 'k3el' not in skip:        # the K21 x per-unit-vector products (as matrix-vector products: four with K21)
                N = N + 0.5 * t.outer(d, d) * (Ko * CV[None, :] + Ko.T * CV[:, None])
                x = (e3 / 2)[:, None] * (Ko * CV[:, None]) * g[None, :]; N = N + x + x.T
            if 'k3x' not in skip: N = N + t.outer(d, d) * Co * K3                                # elementwise product of two matrices
            x = (e3 / 2)[:, None] * (Co * (Ko @ V)[:, None]) * g[None, :]; N = N + x + x.T          # K21 inside the row factor (K21 V)
        dgn = dgn + V * (Ko.T @ gam) + gam * (Ko @ V) - 2 * g * d * ((Ko * Co) @ (g * g)) - 2 * g * g * ((Ko.T * Co) @ (g * d))
    N = N - kap * gg * C
    return t.diagonal_scatter(N, dgn - kap * u['varG']), kap, u


def specific(t, N, C, mu, forms=4):
    """off-diagonal network-specific part: least-squares removal of tr C x C, C^2, mu mu', mu (C mu)' + sym (the diagonal is not used by the K31 plane).
    forms = 3: without C^2 (no n^3 product); forms = 2: tr C x C and mu mu' only."""
    n = N.shape[0]; off = ~t.eye(n, dtype=t.bool); Cmu = C @ mu
    cols = [(t.trace(C) * C)[off]] + ([(C @ C)[off]] if forms == 4 else []) + [t.outer(mu, mu)[off]] + ([(t.outer(mu, Cmu) + t.outer(Cmu, mu))[off]] if forms >= 3 else [])
    X = t.stack(cols, 1)
    co = t.linalg.lstsq(X, N[off][:, None]).solution[:, 0]; R = t.zeros_like(N); R[off] = N[off] - X @ co
    return R, co


def conversion(t, V, C, pp, lam, star, W, bpre, parts='bsm', first=0, last=None):
    """K3V_ab = sum_c V_c kappa3(a,b,c) from the card's own third-order state of the layer: b = bank columns (pushed columns pp, weights lam),
    s = the star (its plane tensor of the layer before, pushed exactly with W; rest[i,k] = T(i,i,k)), m = the b mode Sym(C x bpre)."""
    f64 = t.float64; n = C.shape[0]; out = t.zeros((n, n), dtype=f64)
    if 'b' in parts and lam[first:last].numel() > 0:   # columns first..last (the old tier's columns come first; last = None: to the end)
        p64 = pp[:, first:last].to(f64); c = lam[first:last].to(f64) * (V @ p64); out = out + (p64 * c[None, :]) @ p64.T
    if 's' in parts and star is not None:
        A = star.to(f64); rest = ((A + 2 * t.diag(t.diagonal(A))) / 3).T; ro = rest - t.diag(t.diagonal(rest)); WV = W @ V
        x = (W.T * WV[None, :]) @ (ro @ W); out = out + (W.T * (ro @ WV + t.diagonal(rest) * WV)[None, :]) @ W + x + x.T
    if 'm' in parts:
        CV = C @ V; out = out + C * (bpre @ V) + t.outer(CV, bpre) + t.outer(bpre, CV)
    return out


def install(cd, torch, strength=1.0, rank=0, k21=True, start=1, stop=14, skip=(), conv='', k31v=False, conv_age=0, state_rank=0, law_start=0, partner=0, forms=4, oneside=(0, 0), ownrow=0.0, bank_exact=False, basis='TC', ritz=False, law_ck=0, passive=False):
    t = torch; f64 = t.float64; n = 1024; st = dict(pushed=None, kap=None, Nt=None); info = {}; cd.ncarry_info = info; cd.ncarry_R = {}; cd.ncarry_a = {}
    if conv:           # capture what the conversion needs: the pushed columns and their weights, the incoming star
        push0 = cd.shared['push']
        def push(tt, layer, w, p, lam, previous, Q, F, core, stops, rank=384):
            r = push0(tt, layer, w, p, lam, previous, Q, F, core, stops, rank=rank); st['pp'], st['lam'] = r[0], lam
            # first column of the conversion: conv_age = a > 0 keeps the cohorts born at most a layers ago (cohort j ends at column stops[j]); -1 = the young tier only
            cs_ = [0] + [int(x) for x in stops]; st['first'] = (cs_[max(layer - conv_age, 0)] if conv_age > 0 else (int(r[4]) if conv_age == -1 else 0))
            st['last'] = int(r[4]) if conv_age == -2 else None; st['cols'] = (int(r[0].shape[1]), int(r[4])); return r
        cd.shared['push'] = push
        old_in = cd.hooks.get('state_in')
        def in_hook(l, state, ctx):
            if old_in is not None: state = old_in(l, state, ctx)
            st['star'] = state[4]; return state
        cd.hooks['state_in'] = in_hook
    old_pin = cd.hooks.get('pair_in')
    def eig(C):              # eigen-decomposition of the layer's covariance, once per layer
        if st.get('eig_of') is not C: st['eig'] = t.linalg.eigh(C); st['eig_of'] = C
        return st['eig']
    def pin_hook(l, pin, ctx):
        if old_pin is not None: pin = old_pin(l, pin, ctx)
        mu, Cpre, k21_, k4v, k22, rem = pin; C = Cpre.to(f64); st['Cl'] = C
        if l <= law_start: st['Nt'] = t.zeros((n, n), dtype=f64)
        elif st['pushed'] is not None:
            st['Nt'] = st['pushed'] + 2 * st['kap'] * C
            Bk = st.get('pushed_b') if bank_exact else None
            if Bk is not None: st['Nt'] = st['Nt'] - Bk        # the cuts below act on everything except the bank channel, which the bank's columns carry exactly
            if partner:         # the lead's structure (E32c): a multiple of C plus sum_k (u_k p_k' + p_k u_k') on the top m eigenvectors u_k of the layer's covariance
                a_ = (st['Nt'] * C).sum() / (C * C).sum(); D_ = st['Nt'] - a_ * C; ev_, V_ = t.linalg.eigh(C); U_ = V_[:, -partner:]; UD = U_ @ (U_.T @ D_)
                st['Nt'] = a_ * C + UD + UD.T - (UD @ U_) @ U_.T
            if oneside[0]:      # a cut a native carrier can COMPUTE (one application of the law per basis vector, no full matrix): the one-sided form P D + D P - P D P on the span of
                # [the previous state's r columns pushed through the gates and W | the top-K covariance eigenvectors], then rank r by eigenvalue inside that form
                r_, K_ = oneside; a_ = (st['Nt'] * C).sum() / (C * C).sum(); D_ = st['Nt'] - a_ * C
                cols = [st[k_] for k_, ch_ in (('TY', 'T'), ('TG', 'G'), ('TR', 'R')) if ch_ in basis and st.get(k_) is not None]
                if (K_ and 'C' in basis) or not cols: cols.append(eig(C)[1][:, -(K_ or r_):])
                Q_ = t.linalg.qr(t.cat(cols, 1))[0]; DQ = D_ @ Q_
                if ritz: D1 = Q_ @ (Q_.T @ DQ) @ Q_.T          # Rayleigh-Ritz: both legs in the span
                else: D1 = Q_ @ DQ.T; D1 = D1 + D1.T - Q_ @ (Q_.T @ DQ) @ Q_.T          # one-sided: at least one leg in the span
                info.setdefault('basis', {})[l] = int(Q_.shape[1])
                lam_, V_ = t.linalg.eigh((D1 + D1.T) / 2); o_ = lam_.abs().argsort(descending=True)[:r_]; st['Y'] = V_[:, o_]; st['Nt'] = a_ * C + (V_[:, o_] * lam_[o_][None, :]) @ V_[:, o_].T
            if state_rank:      # the state a native carrier would hold: one multiple of C plus a rank-r matrix
                a_ = (st['Nt'] * C).sum() / (C * C).sum(); D_ = st['Nt'] - a_ * C; lam_, V_ = t.linalg.eigh((D_ + D_.T) / 2); o_ = lam_.abs().argsort(descending=True)[:state_rank]
                st['Nt'] = a_ * C + (V_[:, o_] * lam_[o_][None, :]) @ V_[:, o_].T
            if Bk is not None: st['Nt'] = st['Nt'] + Bk
            st['B'] = Bk
            if start <= l <= stop and (strength or passive):      # passive: the matrix is carried and recorded (cd.ncarry_R) but no plane is added
                if forms >= 2: R, co = specific(t, st['Nt'], C, mu.to(f64), forms)
                else:               # only the multiple of C is removed (no C^2 product, no mean forms)
                    off_ = ~t.eye(n, dtype=t.bool); a1_ = (st['Nt'][off_] * C[off_]).sum() / (C[off_] * C[off_]).sum(); R = (st['Nt'] - a1_ * C) * off_; co = a1_.reshape(1) / t.trace(C)
                cd.ncarry_R[l] = R; cd.ncarry_a[l] = (float((st['Nt'] * C).sum() / (C * C).sum() / t.trace(C)), float(co[0]))      # multiple of C: one-form fit, and as fitted with `forms`
                if rank:
                    lam, V_ = t.linalg.eigh((R + R.T) / 2); o = lam.abs().argsort(descending=True)[:rank]; R = (V_[:, o] * lam[o][None, :]) @ V_[:, o].T; R = R - t.diag(t.diagonal(R))
                s = t.diagonal(C).clamp_min(1e-10).sqrt(); rm = strength * (3.0 / n) * R / (s ** 3)[:, None] / s[None, :]
                info[l] = dict(remainder31_over_T31=float(rm.norm() / (((k4v * t.ones(n, dtype=f64)) / s ** 4)[:, None] * (C - t.diag(t.diagonal(C))) / t.outer(s, s)).norm()),
                               iso_coef=float(co[0]), kappa_over_trC=float(st['kap'] / t.trace(C)))
                if strength: rem = rm if rem is None else rem + rm
        return (mu, Cpre, k21_, k4v, k22, rem)
    cd.hooks['pair_in'] = pin_hook
    old_out = cd.hooks.get('state_out')
    def out_hook(l, out, ctx):
        if old_out is not None: out = old_out(l, out, ctx)
        if law_start <= l <= 14 and st['Nt'] is not None:
            C = ctx['Cpre'].to(f64); mu = ctx['mu'].to(f64); m = out[0].detach().to(f64)
            K3V = K31V = None
            if l >= 1 and (conv or k31v):
                u0 = unit_stats(t, C, mu, m)
                if conv: K3V = conversion(t, u0['V'], C, st['pp'], st['lam'], st['star'], cd.ws[l], ctx['bpre'].to(f64), conv, st['first'], st['last']); info.setdefault('cols', {})[l] = st['cols']
                if k31v:
                    CV = C @ u0['V']; rr = ctx['radial'].to(f64) / 3; K31V = rr * (u0['v'][:, None] * CV[None, :] + 2 * C * CV[:, None])
            Cl = C
            if law_ck:           # inside the law the covariance is replaced by its diagonal plus its top-K eigenpairs (an application to a vector then needs no product with C)
                ev_, V_ = eig(st['Cl'] if st.get('Cl') is not None else C); X_ = (V_[:, -law_ck:] * ev_[-law_ck:][None, :]) @ V_[:, -law_ck:].T; Cl = X_ - t.diag(t.diagonal(X_)) + t.diag(t.diagonal(C))
            N, kap, u_ = law(t, Cl, mu, m, st['Nt'], K21=ctx['k21'].to(f64) if (k21 and l >= 1) else None, K3V=K3V, K31V=K31V, skip=skip, ownrow=ownrow)
            W = cd.ws[l + 1]; st['pushed'] = 2 * (W.T @ N @ W); st['kap'] = kap
            if oneside[0] and st.get('Y') is not None: st['TY'] = W.T.to(f64) @ (u_['g'][:, None] * st['Y'])        # the state's columns, transported exactly
            if oneside[0] and oneside[1] and ('G' in basis or 'R' in basis):      # structured basis for the fresh part (lead 11:49Z): the layer's top-K covariance eigenvectors
                UK = eig(st['Cl'] if st.get('Cl') is not None else C)[1][:, -oneside[1]:]; Co_ = C - t.diag(t.diagonal(C))          # gated / row-scaled and pushed through W
                ar_ = u_['d'] * (Co_ @ u_['V']) + (u_['e3'] / 2) * (u_['gam'].mean() * t.diagonal(st['Nt']) + 2 * ((Co_ * Co_) @ u_['gam'])) + u_['S3']
                st['TG'] = W.T.to(f64) @ (u_['g'][:, None] * UK); st['TR'] = W.T.to(f64) @ (ar_[:, None] * UK)
            if bank_exact and K3V is not None:      # the bank channel of this step, kept apart: g g' o (gbar B + K3V), B = the bank part of the carried matrix
                Bc = st.get('B'); Nb = t.outer(u_['g'], u_['g']) * (K3V if Bc is None else K3V + u_['gam'].mean() * Bc); st['pushed_b'] = 2 * (W.T @ Nb @ W)
        return out
    cd.hooks['state_out'] = out_hook
    return cd
