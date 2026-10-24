"""TAG opus-physicist NCAR_BANK (2026-10-02): the BANK CHANNEL of the second-chaos matrix as a carrier module with the interface of the lead's
O/ncar3.py / O/ncar4.py (install(cd, torch, st, k31=..., **arguments); st = the harness side table with st['pp'] = the pushed bank columns of the
current layer, st['lam_in'] = their weights as given to the push, st['old'] = the number of old-tier columns).

What it carries: the part of Ntilde_ij = sum_k kappa4(z_i, z_j, z_k, z_k) that the third cumulant makes and the gates transport,
    Nt(l+1) = 2 W' [ g g' o ( transport x gbar Nt(l) + K3V(l) ) ] W,     K3V = sum_r lam_r (V . pp_r) pp_r pp_r',   V = 2 m (1 - g),  gbar = mean(g - m d),
and nothing else of the one-layer law (no Gaussian births, no tilt row forms, no own (3,1), no K21 terms, no gain term): the arm 'convonly' of
O/physicist/ncarry.py. Use: the K31 extra plane k31 x (3/n) R_ij / (sigma_i^3 sigma_j) through the pair map's remainder31 input, R = Nt minus its
forms (least squares on the off-diagonal: 4 = tr C x C, C^2, mu mu', mu (C mu)' + sym; 3 = without C^2; 2 = tr C x C, mu mu'; 1 = tr C x C only).
Measured (physicist RESULT.md T22; torch, no refit, S38 physics, strength 1, forms 4, all columns, final-layer MSE): six held-out ranks
651 378 424 328 287 641: -5.70 -2.81 -2.60 -3.29 -4.38 -0.98%, pooled -3.27%; network 35 -4.32%.
This is the RESEARCH form (full n x n matrices: one product over all pushed columns and two products with W per layer). The same recursion needs no
n x n product natively: with pp_r(l+1) = W' (alive g o pp_r(l)) (the bank's own transport) it is a second weight per bank column,
    w_r <- 2 (transport x gbar w_r + lam_r (V . pp_r)),   new columns start at 0,   Nt(l+1) = sum_r w_r pp_r(l+1) pp_r(l+1)',
and the old tier keeps it as a second r x r core next to its own.

install(cd, torch, st, k31=1.0, lo=1, hi=14, forms=4, conv=1, transport=1.0, rank=0, columns=False, top=0)
   columns    False: the matrix recursion above (the matrix is transported exactly). True: the NATIVE form, a second weight per bank column: w_r <- 2 (transport x gbar w_r +
              lam_r (V . pp_r)) on the columns pushed into the layer, new columns start at 0, and the matrix of a layer is sum_r w_r pp_r pp_r' on ITS pushed columns (so the
              bank's own transport, alive mask and joins act on it); no state besides the weights.
   top        with columns: m > 0 builds the plane from the m columns with the largest |w_r| |pp_r|^2 only (a rank-m plane from explicit columns; the weights of all columns
              are still carried)
   conv       1 all pushed columns, 2 the young tier only, 3 the old tier only
   transport  factor on the carried part (0 = no state: the matrix of a layer is the one-layer conversion of the layer before)
   rank       r > 0: the plane uses only the r largest |eigenvalue| directions of R (the state stays exact)
cd.ncar_info[l] = dict(plane_over_T31, iso) for the log."""
import math


def specific(t, N, C, mu, forms=4):
    n = N.shape[0]; off = ~t.eye(n, dtype=t.bool)
    if forms <= 1:
        a = (N[off] * C[off]).sum() / (C[off] * C[off]).sum(); return (N - a * C) * off, a
    Cmu = C @ mu
    cols = [(t.trace(C) * C)[off]] + ([(C @ C)[off]] if forms == 4 else []) + [t.outer(mu, mu)[off]] + ([(t.outer(mu, Cmu) + t.outer(Cmu, mu))[off]] if forms >= 3 else [])
    X = t.stack(cols, 1); co = t.linalg.lstsq(X, N[off][:, None]).solution[:, 0]; R = t.zeros_like(N); R[off] = N[off] - X @ co
    return R, co[0] * t.trace(C)


def install(cd, torch, st, k31=1.0, lo=1, hi=14, forms=4, conv=1, transport=1.0, rank=0, columns=False, top=0):
    t = torch; f64 = t.float64; n = 1024; S = dict(pushed=None, w=None); info = {}; cd.ncar_info = info
    old_pin = cd.hooks.get('pair_in')

    def pin_hook(l, pin, ctx):
        if old_pin is not None: pin = old_pin(l, pin, ctx)
        mu, Cpre, k21_, k4v, k22, rem = pin
        if columns and S['w'] is not None and st.get('pp') is not None:      # the matrix of this layer from the weights and the layer's pushed columns
            pp = st['pp'].detach().to(f64); w = S['w'][:pp.shape[1]]
            if top and top < w.numel():
                sel = (w.abs() * (pp * pp).sum(0)).argsort(descending=True)[:top]; pp, w = pp[:, sel], w[sel]
            S['pushed'] = (pp * w[None, :]) @ pp.T
        if S['pushed'] is not None and lo <= l <= hi and k31:
            C = Cpre.detach().to(f64); R, a = specific(t, S['pushed'], C, mu.detach().to(f64), forms)
            if rank:
                lam, V_ = t.linalg.eigh((R + R.T) / 2); o = lam.abs().argsort(descending=True)[:rank]; R = (V_[:, o] * lam[o][None, :]) @ V_[:, o].T; R = R - t.diag(t.diagonal(R))
            s = t.diagonal(C).clamp_min(1e-10).sqrt(); plane = (3.0 / n) * R / (s ** 3)[:, None] / s[None, :]
            info[l] = dict(plane_norm=float(plane.norm()), iso=float(a / t.trace(C)))
            rm = k31 * plane
            rem = rm if rem is None else rem + rm
        return (mu, Cpre, k21_, k4v, k22, rem)
    cd.hooks['pair_in'] = pin_hook
    old_out = cd.hooks.get('state_out')

    def out_hook(l, out, ctx):
        if old_out is not None: out = old_out(l, out, ctx)
        if 1 <= l <= 14 and st.get('pp') is not None and st['pp'].shape[1]:
            C = ctx['Cpre'].detach().to(f64); mu = ctx['mu'].detach().to(f64); m = out[0].detach().to(f64)
            v = t.diagonal(C).clamp_min(1e-12); s = v.sqrt(); al = mu / s; ph = t.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = t.erfc(-al / math.sqrt(2)) / 2
            gb = (g - m * ph / s).mean(); V = 2 * m * (1 - g); old = int(st.get('old', 0))
            first, last = (old, None) if conv == 2 else ((0, old) if conv == 3 else (0, None))
            pp = st['pp'].detach().to(f64)[:, first:last]; lam = st['lam_in'].detach().to(f64)[first:last][:pp.shape[1]]
            if columns:
                ncol = st['pp'].shape[1]; c = t.zeros(ncol, dtype=f64); c[first:last] = lam * (V @ pp)
                w = c if S['w'] is None else c + transport * gb * S['w'][:ncol]
                S['w'] = t.cat((2 * w, t.zeros(max(int(out[3].numel()) - ncol, 0), dtype=f64))); info.setdefault('cols', {})[l] = (ncol, old)
                return out
            N = (pp * (lam * (V @ pp))[None, :]) @ pp.T
            if S['pushed'] is not None and transport: N = N + transport * gb * S['pushed']
            W = cd.ws[l + 1].to(f64); S['pushed'] = 2 * (W.T @ (t.outer(g, g) * N) @ W)
        return out
    cd.hooks['state_out'] = out_hook
    return cd
