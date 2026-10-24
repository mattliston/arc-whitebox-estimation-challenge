"""TAG opus-physicist NLAW: the one-layer law of the second-chaos matrix Ntilde_ij = sum_k kappa4(z_i, z_j, z_k, z_k), in numpy.
Certified at layer 0 -> 1 (t19k_law_cert.py: 97% of the network-specific part, nothing fitted) and at depth one step from truth
(t19n0_law_depth_partial.py: 88-95%). Everything has its predicted coefficient.

  Ntilde(l+1) = 2 W' N W + 2 kappa C(l+1),     W = weights into layer l+1,  N built at layer l from
     C, mu        pre-activation covariance and mean;  m = post mean (host);  g, d = phi / sigma, gamma = g - m d, V = 2 m (1 - g),
                  S2 = 2 (gamma - g^2), S3, S4 (hub coefficients), e = - alpha phi / v
     Nt           the matrix of layer l (carried);  K21 the third-cumulant plane (row index doubled);  K3V_ab = sum_c V_c kappa3(a,b,c) (optional);
     K31V_ab = sum_c V_c kappa4(a,a,b,c) (optional)
  tilt of the covariance by the norm of the layer's outputs:   P = gbar Nt + 2 C Gamma C + K3V,   kappa = <P, C> / <C, C>   (the pure-gain part)
  off the diagonal
     N_ab = g_a g_b [gbar Nt + K3V + C S2 C]_ab                                   transport, conversion, Gaussian births
          + C_ab [d_a g_b (C V)_a + (e_a / 2) g_b Pv_a] + transpose               mean tilt and variance tilt, Pv_a = gbar Nt_aa + 2 sum_c gamma_c C_ac^2
          + C_ab (g_b S3_a + g_a S3_b)                                            the units' own (3,1) cumulant
          + (d_a / 2) g_b K31V_ab + transpose
          + third cumulant x covariance (the parts a K21 plane gives, see k3c below)
          - kappa g_a g_b C_ab                                                    homogeneity: the gain part is carried by 2 kappa C(l+1) instead
  on the diagonal
     N_aa = S4_a + V_a ((C V)_a - V_a v_a) + sum_c (2 gamma_a gamma_c - 2 g_a^2 g_c^2) C_ac^2 + gamma_a gbar Nt_aa
          + V_a (K21' gamma)_a + gamma_a (K21 V)_a - 2 g_a d_a ((K21 o C) g^2)_a - 2 g_a^2 ((K21' o C)(g d))_a - kappa VarG_a
law(C, mu, m, Nt, K21=None, K3V=None, K31V=None, skip=()) -> (N, kappa);  step(...) = 2 W' N W + 2 kappa C_next."""
import math
import numpy as np


def unit_stats(C, mu, m):
    v = np.diag(C).copy(); s = np.sqrt(v); al = mu / s; ph = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = np.array([math.erfc(-a / math.sqrt(2)) / 2 for a in al]); d = ph / s
    r2 = v * ((1 + al * al) * g + al * ph); r3 = v * s * ((al ** 3 + 3 * al) * g + (al * al + 2) * ph); r4 = v * v * ((al ** 4 + 6 * al * al + 3) * g + (al ** 3 + 5 * al) * ph)
    gam = g - m * d
    return dict(v=v, s=s, al=al, ph=ph, g=g, d=d, gam=gam, V=2 * m * (1 - g), S2=2 * (gam - g * g), S3=3 * (1 - g) * (r2 - 2 * m * m),
                S4=r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2, e3=-al * ph / v, varG=r2 - (s * ph + mu * g) ** 2)


def law(C, mu, m, Nt, K21=None, K3V=None, K31V=None, skip=()):
    n = len(mu); u = unit_stats(C, mu, m); v, s, al, g, d, gam, V, S2, S3, S4, e3 = (u[k] for k in ('v', 's', 'al', 'g', 'd', 'gam', 'V', 'S2', 'S3', 'S4', 'e3'))
    gg = np.outer(g, g); Co = C - np.diag(v); CVo = Co @ V; CV = CVo + v * V; gb = float(gam.mean()); z = np.zeros((n, n))
    CVr = CVo if 'cvown' in skip else CV          # 'cvown': the row form without the unit's own term (c = a), which the own (3,1) term already holds
    K3 = z if K3V is None else K3V
    P = gb * Nt + 2 * (C * gam[None, :]) @ C + K3; kap = 0.0 if 'gain' in skip else float((P * C).sum() / (C * C).sum())
    N = gg * (gb * Nt + K3) if 'nobirth' in skip else gg * (gb * Nt + K3 + (Co * S2[None, :]) @ Co)
    if 'rows' not in skip: N = N + Co * (d * CVr)[:, None] * g[None, :] + Co * g[:, None] * (d * CVr)[None, :]
    if 'vtilt' not in skip:
        Pv = gb * np.diag(Nt) + 2 * ((Co * Co) @ gam); x = Co * ((e3 / 2) * Pv)[:, None] * g[None, :]; N = N + x + x.T
    if 'own31' not in skip: N = N + Co * (S3[:, None] * g[None, :] + g[:, None] * S3[None, :])
    if K31V is not None and 'k31v' not in skip:
        x = (d / 2)[:, None] * K31V * g[None, :]; N = N + x + x.T
    dgn = S4 + V * (CV - V * v) + ((2 * gam[:, None] * gam[None, :] - 2 * (g * g)[:, None] * (g * g)[None, :]) * Co * Co).sum(1) + gam * gb * np.diag(Nt)
    if K21 is not None and 'k3c' not in skip:
        Ko = K21 - np.diag(np.diag(K21)); q1 = d * (1 - g + m * al / s)
        if 'k3off' not in skip:          # off-diagonal K21 terms; 'k3prod' drops only the two groups that need K21 x C products, 'k3x' the elementwise (d d') o Co o K3V
            if 'k3prod' not in skip:
                N = N + gg * ((Ko.T * q1[None, :]) @ Co + (Co * q1[None, :]) @ Ko)
                x = d[:, None] * ((Ko * (S2 / 2)[None, :]) @ Co) * g[None, :]; N = N + x + x.T
            x = d[:, None] * (Co * (Ko.T @ gam)[:, None]) * g[None, :]; N = N + x + x.T
            if 'k3el' not in skip:        # the elementwise products of K21 with per-unit vectors (as matrix-vector products: four with K21)
                N = N + 0.5 * np.outer(d, d) * (Ko * CV[None, :] + Ko.T * CV[:, None])
                x = (e3 / 2)[:, None] * (Ko * CV[:, None]) * g[None, :]; N = N + x + x.T
            if 'k3x' not in skip: N = N + np.outer(d, d) * Co * K3
            x = (e3 / 2)[:, None] * (Co * (Ko @ V)[:, None]) * g[None, :]; N = N + x + x.T
        dgn = dgn + V * (Ko.T @ gam) + gam * (Ko @ V) - 2 * g * d * ((Ko * Co) @ (g * g)) - 2 * g * g * ((Ko.T * Co) @ (g * d))
    N = N - kap * gg * C
    np.fill_diagonal(N, dgn - kap * u['varG'])
    return N, kap


def step(W, C, mu, m, Nt, C_next, **kw):
    N, kap = law(C, mu, m, Nt, **kw)
    return 2 * (W.T @ N @ W) + 2 * kap * C_next, kap


def specific(N, C, mu):
    """network-specific part of a matrix at a layer with covariance C and mean mu (the lead's forms removed)."""
    n = len(mu); off = ~np.eye(n, dtype=bool); dg = np.eye(n, dtype=bool); Cmu = C @ mu
    forms = dict(trC_C=np.trace(C) * C, C2=C @ C, mumu=np.outer(mu, mu), muCmu=np.outer(mu, Cmu) + np.outer(Cmu, mu), eye=np.eye(n)); R = N.copy()
    for mask, names in ((off, ('trC_C', 'C2', 'mumu', 'muCmu')), (dg, ('trC_C', 'C2', 'mumu', 'eye'))):
        X = np.stack([forms[k][mask] for k in names], 1); R[mask] = N[mask] - X @ np.linalg.lstsq(X, N[mask], rcond=None)[0]
    return R
