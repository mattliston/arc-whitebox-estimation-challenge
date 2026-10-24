"""TAG opus-physicist T2 library: plane algebra for the third-order budget (numpy float64).
Conventions: W[i, a] maps post unit i of layer l to pre unit a of layer l+1 (h' = x @ W). A (2,1) plane P has P[i, k] = T(i, i, k)
(first index doubled), diagonal P[i, i] = T(i, i, i).
  Pi(W, P)              push of a tensor supported on the plane patterns only: [a, b] = sum_ijk W_ia W_ja W_kb T(i,j,k)
  hub_push(W, p, w)     full push (coincident entries included) of the symmetrised hub tensor T(i,j,k) = H(i,j;k) + H(i,k;j) + H(j,k;i),
                        H(i,j;k) = w_k p_ik p_jk  (legs p[:, k] of hub k; p_kk must be 0)
  hub_plane(p, w)       its plane: P[i, k] = w_k p_ik^2
  hub_D(W, p, w)        distinct-index content pushed to the next plane = hub_push - Pi(hub_plane)
  cp_plane(P, lam)      plane of sum_r lam_r p_r^(x)3
  k21_from_sums(...)    (2,1) cumulant plane from accumulated sums of nearly-centred variables
  feed_push(...)        push of the fourth-cumulant feed tensor F(i,j;k) = (d_k/2) g_i g_j Q(i,j,k,k), from Monte Carlo sums or from a
                        model fourth cumulant given by its (2,2)-type cross plane
All formulas are certified against brute force in t2_cert.py."""
import numpy as np


def Pi(W, P):
    A = 3 * P.T - 2 * np.diag(np.diag(P)); wt = W.T; pushed = wt @ A
    res = (2 * ((pushed * wt) @ W) + (wt * wt) @ pushed.T) / 3
    np.fill_diagonal(res, (pushed * wt * wt).sum(1)); return res


def cp_plane(P, lam):
    return ((P * P) * lam[None, :]) @ P.T


def hub_push(W, p, w):
    U = W.T @ p                                   # U[a, k] = sum_i W_ia p_ik
    return (U * U) @ (w[:, None] * W) + 2 * ((W.T * w[None, :]) * U) @ U.T


def hub_plane(p, w):
    return (p * p) * w[None, :]


def hub_D(W, p, w):
    return hub_push(W, p, w) - Pi(W, hub_plane(p, w))


def k21_from_sums(M21, S1, S2, cov):
    """K21[i,j] = E[u_i^2 u_j] - 2 m_i E[u_i u_j] - m_j E[u_i^2] + 2 m_i^2 m_j with E[u_i u_j] = cov_ij + m_i m_j (cov: reference
    covariance; it multiplies the residual mean m only)."""
    m = S1
    return M21 - 2 * m[:, None] * (cov + np.outer(m, m)) - S2[:, None] * m[None, :] + 2 * (m * m)[:, None] * m[None, :]


def feed_push(W, d, A1, A2, A3, vy, vz, Sq, covy):
    """full push of the symmetrised feed tensor from sums of centred variables y (gate-linear next pre-activation) and z:
    A1[a,k] = E[y_a^2 z_k^2], A2[a,k] = E[y_a z_k], A3[a,b] = E[y_a q_a y_b] (q = (d o z^2) W - reference mean), vy = E[y^2], vz = E[z^2],
    Sq = E[q] (residual), covy = reference Cov(y_a, y_b)."""
    k4 = A1 - np.outer(vy, vz) - 2 * A2 * A2                              # kappa4(y_a, y_a, z_k, z_k)
    term1 = k4 @ ((d / 2)[:, None] * W)
    term2 = A3 - covy * Sq[:, None] - 2 * (A2 * (W.T * d[None, :])) @ A2.T
    return term1 + term2


def feed_plane(d, g, K22, K31, k4):
    """plane of the feed tensor: T(i,i,k) = (1/2) d_k g_i^2 K22[i,k] + d_i g_i g_k K31[i,k]; T(i,i,i) = (3/2) d_i g_i^2 k4_i."""
    P = 0.5 * (g * g)[:, None] * K22 * d[None, :] + (d * g)[:, None] * K31 * g[None, :]
    np.fill_diagonal(P, 1.5 * d * g * g * k4); return P


def feed_push_model(W, d, g, C, c22, c11):
    """the same push for a MODEL fourth cumulant Q(i,j,k,k) = c22 * C_ij v_k + c11 * C_ik C_jk (isotropic form: c22 = radial/3,
    c11 = 2 radial/3). y = (g o z) W."""
    v = np.diag(C); G = (g[:, None] * W)                                    # y_a = sum_i z_i G_ia
    CG = C @ G; covy = G.T @ CG; vy = np.diag(covy); A2 = CG.T               # A2[a,k] = Cov(y_a, z_k)
    k4 = c22 * np.outer(vy, v) + c11 * A2 * A2
    term1 = k4 @ ((d / 2)[:, None] * W)
    q0 = (v * d) @ W
    term2 = c22 * covy * q0[:, None] + c11 * (A2 * (W.T * d[None, :])) @ A2.T
    return term1 + term2


def feed_plane_model(d, g, C, c22, c11):
    v = np.diag(C)
    K22 = c22 * np.outer(v, v) + c11 * C * C; K31 = (c22 + c11) * v[:, None] * C; k4 = (c22 + c11) * v * v
    return feed_plane(d, g, K22, K31, k4)
