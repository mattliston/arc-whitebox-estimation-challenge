"""Virtual-location diagonal derivatives of the existing30 node features."""
import numpy as np
from radial_mehler import derivatives


def feature_diagonal(mu, pre, K21, k4, k22, output):
    m, cov, _, _, aux = output
    n = len(mu)
    s, a, pdf, g = (aux[k] for k in ('sigma', 'alpha', 'pdf', 'gate'))
    v = s*s
    k3 = np.diag(K21)
    A4 = (a*a-1)*pdf/s**3
    A5 = (-a**3+3*a)*pdf/s**4
    dm = g+k3*A4/6+k4*A5/24
    dv = 2*m*(1-dm)
    dk3 = 3*(1-dm)*(aux['post_v']-m*m)
    dk4 = 4*(1-dm)*(aux['post_k3']-3*m*aux['post_v']+m**3)
    rho = pre/np.outer(s, s)
    diagonal_rho = np.diag(rho).copy()
    np.fill_diagonal(rho, 0)
    T21 = K21/(v[:, None]*s[None, :])
    np.fill_diagonal(T21, 0)
    T22 = k22/np.outer(v, v)
    t3, t4 = k3/s**3, k4/v**2
    T31 = t4[:, None]*rho
    A, _ = derivatives(a, pdf, g, aux['gaussian_mean']/s, 7)
    response = np.zeros_like(pre)
    rr = np.ones_like(pre)
    for r in range(7):
        if r:
            response += rr*np.outer(A[r+1], A[r])
        if r <= 2:
            ec = (T21*np.outer(A[r+3], A[r+1])+
                  T21.T*np.outer(A[r+2], A[r+2]))/2
            ec += T22*np.outer(A[r+3], A[r+2])/4
            ec += (T31*np.outer(A[r+4], A[r+1])+
                   T31.T*np.outer(A[r+2], A[r+3]))/6
            if r:
                ec += (np.outer(t3*A[r+4], A[r])+
                       np.outer(A[r+1], t3*A[r+3]))/6
                ec += (np.outer(t4*A[r+5], A[r])+
                       np.outer(A[r+1], t4*A[r+4]))/24
            response += rr*ec
        rr = rr*rho/(r+1)
    # Partial derivative with respect to the ROW's mu, not simultaneous
    # shifts of both endpoints. Restore the total diagonal derivative.
    dc = response*s[None, :]
    np.fill_diagonal(dc, dv)
    pv = np.maximum(aux['post_v'], 1e-12)
    scale = np.sqrt(pv)
    normalized = cov/np.outer(scale, scale)
    dlogv = np.where(aux['post_v'] > 1e-12, dv/pv, 0)
    dnormalized = dc/np.outer(scale, scale)-.5*normalized*dlogv[:, None]
    dnormalized[np.diag_indices(n)] -= .5*np.diag(normalized)*dlogv
    dnormalized *= (normalized > -1) & (normalized < 1)
    normalized = np.clip(normalized, -1, 1)
    row_rms = np.sqrt(np.mean(normalized*normalized, axis=1))
    dx = np.zeros((n, 30), np.float64)
    dx[:, 3] = 1/s
    dx[:, 5] = pdf/s
    dx[:, 6] = -a*pdf/s
    dx[:, 7] = g/s
    dx[:, 8] = dm/s
    dx[:, 9] = (dm-g)/s
    dx[:, 11] = dv/v
    dx[:, 20] = dk3/s**3
    dx[:, 21] = dk4/s**4
    dx[:, 26] = diagonal_rho/(n*s)
    dx[:, 27] = np.sum(normalized*dnormalized, axis=1)/(n*np.maximum(row_rms, 1e-30))
    dx[:, 28] = np.mean(dnormalized, axis=1)
    dx[:, 29] = dm/(n*s)
    return dx, dm, dc
