"""Fixed target-free primitives for a one-cell nonlinear pair-memory screen."""
import numpy as np


def normalized(a):
    a = np.asarray(a, dtype=np.float64)
    return a / max(float(np.sqrt(np.mean(a*a))), 1e-3)


def offdiag(a):
    result = np.array(a, dtype=np.float64, copy=True)
    np.fill_diagonal(result, 0.)
    return result


def star_action(d, v):
    edge = np.array(d, dtype=np.float64, copy=True)
    np.fill_diagonal(edge, np.diag(d)/3)
    q = edge @ v
    return (v*v).T @ q + 2*(v*q).T @ v


def memory_source(d):
    weighted = np.diag(d)[:, None] * d
    return offdiag((weighted+weighted.T)/2)


def full_cubic_covariance_response(delta_d, covariance, sigma, alpha, pdf, gate):
    """Exact r=0..2 offdiagonal K21 response of the frozen bank pair map.

    delta_d is scaled by sigma_i^2 sigma_j, with zero diagonal. Other
    cumulants, all marginals, and the learned mean-feedback term stay fixed.
    This does not differentiate/recompute the GNN, radial update or next
    nonlinear state. The bank uses original diag(C) in its T21 denominator;
    retain the sigma^2/v factor in case its sigma floor was active.
    """
    delta_d = np.asarray(delta_d, dtype=np.float64)
    variance = np.diag(covariance)
    if (np.any(variance <= 0) or np.any(sigma <= 0)
            or not np.all(np.isfinite(delta_d)) or np.any(np.diag(delta_d) != 0)):
        raise ValueError('Invalid response inputs; no moment repair permitted')
    rho = offdiag(covariance/np.outer(sigma, sigma))
    a1, a2, a3, a4 = gate, pdf, -alpha*pdf, (alpha*alpha-1)*pdf
    coefficient = (np.outer(a2, a1) + rho*np.outer(a3, a2)
                   + .5*rho*rho*np.outer(a4, a3))
    physical = delta_d * (sigma*sigma/variance)[:, None]
    half = .5*np.outer(sigma, sigma)*coefficient*physical
    result = half+half.T
    np.fill_diagonal(result, 0.)
    return result


def next_gaussian_mean_response(delta_c, weight, next_sigma, next_pdf):
    delta_v = np.sum(weight*(delta_c@weight), axis=0)
    return delta_v*next_pdf/(2*next_sigma)
