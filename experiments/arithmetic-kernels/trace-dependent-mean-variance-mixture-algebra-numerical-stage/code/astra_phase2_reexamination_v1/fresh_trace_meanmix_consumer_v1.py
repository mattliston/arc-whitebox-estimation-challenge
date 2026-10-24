"""Paired trace-mixture raw-moment residual; no state updates or targets."""
import numpy as np
from scipy.special import ndtr
from radial_mehler import derivatives
from gamma_radial_pair_v1 import gamma_nodes
from fresh_trace_meanmix_v1 import planes


def gaussian_first_raw_second(mu, C):
    """Same Gaussian Hermite6 correlation truncation as the parent."""
    v = np.diag(C)
    if np.min(v) <= 0:
        raise ValueError('Nonpositive conditional marginal variance.')
    sigma = np.sqrt(v)
    alpha = mu/sigma
    pdf = np.exp(-alpha*alpha/2)/np.sqrt(2*np.pi)
    gate = ndtr(alpha)
    mean = mu*gate+sigma*pdf
    scale = np.outer(sigma, sigma)
    rho = C/scale
    if np.max(np.abs(rho)) > 1+1e-10:
        raise ValueError('A conditional bivariate Gaussian is not PSD.')
    np.fill_diagonal(rho, 0.)
    A, _ = derivatives(alpha, pdf, gate, mean/sigma, 6)
    covariance = np.zeros_like(C)
    rr = np.ones_like(C)
    for r in range(1, 7):
        rr *= rho/r
        covariance += rr*np.outer(A[r], A[r])
    raw2 = covariance*scale+np.outer(mean, mean)
    np.fill_diagonal(raw2, (mu*mu+v)*gate+mu*sigma*pdf)
    return mean, raw2


def edgeworth_first_raw_second(mu, C, K21, K31, K22, k4):
    """Linear-in-cumulants increment; actual K31, all raw local terms."""
    v = np.diag(C)
    if np.min(v) <= 0:
        raise ValueError('Nonpositive reference marginal variance.')
    sigma = np.sqrt(v)
    alpha = mu/sigma
    pdf = np.exp(-alpha*alpha/2)/np.sqrt(2*np.pi)
    gate = ndtr(alpha)
    mean = mu*gate+sigma*pdf
    k3 = np.diag(K21)
    d = pdf/sigma
    e = -alpha*pdf/v
    f = (alpha*alpha-1)*pdf/(v*sigma)
    delta_mean = k3*e/6+k4*f/24
    scale = np.outer(sigma, sigma)
    rho = C/scale
    np.fill_diagonal(rho, 0.)
    T21 = K21/(v[:, None]*sigma[None, :])
    T31 = K31/(sigma[:, None]**3*sigma[None, :])
    T22 = K22/np.outer(v, v)
    t3, t4 = k3/sigma**3, k4/v**2
    A, _ = derivatives(alpha, pdf, gate, mean/sigma, 7)
    delta_raw2 = np.zeros_like(C)
    rr = np.ones_like(C)
    for r in range(3):
        terms = (T21*np.outer(A[r+2], A[r+1])+T21.T*np.outer(A[r+1], A[r+2]))/2
        terms += T22*np.outer(A[r+2], A[r+2])/4
        terms += (T31*np.outer(A[r+3], A[r+1])+T31.T*np.outer(A[r+1], A[r+3]))/6
        # Include r=0 here: these are RAW moments, not covariance.
        terms += (np.outer(t3*A[r+3], A[r])+np.outer(A[r], t3*A[r+3]))/6
        terms += (np.outer(t4*A[r+4], A[r])+np.outer(A[r], t4*A[r+4]))/24
        delta_raw2 += rr*terms
        rr *= rho/(r+1)
    delta_raw2 *= scale
    np.fill_diagonal(delta_raw2, k3*d/3+k4*e/12)
    return delta_mean, delta_raw2


def trace_increment(mu, C, b, B, q, count=8):
    """No clipping, fitting, or silent covariance repair."""
    if q == 0 and not np.any(b):
        return np.zeros_like(mu), np.zeros_like(C)
    if not np.isfinite(q) or q <= 0:
        raise ValueError('This trace mixture requires q>0.')
    a = b/q
    V = B-np.outer(b, b)/q
    D = C-B
    nodes, weights = gamma_nodes(3*q, count)
    delta_mean = np.zeros_like(mu)
    delta_raw2 = np.zeros_like(C)
    for s, weight in zip(nodes, weights):
        mb, rb = gaussian_first_raw_second(mu+a*(s-1), D+s*V)
        mz, rz = gaussian_first_raw_second(mu, D+s*B)
        delta_mean += weight*(mb-mz)
        delta_raw2 += weight*(rb-rz)
    mixed = planes(b, B, q)
    zero = planes(np.zeros_like(b), B, q)
    increments = [x-y for x, y in zip(mixed, zero)]
    em, er = edgeworth_first_raw_second(mu, C, *increments)
    return delta_mean-em, delta_raw2-er
