"""Directed-rho exact bank response; all predictor primitives remain frozen v1.

This additive version does not symmetrize a stored covariance or its output.
It is algebraically the full r=0..2 offdiagonal K21 action of radial_mehler,
including when finite-precision covariance has rho_ij != rho_ji.
"""
import numpy as np

from pair_memory_action_physics_v1 import (
    memory_source, next_gaussian_mean_response, normalized, offdiag, star_action,
)


def full_cubic_covariance_response(delta_d, covariance, sigma, alpha, pdf, gate):
    """Hold all marginals, other cumulants, and learned mean feedback fixed.

    Standardized delta_d uses sigma_i**2*sigma_j units, whereas the bank
    divides its T21 by original variance_i*sigma_j. No variance, symmetry,
    PSD, output-domain, or response repair is performed here.
    """
    delta_d = np.asarray(delta_d, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    sigma, alpha, pdf, gate = (np.asarray(x, dtype=np.float64)
                               for x in (sigma, alpha, pdf, gate))
    if (delta_d.ndim != 2 or delta_d.shape[0] != delta_d.shape[1]
            or covariance.shape != delta_d.shape):
        raise ValueError("Expected equal square delta/covariance matrices")
    n = delta_d.shape[0]
    arrays = (delta_d, covariance, sigma, alpha, pdf, gate)
    if (any(value.shape != (n,) for value in (sigma, alpha, pdf, gate))
            or not all(np.isfinite(value).all() for value in arrays)):
        raise ValueError("Invalid/nonfinite exact-response inputs")
    variance = np.diag(covariance)
    if (np.any(variance <= 0) or np.any(sigma <= 0)
            or np.any(pdf < 0) or np.any(gate < 0) or np.any(gate > 1)
            or np.any(np.diag(delta_d) != 0)):
        raise ValueError("Invalid exact-response domain; no moment repair permitted")
    rho = offdiag(covariance/np.outer(sigma, sigma))
    a1, a2, a3, a4 = gate, pdf, -alpha*pdf, (alpha*alpha-1)*pdf
    forward = (np.outer(a2, a1) + rho*np.outer(a3, a2)
               + .5*rho*rho*np.outer(a4, a3))
    # Endpoint derivatives swap, but the SAME rho_ij remains at output (i,j).
    reverse = (np.outer(a1, a2) + rho*np.outer(a2, a3)
               + .5*rho*rho*np.outer(a3, a4))
    physical = delta_d * (sigma*sigma/variance)[:, None]
    result = .5*np.outer(sigma, sigma)*(forward*physical + reverse*physical.T)
    np.fill_diagonal(result, 0.)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite exact response; no output repair permitted")
    return result
