"""Offline raw-to-centered third-plane arithmetic with a storage-error bound."""
import numpy as np


def central_plane(mu, m2, raw11, raw21):
    return (raw21.astype(np.float64)-2*mu[:,None]*raw11.astype(np.float64)
            -m2[:,None]*mu[None,:]+2*(mu*mu)[:,None]*mu[None,:])


def raw_f32_rounding_bound(mu, raw11, raw21):
    """Given fixed f64 marginals, bound only each matrix's half-ULP storage error."""
    u11 = .5*np.spacing(np.abs(raw11).astype(np.float32)).astype(np.float64)
    u21 = .5*np.spacing(np.abs(raw21).astype(np.float32)).astype(np.float64)
    return u21+2*np.abs(mu)[:,None]*u11


def gaussian_third_variance(C, count):
    v = np.diag(C)
    return (2*(v*v)[:,None]*v[None,:]+4*v[:,None]*C*C)/count
