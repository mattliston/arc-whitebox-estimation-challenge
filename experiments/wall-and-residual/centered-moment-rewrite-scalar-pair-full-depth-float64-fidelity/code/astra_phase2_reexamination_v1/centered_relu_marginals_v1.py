"""Equivalent ReLU Edgeworth marginals using centered Gaussian corrections."""
import numpy as np
from scipy.special import ndtr


def gaussian_central(alpha, sigma):
    """Gaussian ReLU central cumulants, using the smaller reflected tail."""
    a = np.abs(alpha)
    phi = np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
    tail = ndtr(-a)
    # Raw moments of ReLU(N(-a,1)); every term is tail-sized.
    u = phi - a * tail
    r2 = (a*a + 1) * tail - a * phi
    r3 = (a*a + 2) * phi - a * (a*a + 3) * tail
    r4 = (a**4 + 6*a*a + 3) * tail - a * (a*a + 5) * phi
    vu = r2 - u*u
    c3u = r3 - 3*u*r2 + 2*u**3
    c4u = r4 - 4*u*r3 + 6*u*u*r2 - 3*u**4 - 3*vu*vu
    positive = alpha >= 0
    g = 1 - tail
    variance = np.where(positive, 1 - 2*tail + vu, vu)
    c3 = np.where(positive, 3*phi - 6*u*g + c3u, c3u)
    c4 = np.where(positive, -4*a*phi + 12*(tail*g-u*phi) + 12*g*(2*u*u-r2) + c4u, c4u)
    return variance * sigma**2, c3 * sigma**3, c4 * sigma**4


def marginals(mu, variance, k3, k4):
    variance = np.maximum(variance, 1e-10)
    sigma = np.sqrt(variance)
    alpha = mu / sigma
    pdf = np.exp(-alpha * alpha / 2) / np.sqrt(2 * np.pi)
    gate = ndtr(alpha)
    upper = ndtr(-alpha)
    mg = sigma * pdf + mu * gate
    vg, c3g, c4g = gaussian_central(alpha, sigma)
    d = pdf / sigma
    e = -alpha * pdf / variance
    f = (alpha * alpha - 1) * pdf / (variance * sigma)
    delta = k3 * e / 6 + k4 * f / 24
    # Corrections to powers centered about the fixed Gaussian mean mg.
    z2 = k3 * (d - mg*e) / 3 + k4 * (e - mg*f) / 12
    z3 = k3 * (gate - mg*d + .5*mg*mg*e) + k4 * (d/4 - mg*e/4 + mg*mg*f/8)
    z4 = k3 * (4*mg*upper + 2*mg*mg*d - (2/3)*mg**3*e)
    z4 += k4 * (gate - mg*d + .5*mg*mg*e - mg**3*f/6)
    dv = z2 - delta*delta
    post_variance = vg + dv
    post_k3 = c3g + z3 - 3*delta*(vg+z2) + 2*delta**3
    post_k4 = c4g + z4 - 4*delta*(c3g+z3) + 6*delta*delta*(vg+z2) - 3*delta**4 - 3*dv*(2*vg+dv)
    mean = mg + delta
    aux = dict(sigma=sigma, alpha=alpha, pdf=pdf, gate=gate, gaussian_mean=mg,
               pre_v=variance, pre_k3=k3, pre_k4=k4, post_v=post_variance,
               post_k3=post_k3, post_k4=post_k4, mean_shift=delta,
               upper_probability=upper, gaussian_variance=vg,
               gaussian_k3=c3g, gaussian_k4=c4g)
    return mean, aux
