"""Inert FP64 virtual-location reference for the current409f node features.

This is an OFFLINE diagnostic, not a submitted estimator.  It differentiates
a continuous extension of the existing formula: casts are identities, fixed
FP32 W statistics are constants, and all other arithmetic is FP64.  C, K21,
k4, K22 and the previous GNN state stay fixed.  Entry (i,k) of each feature
jet differentiates feature k of node i with respect to mu_i only.  In
particular the covariance jet shifts its LEFT endpoint; its diagonal is
replaced by the TOTAL scalar-variance derivative.
"""
from dataclasses import dataclass
from math import erfc, sqrt

import numpy as np


def normal_cdf(value):
    """Stable Gaussian CDF through libm; OFFPY intentionally has no SciPy."""
    value = np.asarray(value, np.float64)
    return np.fromiter((.5 * erfc(-float(t) / sqrt(2)) for t in value.flat),
                       dtype=np.float64, count=value.size).reshape(value.shape)


@dataclass
class Jet:
    __array_priority__ = 1000

    value: np.ndarray
    first: np.ndarray
    second: np.ndarray

    @staticmethod
    def constant(value):
        value = np.asarray(value, dtype=np.float64)
        return Jet(value, np.zeros_like(value), np.zeros_like(value))

    def __add__(self, other):
        other = other if isinstance(other, Jet) else Jet.constant(other)
        return Jet(self.value + other.value, self.first + other.first,
                   self.second + other.second)

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.value, -self.first, -self.second)

    def __sub__(self, other):
        return self + (-other if isinstance(other, Jet) else -np.asarray(other))

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, Jet) else Jet.constant(other)
        return Jet(self.value * other.value,
                   self.first * other.value + self.value * other.first,
                   self.second * other.value + 2 * self.first * other.first
                   + self.value * other.second)

    __rmul__ = __mul__

    def __pow__(self, power):
        value = self.value ** power
        if power == 0:
            return Jet.constant(value)
        d = power * self.value ** (power - 1)
        dd = np.zeros_like(value) if power == 1 else (
            power * (power - 1) * self.value ** (power - 2))
        return Jet(value, d * self.first,
                   dd * self.first * self.first + d * self.second)

    def __truediv__(self, other):
        if isinstance(other, Jet):
            return self * (other ** -1)
        return Jet(self.value / other, self.first / other, self.second / other)


def exp(x):
    value = np.exp(x.value)
    return Jet(value, value * x.first,
               value * (x.second + x.first * x.first))


def cdf(x):
    density = np.exp(-.5 * x.value * x.value) / np.sqrt(2 * np.pi)
    return Jet(normal_cdf(x.value), density * x.first,
               density * (x.second - x.value * x.first * x.first))


def floor(x, threshold):
    """Zero derivatives below OR exactly at the floor; report boundaries upstream."""
    active = x.value > threshold
    return Jet(np.maximum(x.value, threshold), x.first * active,
               x.second * active)


def clip(x, lower, upper):
    active = (x.value > lower) & (x.value < upper)
    return Jet(np.clip(x.value, lower, upper), x.first * active,
               x.second * active)


def row_mean(x):
    return Jet(*(np.mean(t, axis=1) for t in (x.value, x.first, x.second)))


def raw_jets(mu, pre_cov, k21, k4):
    """Exact continuous derivatives of the bank's scalar source, including its floor."""
    v = np.diag(pre_cov).copy()
    sv = np.maximum(v, 1e-10)
    sigma = np.sqrt(sv)
    shift = Jet(np.asarray(mu, np.float64), np.ones_like(mu), np.zeros_like(mu))
    alpha = shift / sigma
    density = exp(-.5 * alpha * alpha) / np.sqrt(2 * np.pi)
    gate = cdf(alpha)
    mg = sigma * density + shift * gate
    r2g = (shift * shift + sv) * gate + shift * sigma * density
    r3g = shift * (shift * shift + 3 * sv) * gate + (shift * shift + 2 * sv) * sigma * density
    r4g = (shift ** 4 + 6 * shift * shift * sv + 3 * sv * sv) * gate
    r4g = r4g + shift * (shift * shift + 5 * sv) * sigma * density
    d = density / sigma
    e = -alpha * density / v
    f = (alpha * alpha - 1) * density / (v * sigma)
    k3 = np.diag(k21).copy()
    mean = mg + k3 * e / 6 + k4 * f / 24
    r2 = r2g + k3 * d / 3 + k4 * e / 12
    r3 = r3g + k3 * gate + k4 * d / 4
    r4 = r4g + 4 * k3 * mg + k4 * gate
    var = r2 - mean * mean
    post_k3 = r3 - 3 * mean * r2 + 2 * mean ** 3
    post_k4 = r4 - 4 * mean * r3 + 6 * mean * mean * r2 - 3 * mean ** 4 - 3 * var * var
    return dict(sigma=sigma, alpha=alpha, pdf=density, gate=gate,
                gaussian_mean=mg, pre_v=v, pre_cov=pre_cov, pre_k3=k3,
                pre_k4=np.broadcast_to(k4, (len(mu),)).copy(), mean=mean,
                post_v=var, post_k3=post_k3, post_k4=post_k4)


def hermite_response(alpha, density, gate, mean_over_sigma, order=8):
    table = [mean_over_sigma, gate]
    hm2, hm1 = np.ones_like(alpha), alpha
    for r in range(2, order + 1):
        if r == 2:
            h = hm2
        elif r == 3:
            h = hm1
        else:
            h = alpha * hm1 - (r - 3) * hm2
            hm2, hm1 = hm1, h
        table.append((-1) ** (r - 2) * h * density)
    return table


def covariance_left_jets(pre_cov, k21, k4, k22, aux):
    """Same covariance polynomial, three channels shifted 0,1,2 on the left."""
    sigma, v = aux['sigma'], aux['pre_v']
    rho = pre_cov / np.outer(sigma, sigma)
    np.fill_diagonal(rho, 0)
    t21 = k21 / (v[:, None] * sigma[None, :])
    np.fill_diagonal(t21, 0)
    t22 = k22 / np.outer(v, v)
    t3 = aux['pre_k3'] / sigma ** 3
    t4 = np.broadcast_to(k4, (len(sigma),)) / v ** 2
    t31 = t4[:, None] * rho
    a = hermite_response(aux['alpha'].value, aux['pdf'].value,
                        aux['gate'].value, aux['gaussian_mean'].value / sigma)
    ans = np.zeros((3, len(sigma), len(sigma)), np.float64)
    rr = np.ones_like(pre_cov)
    for r in range(7):
        for j in range(3):
            if r:
                ans[j] += rr * np.outer(a[r+j], a[r])
            if r <= 2:
                ec = .5 * (t21 * np.outer(a[r+2+j], a[r+1])
                           + t21.T * np.outer(a[r+1+j], a[r+2]))
                ec += .25 * t22 * np.outer(a[r+2+j], a[r+2])
                ec += (t31 * np.outer(a[r+3+j], a[r+1])
                       + t31.T * np.outer(a[r+1+j], a[r+3])) / 6
                if r:
                    ec += (np.outer(t3 * a[r+3+j], a[r])
                           + np.outer(a[r+j], t3 * a[r+3])) / 6
                    ec += (np.outer(t4 * a[r+4+j], a[r])
                           + np.outer(a[r+j], t4 * a[r+4])) / 24
                ans[j] += rr * ec
        if r < 6:
            rr = rr * rho / (r + 1)
    ans[0] *= np.outer(sigma, sigma)
    ans[1] *= sigma[None, :]
    ans[2] *= sigma[None, :] / sigma[:, None]
    var = aux['post_v']
    for target, diagonal in zip(ans, (var.value, var.first, var.second)):
        np.fill_diagonal(target, diagonal)
    return Jet(*ans)


def features(l, mu, pre_cov, k21, k4, k22, weights):
    """Return x, diagonal dx, diagonal d²x and scalar/covariance shadow state."""
    n = len(mu)
    aux = raw_jets(mu, pre_cov, k21, k4)
    sigma = aux['sigma']
    v = sigma * sigma
    cov = covariance_left_jets(pre_cov, k21, k4, k22, aux)
    rho = pre_cov / np.outer(sigma, sigma)
    skplane = k21 / (v[:, None] * sigma[None, :])
    post_v = floor(aux['post_v'], 1e-12)
    invscale = post_v ** -.5
    denominator = invscale.value[:, None] * invscale.value[None, :]
    ratio = Jet(cov.value * denominator,
                cov.first * denominator + cov.value * invscale.first[:, None] * invscale.value[None, :],
                cov.second * denominator
                + 2 * cov.first * invscale.first[:, None] * invscale.value[None, :]
                + cov.value * invscale.second[:, None] * invscale.value[None, :])
    total_diagonal = aux['post_v'] / post_v
    for matrix, diagonal in zip((ratio.value, ratio.first, ratio.second),
                                (total_diagonal.value, total_diagonal.first, total_diagonal.second)):
        np.fill_diagonal(matrix, diagonal)
    ratio = clip(ratio, -1, 1)
    mean_square = row_mean(ratio * ratio)
    # A zero-RMS row has the explicitly chosen zero jet, not division by zero.
    nonzero_rms = mean_square.value > 0
    safe_square = Jet(np.where(nonzero_rms, mean_square.value, 1.),
                      np.where(nonzero_rms, mean_square.first, 0.),
                      np.where(nonzero_rms, mean_square.second, 0.))
    rms = safe_square ** .5
    rms.value = np.sqrt(mean_square.value)
    rms.first = np.where(nonzero_rms, rms.first, 0)
    rms.second = np.where(nonzero_rms, rms.second, 0)
    w = weights.astype(np.float32)
    w2 = w*w
    w3 = w2*w
    w4 = w2*w2
    ones = np.ones(n)
    mean = aux['mean']
    gaussian_mean = aux['gaussian_mean']
    const = Jet.constant
    gate_message = Jet(rho @ aux['alpha'].value / n,
                       np.diag(rho) / (n*sigma), np.zeros(n))
    global_mean = Jet(ones * np.mean(mean.value / sigma),
                      mean.first / (n*sigma), mean.second / (n*sigma))
    vals = (const(ones * (2*l/15-1)), const(ones*(l/16)), const(ones),
            aux['alpha'], const(np.log(sigma)-np.mean(np.log(sigma))),
            aux['gate'], aux['pdf'], gaussian_mean/sigma, mean/sigma,
            (mean-gaussian_mean)/sigma, const(aux['pre_k3']/(v*sigma)),
            aux['post_v']/v, const(aux['pre_k4']/(v*v)),
            const(np.mean(rho,axis=1)), const(np.sqrt(np.mean(rho*rho,axis=1))),
            const(np.mean(np.abs(rho),axis=1)),
            const(np.sum(w,axis=0)*.7071067811865475),
            const(np.sum(w2-.001953125,axis=0)*11.313708498984761),
            const(np.sum(w3,axis=0)*93.47798314754834),
            const(np.sum(w4-1.1444091796875e-05,axis=0)*836.0924988699916),
            aux['post_k3']/(v*sigma), aux['post_k4']/(v*v),
            const(np.sqrt(np.sum(skplane*skplane,axis=1))),
            const(np.sqrt(np.sum(skplane*skplane,axis=0))),
            const(np.sum(skplane,axis=1)/np.sqrt(n)),
            const(np.sum(skplane,axis=0)/np.sqrt(n)), gate_message,
            rms, row_mean(ratio), global_mean)
    result = Jet(*(np.stack([getattr(q,k) for q in vals],axis=1)
                   for k in ('value','first','second')))
    boundaries = dict(pre_sigma_floor=int(np.count_nonzero(aux['pre_v']==1e-10)),
                      post_variance_floor=int(np.count_nonzero(aux['post_v'].value==1e-12)),
                      diagonal_lower_clip=int(np.count_nonzero(aux['post_v'].value==-1e-12)),
                      negative_post_variance=int(np.count_nonzero(aux['post_v'].value<0)),
                      offdiagonal_post_clip=int(np.count_nonzero(
                          (np.abs(cov.value*denominator)==1)&~np.eye(n,dtype=bool))),
                      zero_rms=int(np.count_nonzero(mean_square.value==0)))
    return result, aux, cov, boundaries
