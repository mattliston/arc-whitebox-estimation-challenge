"""Offline Torch port of the certified centered scalar and pair identities.

Same moment truncation and original asymmetric-covariance convention.
No NumPy calculation, CPU fallback, fitting or target access in this module.
"""
import math
import torch
from full_carrier_autodiff_v1 import diagonal


def normal_cdf(alpha):
    return torch.erfc(-alpha / math.sqrt(2)) / 2


def gaussian_central(alpha, sigma):
    positive = alpha >= 0
    # abs has the WRONG derivative at zero for this piecewise identity.
    a = torch.where(positive, alpha, -alpha)
    phi = torch.exp(-a*a/2) / math.sqrt(2*math.pi)
    tail = normal_cdf(-a)
    u = phi - a*tail
    r2 = (a*a+1)*tail - a*phi
    r3 = (a*a+2)*phi - a*(a*a+3)*tail
    r4 = (a**4+6*a*a+3)*tail - a*(a*a+5)*phi
    vu = r2-u*u
    c3u = r3-3*u*r2+2*u**3
    c4u = r4-4*u*r3+6*u*u*r2-3*u**4-3*vu*vu
    g = 1-tail
    variance = torch.where(positive, 1-2*tail+vu, vu)
    c3 = torch.where(positive, 3*phi-6*u*g+c3u, c3u)
    c4 = torch.where(positive, -4*a*phi+12*(tail*g-u*phi)+12*g*(2*u*u-r2)+c4u, c4u)
    return variance*sigma**2, c3*sigma**3, c4*sigma**4


def marginals(mu, variance, k3, k4):
    variance = variance.clamp_min(1e-10)
    sigma = variance.sqrt()
    alpha = mu/sigma
    pdf = torch.exp(-alpha*alpha/2) / math.sqrt(2*math.pi)
    gate, upper = normal_cdf(alpha), normal_cdf(-alpha)
    mg = sigma*pdf+mu*gate
    vg, c3g, c4g = gaussian_central(alpha, sigma)
    d = pdf/sigma
    e = -alpha*pdf/variance
    f = (alpha*alpha-1)*pdf/(variance*sigma)
    delta = k3*e/6+k4*f/24
    z2 = k3*(d-mg*e)/3+k4*(e-mg*f)/12
    z3 = k3*(gate-mg*d+.5*mg*mg*e)+k4*(d/4-mg*e/4+mg*mg*f/8)
    z4 = k3*(4*mg*upper+2*mg*mg*d-(2/3)*mg**3*e)
    z4 = z4+k4*(gate-mg*d+.5*mg*mg*e-mg**3*f/6)
    dv = z2-delta*delta
    post_variance = vg+dv
    post_k3 = c3g+z3-3*delta*(vg+z2)+2*delta**3
    post_k4 = c4g+z4-4*delta*(c3g+z3)+6*delta*delta*(vg+z2)-3*delta**4-3*dv*(2*vg+dv)
    mean = mg+delta
    aux = dict(sigma=sigma, alpha=alpha, pdf=pdf, gate=gate, gaussian_mean=mg,
               pre_v=variance, pre_k3=k3, pre_k4=k4, post_v=post_variance,
               post_k3=post_k3, post_k4=post_k4, mean_shift=delta,
               upper_probability=upper, gaussian_variance=vg,
               gaussian_k3=c3g, gaussian_k4=c4g)
    return mean, aux


def nonlinear(mu, cov, k21, radial):
    v = torch.diagonal(cov)
    k3 = torch.diagonal(k21)
    k4 = radial*v*v
    m, aux = marginals(mu, v, k3, k4)
    s, a, pdf, g, mg = (aux[k] for k in ('sigma', 'alpha', 'pdf', 'gate', 'gaussian_mean'))
    zero = torch.zeros_like(s)
    rho = diagonal(cov/torch.outer(s, s), zero)
    t21 = diagonal(k21/(v[:, None]*s[None, :]), zero)
    k22 = radial*(torch.outer(v, v)+2*cov*cov)/3
    t22 = k22/torch.outer(v, v)
    t3, t4 = k3/s**3, k4/v**2
    t31 = t4[:, None]*rho
    A = [mg/s, g, pdf, -a*pdf, (a*a-1)*pdf,
         (-a**3+3*a)*pdf, (a**4-6*a*a+3)*pdf]
    q = m/s
    F = [zero, 2*(aux['upper_probability']*mg/s-g*aux['mean_shift']/s),
         2*(g-q*pdf)]
    F += [2*(A[r-1]-q*A[r]) for r in range(3, 7)]

    def pair(X, Y, asymmetry=False):
        ans = torch.zeros_like(cov)
        rr = torch.ones_like(cov)
        for r in range(7):
            weight = rr-rr.T if asymmetry else rr
            if r:
                ans = ans+weight*torch.outer(X[r], Y[r])
            if r <= 2:
                ec = (t21*torch.outer(X[r+2], Y[r+1])+t21.T*torch.outer(X[r+1], Y[r+2]))/2
                ec = ec+t22*torch.outer(X[r+2], Y[r+2])/4
                ec = ec+(t31*torch.outer(X[r+3], Y[r+1])+t31.T*torch.outer(X[r+1], Y[r+3]))/6
                if r:
                    ec = ec+(torch.outer(t3*X[r+3], Y[r])+torch.outer(X[r], t3*Y[r+3]))/6
                    ec = ec+(torch.outer(t4*X[r+4], Y[r])+torch.outer(X[r], t4*Y[r+4]))/24
                ans = ans+weight*ec
                if asymmetry:
                    ans = ans+rr.T*(t22-t22.T)*torch.outer(X[r+2], Y[r+2])/4
            rr = rr*rho/(r+1)
        return ans

    c = pair(A, A)*torch.outer(s, s)
    post21 = pair(F, A)*torch.outer(v, s)
    post22 = pair(F, F)*torch.outer(v, v)-2*c*c
    asym = pair(A, F, asymmetry=True)*torch.outer(s, v)
    post22 = post22+2*m[:, None]*asym+4*torch.outer(m, m)*(c-c.T)
    post22 = diagonal(post22, aux['post_k4'])
    c = diagonal(c, aux['post_v'])
    post21 = diagonal(post21, aux['post_k3'])
    new_radial = 3*post22.sum()/(torch.trace(c)**2+2*(c*c).sum())
    aux['pre_cov'] = cov
    aux['birth_beta'] = 2*(aux['upper_probability']*mg-g*aux['mean_shift'])
    return m, c, post21, new_radial, aux
