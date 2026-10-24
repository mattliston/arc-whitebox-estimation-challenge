"""Explicit read-only Gaussian error-memory recurrence; NumPy reference."""
import numpy as np


def analytic_basis(x):
    a, g, p, ms, t3, t4 = (x[:, :, k] for k in (3, 5, 6, 8, 10, 12))
    return np.stack((g, p, a*p, (a*a-1)*p, ms, a*g,
                     t3*g, t3*p, t3*a*p, t3*ms,
                     t4*g, t4*p, t4*a*p, t4*ms), axis=-1)


def tangent(mu, pre, u, V, ndtr):
    s = np.sqrt(np.diag(pre))
    a = mu/s
    pdf = np.exp(-a*a/2)/np.sqrt(2*np.pi)
    g = ndtr(a)
    mg = s*pdf+mu*g
    z = np.diag(V)
    dg = pdf*u/s-a*pdf*z/(2*s*s)
    e = g*u+.5*pdf*z/s
    E = g[:, None]*g[None, :]*V+pre*(dg[:, None]*g[None, :]+g[:, None]*dg[None, :])
    np.fill_diagonal(E, (g-mg*pdf/s)*z+2*mg*(1-g)*u)
    return e, E


def predict(data, theta, feature_scale, memory=True):
    W, C, sigma, x, alive = (data[k] for k in ('weights', 'pre_cov', 'sigma', 'x', 'alive'))
    dtype = theta.dtype
    W, C, sigma, x, alive = (np.asarray(t, dtype=dtype) for t in (W, C, sigma, x, alive))
    F = analytic_basis(x)/np.asarray(feature_scale, dtype=dtype)
    n = W.shape[1]
    e, E = np.zeros(n, dtype), np.zeros((n, n), dtype)
    coef = (theta+theta.transpose(0, 2, 1))/2
    first = 0 if memory else 14
    for l in range(first, len(W)):
        w, c, s = W[l], C[l], sigma[l]
        a, g, p, mg = (x[l, :, k] for k in (3, 5, 6, 7))
        mg = mg*s
        u = w.T@e
        V = w.T@E@w
        z = np.diag(V)
        e = g*u+.5*p*z/s
        if l == len(W)-1:
            break
        dg = p*u/s-a*p*z/(2*s*s)
        E = g[:, None]*g[None, :]*V+c*(dg[:, None]*g[None, :]+g[:, None]*dg[None, :])
        np.fill_diagonal(E, (g-mg*p/s)*z+2*mg*(1-g)*u)
        if l >= 1:
            f = F[l]
            r = c/(s[:, None]*s[None, :])
            source = f@coef[0]@f.T+(f@coef[1]@f.T)*r+(f@coef[2]@f.T)*(r*r)
            scale = s*alive[l]
            E += np.asarray(1e-4, dtype)*scale[:, None]*scale[None, :]*source
    return e
