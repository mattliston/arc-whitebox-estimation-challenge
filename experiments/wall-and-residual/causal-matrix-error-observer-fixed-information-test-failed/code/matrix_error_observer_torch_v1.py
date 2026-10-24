"""Differentiable implementation of the explicit observer, not full-card AD."""
import torch


def basis(x):
    a, g, p, ms, t3, t4 = (x[:, :, k] for k in (3, 5, 6, 8, 10, 12))
    return torch.stack((g, p, a*p, (a*a-1)*p, ms, a*g,
                        t3*g, t3*p, t3*a*p, t3*ms,
                        t4*g, t4*p, t4*a*p, t4*ms), dim=-1)


def predict(data, theta, feature_scale, memory=True):
    W, C, sigma, x, alive = (data[k] for k in ('weights', 'pre_cov', 'sigma', 'x', 'alive'))
    F = basis(x)/feature_scale
    n = W.shape[1]
    e = torch.zeros(n, dtype=theta.dtype, device=theta.device)
    E = torch.zeros((n, n), dtype=theta.dtype, device=theta.device)
    coef = (theta+theta.transpose(1, 2))/2
    first = 0 if memory else 14
    for l in range(first, len(W)):
        w, c, s = W[l], C[l], sigma[l]
        a, g, p, mg = (x[l, :, k] for k in (3, 5, 6, 7))
        mg = mg*s
        u = w.T@e
        V = w.T@E@w
        z = torch.diagonal(V)
        e = g*u+.5*p*z/s
        if l == len(W)-1:
            break
        dg = p*u/s-a*p*z/(2*s*s)
        E = g[:, None]*g[None, :]*V+c*(dg[:, None]*g[None, :]+g[:, None]*dg[None, :])
        diagonal = (g-mg*p/s)*z+2*mg*(1-g)*u
        E = E+torch.diag_embed(diagonal-torch.diagonal(E))
        if l >= 1:
            f = F[l]
            r = c/(s[:, None]*s[None, :])
            source = f@coef[0]@f.T+(f@coef[1]@f.T)*r+(f@coef[2]@f.T)*(r*r)
            scale = s*alive[l]
            E = E+1e-4*scale[:, None]*scale[None, :]*source
    return e
