"""Gaussian reference and analytically centered single-site control, host study."""
import math
import numpy as np
from scipy.special import ndtr

def gaussian_mean(mu, variance):
    mu, variance = np.broadcast_arrays(np.asarray(mu, dtype=np.float64),
                                       np.asarray(variance, dtype=np.float64))
    assert np.all(variance >= 0) and np.isfinite(mu).all() and np.isfinite(variance).all()
    s = np.sqrt(variance)
    a = np.divide(mu, s, out=np.zeros_like(mu), where=s > 0)
    ans = mu*ndtr(a)+s*np.exp(-a*a/2)/math.sqrt(2*math.pi)
    return np.where(s > 0, ans, np.maximum(mu, 0))

def nonnegative_roundoff(value, scale, label):
    value = np.asarray(value, dtype=np.float64)
    tolerance = 512*np.finfo(np.float64).eps*np.maximum(np.asarray(scale), 1e-30)
    assert np.isfinite(value).all() and np.all(value >= -tolerance), (label, float(value.min()))
    report = dict(label=label, minimum=float(value.min()), negatives=int(np.sum(value < 0)),
                  max_tolerance=float(np.max(tolerance)))
    return np.maximum(value, 0), report

def reference(weights, keep_matrices=False, progress=None):
    n = weights.shape[1]
    mprev = np.zeros(n, dtype=np.float64)
    Cprev = np.eye(n, dtype=np.float64)
    Lprev = np.eye(n, dtype=np.float64)
    rows, reports, matrices = [], [], []
    for layer, w0 in enumerate(weights):
        w = np.asarray(w0, dtype=np.float64)
        mu = mprev@w
        Cpre = w.T@(Cprev@w)
        Cpre = (Cpre+Cpre.T)/2
        Lpre = w.T@Lprev
        v = np.diag(Cpre).copy()
        assert (v > 0).all()
        linear_v = np.sum(Lpre*Lpre, axis=1)
        d, report = nonnegative_roundoff(v-linear_v, v+linear_v, f'd{layer}')
        if layer == 0:
            assert np.max(np.abs(v-linear_v)/(v+linear_v)) < 1e-13
            d[:] = 0
        reports.append(report)
        s = np.sqrt(v); a = mu/s
        g = ndtr(a); pdf = np.exp(-a*a/2)/math.sqrt(2*math.pi)
        m = mu*g+s*pdf
        target_var = (mu*mu+v)*g+mu*s*pdf-m*m
        A = (g, pdf, -a*pdf, (a*a-1)*pdf,
             -(a*a*a-3*a)*pdf, (a**4-6*a*a+3)*pdf)
        rho = Cpre/np.outer(s,s)
        power = np.ones_like(rho)
        cov = np.zeros_like(rho)
        diag_sum = np.zeros(n, dtype=np.float64)
        for r, ar in enumerate(A, start=1):
            power = power*rho/r
            cov += power*np.outer(ar, ar)
            diag_sum += ar*ar/math.factorial(r)
        cov *= np.outer(s,s)
        tail, report = nonnegative_roundoff(target_var-v*diag_sum,
                                           v+mu*mu+m*m, f'tail{layer}')
        reports.append(report)
        cov[np.diag_indices(n)] += tail
        if keep_matrices:
            matrices.append(dict(Cpre=Cpre.copy(), Cpost=cov.copy(), Lpre=Lpre.copy(),
                                 noise=cov-np.outer(g,g)*Cpre))
        rows.append(dict(mu=mu, v=v, d=d, linear_v=linear_v, g=g, m=m))
        mprev, Cprev, Lprev = m, cov, g[:,None]*Lpre
        if progress: progress(layer, float(d.min()), float(tail.min()))
    return {k:np.stack([r[k] for r in rows]) for k in rows[0]}, reports, matrices

def coupled(weights, ref, X, keep_layers=False):
    """Both fixed controls on common pairs; production pair stream needs only tavg."""
    plus, minus = X, -X
    linear = X
    tp = np.zeros_like(X); ta = np.zeros_like(X)
    snapshots = []
    for l, w0 in enumerate(weights):
        w = np.asarray(w0, dtype=np.float64)
        if l == 0:
            y = X@w
            plus, minus = np.maximum(y,0), np.maximum(-y,0)
        else:
            plus = np.maximum(plus@w,0)
            minus = np.maximum(minus@w,0)
            y = linear@w
        g, mu, d, m = (ref[k][l] for k in ('g','mu','d','m'))
        Mp = gaussian_mean(mu+y,d)
        Mm = gaussian_mean(mu-y,d)
        rp = Mp-m-g*y
        ra = (Mp+Mm)/2-m
        if l == 0:
            tp, ta = rp, ra
        else:
            tp = (tp@w)*g+rp
            ta = (ta@w)*g+ra
        linear = y*g
        sp, sa = m+linear+tp, m+ta
        result = dict(plain=plus, antithetic=(plus+minus)/2,
                      ordinary_residual=plus-sp, antithetic_residual=(plus+minus)/2-sa,
                      linear_residual=plus-(m+linear))
        if keep_layers: snapshots.append({k:v.copy() for k,v in result.items()})
    return result, snapshots
