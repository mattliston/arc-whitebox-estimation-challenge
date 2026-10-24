"""Independent input-only Gaussian trajectory; no bank state mutations."""
import numpy as np
from k3_radial_numpy import gaussian_raw
from radial_mehler import derivatives


def gaussian_step(mu, pre_cov):
    v = np.diag(pre_cov).copy()
    assert np.all(np.isfinite(v)) and np.min(v) > 0
    s, a, pdf, gate, mean, r2, _, _ = gaussian_raw(mu, v)
    A, _ = derivatives(a, pdf, gate, mean/s, 6)
    scale = np.outer(s, s)
    rho = pre_cov/scale
    np.fill_diagonal(rho, 0)
    power = rho.copy()
    cov = np.zeros_like(pre_cov)
    for order in range(1, 7):
        cov += power*np.outer(A[order], A[order])
        power = power*rho/(order+1)
    cov *= scale
    np.fill_diagonal(cov, r2-mean*mean)
    assert np.all(np.isfinite(cov)) and np.min(np.diag(cov)) >= -1e-9
    return mean, cov, dict(sigma=s, alpha=a, gate=gate)


def trajectory(weights):
    depth, n, n2 = weights.shape
    assert n == n2
    mean = np.zeros(n, np.float64)
    cov = np.eye(n, dtype=np.float64)
    history = []
    for layer in range(depth):
        w = weights[layer].astype(np.float32)
        mu = w.astype(np.float64).T@mean
        pre_cov = (w.T@cov.astype(np.float32)@w).astype(np.float64)
        mean, cov, aux = gaussian_step(mu, pre_cov)
        history.append(dict(layer=layer, min_pre_variance=float(np.min(np.diag(pre_cov))),
                            min_post_variance=float(np.min(np.diag(cov))),
                            mean_abs=float(np.mean(np.abs(mean))),
                            covariance_asymmetry=float(np.max(np.abs(pre_cov-pre_cov.T)))))
    return dict(mu=mu, pre_cov=pre_cov, mean=mean, **aux), history


def disagreement(shadow, bank, bank_pre_cov):
    s = bank['sigma'].astype(np.float64)
    alpha = bank['x'][:, 3].astype(np.float64)
    gate = bank['x'][:, 5].astype(np.float64)
    host = bank['host'].astype(np.float64)
    delta_rho = (shadow['pre_cov']-bank_pre_cov)/np.outer(s, s)
    np.fill_diagonal(delta_rho, 0)
    n = len(s)
    features = np.stack(((shadow['mu']-alpha*s)/s,
                         np.log(shadow['sigma']/s),
                         (shadow['mean']-host)/s,
                         shadow['gate']-gate,
                         delta_rho.mean(axis=1),
                         np.sqrt(np.mean(delta_rho*delta_rho, axis=1)),
                         delta_rho@alpha/n,
                         delta_rho@(host/s)/n), axis=1).astype(np.float32)
    assert features.shape == (n, 8) and np.all(np.isfinite(features))
    return features
