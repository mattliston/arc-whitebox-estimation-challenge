"""Input-only approximate Dirichlet Gram; never mutates the bank state.

The first-layer gate formula is exact on an exact Gaussian Gram. Later
gate/gradient factorization and degree-five gate expansion are approximations.
No PSD or true-Jacobian fidelity claim is made for rounded wide bank inputs.
"""
import math

import numpy as np
from scipy.special import ndtr


def covariance_coordinates(mu, raw_cov):
    mu = np.asarray(mu, dtype=np.float64)
    raw_cov = np.asarray(raw_cov, dtype=np.float64)
    assert raw_cov.shape == (len(mu), len(mu))
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(raw_cov))
    cov = (raw_cov + raw_cov.T) * 0.5
    variance = np.diag(cov).copy()
    assert np.min(variance) > 0
    sigma = np.sqrt(variance)
    rho = cov / np.outer(sigma, sigma)
    np.fill_diagonal(rho, 0.0)
    offdiag_max = float(np.max(np.abs(rho)))
    assert offdiag_max < 1.0, offdiag_max
    np.fill_diagonal(rho, 1.0)
    return cov, sigma, mu / sigma, rho, offdiag_max


def zero_mean_gate(rho):
    rho = np.asarray(rho, dtype=np.float64)
    assert np.all(np.isfinite(rho)) and np.max(np.abs(rho)) <= 1.0
    return 0.25 + np.arcsin(rho) / (2.0 * math.pi)


def gate_hermite5(alpha, rho):
    """Six Hermite terms (degrees0..5), with exact Gaussian gate diagonal."""
    alpha = np.asarray(alpha, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    gate = ndtr(alpha)
    pdf = np.exp(-0.5 * alpha * alpha) / math.sqrt(2.0 * math.pi)
    a2 = alpha * alpha
    coefficients = (
        gate, pdf, -alpha * pdf, (a2 - 1.0) * pdf,
        (3.0 * alpha - alpha * a2) * pdf,
        (a2 * a2 - 6.0 * a2 + 3.0) * pdf,
    )
    result = np.outer(gate, gate)
    power = np.ones_like(rho)
    for degree in range(1, 6):
        power = power * rho / degree
        result += power * np.outer(coefficients[degree], coefficients[degree])
    np.fill_diagonal(result, gate)
    assert np.all(np.isfinite(result))
    return result


def transport(gram, weight, product_dtype=np.float32):
    """Production uses FP32 products; FP64 option is for algebra fixtures."""
    w = np.asarray(weight, dtype=product_dtype)
    k = np.asarray(gram, dtype=product_dtype)
    pushed = np.asarray(w.T @ k @ w, dtype=np.float64)
    result = (pushed + pushed.T) * 0.5
    assert np.all(np.isfinite(result)) and np.min(np.diag(result)) > 0
    return result


def observer(weights, pre_covariances, raw_features, sigmas):
    depth, n, n2 = weights.shape
    assert depth == 16 and n == n2
    assert pre_covariances.shape == (depth, n, n)
    assert raw_features.shape[:2] == sigmas.shape == (depth, n)
    assert raw_features.shape[2] == 30
    history = []
    gram = None
    for layer in range(depth):
        if layer > 0:
            pre_gram = transport(gram, weights[layer])
        if layer == depth - 1:
            history.append(dict(layer=layer, outgoing_gate=False,
                                min_pre_gram_diagonal=float(np.min(np.diag(pre_gram))),
                                max_pre_gram_diagonal=float(np.max(np.diag(pre_gram)))))
            return pre_gram, history
        mu = raw_features[layer, :, 3].astype(np.float64) * sigmas[layer].astype(np.float64)
        cov, sigma, alpha, rho, offdiag_max = covariance_coordinates(mu, pre_covariances[layer])
        if layer == 0:
            assert np.count_nonzero(mu) == 0
            pre_gram = cov
            q = zero_mean_gate(rho)
        else:
            q = gate_hermite5(alpha, rho)
        gram = q * pre_gram
        assert np.all(np.isfinite(gram)) and np.min(np.diag(gram)) > 0
        history.append(dict(layer=layer, outgoing_gate=True,
                            max_abs_source_rho=offdiag_max,
                            min_source_variance=float(np.min(sigma * sigma)),
                            min_pre_gram_diagonal=float(np.min(np.diag(pre_gram))),
                            min_post_gram_diagonal=float(np.min(np.diag(gram))),
                            max_post_gram_diagonal=float(np.max(np.diag(gram))),
                            symmetry_error=float(np.max(np.abs(gram - gram.T)))))
    raise AssertionError("unreachable")


def final_features(pre_gram, full_pre_cov, bank_final):
    sigma = bank_final['sigma'].astype(np.float64)
    alpha = bank_final['x'][:, 3].astype(np.float64)
    gate = bank_final['x'][:, 5].astype(np.float64)
    host = bank_final['host'].astype(np.float64)
    assert np.all(np.isfinite(sigma)) and np.min(sigma) > 0
    assert np.all(np.isfinite(pre_gram)) and np.min(np.diag(pre_gram)) > 0
    c = np.asarray(full_pre_cov, dtype=np.float64)
    c = (c + c.T) * 0.5
    scale = np.outer(sigma, sigma)
    rho = c / scale
    difference = (pre_gram - c) / scale
    np.fill_diagonal(rho, 0.0)
    np.fill_diagonal(difference, 0.0)
    n = len(sigma)
    values = np.stack((
        np.log(np.diag(pre_gram) / (sigma * sigma)),
        np.mean(difference, axis=1),
        np.sqrt(np.mean(difference * difference, axis=1)),
        difference @ alpha / n,
        difference @ (host / sigma) / n,
        difference @ gate / n,
        np.mean(difference * rho, axis=1),
        np.mean(difference * rho * rho, axis=1),
    ), axis=1).astype(np.float32)
    assert values.shape == (n, 8) and np.all(np.isfinite(values))
    return values
