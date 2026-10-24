"""Formal Gamma variance-mixture consumer; no labels or state propagation."""
import numpy as np
from radial_mehler import make_nonlinear

CONDITIONAL = make_nonlinear(6, 2, True, True)


def gamma_nodes(radial, count=8):
    if radial == 0:
        return np.ones(1), np.ones(1)
    if not np.isfinite(radial) or radial < 0:
        raise ValueError('A negative/nonfinite radial cumulant is not this ansatz.')
    shape = 3.0 / radial
    i = np.arange(count, dtype=np.float64)
    diagonal = (2*i+shape)/shape
    j = np.arange(1, count, dtype=np.float64)
    off = np.sqrt(j*(j+shape-1))/shape
    jacobi = np.diag(diagonal)+np.diag(off, 1)+np.diag(off, -1)
    nodes, eigenvectors = np.linalg.eigh(jacobi)
    weights = eigenvectors[0]**2
    assert np.min(nodes) > 0 and np.min(weights) >= 0
    return nodes, weights


def pair(mu, covariance, third21, radial, count=8):
    n = len(mu)
    zero = np.zeros(n)
    zero22 = np.zeros_like(covariance)
    nodes, weights = gamma_nodes(radial, count)
    third_normalizer = float(weights @ nodes**1.5)
    means, average_covariance = [], np.zeros_like(covariance)
    for scale, weight in zip(nodes, weights):
        out = CONDITIONAL(mu, scale*covariance,
                          scale**1.5/third_normalizer*third21,
                          zero, zero22, 'covariance')
        means.append(out[0])
        average_covariance += weight*out[1]
    means = np.asarray(means)
    mean = weights @ means
    deviations = means-mean
    average_covariance += (deviations.T*weights) @ deviations
    return mean, average_covariance
