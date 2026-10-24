"""Causal row/cohort gain tangent with exact third-pair response templates."""
import numpy as np
from fourth_plane_pair_span_v1 import PlaneOperator
from third_endpoint_score_risk_v1 import metric_action


def cohort_planes(p, lam, counts, active):
    n = len(p)
    out = np.zeros((len(counts), n, n), np.float64)
    offset = 0
    for b, count in enumerate(counts):
        stop = offset + 2 * int(count)
        block = p[np.ix_(active, np.arange(offset, stop))].astype(np.float64)
        out[b][np.ix_(active, active)] = (block * block) @ (block * lam[offset:stop].astype(np.float64)).T
        np.fill_diagonal(out[b], 0)
        offset = stop
    assert offset == p.shape[1] == len(lam)
    return out


def plane_action(planes, gains, finite=False):
    if finite:
        return np.sum(planes * ((1 + gains.T[:, :, None]) ** 2 * (1 + gains.T[:, None, :]) - 1), axis=0)
    return np.sum(planes * (2 * gains.T[:, :, None] + gains.T[:, None, :]), axis=0)


class CohortGainOperator(PlaneOperator):
    def __init__(self, H, standardized_planes, kernel, u, beta, ridge=.001):
        self.H, self.u, self.beta, self.ridge = H, u, beta, ridge
        raw = standardized_planes * kernel[None, :, :]
        D = 2 * raw + raw.transpose(0, 2, 1)
        self.covariance_scale = float(np.sqrt(2 * np.mean(np.sum(D * D, axis=0))))
        assert self.covariance_scale > 0
        self.templates = D / self.covariance_scale
        self.shape = (H.shape[1], len(D))
        self.scale = float(len(H) ** 2)
        self.preconditioner = 2 * (H * H).T @ np.sum(self.templates * self.templates, axis=2).T / self.scale + ridge

    def gains(self, theta):
        return self.H @ theta

    def forward(self, theta):
        raw = np.einsum('bij,ib->ij', self.templates, self.gains(theta), optimize=False)
        return raw + raw.T

    def adjoint(self, matrix):
        rows = np.einsum('bij,ij->ib', self.templates, matrix + matrix.T, optimize=False)
        return self.H.T @ rows

    def metric(self, matrix):
        return metric_action(matrix, self.u, self.beta)

    def normal(self, theta):
        return self.adjoint(self.metric(self.forward(theta))) / self.scale + self.ridge * theta

    def fit(self, target, tolerance=1e-7, max_iterations=1000):
        return PlaneOperator.fit(self, self.metric(target), tolerance, max_iterations)
