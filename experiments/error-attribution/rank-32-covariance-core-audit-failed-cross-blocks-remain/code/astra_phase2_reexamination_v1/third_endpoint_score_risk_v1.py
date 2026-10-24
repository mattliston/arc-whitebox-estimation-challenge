"""Endpoint covariance operator with certified Gaussian score-risk metric."""
import numpy as np
from fourth_plane_pair_span_v1 import PlaneOperator


def metric_action(E, u, beta):
    E = (E + E.T) / 2
    Eu = E @ u
    uEu = float(u @ Eu)
    uu = np.outer(u, u)
    AEA = E - beta * (np.outer(u, Eu) + np.outer(Eu, u)) + beta * beta * uEu * uu
    trace = float(np.trace(E)) - beta * uEu
    return AEA + .5 * trace * (np.eye(len(E)) - beta * uu)


class ScoreRiskOperator(PlaneOperator):
    def __init__(self, H, powers, kernel, u, beta, ridge=.001):
        self.H, self.powers = H, powers
        self.symmetric, self.ridge = False, ridge
        self.shape = (len(powers), H.shape[1], H.shape[1])
        self.scale = float(len(H) ** 2)
        self.covariance_scale = float(np.sqrt(2 * np.mean(kernel * kernel)))
        assert self.covariance_scale > 0 and 0 <= beta < 1
        self.kernel = kernel / self.covariance_scale
        self.u, self.beta = u, beta
        square = H * H
        self.preconditioner = np.stack([
            2 * square.T @ ((r * self.kernel) ** 2) @ square / self.scale + ridge
            for r in powers])

    def plane(self, theta):
        return PlaneOperator.forward(self, theta)

    def forward(self, theta):
        raw = self.kernel * self.plane(theta)
        return raw + raw.T

    def adjoint(self, matrix):
        return PlaneOperator.adjoint(self, self.kernel * (matrix + matrix.T))

    def metric(self, matrix):
        return metric_action(matrix, self.u, self.beta)

    def normal(self, theta):
        return self.adjoint(self.metric(self.forward(theta))) / self.scale + self.ridge * theta

    def fit(self, target, tolerance=1e-7, max_iterations=1000):
        return PlaneOperator.fit(self, self.metric(target), tolerance, max_iterations)
