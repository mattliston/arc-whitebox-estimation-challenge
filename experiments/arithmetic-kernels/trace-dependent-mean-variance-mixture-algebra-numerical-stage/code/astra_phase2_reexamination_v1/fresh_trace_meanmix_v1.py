"""Exact probability-model identities; no labels, fitting, or rollout."""
from itertools import combinations
import numpy as np
from numpy.polynomial import Polynomial


def gamma_moments(q, degree):
    result = np.ones(degree + 1)
    for k in range(1, degree + 1):
        result[k] = result[k - 1] * (1 + (k - 1) * q)
    return result


def conditional_raw_polynomial(indices, a, V, D):
    """Gaussian Wick recursion conditioned on S; polynomial in S."""
    if not indices:
        return Polynomial([1.0])
    i, rest = indices[0], indices[1:]
    result = Polynomial([-a[i], a[i]]) * conditional_raw_polynomial(rest, a, V, D)
    for pos, j in enumerate(rest):
        remaining = rest[:pos] + rest[pos + 1:]
        result += Polynomial([D[i, j], V[i, j]]) * conditional_raw_polynomial(remaining, a, V, D)
    return result


def central_raw(indices, b, B, C, q):
    V = B - np.outer(b, b) / q
    D = C - B
    poly = conditional_raw_polynomial(tuple(indices), b / q, V, D)
    return float(poly.coef @ gamma_moments(q, len(poly.coef) - 1))


def k3_tensor(b, B, q):
    return (b[:, None, None] * B[None, :, :]
            + b[None, :, None] * B[:, None, :]
            + b[None, None, :] * B[:, :, None]
            - np.einsum('i,j,k->ijk', b, b, b) / q)


def k4_entry(indices, b, B, q):
    i, j, k, l = indices
    gaussian = B[i, j] * B[k, l] + B[i, k] * B[j, l] + B[i, l] * B[j, k]
    bbB = 0.0
    for left in combinations(range(4), 2):
        right = [r for r in range(4) if r not in left]
        bbB += b[indices[left[0]]] * b[indices[left[1]]] * B[indices[right[0]], indices[right[1]]]
    return q * gaussian + bbB - 3 * np.prod(b[list(indices)]) / q


def planes(b, B, q):
    """T_ii,j, Q_iii,j, Q_ii,jj and marginal Q_iiii."""
    v = np.diag(B)
    third21 = 2 * b[:, None] * B + v[:, None] * b[None, :] - b[:, None] ** 2 * b[None, :] / q
    fourth31 = (3 * q * v[:, None] * B + 3 * b[:, None] ** 2 * B
                + 3 * (b * v)[:, None] * b[None, :] - 3 * b[:, None] ** 3 * b[None, :] / q)
    fourth22 = (q * (v[:, None] * v[None, :] + 2 * B * B)
                + b[:, None] ** 2 * v[None, :] + v[:, None] * b[None, :] ** 2
                + 4 * b[:, None] * b[None, :] * B - 3 * b[:, None] ** 2 * b[None, :] ** 2 / q)
    return third21, fourth31, fourth22, 3 * q * v * v + 6 * b * b * v - 3 * b ** 4 / q


def directional_cumulants(direction, b, B, q):
    x = float(direction @ b)
    y = float(direction @ B @ direction)
    return {3: 3*x*y-x**3/q,
            4: 3*q*y*y+6*x*x*y-3*x**4/q,
            5: 30*q*x*y*y-6*x**5/q,
            6: 30*q*q*y**3+180*q*x*x*y*y-90*x**4*y}
