#!/usr/bin/env python3
"""Exact small-width checks for the Gaussian (1,2,3) diagram shell."""

from __future__ import annotations

import itertools
import math

import numpy as np


def diagram(p: int, q: int, r: int, rij: float, rik: float, rjk: float) -> float:
    xij = (p + q - r) // 2
    xik = (p + r - q) // 2
    xjk = (q + r - p) // 2
    if (
        p + q + r & 1
        or min(xij, xik, xjk) < 0
        or 2 * xij != p + q - r
        or 2 * xik != p + r - q
        or 2 * xjk != q + r - p
    ):
        return 0.0
    numerator = math.factorial(p) * math.factorial(q) * math.factorial(r)
    denominator = (
        math.factorial(xij)
        * math.factorial(xik)
        * math.factorial(xjk)
    )
    return numerator / denominator * rij**xij * rik**xik * rjk**xjk


def explicit_directional(c: np.ndarray, correlation: np.ndarray, weight: np.ndarray) -> float:
    n = weight.size
    value = 0.0
    for p, q, r in itertools.permutations((1, 2, 3)):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    value += (
                        weight[i] * c[p, i] / math.factorial(p)
                        * weight[j] * c[q, j] / math.factorial(q)
                        * weight[k] * c[r, k] / math.factorial(r)
                        * diagram(p, q, r, correlation[i, j], correlation[i, k], correlation[j, k])
                    )
    return value


def fast_directional(c: np.ndarray, correlation: np.ndarray, weight: np.ndarray) -> float:
    b1 = c[1] * weight
    b2 = 0.5 * c[2] * weight
    b3 = (1.0 / 6.0) * c[3] * weight
    return float(36.0 * np.sum(b3 * (correlation @ b1) * ((correlation**2) @ b2)))


def explicit_h(c: np.ndarray, correlation: np.ndarray) -> np.ndarray:
    n = correlation.shape[0]
    h = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for p, q, r in itertools.permutations((1, 2, 3)):
                h[i, j] += (
                    c[p, i] / math.factorial(p)
                    * c[q, i] / math.factorial(q)
                    * c[r, j] / math.factorial(r)
                    * diagram(p, q, r, 1.0, correlation[i, j], correlation[i, j])
                )
    return h


def fast_h(c: np.ndarray, correlation: np.ndarray) -> np.ndarray:
    c1, c2, c3 = c[1], c[2], c[3]
    return (
        (c2 * c3)[:, None] * c1[None, :] * correlation
        + (c1 * c3)[:, None] * c2[None, :] * correlation**2
        + (c1 * c2)[:, None] * c3[None, :] * correlation**3
    )


def check_degree4_birth(c: np.ndarray, correlation: np.ndarray, weight: np.ndarray) -> None:
    # Exact (1,1,2) Hermite shell.
    b1 = c[1] * weight
    b2 = 0.5 * c[2] * weight
    shell = 3.0 * 2.0 * np.sum(b2 * (correlation @ b1) ** 2)
    # CP spelling after converting C=diag(sigma)Rdiag(sigma), with arbitrary
    # positive sigma and gate/pdf chosen to reproduce c1/c2.
    sigma = np.linspace(0.7, 1.3, weight.size)
    gate = c[1] / sigma
    pdf = c[2] / sigma
    covariance = sigma[:, None] * correlation * sigma[None, :]
    p = gate[:, None] * covariance
    cp = 3.0 * np.sum((pdf / sigma) * (p.T @ weight) ** 2 * weight)
    np.testing.assert_allclose(shell, cp, rtol=2e-13, atol=2e-13)


def main() -> None:
    rng = np.random.default_rng(202608261236)
    n = 5
    x = rng.normal(size=(n, n + 3))
    correlation = x @ x.T
    scale = np.sqrt(np.diag(correlation))
    correlation /= scale[:, None] * scale[None, :]
    c = np.zeros((4, n), dtype=np.float64)
    c[1:] = rng.normal(size=(3, n))
    weight = rng.normal(size=n)
    direct = explicit_directional(c, correlation, weight)
    fast = fast_directional(c, correlation, weight)
    np.testing.assert_allclose(fast, direct, rtol=2e-13, atol=2e-13)
    h_direct = explicit_h(c, correlation)
    h_fast = fast_h(c, correlation)
    np.testing.assert_allclose(h_fast, h_direct, rtol=2e-13, atol=2e-13)
    # The lift must reproduce all repeated slices and only those slices.
    diagonal = np.diag(h_fast)
    directed = h_fast.copy()
    np.fill_diagonal(directed, 0.0)
    lift = np.sum(diagonal * weight**3 + 3.0 * weight**2 * (directed @ weight))
    quotient = fast - lift
    check_degree4_birth(c, correlation, weight)
    print(
        {
            "status": "PASS",
            "explicit": direct,
            "fast": fast,
            "lift": float(lift),
            "quotient": float(quotient),
            "max_h_abs_error": float(np.max(np.abs(h_direct - h_fast))),
            "degree4_equals_cp_birth": True,
        }
    )


if __name__ == "__main__":
    main()
