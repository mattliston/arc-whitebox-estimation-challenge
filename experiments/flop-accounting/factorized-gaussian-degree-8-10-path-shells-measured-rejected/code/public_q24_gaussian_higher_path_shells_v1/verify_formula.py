#!/usr/bin/env python3
"""Exact small-width checks for the D8/D10 Gaussian path shells."""

from __future__ import annotations

import itertools
import math

import numpy as np


PATHS = ((1, 3, 4), (2, 2, 4), (1, 4, 5), (2, 3, 5))


def diagram(p: int, q: int, r: int, rij: float, rik: float, rjk: float) -> float:
    xij = (p + q - r) // 2
    xik = (p + r - q) // 2
    xjk = (q + r - p) // 2
    if (
        (p + q + r) & 1
        or min(xij, xik, xjk) < 0
        or 2 * xij != p + q - r
        or 2 * xik != p + r - q
        or 2 * xjk != q + r - p
    ):
        return 0.0
    return (
        math.factorial(p)
        * math.factorial(q)
        * math.factorial(r)
        / (math.factorial(xij) * math.factorial(xik) * math.factorial(xjk))
        * rij**xij
        * rik**xik
        * rjk**xjk
    )


def unique_permutations(values: tuple[int, int, int]) -> tuple[tuple[int, int, int], ...]:
    return tuple(sorted(set(itertools.permutations(values))))


def explicit_directional(
    degrees: tuple[int, int, int], c: np.ndarray, correlation: np.ndarray, weight: np.ndarray
) -> float:
    value = 0.0
    n = weight.size
    for p, q, r in unique_permutations(degrees):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    value += (
                        weight[i]
                        * c[p, i]
                        / math.factorial(p)
                        * weight[j]
                        * c[q, j]
                        / math.factorial(q)
                        * weight[k]
                        * c[r, k]
                        / math.factorial(r)
                        * diagram(
                            p,
                            q,
                            r,
                            correlation[i, j],
                            correlation[i, k],
                            correlation[j, k],
                        )
                    )
    return value


def fast_directional(
    degrees: tuple[int, int, int], c: np.ndarray, correlation: np.ndarray, weight: np.ndarray
) -> float:
    a, b, high = degrees
    ba = c[a] * weight / math.factorial(a)
    bb = c[b] * weight / math.factorial(b)
    bh = c[high] * weight / math.factorial(high)
    multiplicity = len(unique_permutations(degrees))
    return float(
        multiplicity
        * math.factorial(high)
        * np.sum(
            bh
            * ((correlation**a) @ ba)
            * ((correlation**b) @ bb)
        )
    )


def explicit_h(
    degrees: tuple[int, int, int], c: np.ndarray, correlation: np.ndarray
) -> np.ndarray:
    n = correlation.shape[0]
    h = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for p, q, r in unique_permutations(degrees):
                h[i, j] += (
                    c[p, i]
                    / math.factorial(p)
                    * c[q, i]
                    / math.factorial(q)
                    * c[r, j]
                    / math.factorial(r)
                    * diagram(p, q, r, 1.0, correlation[i, j], correlation[i, j])
                )
    return h


def fast_h(
    degrees: tuple[int, int, int], c: np.ndarray, correlation: np.ndarray
) -> np.ndarray:
    a, b, high = degrees
    if a != b:
        scale = 2.0 / (math.factorial(a) * math.factorial(b))
        return scale * (
            (c[b] * c[high])[:, None] * c[a][None, :] * correlation**a
            + (c[a] * c[high])[:, None] * c[b][None, :] * correlation**b
            + (c[a] * c[b])[:, None] * c[high][None, :] * correlation**high
        )
    scale = 1.0 / math.factorial(a) ** 2
    return scale * (
        2.0
        * (c[a] * c[high])[:, None]
        * c[a][None, :]
        * correlation**a
        + c[a][:, None] ** 2 * c[high][None, :] * correlation**high
    )


def main() -> None:
    rng = np.random.default_rng(202608261743)
    n = 5
    x = rng.normal(size=(n, n + 4))
    correlation = x @ x.T
    scale = np.sqrt(np.diag(correlation))
    correlation /= scale[:, None] * scale[None, :]
    c = np.zeros((6, n), dtype=np.float64)
    c[1:] = rng.normal(size=(5, n))
    weight = rng.normal(size=n)
    rows = []
    for degrees in PATHS:
        direct = explicit_directional(degrees, c, correlation, weight)
        fast = fast_directional(degrees, c, correlation, weight)
        np.testing.assert_allclose(fast, direct, rtol=4e-13, atol=4e-13)
        h_direct = explicit_h(degrees, c, correlation)
        h_fast = fast_h(degrees, c, correlation)
        np.testing.assert_allclose(h_fast, h_direct, rtol=4e-13, atol=4e-13)
        diagonal = np.diag(h_fast)
        lift_explicit = 0.0
        for i in range(n):
            lift_explicit += h_fast[i, i] * weight[i] ** 3
            for j in range(n):
                if i != j:
                    lift_explicit += 3.0 * h_fast[i, j] * weight[i] ** 2 * weight[j]
        lift_fast = np.sum(
            3.0 * weight**2 * (h_fast @ weight)
            - 2.0 * diagonal * weight**3
        )
        np.testing.assert_allclose(lift_fast, lift_explicit, rtol=4e-13, atol=4e-13)
        rows.append(
            {
                "degrees": degrees,
                "explicit": direct,
                "fast": fast,
                "lift": float(lift_fast),
                "quotient": float(fast - lift_fast),
                "max_h_abs_error": float(np.max(np.abs(h_direct - h_fast))),
            }
        )
    print({"status": "PASS", "rows": rows})


if __name__ == "__main__":
    main()
