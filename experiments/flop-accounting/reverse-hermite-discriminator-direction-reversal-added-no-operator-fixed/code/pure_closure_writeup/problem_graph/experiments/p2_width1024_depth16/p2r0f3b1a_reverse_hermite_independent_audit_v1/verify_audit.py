#!/usr/bin/env python3
"""Dependency-free exact checks for the independent P2R0F3B1A audit.

This file never opens network targets.  Every arithmetic identity below is
checked with ``fractions.Fraction``; the combinatorial counts are integers.
"""

from __future__ import annotations

import json
from fractions import Fraction as F
from math import comb


def matmul(a, b):
    return [
        [sum((a[i][k] * b[k][j] for k in range(len(b))), F(0))
         for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def row_mat(v, a):
    return [
        sum((v[i] * a[i][j] for i in range(len(v))), F(0))
        for j in range(len(a[0]))
    ]


def mat_col(a, v):
    return [
        sum((a[i][j] * v[j] for j in range(len(v))), F(0))
        for i in range(len(a))
    ]


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), F(0))


def duhamel_check():
    """Forward source transport equals one adjoint contraction per output."""
    n = 3
    depth = 5
    js = []
    for layer in range(depth):
        js.append([
            [F((layer + 2) * (i + 1) - (j + 1), 11 + layer)
             for j in range(n)]
            for i in range(n)
        ])
    sources = [
        [F((layer + 1) * (i + 2) - 3, 13 + i + layer) for i in range(n)]
        for layer in range(depth)
    ]

    forward = [F(0) for _ in range(n)]
    for layer, source in enumerate(sources):
        carried = list(source)
        for future in range(layer + 1, depth):
            carried = row_mat(carried, js[future])
        forward = [x + y for x, y in zip(forward, carried)]

    reverse = []
    for endpoint in range(n):
        total = F(0)
        for layer, source in enumerate(sources):
            sensitivity = [F(int(i == endpoint)) for i in range(n)]
            for future in range(depth - 1, layer, -1):
                sensitivity = mat_col(js[future], sensitivity)
            total += dot(source, sensitivity)
        reverse.append(total)
    assert forward == reverse
    return {
        "dimension": n,
        "depth": depth,
        "exact_fraction_equality": True,
        "endpoint_numerators": [x.numerator for x in forward],
        "endpoint_denominators": [x.denominator for x in forward],
    }


def he_at_zero(k):
    """Probabilists' He_k(0), exactly."""
    if k == 0:
        return 1
    if k == 1:
        return 0
    hm2, hm1 = 1, 0
    for degree in range(1, k):
        h = -degree * hm2
        hm2, hm1 = hm1, h
    return hm1


def relu_hermite_check(max_degree=20):
    """Return a_k / phi(0) for ReLU's probabilists-Hermite expansion.

    For k >= 2, a_k = E[Z_+ He_k(Z)] / k!
                         = phi(0) He_{k-2}(0) / k!.
    Thus every even degree is nonzero.
    """
    factorial = 1
    ratios = {}
    nonzero_even = []
    for k in range(max_degree + 1):
        if k > 0:
            factorial *= k
        if k == 0:
            ratio = F(1)
        elif k == 1:
            ratio = None  # a_1=1/2 is not a rational multiple of phi(0)
        else:
            ratio = F(he_at_zero(k - 2), factorial)
        ratios[str(k)] = None if ratio is None else f"{ratio.numerator}/{ratio.denominator}"
        if k >= 2 and k % 2 == 0:
            assert ratio != 0
            nonzero_even.append(k)
        if k >= 3 and k % 2 == 1:
            assert ratio == 0
    return {
        "formula": "a_k/phi(0)=He_{k-2}(0)/k! for k>=2; a_1=1/2",
        "checked_through_degree": max_degree,
        "ratios": ratios,
        "nonzero_even_degrees": nonzero_even,
    }


def make_k_and_w():
    n, outputs = 4, 3
    k = [[F(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            value = F((i + 2) * (j + 3) - 5, 17 + i + j)
            k[i][j] = value
            k[j][i] = value
    w = [
        [F((i + 1) * (a + 2) - 4, 7 + i + 2 * a)
         for a in range(outputs)]
        for i in range(n)
    ]
    return k, w


def tensor_from_k(k):
    """Replicate the ordered K/2 construction used by the supply audit."""
    n = len(k)
    tensor = {}
    for i in range(n):
        for j in range(n):
            if i == j or k[i][j] == 0:
                continue
            half = k[i][j] / 2
            for slots in (
                (i, i, j, j), (i, j, i, j), (i, j, j, i),
                (j, i, i, j), (j, i, j, i), (j, j, i, i),
            ):
                tensor[slots] = tensor.get(slots, F(0)) + half
    return tensor


def pushed_entry(tensor, w, a, b, c, d):
    return sum(
        (value * w[i][a] * w[j][b] * w[k][c] * w[l][d]
         for (i, j, k, l), value in tensor.items()),
        F(0),
    )


def k22_diagram_check():
    """Exact small-n proof of the deployed K22 channel and missing 2F."""
    k, w = make_k_and_w()
    tensor = tensor_from_k(k)
    n, outputs = len(k), len(w[0])
    carried = [[F(0) for _ in range(outputs)] for _ in range(outputs)]
    f_form = [[F(0) for _ in range(outputs)] for _ in range(outputs)]
    for a in range(outputs):
        for b in range(outputs):
            carried[a][b] = sum(
                (w[i][a] ** 2 * k[i][j] * w[j][b] ** 2
                 for i in range(n) for j in range(n)),
                F(0),
            )
            f_form[a][b] = sum(
                (w[i][a] * w[i][b] * k[i][j] * w[j][a] * w[j][b]
                 for i in range(n) for j in range(n)),
                F(0),
            )

            brute22 = pushed_entry(tensor, w, a, a, b, b)
            assert brute22 == carried[a][b] + 2 * f_form[a][b]

            brute4 = pushed_entry(tensor, w, a, a, a, a)
            carried_diag = sum(
                (w[i][a] ** 2 * k[i][j] * w[j][a] ** 2
                 for i in range(n) for j in range(n)),
                F(0),
            )
            assert brute4 == 3 * carried_diag

            brute31 = pushed_entry(tensor, w, a, a, a, b)
            deployed31 = 3 * sum(
                (w[i][a] ** 2 * k[i][j] * w[j][a] * w[j][b]
                 for i in range(n) for j in range(n)),
                F(0),
            )
            assert brute31 == deployed31

    return {
        "input_dimension": n,
        "output_dimension": outputs,
        "exact_fraction_equality": True,
        "k4_diagonal": "3 diag((W^2)^T K W^2), fully carried",
        "k31": "3 sum_ij W_ia^2 K_ij W_ja W_jb, fully carried",
        "k22": "((W^2)^T K W^2)_ab + 2 F_ab; deployed supply omits 2F",
    }


def coefficient_counts():
    rows = []
    for n in (8, 16, 1024):
        for degree in (2, 3, 4, 5, 12):
            homogeneous = comb(n + degree - 1, degree)
            through = comb(n + degree, degree)
            rows.append({
                "n": n,
                "degree": degree,
                "homogeneous_coefficients": homogeneous,
                "through_degree_coefficients": through,
                "through_degree_float32_bytes": 4 * through,
            })
    return rows


def main():
    result = {
        "schema": "p2r0f3b1a.reverse_hermite.independent_audit.exact_checks.v1",
        "target_accesses": 0,
        "checks": {
            "relu_infinite_hermite_tail": relu_hermite_check(),
            "forward_reverse_duhamel": duhamel_check(),
            "k22_collision_diagram": k22_diagram_check(),
            "coefficient_counts": coefficient_counts(),
        },
        "pass": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
