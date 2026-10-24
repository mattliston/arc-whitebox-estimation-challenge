"""Target-free, pure-stdlib algebra checks for FB-EC4-32.

The checks use only tiny literal synthetic tensors.  They do not import the
deployed estimator, FlopScope, a model reader, or any role/target artifact.
"""

from __future__ import annotations

import ast
import math
import struct
from pathlib import Path


HERE = Path(__file__).resolve().parent
ESTIMATOR = HERE / "estimator.py"


def close(left: float, right: float, tolerance: float = 2.0e-11) -> None:
    scale = max(1.0, abs(left), abs(right))
    assert abs(left - right) <= tolerance * scale, (left, right)


def matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[i][k] * right[k][j] for k in range(len(right)))
         for j in range(len(right[0]))]
        for i in range(len(left))
    ]


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def literal_assignment(name: str):
    tree = ast.parse(ESTIMATOR.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal {name}")


def test_literal_tables() -> None:
    inverse_factorial = literal_assignment("_INV_FACTORIAL")
    assert len(inverse_factorial) == 33
    value = 1.0
    for order, frozen in enumerate(inverse_factorial):
        if order > 0:
            value /= order
        close(frozen, value, 2.0e-15)
    assert literal_assignment("_HERMITE_RECURRENCE") == tuple(
        float(value) for value in range(1, 36)
    )
    assert literal_assignment("_DERIVATIVE_CHANNELS") == (
        (0, 0),
        (0, 3), (1, 2), (2, 1), (3, 0),
        (0, 4), (1, 3), (2, 2), (3, 1), (4, 0),
        (0, 6), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (6, 0),
    )


def test_cp3_fused_k21() -> None:
    lambdas = [0.7, -0.4]
    p = [[0.2, -0.1], [0.5, 0.3], [-0.4, 0.8]]
    q = [[0.9, 0.2], [-0.6, 0.7], [0.1, -0.5]]
    n = len(p)
    explicit = [[0.0] * n for _ in range(n)]
    fused = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            explicit[i][j] = sum(
                lambdas[r]
                * (
                    p[i][r] * p[i][r] * q[j][r]
                    + 2.0 * p[i][r] * q[i][r] * p[j][r]
                )
                for r in range(len(lambdas))
            )
            fused[i][j] = sum(
                (p[i][r] * p[i][r]) * (q[j][r] * lambdas[r])
                + 2.0 * (p[i][r] * q[i][r]) * (p[j][r] * lambdas[r])
                for r in range(len(lambdas))
            )
            close(explicit[i][j], fused[i][j])


def cubic_entry(star: list[list[float]], i: int, j: int, k: int) -> float:
    if i == j == k:
        return star[i][i]
    if i == j:
        return star[i][k]
    if i == k:
        return star[i][j]
    if j == k:
        return star[j][i]
    return 0.0


def test_cubic_star_push() -> None:
    star = [[0.7, -0.2, 0.4], [0.1, -0.3, 0.6], [-0.5, 0.8, 0.2]]
    weight = [[0.3, -0.4, 0.2], [0.5, 0.1, -0.7], [-0.2, 0.6, 0.9]]
    n = len(star)
    diagonal = [star[i][i] for i in range(n)]
    directed = [[star[i][j] if i != j else 0.0 for j in range(n)] for i in range(n)]
    transported = matmul(directed, weight)
    for a in range(n):
        pushed_d = sum(
            diagonal[i] * weight[i][a] ** 3
            + 3.0 * weight[i][a] ** 2 * transported[i][a]
            for i in range(n)
        )
        explicit_d = sum(
            cubic_entry(star, i, j, k)
            * weight[i][a] * weight[j][a] * weight[k][a]
            for i in range(n) for j in range(n) for k in range(n)
        )
        close(pushed_d, explicit_d)
        for b in range(n):
            pushed_k21 = sum(
                diagonal[i] * weight[i][a] ** 2 * weight[i][b]
                + 2.0 * weight[i][a] * transported[i][a] * weight[i][b]
                + weight[i][a] ** 2 * transported[i][b]
                for i in range(n)
            )
            explicit_k21 = sum(
                cubic_entry(star, i, j, k)
                * weight[i][a] * weight[j][a] * weight[k][b]
                for i in range(n) for j in range(n) for k in range(n)
            )
            close(pushed_k21, explicit_k21)


def quartic_star_entry(star: list[list[float]], indices: tuple[int, int, int, int]) -> float:
    counts = {value: indices.count(value) for value in set(indices)}
    if len(counts) == 1:
        coordinate = indices[0]
        return star[coordinate][coordinate]
    if sorted(counts.values()) == [1, 3]:
        repeated = next(key for key, count in counts.items() if count == 3)
        singleton = next(key for key, count in counts.items() if count == 1)
        return star[repeated][singleton]
    return 0.0


def pairing_entry(covariance: list[list[float]], i: int, j: int, k: int, ell: int) -> float:
    return (
        covariance[i][j] * covariance[k][ell]
        + covariance[i][k] * covariance[j][ell]
        + covariance[i][ell] * covariance[j][k]
    )


def contract4(
    star: list[list[float]],
    alpha: float,
    covariance: list[list[float]],
    weight: list[list[float]],
    outputs: tuple[int, int, int, int],
) -> float:
    n = len(star)
    total = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for ell in range(n):
                    tensor = quartic_star_entry(star, (i, j, k, ell))
                    tensor += alpha * pairing_entry(covariance, i, j, k, ell)
                    total += (
                        tensor
                        * weight[i][outputs[0]]
                        * weight[j][outputs[1]]
                        * weight[k][outputs[2]]
                        * weight[ell][outputs[3]]
                    )
    return total


def test_coherent_quartic_push_and_gauge() -> None:
    star = [[0.5, -0.2, 0.3], [0.4, -0.6, 0.1], [-0.7, 0.8, 0.2]]
    covariance = [[1.2, 0.2, -0.1], [0.2, 0.9, 0.15], [-0.1, 0.15, 1.1]]
    weight = [[0.3, -0.4, 0.2], [0.5, 0.1, -0.7], [-0.2, 0.6, 0.9]]
    alpha = 0.37
    n = len(star)
    diagonal = [star[i][i] for i in range(n)]
    directed = [[star[i][j] if i != j else 0.0 for j in range(n)] for i in range(n)]
    transported = matmul(directed, weight)
    pushed_covariance = matmul(transpose(weight), matmul(covariance, weight))

    for a in range(n):
        star_d = sum(
            diagonal[i] * weight[i][a] ** 4
            + 4.0 * weight[i][a] ** 3 * transported[i][a]
            for i in range(n)
        )
        formula_d = star_d + 3.0 * alpha * pushed_covariance[a][a] ** 2
        close(formula_d, contract4(star, alpha, covariance, weight, (a, a, a, a)))
        for b in range(n):
            star_a = sum(
                diagonal[i] * weight[i][a] ** 3 * weight[i][b]
                + weight[i][a] ** 3 * transported[i][b]
                + 3.0 * weight[i][a] ** 2 * transported[i][a] * weight[i][b]
                for i in range(n)
            )
            formula_a = star_a + 3.0 * alpha * pushed_covariance[a][a] * pushed_covariance[a][b]
            close(formula_a, contract4(star, alpha, covariance, weight, (a, a, a, b)))

            star_b = sum(
                diagonal[i] * weight[i][a] ** 2 * weight[i][b] ** 2
                + 2.0 * (
                    weight[i][a] * weight[i][b] * transported[i][a] * weight[i][b]
                    + weight[i][b] * weight[i][a] * transported[i][b] * weight[i][a]
                )
                for i in range(n)
            )
            formula_b = star_b + alpha * (
                pushed_covariance[a][a] * pushed_covariance[b][b]
                + 2.0 * pushed_covariance[a][b] ** 2
            )
            close(formula_b, contract4(star, alpha, covariance, weight, (a, a, b, b)))

    target_d = [0.4, -0.1, 0.7]
    target_a = [[0.0, 0.3, -0.2], [0.6, 0.0, 0.5], [-0.4, 0.2, 0.0]]
    target_b = [[0.0, 0.8, -0.3], [0.8, 0.0, 0.1], [-0.3, 0.1, 0.0]]
    s = [[
        covariance[i][i] * covariance[j][j] + 2.0 * covariance[i][j] ** 2
        for j in range(n)
    ] for i in range(n)]
    numerator = sum(target_b[i][j] * s[i][j] for i in range(n) for j in range(i + 1, n))
    denominator = sum(s[i][j] ** 2 for i in range(n) for j in range(i + 1, n))
    fitted = numerator / denominator
    gauge_d = [target_d[i] - 3.0 * fitted * covariance[i][i] ** 2 for i in range(n)]
    gauge_a = [[
        target_a[i][j] - 3.0 * fitted * covariance[i][i] * covariance[i][j]
        if i != j else 0.0
        for j in range(n)
    ] for i in range(n)]
    for i in range(n):
        close(gauge_d[i] + 3.0 * fitted * covariance[i][i] ** 2, target_d[i])
        for j in range(n):
            if i != j:
                close(
                    gauge_a[i][j] + 3.0 * fitted * covariance[i][i] * covariance[i][j],
                    target_a[i][j],
                )
    normal_residual = sum(
        (target_b[i][j] - fitted * s[i][j]) * s[i][j]
        for i in range(n) for j in range(i + 1, n)
    )
    close(normal_residual, 0.0)


def cp4_planes(
    lambdas: list[float],
    factors: list[list[float]],
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    n = len(factors)
    rank = len(lambdas)
    diagonal = [
        sum(lambdas[r] * factors[i][r] ** 4 for r in range(rank))
        for i in range(n)
    ]
    directed = [[
        sum(
            lambdas[r] * factors[i][r] ** 3 * factors[j][r]
            for r in range(rank)
        ) if i != j else 0.0
        for j in range(n)
    ] for i in range(n)]
    paired = [[
        sum(
            lambdas[r] * factors[i][r] ** 2 * factors[j][r] ** 2
            for r in range(rank)
        ) if i != j else 0.0
        for j in range(n)
    ] for i in range(n)]
    return diagonal, directed, paired


def test_cp4_top_energy_residual_seam() -> None:
    n = 3
    cap = 2 * n
    inherited_lambda = [0.4, -0.3, 0.8, 0.2, -0.6, 0.5]
    inherited_u = [
        [0.9, 0.1, -0.3, 0.7, 0.2, -0.8],
        [0.2, -0.6, 0.5, 0.1, 0.9, 0.4],
        [-0.4, 0.8, 0.2, -0.5, 0.3, 0.6],
    ]
    gate = [0.35, 0.7, 0.55]
    gated_u = [[gate[i] * value for value in inherited_u[i]] for i in range(n)]
    inherited_planes = cp4_planes(inherited_lambda, gated_u)
    target_d = [0.18, -0.07, 0.22]
    birth_lambda = [target_d[i] - inherited_planes[0][i] for i in range(n)]
    identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    candidate_lambda = inherited_lambda + birth_lambda
    candidate_u = [gated_u[i] + identity[i] for i in range(n)]
    candidate_planes = cp4_planes(candidate_lambda, candidate_u)
    for i in range(n):
        close(candidate_planes[0][i], target_d[i])
        for j in range(n):
            close(candidate_planes[1][i][j], inherited_planes[1][i][j])
            close(candidate_planes[2][i][j], inherited_planes[2][i][j])

    norm2 = [sum(candidate_u[i][r] ** 2 for i in range(n)) for r in range(3 * n)]
    energy = [candidate_lambda[r] ** 2 * norm2[r] ** 4 for r in range(3 * n)]
    order = sorted(range(3 * n), key=lambda r: (-energy[r], r))
    keep = order[:cap]
    drop = order[cap:]
    assert len(drop) == n
    assert keep != list(range(cap)), "fixture must distinguish top-energy from FIFO"

    retained_lambda = [candidate_lambda[r] for r in keep]
    retained_u = [[candidate_u[i][r] for r in keep] for i in range(n)]
    dropped_lambda = [candidate_lambda[r] for r in drop]
    dropped_u = [[candidate_u[i][r] for r in drop] for i in range(n)]
    retained_direct = cp4_planes(retained_lambda, retained_u)
    dropped = cp4_planes(dropped_lambda, dropped_u)
    retained_subtract = (
        [target_d[i] - dropped[0][i] for i in range(n)],
        [[inherited_planes[1][i][j] - dropped[1][i][j] for j in range(n)] for i in range(n)],
        [[inherited_planes[2][i][j] - dropped[2][i][j] for j in range(n)] for i in range(n)],
    )
    for plane in range(3):
        if plane == 0:
            for i in range(n):
                close(retained_direct[plane][i], retained_subtract[plane][i])
        else:
            for i in range(n):
                for j in range(n):
                    close(retained_direct[plane][i][j], retained_subtract[plane][i][j])

    # The H4 residual seam subtracts once, and affine assembly adds once.
    target_a = [[0.0, 0.12, -0.03], [0.05, 0.0, 0.09], [-0.07, 0.02, 0.0]]
    target_b = [[0.0, -0.08, 0.04], [-0.08, 0.0, 0.11], [0.04, 0.11, 0.0]]
    for i in range(n):
        host_d = target_d[i] - retained_direct[0][i]
        close(host_d + retained_direct[0][i], target_d[i])
        for j in range(n):
            host_a = target_a[i][j] - retained_direct[1][i][j]
            host_b = target_b[i][j] - retained_direct[2][i][j]
            close(host_a + retained_direct[1][i][j], target_a[i][j])
            close(host_b + retained_direct[2][i][j], target_b[i][j])

    # Births are stored in fp32.  The provider plane must use the stored
    # value, leaving its rounding residual in H4 instead of silently dropping
    # it by pretending the provider diagonal equals the fp64 target.
    target = 0.123456789123
    inherited = -0.031415926535
    stored_birth = struct.unpack("<f", struct.pack("<f", target - inherited))[0]
    actual_provider = inherited + stored_birth
    assert actual_provider != target
    host_residual = target - actual_provider
    close(actual_provider + host_residual, target, 1.0e-15)
    mutated_host_residual = target - target
    assert abs(actual_provider + mutated_host_residual - target) > 1.0e-10

    # Quantify the only deliberately analytic part of the split: A/B are
    # gated from the pre-read planes rather than reread after materializing
    # fp32 gate products.  For normal fixture values, four factor roundings
    # give the standard per-atom bound eta4=(1+2^-24)^4-1.
    f32 = lambda value: struct.unpack("<f", struct.pack("<f", value))[0]
    unit_roundoff = 2.0 ** -24
    eta4 = (1.0 + unit_roundoff) ** 4 - 1.0
    lambda32 = [f32(value) for value in inherited_lambda]
    u32 = [[f32(value) for value in row] for row in inherited_u]
    gate32 = [f32(value) for value in gate]
    pre_planes = cp4_planes(lambda32, u32)
    exact_gated = [[gate32[i] * u32[i][r] for r in range(2 * n)] for i in range(n)]
    stored_gated = [[f32(value) for value in row] for row in exact_gated]
    stored_planes = cp4_planes(lambda32, stored_gated)
    max_relative_bound_ratio = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            analytic_a = gate32[i] ** 3 * gate32[j] * pre_planes[1][i][j]
            analytic_b = gate32[i] ** 2 * gate32[j] ** 2 * pre_planes[2][i][j]
            absolute_a = sum(
                abs(lambda32[r] * exact_gated[i][r] ** 3 * exact_gated[j][r])
                for r in range(2 * n)
            )
            absolute_b = sum(
                abs(lambda32[r] * exact_gated[i][r] ** 2 * exact_gated[j][r] ** 2)
                for r in range(2 * n)
            )
            error_a = abs(stored_planes[1][i][j] - analytic_a)
            error_b = abs(stored_planes[2][i][j] - analytic_b)
            assert error_a <= eta4 * absolute_a + 2.0e-15
            assert error_b <= eta4 * absolute_b + 2.0e-15
            max_relative_bound_ratio = max(
                max_relative_bound_ratio,
                error_a / max(eta4 * absolute_a, 1.0e-30),
                error_b / max(eta4 * absolute_b, 1.0e-30),
            )
    assert max_relative_bound_ratio <= 1.0000001


def test_connected_resets() -> None:
    support = [(-1.0, 0.3, 0.1), (0.2, 1.4, 0.2), (1.1, -0.7, 0.3), (2.0, 0.8, 0.4)]
    mx = sum(x * probability for x, _y, probability in support)
    my = sum(y * probability for _x, y, probability in support)
    moment = lambda p, q: sum((x ** p) * (y ** q) * probability for x, y, probability in support)
    m20, m30, m40 = moment(2, 0), moment(3, 0), moment(4, 0)
    m02 = moment(0, 2)
    m11, m21, m12, m31, m22 = moment(1, 1), moment(2, 1), moment(1, 2), moment(3, 1), moment(2, 2)
    cxx = m20 - mx * mx
    cyy = m02 - my * my
    cxy = m11 - mx * my
    k21_formula = m21 - m20 * my - 2.0 * mx * m11 + 2.0 * mx * mx * my
    k21_direct = sum((x - mx) ** 2 * (y - my) * probability for x, y, probability in support)
    close(k21_formula, k21_direct)
    central31 = m31 - my * m30 - 3.0 * mx * m21 + 3.0 * mx * my * m20 + 3.0 * mx * mx * m11 - 3.0 * mx ** 3 * my
    k31_formula = central31 - 3.0 * cxx * cxy
    k31_direct = sum((x - mx) ** 3 * (y - my) * probability for x, y, probability in support) - 3.0 * cxx * cxy
    close(k31_formula, k31_direct)
    central22 = m22 - 2.0 * my * m21 - 2.0 * mx * m12 + my * my * m20 + mx * mx * m02 + 4.0 * mx * my * m11 - 3.0 * mx * mx * my * my
    k22_formula = central22 - cxx * cyy - 2.0 * cxy * cxy
    k22_direct = sum((x - mx) ** 2 * (y - my) ** 2 * probability for x, y, probability in support) - cxx * cyy - 2.0 * cxy * cxy
    close(k22_formula, k22_direct)
    k4_formula = m40 - 4.0 * mx * m30 - 3.0 * m20 * m20 + 12.0 * mx * mx * m20 - 6.0 * mx ** 4
    k4_direct = sum((x - mx) ** 4 * probability for x, _y, probability in support) - 3.0 * cxx * cxx
    close(k4_formula, k4_direct)


def test_l3_squared_convolution() -> None:
    weights = [0.13, -0.27, 0.41, 0.19]
    literal = [
        0.5 * weights[0] * weights[0],
        weights[0] * weights[1],
        weights[0] * weights[2] + 0.5 * weights[1] * weights[1],
        weights[0] * weights[3] + weights[1] * weights[2],
        weights[1] * weights[3] + 0.5 * weights[2] * weights[2],
        weights[2] * weights[3],
        0.5 * weights[3] * weights[3],
    ]
    for total in range(7):
        explicit = 0.5 * sum(
            weights[left] * weights[right]
            for left in range(4) for right in range(4)
            if left + right == total
        )
        close(literal[total], explicit)
    # Fatal mutation: omitting the w12*w21 cross term must not pass.
    mutated_u33 = weights[0] * weights[3]
    assert abs(mutated_u33 - literal[3]) > 1.0e-6


def test_explicit_univariate_mean() -> None:
    mean = 0.37
    sigma = 1.21
    variance = sigma * sigma
    alpha = mean / sigma
    pdf = math.exp(-0.5 * alpha * alpha) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(alpha / math.sqrt(2.0)))
    gaussian = sigma * pdf + mean * cdf
    k3 = -0.08
    k4 = 0.11
    generic = (
        gaussian
        + k3 * (-alpha * sigma * pdf) / (6.0 * sigma ** 3)
        + k4 * ((alpha * alpha - 1.0) * sigma * pdf) / (24.0 * sigma ** 4)
        + k3 * k3 * ((alpha ** 4 - 6.0 * alpha * alpha + 3.0) * sigma * pdf) / (72.0 * sigma ** 6)
    )
    explicit = (
        gaussian
        + k3 * (-alpha * pdf / variance) / 6.0
        + k4 * ((alpha * alpha - 1.0) * pdf / sigma ** 3) / 24.0
        + k3 * k3 * ((alpha ** 4 - 6.0 * alpha * alpha + 3.0) * pdf / sigma ** 5) / 72.0
    )
    close(generic, explicit)


def test_relu_mean_and_variance_guards() -> None:
    corrected1 = -0.37
    corrected2 = -0.11
    variance_floor = 1.0e-30
    guarded1 = max(corrected1, 0.0)
    guarded2 = max(corrected2, guarded1 * guarded1 + variance_floor)
    assert guarded1 == 0.0
    assert guarded2 - guarded1 * guarded1 >= variance_floor

    corrected1 = 0.8
    corrected2 = 0.2
    guarded1 = max(corrected1, 0.0)
    guarded2 = max(corrected2, guarded1 * guarded1 + variance_floor)
    assert guarded1 == 0.8
    assert guarded2 >= guarded1 * guarded1


def main() -> None:
    test_literal_tables()
    test_cp3_fused_k21()
    test_cubic_star_push()
    test_coherent_quartic_push_and_gauge()
    test_cp4_top_energy_residual_seam()
    test_connected_resets()
    test_l3_squared_convolution()
    test_explicit_univariate_mean()
    test_relu_mean_and_variance_guards()
    print("PASS: target-free FB-EC4-32 algebra")


if __name__ == "__main__":
    main()
