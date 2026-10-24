#!/usr/bin/env python3
"""Asset-free standard real Kerdock set in dimension 1024.

This is the construction in Proposition 7 of Fuchs--Gross--Krahmer--Kueng--
Mixon, arXiv:2105.05879.  For even ``k=10`` it produces 512 alternating
10-by-10 matrices over GF(2).  Each matrix supplies a Walsh/chirp basis in
R^(2^10); adjoining the coordinate basis gives the maximal 513 real MUBs.

The field GF(2^9) uses the primitive polynomial ``t^9+t^4+1``.  No table or
external Kerdock asset is required.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


K = 10
WIDTH = 1 << K
FIELD_DEGREE = K - 1
FIELD_SIZE = 1 << FIELD_DEGREE
FIELD_MASK = FIELD_SIZE - 1
PRIMITIVE_POLYNOMIAL = 0b1000010001  # t^9 + t^4 + 1
KERDOCK_SIZE = 1 << (K - 1)
MAXIMAL_REAL_MUB_COUNT = KERDOCK_SIZE + 1


def gf_multiply(left: int, right: int) -> int:
    """Polynomial-basis multiplication in GF(2^9)."""
    result = 0
    a = int(left) & FIELD_MASK
    b = int(right) & FIELD_MASK
    for _ in range(FIELD_DEGREE):
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & FIELD_SIZE:
            a ^= PRIMITIVE_POLYNOMIAL
    return result & FIELD_MASK


def gf_square(value: int) -> int:
    return gf_multiply(value, value)


def gf_power(value: int, exponent: int) -> int:
    result = 1
    base = int(value) & FIELD_MASK
    power = int(exponent)
    while power:
        if power & 1:
            result = gf_multiply(result, base)
        base = gf_square(base)
        power >>= 1
    return result


def gf_trace(value: int) -> int:
    """Absolute trace GF(2^9)->GF(2)."""
    term = int(value) & FIELD_MASK
    result = term
    for _ in range(1, FIELD_DEGREE):
        term = gf_square(term)
        result ^= term
    if result not in (0, 1):
        raise AssertionError(f"invalid GF(2^9) trace: {result}")
    return result


def primitive_polynomial_receipt() -> dict[str, object]:
    """Exact order check for the polynomial-basis generator t."""
    generator = 2
    order = FIELD_SIZE - 1  # 511 = 7 * 73
    return {
        "polynomial": "t^9+t^4+1",
        "polynomial_binary": bin(PRIMITIVE_POLYNOMIAL),
        "t_power_511": gf_power(generator, order),
        "t_power_73": gf_power(generator, order // 7),
        "t_power_7": gf_power(generator, order // 73),
        "primitive": bool(
            gf_power(generator, order) == 1
            and gf_power(generator, order // 7) != 1
            and gf_power(generator, order // 73) != 1
        ),
    }


def dot(left: tuple[int, int], right: tuple[int, int]) -> int:
    """The nondegenerate form Tr(xy)+alpha*beta on GF(512) x GF(2)."""
    x, alpha = left
    y, beta = right
    return gf_trace(gf_multiply(x, y)) ^ (int(alpha) & int(beta) & 1)


def l_map(s: int, vector: tuple[int, int]) -> tuple[int, int]:
    """Proposition 7: L_s(x,a)=(s^2 x+s Tr(sx)+a s, Tr(sx))."""
    x, alpha = vector
    trace_sx = gf_trace(gf_multiply(s, x))
    first = gf_multiply(gf_square(s), x)
    if trace_sx:
        first ^= s
    if alpha:
        first ^= s
    return first, trace_sx


def standard_kerdock_matrices() -> np.ndarray:
    """Return all 512 standard real Kerdock matrices, shape [512,10,10]."""
    basis = [(1 << index, 0) for index in range(FIELD_DEGREE)] + [(0, 1)]
    matrices = np.empty((KERDOCK_SIZE, K, K), dtype=np.uint8)
    for s in range(KERDOCK_SIZE):
        images = [l_map(s, vector) for vector in basis]
        for row, left in enumerate(basis):
            for column, image in enumerate(images):
                matrices[s, row, column] = dot(left, image)
    return matrices


def gf2_rank(matrix: np.ndarray) -> int:
    """Exact rank of a small binary matrix using packed integer rows."""
    rows = []
    for row in np.asarray(matrix, dtype=np.uint8):
        packed = 0
        for column, value in enumerate(row.tolist()):
            packed |= (int(value) & 1) << column
        rows.append(packed)
    rank = 0
    for column in range(int(matrix.shape[1])):
        pivot = next(
            (index for index in range(rank, len(rows)) if (rows[index] >> column) & 1),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for index in range(len(rows)):
            if index != rank and ((rows[index] >> column) & 1):
                rows[index] ^= rows[rank]
        rank += 1
    return rank


def verify_kerdock_set(matrices: np.ndarray) -> dict[str, object]:
    """Exhaustively check alternation and all 130816 pairwise differences."""
    if matrices.shape != (KERDOCK_SIZE, K, K):
        raise AssertionError(f"matrix schema changed: {matrices.shape}")
    if not np.array_equal(matrices, np.swapaxes(matrices, 1, 2)):
        raise AssertionError("a Kerdock matrix is not symmetric in characteristic two")
    if np.any(np.diagonal(matrices, axis1=1, axis2=2)):
        raise AssertionError("a Kerdock matrix has nonzero diagonal")
    pair_count = 0
    rank_histogram: dict[int, int] = {}
    for first in range(KERDOCK_SIZE):
        for second in range(first + 1, KERDOCK_SIZE):
            rank = gf2_rank(matrices[first] ^ matrices[second])
            rank_histogram[rank] = rank_histogram.get(rank, 0) + 1
            if rank != K:
                raise AssertionError((first, second, rank))
            pair_count += 1
    return {
        "matrix_count": KERDOCK_SIZE,
        "matrix_shape": [K, K],
        "alternating": True,
        "pairwise_differences_checked": pair_count,
        "pairwise_difference_rank_histogram": {
            str(rank): count for rank, count in sorted(rank_histogram.items())
        },
        "real_mub_count_after_identity_basis": MAXIMAL_REAL_MUB_COUNT,
        "normalized_cross_basis_absolute_inner_product": 1.0 / math.sqrt(WIDTH),
        "status": "pass",
    }


def chirp_signs(matrices: np.ndarray, indices: np.ndarray | list[int]) -> np.ndarray:
    """Return (-1)^Q_M(x) for selected Kerdock matrices and all x in F_2^10."""
    selected = np.asarray(indices, dtype=np.int64)
    if np.any(selected < 0) or np.any(selected >= KERDOCK_SIZE):
        raise ValueError("Kerdock index outside 0:511")
    coordinates = np.arange(WIDTH, dtype=np.uint16)
    result = np.empty((selected.size, WIDTH), dtype=np.int8)
    for output_index, matrix_index in enumerate(selected.tolist()):
        matrix = matrices[matrix_index]
        phase = np.zeros(WIDTH, dtype=np.uint8)
        for row in range(K):
            row_bit = (coordinates >> row) & 1
            for column in range(row + 1, K):
                if matrix[row, column]:
                    phase ^= row_bit * ((coordinates >> column) & 1)
        result[output_index] = 1 - 2 * phase.astype(np.int8)
    return result


def fourth_moment_receipt(number_of_bases: int) -> dict[str, object]:
    """Exact global fourth-moment law for any equal-weight MUB subset.

    The normalized order-2 frame potential of any L real MUBs is
    ``(n+L-1)/(L*n^2)``.  Its squared Frobenius distance from the spherical
    fourth-moment tensor is this value minus ``3/(n*(n+2))``.  Consequently,
    no choice of 16 mutually unbiased bases can improve global degree-four
    balance over any other choice of 16.
    """
    bases = int(number_of_bases)
    if bases <= 0:
        raise ValueError("number_of_bases must be positive")
    discrete = (WIDTH + bases - 1.0) / (bases * WIDTH * WIDTH)
    sphere = 3.0 / (WIDTH * (WIDTH + 2.0))
    return {
        "dimension": WIDTH,
        "number_of_bases": bases,
        "normalized_order2_frame_potential": discrete,
        "spherical_order2_frame_potential": sphere,
        "fourth_tensor_squared_frobenius_error": discrete - sphere,
        "frame_potential_ratio_to_sphere": discrete / sphere,
        "selection_invariant_for_equal_weight_mub_subsets": True,
        "exact_projective_2_design": bases == MAXIMAL_REAL_MUB_COUNT,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    matrices = standard_kerdock_matrices()
    result = {
        "schema": "standard-real-kerdock-r1024-proof-v1",
        "source": {
            "paper": "Fuchs et al., Sketching with Kerdock's crayons",
            "arxiv": "2105.05879",
            "construction": "Section 3, Proposition 7",
            "formula": "L_s(x,a)=(s^2*x+s*Tr(s*x)+a*s,Tr(s*x))",
        },
        "field": primitive_polynomial_receipt(),
        "kerdock": verify_kerdock_set(matrices),
        "fourth_moment_l16": fourth_moment_receipt(16),
        "fourth_moment_l513": fourth_moment_receipt(513),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
