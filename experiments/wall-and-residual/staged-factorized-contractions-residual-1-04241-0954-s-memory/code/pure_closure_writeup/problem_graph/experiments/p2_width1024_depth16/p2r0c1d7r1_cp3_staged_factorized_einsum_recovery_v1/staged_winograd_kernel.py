"""Preparation-only P2R0C1D7 staged native-contraction kernel.

This is an isolated kernel, not an estimator.  It factors each matrix axis
into four binary block indices plus one leaf index, applies four explicit
two-operand coefficient contractions to each operand, performs one batched
leaf matmul, and applies four explicit reconstruction contractions.  There is
no all-depth bilinear einsum: every einsum contracts exactly one 2x2
coefficient map with one numerical array.
"""

from __future__ import annotations

import flopscope.numpy as fnp


LEVELS = 4
PAIRWISE_PATH = ((0, 1),)

U_COEFFICIENTS = (
    ((1.0, 0.0), (0.0, 0.0)),
    ((0.0, 1.0), (0.0, 0.0)),
    ((1.0, 1.0), (-1.0, -1.0)),
    ((0.0, 0.0), (0.0, 1.0)),
    ((0.0, 0.0), (1.0, 1.0)),
    ((-1.0, 0.0), (1.0, 1.0)),
    ((1.0, 0.0), (-1.0, 0.0)),
)

V_COEFFICIENTS = (
    ((1.0, 0.0), (0.0, 0.0)),
    ((0.0, 0.0), (1.0, 0.0)),
    ((0.0, 0.0), (0.0, 1.0)),
    ((1.0, -1.0), (-1.0, 1.0)),
    ((-1.0, 1.0), (0.0, 0.0)),
    ((1.0, -1.0), (0.0, 1.0)),
    ((0.0, -1.0), (0.0, 1.0)),
)

W_COEFFICIENTS = (
    (
        (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0),
    ),
    (
        (1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0),
        (1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
    ),
)


def bind_coefficients():
    """Meter and bind the three immutable float32 coefficient tensors."""
    return (
        fnp.asarray(U_COEFFICIENTS, dtype=fnp.float32),
        fnp.asarray(V_COEFFICIENTS, dtype=fnp.float32),
        fnp.asarray(W_COEFFICIENTS, dtype=fnp.float32),
    )


def staged_winograd(a, b, out, coefficients):
    """Evaluate one depth-4 product through 13 native contractions.

    The 12 coefficient contractions are deliberately unrolled.  This keeps
    the dimension-independent Python control surface frozen and makes the
    exact call inventory statically auditable.
    """
    m = int(a.shape[-2])
    k = int(a.shape[-1])
    n = int(b.shape[-1])
    if (
        int(b.shape[-2]) != k
        or m % 16 != 0
        or k % 16 != 0
        or n % 16 != 0
        or tuple(out.shape) != (m, n)
    ):
        raise ValueError("D7 staged kernel requires compatible 16-divisible matrices")
    u, v, w = coefficients

    # Full binary factorization is one view per input.  Keeping all remaining
    # bit indices explicit lets every later stage feed the next without a
    # transpose or an intervening reshape.
    a0 = fnp.reshape(a, (2, 2, 2, 2, m // 16, 2, 2, 2, 2, k // 16))
    b0 = fnp.reshape(b, (2, 2, 2, 2, k // 16, 2, 2, 2, 2, n // 16))

    a1 = fnp.einsum(
        "rae,abcdMefghK->rbcdMfghK", u, a0, optimize=PAIRWISE_PATH
    )
    del a0
    a2 = fnp.einsum(
        "sbf,rbcdMfghK->srcdMghK", u, a1, optimize=PAIRWISE_PATH
    )
    del a1
    a3 = fnp.einsum(
        "tcg,srcdMghK->tsrdMhK", u, a2, optimize=PAIRWISE_PATH
    )
    del a2
    a4 = fnp.einsum(
        "udh,tsrdMhK->utsrMK", u, a3, optimize=PAIRWISE_PATH
    )
    del a3

    b1 = fnp.einsum(
        "rei,efghKijklN->rfghKjklN", v, b0, optimize=PAIRWISE_PATH
    )
    del b0
    b2 = fnp.einsum(
        "sfj,rfghKjklN->srghKklN", v, b1, optimize=PAIRWISE_PATH
    )
    del b1
    b3 = fnp.einsum(
        "tgk,srghKklN->tsrhKlN", v, b2, optimize=PAIRWISE_PATH
    )
    del b2
    b4 = fnp.einsum(
        "uhl,tsrhKlN->utsrKN", v, b3, optimize=PAIRWISE_PATH
    )
    del b3

    p4 = fnp.matmul(a4, b4)
    del a4, b4

    p3 = fnp.einsum(
        "xyu,utsrMN->tsrxMyN", w, p4, optimize=PAIRWISE_PATH
    )
    del p4
    p2 = fnp.einsum(
        "wzt,tsrxMyN->srwxMzyN", w, p3, optimize=PAIRWISE_PATH
    )
    del p3
    p1 = fnp.einsum(
        "vqs,srwxMzyN->rvwxMqzyN", w, p2, optimize=PAIRWISE_PATH
    )
    del p2
    out_blocks = fnp.reshape(
        out, (2, 2, 2, 2, m // 16, 2, 2, 2, 2, n // 16)
    )
    fnp.einsum(
        "pir,rvwxMqzyN->pvwxMiqzyN",
        w,
        p1,
        out=out_blocks,
        optimize=PAIRWISE_PATH,
    )
    del p1, out_blocks
    return out
