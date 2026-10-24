#!/usr/bin/env python3
"""Build the D4 descendant with lawful pair-symmetry and reset-plane CSE."""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1" / "estimator.py"
DESTINATION = HERE / "estimator.py"
EXPECTED_SOURCE_SHA256 = "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e"
EXPECTED_CANDIDATE_SHA256 = "d8e2c1261bd8b77ce9b54b434aaa41f21704146d48e215ccb690dd38224398bb"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


OLD_G = '''def _g_four_all(context):
    """Evaluate all 17 derivative channels for four pair moments together."""
    _features, pair_left, pair_right, rho_powers, inverse_factorial = context
    return fnp.einsum(
        "mdti,mdtj,tij,t->mdij",
        pair_left,
        pair_right,
        rho_powers,
        inverse_factorial,
        optimize=[(0, 3), (0, 1, 2)],
    )


'''


NEW_G = '''def _g_pair_cse(context):
    """Compute 54 lawful channels: all asymmetric and unique symmetric ones."""
    _features, pair_left, pair_right, rho_powers, inverse_factorial = context
    unique = (0, 1, 2, 5, 6, 7, 10, 11, 12, 13)
    asymmetric = fnp.einsum(
        "mdti,mdtj,tij,t->mdij",
        pair_left[1:3],
        pair_right[1:3],
        rho_powers,
        inverse_factorial,
        optimize=[(0, 3), (0, 1, 2)],
    )
    symmetric = fnp.einsum(
        "mdti,mdtj,tij,t->mdij",
        pair_left[::3, unique],
        pair_right[::3, unique],
        rho_powers,
        inverse_factorial,
        optimize=[(0, 3), (0, 1, 2)],
    )
    return asymmetric, symmetric


'''


OLD_FOUR = '''def _four_pair_edgeworth(context, weights, scratch):
    """Return M11, M21, M31 and M22 from six batched derivative groups."""
    weighted_kernels = _g_four_all(context)
    fnp.multiply(
        weights[None, :, :, :],
        weighted_kernels,
        out=weighted_kernels,
    )
    result, group_sum = scratch
    fnp.copyto(result, 0.0)
    for start in range(0, 17, 3):
        stop = min(start + 3, 17)
        fnp.sum(
            weighted_kernels[:, start:stop],
            axis=1,
            out=group_sum,
        )
        fnp.add(result, group_sum, out=result)
    return result[0], result[1], result[2], result[3]


'''


NEW_FOUR = '''def _four_pair_edgeworth(context, weights, scratch):
    """Return four moments while reusing transpose-related symmetric channels."""
    asymmetric, symmetric = _g_pair_cse(context)
    fnp.multiply(
        weights[None, :, :, :],
        asymmetric,
        out=asymmetric,
    )
    unique = (0, 1, 2, 5, 6, 7, 10, 11, 12, 13)
    for position, channel in enumerate(unique):
        fnp.multiply(
            symmetric[:, position],
            weights[channel][None, :, :],
            out=symmetric[:, position],
        )
    p0, p1, p2, p5, p6, p7, p10, p11, p12, p13 = (
        symmetric[:, position] for position in range(10)
    )
    symmetric_full = fnp.stack(
        (
            p0,
            p1,
            p2,
            fnp.transpose(p2, (0, 2, 1)),
            fnp.transpose(p1, (0, 2, 1)),
            p5,
            p6,
            p7,
            fnp.transpose(p6, (0, 2, 1)),
            fnp.transpose(p5, (0, 2, 1)),
            p10,
            p11,
            p12,
            p13,
            fnp.transpose(p12, (0, 2, 1)),
            fnp.transpose(p11, (0, 2, 1)),
            fnp.transpose(p10, (0, 2, 1)),
        ),
        axis=1,
    )

    result, group_sum = scratch
    symmetric_result = result[::3]
    asymmetric_result = result[1:3]
    symmetric_group_sum = group_sum[::3]
    asymmetric_group_sum = group_sum[1:3]
    fnp.copyto(result, 0.0)
    for start in range(0, 17, 3):
        stop = min(start + 3, 17)
        fnp.sum(
            symmetric_full[:, start:stop],
            axis=1,
            out=symmetric_group_sum,
        )
        fnp.add(
            symmetric_result,
            symmetric_group_sum,
            out=symmetric_result,
        )
        fnp.sum(
            asymmetric[:, start:stop],
            axis=1,
            out=asymmetric_group_sum,
        )
        fnp.add(
            asymmetric_result,
            asymmetric_group_sum,
            out=asymmetric_result,
        )
    return result[0], result[1], result[2], result[3]


'''


OLD_RESET = '''    pairing_plane = _off_diagonal(
        fnp.outer(variance, variance) + 2.0 * covariance * covariance,
        diagonal_context,
    )
'''


NEW_RESET = '''    # These two planes were already formed from the same repaired state by
    # _connected_planes; reusing them removes duplicate full-plane products.
    pairing_plane = _off_diagonal(
        variance_outer + 2.0 * covariance2,
        diagonal_context,
    )
'''


def build() -> bytes:
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("passed D4 parent changed")
    text = SOURCE.read_text()
    text = replace_once(text, OLD_G, NEW_G, "pair contraction")
    text = replace_once(text, OLD_FOUR, NEW_FOUR, "pair reduction")
    text = replace_once(text, OLD_RESET, NEW_RESET, "reset pairing reuse")
    return text.encode("utf-8")


def main() -> None:
    payload = build()
    observed = sha256_bytes(payload)
    if EXPECTED_CANDIDATE_SHA256 != "TO_BE_FROZEN" and observed != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError((observed, EXPECTED_CANDIDATE_SHA256))
    DESTINATION.write_bytes(payload)
    print(observed)


if __name__ == "__main__":
    main()
