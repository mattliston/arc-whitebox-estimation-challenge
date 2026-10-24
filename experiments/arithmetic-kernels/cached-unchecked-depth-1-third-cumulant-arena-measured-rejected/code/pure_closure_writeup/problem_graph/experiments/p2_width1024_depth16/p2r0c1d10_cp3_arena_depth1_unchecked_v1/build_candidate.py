#!/usr/bin/env python3
"""Build the Python-overhead-only D1U descendant from frozen D1."""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
D1 = HERE.parent / "p2r0c1d9_cp3_arena_depth1_full_estimator_v1"
SOURCE = D1 / "estimator.py"
OUTPUT = HERE / "estimator.py"

EXPECTED_D1_SHA256 = (
    "f2ec257c891241b57c85803c70c73f365e075711e2b2cd214d9b58250762892e"
)

OLD_HEADER = '''def _cp3_winograd_arena_depth1(a, b, out, arena_views):
    """Sealed one-level fixed-arena Winograd product with explicit out=."""
    m = int(a.shape[-2])
    k = int(a.shape[-1])
    n = int(b.shape[-1])
    if (
        int(b.shape[-2]) != k
        or m % 2 != 0
        or k % 2 != 0
        or n % 2 != 0
        or tuple(out.shape) != (m, n)
    ):
        raise ValueError("D1 arena kernel requires compatible even matrices")

    m2 = m // 2
    k2 = k // 2
    n2 = n // 2
'''

NEW_HEADER = '''def _cp3_winograd_arena_depth1_unchecked(a, b, out, arena_views):
    """Private fixed-contract D1 kernel; callers guarantee compatible shapes."""
    m2 = a.shape[-2] // 2
    k2 = a.shape[-1] // 2
    n2 = b.shape[-1] // 2
'''

OLD_WEIGHT_BIND = '''    if rank == 0:
        return cp_lambda, destination_p[:, :0], destination_q[:, :0]
    _weight32, weight32_t, _weight2, _weight2_t, _weight3, _weight3_t = (
        weight_context
    )
'''

NEW_WEIGHT_BIND = '''    if rank == 0:
        return cp_lambda, destination_p[:, :0], destination_q[:, :0]
    weight32_t = weight_context[1]
'''

OLD_SCRATCH_CREATE = '''        cp3_context = (
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty(_CP3_CAP, dtype=fnp.float32),
            fnp.empty(width * _CP3_CAP, dtype=fnp.float32),
            fnp.empty(width * _CP3_CAP, dtype=fnp.float32),
            cp3_arena,
            _cp3_bind_arena_views(cp3_arena, _CP3_PUSH_ARENA_VIEW_SPECS),
            _cp3_bind_arena_views(cp3_arena, _CP3_READ_ARENA_VIEW_SPECS),
        )
        identity32 = fnp.eye(width, dtype=fnp.float32)
'''

NEW_SCRATCH_CREATE = '''        cp3_context = (
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty(_CP3_CAP, dtype=fnp.float32),
            fnp.empty(width * _CP3_CAP, dtype=fnp.float32),
            fnp.empty(width * _CP3_CAP, dtype=fnp.float32),
            cp3_arena,
            _cp3_bind_arena_views(cp3_arena, _CP3_PUSH_ARENA_VIEW_SPECS),
            _cp3_bind_arena_views(cp3_arena, _CP3_READ_ARENA_VIEW_SPECS),
        )
        weight_power_context = (
            fnp.empty((width, width), dtype=fnp.float32),
            fnp.empty((width, width), dtype=fnp.float32),
        )
        identity32 = fnp.eye(width, dtype=fnp.float32)
'''

OLD_SCRATCH_STORE = '''        _PRODUCTION_SCRATCH_CONTEXT = (
            hermite_context,
            fusion_context,
            cp3_context,
            identity32,
            diagonal_context,
        )
'''

NEW_SCRATCH_STORE = '''        _PRODUCTION_SCRATCH_CONTEXT = (
            hermite_context,
            fusion_context,
            cp3_context,
            weight_power_context,
            identity32,
            diagonal_context,
        )
'''

OLD_SCRATCH_LOAD = '''        (
            hermite_context,
            fusion_context,
            cp3_context,
            identity32,
            diagonal_context,
        ) = _PRODUCTION_SCRATCH_CONTEXT
'''

NEW_SCRATCH_LOAD = '''        (
            hermite_context,
            fusion_context,
            cp3_context,
            weight_power_context,
            identity32,
            diagonal_context,
        ) = _PRODUCTION_SCRATCH_CONTEXT
'''

OLD_WEIGHT_POWERS = '''        weight2 = weight32 * weight32
        weight3 = weight2 * weight32
'''

NEW_WEIGHT_POWERS = '''        weight2, weight3 = weight_power_context
        fnp.multiply(weight32, weight32, out=weight2)
        fnp.multiply(weight2, weight32, out=weight3)
'''

OLD_A_STAGE = '''    fnp.add(a21, a22, out=scratch_s[0])
    fnp.subtract(scratch_s[0], a11, out=scratch_s[1])
    fnp.subtract(a11, a21, out=scratch_s[2])
    fnp.subtract(a12, scratch_s[1], out=scratch_s[3])
    fnp.copyto(transformed_a[0], a11)
    fnp.copyto(transformed_a[1], a12)
    fnp.copyto(transformed_a[2], scratch_s[3])
    fnp.copyto(transformed_a[3], a22)
    fnp.copyto(transformed_a[4], scratch_s[0])
    fnp.copyto(transformed_a[5], scratch_s[1])
    fnp.copyto(transformed_a[6], scratch_s[2])
'''

NEW_A_STAGE = '''    s0 = scratch_s[0]
    s1 = scratch_s[1]
    s2 = scratch_s[2]
    s3 = scratch_s[3]
    fnp.add(a21, a22, out=s0)
    fnp.subtract(s0, a11, out=s1)
    fnp.subtract(a11, a21, out=s2)
    fnp.subtract(a12, s1, out=s3)
    fnp.copyto(transformed_a[0], a11)
    fnp.copyto(transformed_a[1], a12)
    fnp.copyto(transformed_a[2], s3)
    fnp.copyto(transformed_a[3], a22)
    fnp.copyto(transformed_a[4], s0)
    fnp.copyto(transformed_a[5], s1)
    fnp.copyto(transformed_a[6], s2)
'''

OLD_B_STAGE = '''    fnp.subtract(b12, b11, out=scratch_t[0])
    fnp.subtract(b22, scratch_t[0], out=scratch_t[1])
    fnp.subtract(b22, b12, out=scratch_t[2])
    fnp.subtract(scratch_t[1], b21, out=scratch_t[3])
    fnp.copyto(transformed_b[0], b11)
    fnp.copyto(transformed_b[1], b21)
    fnp.copyto(transformed_b[2], b22)
    fnp.copyto(transformed_b[3], scratch_t[3])
    fnp.copyto(transformed_b[4], scratch_t[0])
    fnp.copyto(transformed_b[5], scratch_t[1])
    fnp.copyto(transformed_b[6], scratch_t[2])
'''

NEW_B_STAGE = '''    t0 = scratch_t[0]
    t1 = scratch_t[1]
    t2 = scratch_t[2]
    t3 = scratch_t[3]
    fnp.subtract(b12, b11, out=t0)
    fnp.subtract(b22, t0, out=t1)
    fnp.subtract(b22, b12, out=t2)
    fnp.subtract(t1, b21, out=t3)
    fnp.copyto(transformed_b[0], b11)
    fnp.copyto(transformed_b[1], b21)
    fnp.copyto(transformed_b[2], b22)
    fnp.copyto(transformed_b[3], t3)
    fnp.copyto(transformed_b[4], t0)
    fnp.copyto(transformed_b[5], t1)
    fnp.copyto(transformed_b[6], t2)
'''

OLD_RECONSTRUCTION = '''    fnp.add(products[0], products[1], out=c11)
    fnp.add(products[0], products[5], out=u_scratch[0])
    fnp.add(u_scratch[0], products[6], out=u_scratch[1])
    fnp.add(u_scratch[0], products[4], out=u_scratch[2])
    fnp.add(u_scratch[2], products[2], out=c12)
    fnp.subtract(u_scratch[1], products[3], out=c21)
    fnp.add(u_scratch[1], products[4], out=c22)
'''

NEW_RECONSTRUCTION = '''    p0 = products[0]
    p1 = products[1]
    p2 = products[2]
    p3 = products[3]
    p4 = products[4]
    p5 = products[5]
    p6 = products[6]
    u0 = u_scratch[0]
    u1 = u_scratch[1]
    u2 = u_scratch[2]
    fnp.add(p0, p1, out=c11)
    fnp.add(p0, p5, out=u0)
    fnp.add(u0, p6, out=u1)
    fnp.add(u0, p4, out=u2)
    fnp.add(u2, p2, out=c12)
    fnp.subtract(u1, p3, out=c21)
    fnp.add(u1, p4, out=c22)
'''


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AssertionError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    source = SOURCE.read_bytes()
    observed = sha256(source)
    if observed != EXPECTED_D1_SHA256:
        raise AssertionError((observed, EXPECTED_D1_SHA256))
    text = source.decode("utf-8")
    text = replace_once(text, OLD_HEADER, NEW_HEADER, "unchecked header")
    text = replace_once(text, OLD_WEIGHT_BIND, NEW_WEIGHT_BIND, "weight bind")
    text = replace_once(
        text,
        OLD_SCRATCH_CREATE,
        NEW_SCRATCH_CREATE,
        "persistent weight workspaces",
    )
    text = replace_once(
        text,
        OLD_SCRATCH_STORE,
        NEW_SCRATCH_STORE,
        "scratch tuple store",
    )
    text = replace_once(
        text,
        OLD_SCRATCH_LOAD,
        NEW_SCRATCH_LOAD,
        "scratch tuple load",
    )
    text = replace_once(
        text,
        OLD_WEIGHT_POWERS,
        NEW_WEIGHT_POWERS,
        "reused weight powers",
    )
    text = replace_once(text, OLD_A_STAGE, NEW_A_STAGE, "cached S views")
    text = replace_once(text, OLD_B_STAGE, NEW_B_STAGE, "cached T views")
    text = replace_once(
        text,
        OLD_RECONSTRUCTION,
        NEW_RECONSTRUCTION,
        "cached product/U views",
    )
    expected_calls = text.count("_cp3_winograd_arena_depth1(")
    if expected_calls != 4:
        raise AssertionError(("kernel call sites", expected_calls))
    text = text.replace(
        "_cp3_winograd_arena_depth1(",
        "_cp3_winograd_arena_depth1_unchecked(",
    )
    if text.count("_cp3_winograd_arena_depth1_unchecked(") != 5:
        raise AssertionError("D1U definition/call count changed")
    OUTPUT.write_bytes(text.encode("utf-8"))
    print(sha256(OUTPUT.read_bytes()))


if __name__ == "__main__":
    main()
