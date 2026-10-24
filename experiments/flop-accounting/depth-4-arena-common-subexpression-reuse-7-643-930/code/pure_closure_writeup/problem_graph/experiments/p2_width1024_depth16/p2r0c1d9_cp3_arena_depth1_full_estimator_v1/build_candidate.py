#!/usr/bin/env python3
"""Build the exact depth-1 arena descendant of graded no-eigh #328439.

This is a source-only mechanical builder.  It verifies the incumbent and
depth-ladder authorities, writes estimator.py, and never imports an estimator
or opens a fixture, target, score, board result, or network resource.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "public_noeigh_card_v1" / "estimator.py"
DEPTH_LADDER = P2 / "p2r0c1d7_cp3_arena_depth_ladder_v1" / "DEPTH_LADDER_RECEIPT.json"
DESTINATION = HERE / "estimator.py"

EXPECTED_SOURCE_SHA256 = (
    "373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7"
)
EXPECTED_DEPTH_LADDER_SHA256 = (
    "98b48f71588c1bb332fbea6608b4e5d32b7b433164eb276143640a2e2858ae19"
)
EXPECTED_CANDIDATE_SHA256 = (
    "f2ec257c891241b57c85803c70c73f365e075711e2b2cd214d9b58250762892e"
)

WIDTH = 1_024
LEVELS = 1
MAX_PUSH_RANK = 15_360
MAX_READ_RANK = 14_336
ARENA_ELEMENTS = 151_257_088
EXPECTED_HOT_SAVING = 118_525_263_872
EXPECTED_COLD_BINDING_BILL = 202_375_168
Q = Fraction(7, 4)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one exact anchor, found {count}")
    return text.replace(old, new, 1)


def integer(value: Fraction) -> int:
    if value.denominator != 1:
        raise AssertionError(value)
    return value.numerator


def elements(shape: tuple[int, ...]) -> int:
    result = 1
    for extent in shape:
        result *= extent
    return result


def make_layout(m: int, k: int, n: int) -> tuple[int, tuple, int]:
    """Reproduce the sealed D1 ladder's physical layout and eight views."""
    a = m * k
    b = k * n
    c = m * n
    cursor = 0
    regions: dict[str, tuple[int, int]] = {}

    def region(name: str, size: Fraction | int) -> None:
        nonlocal cursor
        count = integer(size) if isinstance(size, Fraction) else int(size)
        regions[name] = (cursor, cursor + count)
        cursor += count

    region("a1", Q * a)
    region("b1", Q * b)
    region("dependency", max(a, b))
    region("leaf", Q * c)
    region("product0", c)
    region("product1", c)
    region("u", c)
    region("output0", c)
    region("output1", c)

    specs: list[tuple[int, int, tuple[int, ...]]] = []

    def view(owner: str, shape: tuple[int, ...]) -> None:
        start, owner_stop = regions[owner]
        stop = start + elements(shape)
        if stop > owner_stop:
            raise AssertionError((owner, shape, start, stop, owner_stop))
        specs.append((start, stop, shape))

    view("a1", (7, m // 2, k // 2))
    view("b1", (7, k // 2, n // 2))
    view("dependency", (4, m // 2, k // 2))
    view("dependency", (4, k // 2, n // 2))
    view("leaf", (7, m // 2, n // 2))
    view("u", (3, m // 2, n // 2))
    view("output0", (m, n))
    view("output1", (m, n))
    if len(specs) != 8:
        raise AssertionError(len(specs))
    reshape_bill = sum(stop - start for start, stop, _shape in specs)
    return cursor, tuple(specs), reshape_bill


def constants_text(push_specs: tuple, read_specs: tuple) -> str:
    return (
        "_CP3_CAP = 16384\n"
        "_CP3_WINOGRAD_LEVELS = 1\n"
        "_CP3_WINOGRAD_ARENA_ELEMENTS = 151257088\n"
        f"_CP3_PUSH_ARENA_VIEW_SPECS = {push_specs!r}\n"
        f"_CP3_READ_ARENA_VIEW_SPECS = {read_specs!r}\n"
        "_CP4_CAP = 2048\n"
    )


NEW_FUNCTIONS = '''def _cp3_bind_arena_views(arena, view_specs):
    """Bind the sealed D1 arena views once on the first prediction."""
    return tuple(
        fnp.reshape(arena[start:stop], shape)
        for start, stop, shape in view_specs
    )


def _cp3_winograd_arena_depth1(a, b, out, arena_views):
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
    a11 = a[:m2, :k2]
    a12 = a[:m2, k2:]
    a21 = a[m2:, :k2]
    a22 = a[m2:, k2:]
    b11 = b[:k2, :n2]
    b12 = b[:k2, n2:]
    b21 = b[k2:, :n2]
    b22 = b[k2:, n2:]

    transformed_a = arena_views[0][..., :m2, :k2]
    transformed_b = arena_views[1][..., :k2, :n2]
    scratch_s = arena_views[2][..., :m2, :k2]
    scratch_t = arena_views[3][..., :k2, :n2]

    fnp.add(a21, a22, out=scratch_s[0])
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

    fnp.subtract(b12, b11, out=scratch_t[0])
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

    products = arena_views[4][..., :m2, :n2]
    fnp.matmul(transformed_a, transformed_b, out=products)
    u_scratch = arena_views[5][..., :m2, :n2]
    c11 = out[:m2, :n2]
    c12 = out[:m2, n2:]
    c21 = out[m2:, :n2]
    c22 = out[m2:, n2:]
    fnp.add(products[0], products[1], out=c11)
    fnp.add(products[0], products[5], out=u_scratch[0])
    fnp.add(u_scratch[0], products[6], out=u_scratch[1])
    fnp.add(u_scratch[0], products[4], out=u_scratch[2])
    fnp.add(u_scratch[2], products[2], out=c12)
    fnp.subtract(u_scratch[1], products[3], out=c21)
    fnp.add(u_scratch[1], products[4], out=c22)
    return out


def _cp_read(cp_lambda, cp_p, cp_q, product_a, product_b, arena_views):
    rank = int(cp_lambda.shape[0])
    if rank == 0:
        width = int(cp_p.shape[0])
        return fnp.zeros((width, width), dtype=fnp.float64)
    width = int(cp_p.shape[0])
    active_a = fnp.reshape(product_a[:width * rank], (width, rank))
    active_b = fnp.reshape(product_b[:width * rank], (width, rank))
    fnp.multiply(cp_p, cp_p, out=active_a)
    fnp.multiply(cp_q, cp_lambda[None, :], out=active_b)
    first = arena_views[6]
    _cp3_winograd_arena_depth1(
        active_a, fnp.transpose(active_b), first, arena_views
    )
    fnp.multiply(cp_p, cp_q, out=active_a)
    fnp.multiply(cp_p, cp_lambda[None, :], out=active_b)
    second = arena_views[7]
    _cp3_winograd_arena_depth1(
        active_a, fnp.transpose(active_b), second, arena_views
    )
    return fnp.asarray(first + 2.0 * second, dtype=fnp.float64)


def _cp_k3(cp_lambda, cp_p, cp_q, product_a):
    """Marginal-only final CP read; no n-by-n K21 plane is materialized."""
    rank = int(cp_lambda.shape[0])
    width = int(cp_p.shape[0])
    active_a = fnp.reshape(product_a[:width * rank], (width, rank))
    fnp.multiply(cp_p, cp_p, out=active_a)
    fnp.multiply(active_a, cp_q, out=active_a)
    return fnp.asarray(
        3.0 * (active_a @ cp_lambda),
        dtype=fnp.float64,
    )


def _cp_push(
    cp_lambda, cp_p, cp_q, weight_context, destination_p, destination_q,
    arena_views,
):
    rank = int(cp_lambda.shape[0])
    if rank == 0:
        return cp_lambda, destination_p[:, :0], destination_q[:, :0]
    _weight32, weight32_t, _weight2, _weight2_t, _weight3, _weight3_t = (
        weight_context
    )
    pushed_p = destination_p[:, :rank]
    pushed_q = destination_q[:, :rank]
    _cp3_winograd_arena_depth1(
        weight32_t, cp_p, pushed_p, arena_views
    )
    _cp3_winograd_arena_depth1(
        weight32_t, cp_q, pushed_q, arena_views
    )
    return cp_lambda, pushed_p, pushed_q


'''


def build_candidate(push_specs: tuple, read_specs: tuple) -> bytes:
    source_bytes = SOURCE.read_bytes()
    observed = sha256_bytes(source_bytes)
    if observed != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"incumbent source hash mismatch: {observed}")
    text = source_bytes.decode("utf-8")

    text = replace_once(
        text,
        "_CP3_CAP = 16384\n_CP4_CAP = 2048\n",
        constants_text(push_specs, read_specs),
        "D1 constants",
    )

    function_start = text.index("def _cp_read(")
    function_end = text.index("def _cp_post_read(", function_start)
    if text.count("def _cp_read(") != 1 or text.count("def _cp_post_read(") != 1:
        raise RuntimeError("CP3 function boundary is not unique")
    text = text[:function_start] + NEW_FUNCTIONS + text[function_end:]

    text = replace_once(
        text,
        '''        cp3_context = (
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty(_CP3_CAP, dtype=fnp.float32),
            fnp.empty(width * _CP3_CAP, dtype=fnp.float32),
            fnp.empty(width * _CP3_CAP, dtype=fnp.float32),
        )
''',
        '''        cp3_arena = fnp.empty(
            _CP3_WINOGRAD_ARENA_ELEMENTS, dtype=fnp.float32
        )
        cp3_context = (
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty(_CP3_CAP, dtype=fnp.float32),
            fnp.empty(width * _CP3_CAP, dtype=fnp.float32),
            fnp.empty(width * _CP3_CAP, dtype=fnp.float32),
            cp3_arena,
            _cp3_bind_arena_views(cp3_arena, _CP3_PUSH_ARENA_VIEW_SPECS),
            _cp3_bind_arena_views(cp3_arena, _CP3_READ_ARENA_VIEW_SPECS),
        )
''',
        "persistent D1 arena and views",
    )

    text = replace_once(
        text,
        '''            cp3_context[0][cp_bank_index],
            cp3_context[1][cp_bank_index],
        )
''',
        '''            cp3_context[0][cp_bank_index],
            cp3_context[1][cp_bank_index],
            cp3_context[6],
        )
''',
        "loop D1 push views",
    )

    text = replace_once(
        text,
        '''            cp3_context[3],
            cp3_context[4],
        )
''',
        '''            cp3_context[3],
            cp3_context[4],
            cp3_context[7],
        )
''',
        "loop D1 read views",
    )
    return text.encode("utf-8")


def mm(m: int, k: int, n: int) -> int:
    return m * n * (2 * k - 1)


def d1(m: int, k: int, n: int) -> int:
    return (
        7 * mm(m // 2, k // 2, n // 2)
        + (11 * (m * k + k * n) + 7 * m * n) // 4
    )


def verify_accounting(push_bill: int, read_bill: int) -> None:
    roster = (
        [(WIDTH, WIDTH, WIDTH * layer) for layer in range(1, 16) for _ in range(2)]
        + [(WIDTH, WIDTH * layer, WIDTH) for layer in range(1, 15) for _ in range(2)]
    )
    dense = sum(mm(*shape) for shape in roster)
    arena = sum(d1(*shape) for shape in roster)
    if len(roster) != 58:
        raise AssertionError(len(roster))
    if dense != 966_086_623_232:
        raise AssertionError(dense)
    if arena != 847_561_359_360:
        raise AssertionError(arena)
    if dense - arena != EXPECTED_HOT_SAVING:
        raise AssertionError(dense - arena)
    if push_bill + read_bill != EXPECTED_COLD_BINDING_BILL:
        raise AssertionError(push_bill + read_bill)


def main() -> None:
    if sha256(DEPTH_LADDER) != EXPECTED_DEPTH_LADDER_SHA256:
        raise RuntimeError("depth-ladder authority hash mismatch")
    push_elements, push_specs, push_bill = make_layout(
        WIDTH, WIDTH, MAX_PUSH_RANK
    )
    read_elements, read_specs, read_bill = make_layout(
        WIDTH, MAX_READ_RANK, WIDTH
    )
    if push_elements != ARENA_ELEMENTS:
        raise AssertionError(push_elements)
    if read_elements != 73_138_176:
        raise AssertionError(read_elements)
    if push_bill != 116_916_224 or read_bill != 85_458_944:
        raise AssertionError((push_bill, read_bill))
    verify_accounting(push_bill, read_bill)

    candidate = build_candidate(push_specs, read_specs)
    candidate_sha = sha256_bytes(candidate)
    if (
        EXPECTED_CANDIDATE_SHA256 != "TO_BE_FROZEN"
        and candidate_sha != EXPECTED_CANDIDATE_SHA256
    ):
        raise RuntimeError(f"candidate source hash mismatch: {candidate_sha}")
    DESTINATION.write_bytes(candidate)
    print(candidate_sha)


if __name__ == "__main__":
    main()
