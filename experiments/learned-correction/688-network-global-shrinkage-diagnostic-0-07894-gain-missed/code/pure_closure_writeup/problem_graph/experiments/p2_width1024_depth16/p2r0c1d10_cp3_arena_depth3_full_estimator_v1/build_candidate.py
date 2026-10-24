#!/usr/bin/env python3
"""Mechanically replace the audited D1 CP3 arena by audited depth three.

The scientific closure is unchanged.  ``--cp4-cap`` is deliberately a build
parameter so a separately selected cap can be stacked without touching this
kernel.  The default builds the standalone cap-2048 discriminator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "p2r0c1d9_cp3_arena_depth1_full_estimator_v1" / "estimator.py"
LADDER = P2 / "p2r0c1d7_cp3_arena_depth_ladder_v1" / "DEPTH_LADDER_RECEIPT.json"
DEFAULT_DESTINATION = HERE / "estimator.py"
EXPECTED_SOURCE_SHA256 = "f2ec257c891241b57c85803c70c73f365e075711e2b2cd214d9b58250762892e"
EXPECTED_LADDER_SHA256 = "98b48f71588c1bb332fbea6608b4e5d32b7b433164eb276143640a2e2858ae19"
EXPECTED_DEFAULT_SHA256 = "1771a616c86507ce1aad75dced22be57caddb93f5121034c5022d4a11c62c8c4"

WIDTH = 1_024
LEVELS = 3
MAX_PUSH_RANK = 15_360
MAX_READ_RANK = 14_336
EXPECTED_PUSH_ELEMENTS = 479_084_544
EXPECTED_READ_ELEMENTS = 360_955_904
EXPECTED_ARENA_ROSTER = 660_308_197_376
EXPECTED_HOT_SAVING = 305_778_425_856


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def product(shape: tuple[int, ...]) -> int:
    result = 1
    for extent in shape:
        result *= int(extent)
    return result


def integer(value: Fraction) -> int:
    if value.denominator != 1:
        raise AssertionError(value)
    return value.numerator


def layout(m: int, k: int, n: int, levels: int) -> tuple[int, tuple, int]:
    """Byte-for-byte transcription of the sealed depth-ladder layout."""
    q = Fraction(7, 4)
    a, b, c = m * k, k * n, m * n
    regions: dict[str, tuple[int, int]] = {}
    cursor = 0

    def add(name: str, size: Fraction | int) -> None:
        nonlocal cursor
        count = integer(size) if isinstance(size, Fraction) else int(size)
        regions[name] = (cursor, cursor + count)
        cursor += count

    for level in range(1, levels + 1):
        add(f"a{level}", integer(q**level * a))
        add(f"b{level}", integer(q**level * b))
    cap = q ** (levels - 1)
    add("dependency", integer(cap * max(a, b)))
    add("leaf", integer(q**levels * c))
    add("product0", integer(cap * c))
    add("product1", integer(cap * c))
    add("u", integer(cap * c))
    add("output0", c)
    add("output1", c)
    specs: list[tuple[int, int, tuple[int, ...]]] = []

    def view(owner: str, shape: tuple[int, ...]) -> None:
        start, owner_stop = regions[owner]
        stop = start + product(shape)
        if stop > owner_stop:
            raise AssertionError((owner, shape, stop, owner_stop))
        specs.append((start, stop, shape))

    for level in range(1, levels + 1):
        prefix = (7,) * level
        view(f"a{level}", prefix + (m // 2**level, k // 2**level))
        view(f"b{level}", prefix + (k // 2**level, n // 2**level))
    for level in range(1, levels + 1):
        prefix = (7,) * (level - 1)
        view("dependency", (4,) + prefix + (m // 2**level, k // 2**level))
    for level in range(1, levels + 1):
        prefix = (7,) * (level - 1)
        view("dependency", (4,) + prefix + (k // 2**level, n // 2**level))
    view("leaf", (7,) * levels + (m // 2**levels, n // 2**levels))
    for unwind, remaining in enumerate(range(levels - 1, 0, -1)):
        view(
            "product0" if unwind % 2 == 0 else "product1",
            (7,) * remaining + (m // 2**remaining, n // 2**remaining),
        )
    for level in range(levels, 0, -1):
        view(
            "u",
            (3,) + (7,) * (level - 1)
            + (m // 2**level, n // 2**level),
        )
    view("output0", (m, n))
    view("output1", (m, n))
    if len(specs) != 6 * levels + 2:
        raise AssertionError(len(specs))
    return cursor, tuple(specs), sum(stop - start for start, stop, _ in specs)


def mm(m: int, k: int, n: int) -> int:
    return m * n * (2 * k - 1)


def arena_flops(m: int, k: int, n: int, levels: int) -> int:
    return 7**levels * mm(m // 2**levels, k // 2**levels, n // 2**levels) + sum(
        7**level * (11 * (m * k + k * n) + 7 * m * n) // 4 ** (level + 1)
        for level in range(levels)
    )


KERNEL = '''def _cp3_winograd_arena_depth3(a, b, out, arena_views):
    """Audited three-level fixed-arena Winograd product with explicit out=."""
    levels = _CP3_WINOGRAD_LEVELS
    m = int(a.shape[-2])
    k = int(a.shape[-1])
    n = int(b.shape[-1])
    if (
        int(b.shape[-2]) != k
        or m % 2**levels != 0
        or k % 2**levels != 0
        or n % 2**levels != 0
        or tuple(out.shape) != (m, n)
    ):
        raise ValueError("D3 arena kernel requires compatible /8 matrices")

    cursor = 0
    a_views = arena_views[cursor : cursor + 2 * levels : 2]
    b_views = arena_views[cursor + 1 : cursor + 2 * levels : 2]
    cursor += 2 * levels
    dependency_a = arena_views[cursor : cursor + levels]
    cursor += levels
    dependency_b = arena_views[cursor : cursor + levels]
    cursor += levels
    leaf_view = arena_views[cursor]
    cursor += 1
    intermediates = arena_views[cursor : cursor + levels - 1]
    cursor += levels - 1
    u_scratch = arena_views[cursor : cursor + levels]

    current_a, current_b = a, b
    for index in range(levels):
        level = index + 1
        m2, k2, n2 = m // 2**level, k // 2**level, n // 2**level
        a11 = current_a[..., :m2, :k2]
        a12 = current_a[..., :m2, k2 : 2 * k2]
        a21 = current_a[..., m2 : 2 * m2, :k2]
        a22 = current_a[..., m2 : 2 * m2, k2 : 2 * k2]
        b11 = current_b[..., :k2, :n2]
        b12 = current_b[..., :k2, n2 : 2 * n2]
        b21 = current_b[..., k2 : 2 * k2, :n2]
        b22 = current_b[..., k2 : 2 * k2, n2 : 2 * n2]

        scratch_a = dependency_a[index][..., :m2, :k2]
        fnp.add(a21, a22, out=scratch_a[0])
        fnp.subtract(scratch_a[0], a11, out=scratch_a[1])
        fnp.subtract(a11, a21, out=scratch_a[2])
        fnp.subtract(a12, scratch_a[1], out=scratch_a[3])
        next_a = a_views[index][..., :m2, :k2]
        for destination, source in zip(
            next_a, (a11, a12, scratch_a[3], a22, scratch_a[0], scratch_a[1], scratch_a[2])
        ):
            fnp.copyto(destination, source)

        scratch_b = dependency_b[index][..., :k2, :n2]
        fnp.subtract(b12, b11, out=scratch_b[0])
        fnp.subtract(b22, scratch_b[0], out=scratch_b[1])
        fnp.subtract(b22, b12, out=scratch_b[2])
        fnp.subtract(scratch_b[1], b21, out=scratch_b[3])
        next_b = b_views[index][..., :k2, :n2]
        for destination, source in zip(
            next_b, (b11, b21, b22, scratch_b[3], scratch_b[0], scratch_b[1], scratch_b[2])
        ):
            fnp.copyto(destination, source)
        current_a, current_b = next_a, next_b

    current = leaf_view[..., : m // 2**levels, : n // 2**levels]
    fnp.matmul(current_a, current_b, out=current)
    for unwind, level in enumerate(range(levels, 0, -1)):
        rows, columns = m // 2**level, n // 2**level
        destination = intermediates[unwind] if level > 1 else out
        destination = destination[..., : 2 * rows, : 2 * columns]
        p0, p1, p2, p3, p4, p5, p6 = (current[index] for index in range(7))
        scratch = u_scratch[unwind][..., :rows, :columns]
        top_left = destination[..., :rows, :columns]
        top_right = destination[..., :rows, columns : 2 * columns]
        bottom_left = destination[..., rows : 2 * rows, :columns]
        bottom_right = destination[..., rows : 2 * rows, columns : 2 * columns]
        fnp.add(p0, p1, out=top_left)
        fnp.add(p0, p5, out=scratch[0])
        fnp.add(scratch[0], p6, out=scratch[1])
        fnp.add(scratch[0], p4, out=scratch[2])
        fnp.add(scratch[2], p2, out=top_right)
        fnp.subtract(scratch[1], p3, out=bottom_left)
        fnp.add(scratch[1], p4, out=bottom_right)
        current = destination
    return out


'''


def build(cp4_cap: int) -> tuple[bytes, dict[str, int]]:
    if not 1 <= cp4_cap <= 2048:
        raise ValueError("cp4 cap must be in [1, 2048]")
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("D1 source authority changed")
    if sha256(LADDER) != EXPECTED_LADDER_SHA256:
        raise RuntimeError("depth-ladder authority changed")
    push_elements, push_specs, push_bill = layout(WIDTH, WIDTH, MAX_PUSH_RANK, LEVELS)
    read_elements, read_specs, read_bill = layout(WIDTH, MAX_READ_RANK, WIDTH, LEVELS)
    if (push_elements, read_elements) != (EXPECTED_PUSH_ELEMENTS, EXPECTED_READ_ELEMENTS):
        raise AssertionError((push_elements, read_elements))

    roster = (
        [(WIDTH, WIDTH, WIDTH * layer) for layer in range(1, 16) for _ in range(2)]
        + [(WIDTH, WIDTH * layer, WIDTH) for layer in range(1, 15) for _ in range(2)]
    )
    dense = sum(mm(*shape) for shape in roster)
    arena = sum(arena_flops(*shape, LEVELS) for shape in roster)
    if dense != 966_086_623_232 or arena != EXPECTED_ARENA_ROSTER:
        raise AssertionError((dense, arena))
    if dense - arena != EXPECTED_HOT_SAVING:
        raise AssertionError(dense - arena)

    text = SOURCE.read_text()
    constants_start = text.index("_CP3_CAP = 16384\n")
    constants_end = text.index("_Q24_TERMS = 7\n", constants_start)
    constants = (
        "_CP3_CAP = 16384\n"
        f"_CP3_WINOGRAD_LEVELS = {LEVELS}\n"
        f"_CP3_WINOGRAD_ARENA_ELEMENTS = {push_elements}\n"
        f"_CP3_PUSH_ARENA_VIEW_SPECS = {push_specs!r}\n"
        f"_CP3_READ_ARENA_VIEW_SPECS = {read_specs!r}\n"
        f"_CP4_CAP = {cp4_cap}\n"
    )
    text = text[:constants_start] + constants + text[constants_end:]
    kernel_start = text.index("def _cp3_winograd_arena_depth1(")
    read_start = text.index("def _cp_read(", kernel_start)
    text = text[:kernel_start] + KERNEL + text[read_start:]
    if text.count("_cp3_winograd_arena_depth1") != 4:
        raise AssertionError(text.count("_cp3_winograd_arena_depth1"))
    text = text.replace("_cp3_winograd_arena_depth1", "_cp3_winograd_arena_depth3")
    if text.count("first = arena_views[6]") != 1 or text.count("second = arena_views[7]") != 1:
        raise AssertionError("D1 output view anchors changed")
    text = text.replace("first = arena_views[6]", "first = arena_views[-2]", 1)
    text = text.replace("second = arena_views[7]", "second = arena_views[-1]", 1)
    payload = text.encode("utf-8")
    return payload, {
        "push_arena_elements": push_elements,
        "push_arena_bytes": 4 * push_elements,
        "read_arena_elements": read_elements,
        "read_arena_bytes": 4 * read_elements,
        "push_binding_flops": push_bill,
        "read_binding_flops": read_bill,
        "cold_binding_flops": push_bill + read_bill,
        "hot_cp3_roster_flops": arena,
        "hot_cp3_saving_flops": dense - arena,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp4-cap", type=int, default=2048)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    payload, accounting = build(args.cp4_cap)
    observed = sha256_bytes(payload)
    if args.cp4_cap == 2048 and args.destination == DEFAULT_DESTINATION:
        if EXPECTED_DEFAULT_SHA256 != "TO_BE_FROZEN" and observed != EXPECTED_DEFAULT_SHA256:
            raise RuntimeError((observed, EXPECTED_DEFAULT_SHA256))
    args.destination.write_bytes(payload)
    print(json.dumps({"source_sha256": observed, "cp4_cap": args.cp4_cap, **accounting}, sort_keys=True))


if __name__ == "__main__":
    main()
