"""Exact depth-5 Winograd kernel with a lifetime-packed fixed arena.

The arithmetic schedule is the audited D1-D4 arena schedule at depth five.
Only physical storage ownership changes: odd/even transform levels alternate,
the leaf reuses storage whose transform lifetime has ended, and unwind buffers
reuse dead transform/leaf storage.  Every live interval remains disjoint.
"""

from __future__ import annotations

from fractions import Fraction
from math import prod

import flopscope.numpy as fnp


LEVELS = 5
RADIX = 7


def _exact(value: Fraction) -> int:
    if value.denominator != 1:
        raise AssertionError(value)
    return value.numerator


def packed_layout(m: int, k: int, n: int):
    """Return arena elements, named intervals, and 30 persistent views."""
    if any(extent % 2**LEVELS for extent in (m, k, n)):
        raise ValueError("packed D5 requires all matrix extents divisible by 32")
    q = Fraction(7, 4)
    a_size, b_size, c_size = m * k, k * n, m * n
    a_odd = _exact(q**5 * a_size)
    a_even = _exact(q**4 * a_size)
    b_odd = _exact(q**5 * b_size)
    b_even = _exact(q**4 * b_size)
    dependency = _exact(q**4 * max(a_size, b_size))
    leaf = _exact(q**5 * c_size)
    product = _exact(q**4 * c_size)

    intervals = {}
    cursor = 0

    def append(name: str, size: int):
        nonlocal cursor
        intervals[name] = (cursor, cursor + size)
        cursor += size

    # Transform-phase peak.  Odd/even levels must coexist only across the
    # transition that creates the next level, so each parity owns one maximum
    # buffer rather than one buffer per level.
    append("a_odd", a_odd)
    append("b_odd", b_odd)
    odd_stop = cursor
    append("a_even", a_even)
    append("b_even", b_even)
    append("dependency", dependency)
    arena_elements = cursor

    free_transform_bytes = a_even + b_even + dependency
    if leaf > free_transform_bytes:
        raise AssertionError("leaf does not fit after even/dependency lifetime")
    leaf_start = arena_elements - leaf
    intervals["leaf"] = (leaf_start, arena_elements)
    if 2 * product > leaf_start:
        raise AssertionError("two live unwind buffers collide with leaf")
    intervals["product0"] = (0, product)
    intervals["u"] = (product, 2 * product)
    intervals["product1"] = (leaf_start, leaf_start + product)
    if odd_stop > leaf_start:
        raise AssertionError("leaf collides with live odd transform buffers")

    specs = []

    def view(owner: str, shape):
        start, owner_stop = intervals[owner]
        stop = start + prod(shape)
        if stop > owner_stop:
            raise AssertionError((owner, shape, stop, owner_stop))
        specs.append((owner, start, stop, tuple(shape)))

    for level in range(1, LEVELS + 1):
        prefix = (RADIX,) * level
        parity = "odd" if level % 2 else "even"
        view(f"a_{parity}", prefix + (m // 2**level, k // 2**level))
        view(f"b_{parity}", prefix + (k // 2**level, n // 2**level))
    for level in range(1, LEVELS + 1):
        prefix = (RADIX,) * (level - 1)
        view(
            "dependency",
            (4,) + prefix + (m // 2**level, k // 2**level),
        )
    for level in range(1, LEVELS + 1):
        prefix = (RADIX,) * (level - 1)
        view(
            "dependency",
            (4,) + prefix + (k // 2**level, n // 2**level),
        )
    view("leaf", (RADIX,) * LEVELS + (m // 32, n // 32))
    for unwind, remaining in enumerate(range(LEVELS - 1, 0, -1)):
        view(
            "product0" if unwind % 2 == 0 else "product1",
            (RADIX,) * remaining
            + (m // 2**remaining, n // 2**remaining),
        )
    for level in range(LEVELS, 0, -1):
        view(
            "u",
            (3,)
            + (RADIX,) * (level - 1)
            + (m // 2**level, n // 2**level),
        )
    if len(specs) != 6 * LEVELS:
        raise AssertionError(len(specs))
    return arena_elements, intervals, tuple(specs)


def bind(arena, specs):
    return tuple(
        fnp.reshape(arena[start:stop], shape)
        for _owner, start, stop, shape in specs
    )


def split_views(views):
    cursor = 0
    a_views = views[cursor : cursor + 2 * LEVELS : 2]
    b_views = views[cursor + 1 : cursor + 2 * LEVELS : 2]
    cursor += 2 * LEVELS
    dependency_a = views[cursor : cursor + LEVELS]
    cursor += LEVELS
    dependency_b = views[cursor : cursor + LEVELS]
    cursor += LEVELS
    leaf = views[cursor]
    cursor += 1
    intermediates = views[cursor : cursor + LEVELS - 1]
    cursor += LEVELS - 1
    u_scratch = views[cursor : cursor + LEVELS]
    cursor += LEVELS
    if cursor != len(views):
        raise AssertionError((cursor, len(views)))
    return (
        a_views,
        b_views,
        dependency_a,
        dependency_b,
        leaf,
        intermediates,
        u_scratch,
    )


def packed_product(a, b, out, arena_views):
    """Audited five-level fixed-arena Winograd product with explicit out=."""
    (
        a_views,
        b_views,
        dependency_a,
        dependency_b,
        leaf_view,
        intermediates,
        u_scratch,
    ) = split_views(arena_views)
    levels = LEVELS
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
        raise ValueError("packed D5 requires compatible /32 matrices")

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
            next_a,
            (a11, a12, scratch_a[3], a22, scratch_a[0], scratch_a[1], scratch_a[2]),
        ):
            fnp.copyto(destination, source)

        scratch_b = dependency_b[index][..., :k2, :n2]
        fnp.subtract(b12, b11, out=scratch_b[0])
        fnp.subtract(b22, scratch_b[0], out=scratch_b[1])
        fnp.subtract(b22, b12, out=scratch_b[2])
        fnp.subtract(scratch_b[1], b21, out=scratch_b[3])
        next_b = b_views[index][..., :k2, :n2]
        for destination, source in zip(
            next_b,
            (b11, b21, b22, scratch_b[3], scratch_b[0], scratch_b[1], scratch_b[2]),
        ):
            fnp.copyto(destination, source)
        current_a, current_b = next_a, next_b

    current = leaf_view[..., : m // 2**levels, : n // 2**levels]
    fnp.matmul(current_a, current_b, out=current)
    for unwind, level in enumerate(range(levels, 0, -1)):
        rows, columns = m // 2**level, n // 2**level
        destination = intermediates[unwind] if level > 1 else out
        destination = destination[..., : 2 * rows, : 2 * columns]
        p0, p1, p2, p3, p4, p5, p6 = (
            current[index] for index in range(RADIX)
        )
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
