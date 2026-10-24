"""Isolated tiny-shape prototype of the D6 fixed-arena D4 kernel.

This module imports no estimator, fixture, or repository helper.  All
large-array writes in the hot path are explicit FlopScope ``out=`` writes.
Shape changes are performed only while binding persistent arena views and are
therefore billed once, separately from the hot kernel inventory.
"""

from __future__ import annotations

from dataclasses import dataclass

import flopscope.numpy as fnp


DEPTH = 4
RADIX = 7


@dataclass(frozen=True)
class Region:
    name: str
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class ArenaLayout:
    max_m: int
    max_k: int
    max_n: int
    regions: tuple[Region, ...]
    total_elements: int

    def region(self, name: str) -> Region:
        for item in self.regions:
            if item.name == name:
                return item
        raise KeyError(name)


@dataclass(frozen=True)
class ArenaViews:
    arena: object
    layout: ArenaLayout
    a_levels: tuple[object, ...]
    b_levels: tuple[object, ...]
    dependency_a: tuple[object, ...]
    dependency_b: tuple[object, ...]
    leaf_product: object
    product_bank0_l3: object
    product_bank0_l1: object
    product_bank1_l2: object
    u_scratch: tuple[object, ...]
    output_banks: tuple[object, object]


def _exact_scaled(base: int, numerator: int, denominator: int) -> int:
    value = base * numerator
    if value % denominator:
        raise ValueError("nonintegral arena term")
    return value // denominator


def make_layout(max_m: int, max_k: int, max_n: int) -> ArenaLayout:
    """Bind the conservative D6 admission formula to exact flat intervals."""
    for value in (max_m, max_k, max_n):
        if value <= 0 or value % (2**DEPTH):
            raise ValueError("all maximum dimensions must be positive multiples of 16")

    a_size = max_m * max_k
    b_size = max_k * max_n
    c_size = max_m * max_n
    regions: list[Region] = []
    cursor = 0

    def add(name: str, size: int) -> None:
        nonlocal cursor
        if size <= 0:
            raise ValueError("arena regions must be nonempty")
        regions.append(Region(name, cursor, cursor + size))
        cursor += size

    for level in range(1, DEPTH + 1):
        add(
            f"a_level_{level}",
            _exact_scaled(a_size, RADIX**level, 4**level),
        )
        add(
            f"b_level_{level}",
            _exact_scaled(b_size, RADIX**level, 4**level),
        )
    add(
        "dependency",
        _exact_scaled(max(a_size, b_size), RADIX**3, 4**3),
    )
    add("leaf_product", _exact_scaled(c_size, RADIX**4, 4**4))
    add("product_bank_0", _exact_scaled(c_size, RADIX**3, 4**3))
    add("product_bank_1", _exact_scaled(c_size, RADIX**3, 4**3))
    add("u_scratch", _exact_scaled(c_size, RADIX**3, 4**3))
    add("output_bank_0", c_size)
    add("output_bank_1", c_size)
    return ArenaLayout(max_m, max_k, max_n, tuple(regions), cursor)


def allocate_arena(layout: ArenaLayout):
    """Allocate the one and only flat float32 arena."""
    return fnp.empty((layout.total_elements,), dtype=fnp.float32)


def _bound_view(arena, region: Region, shape: tuple[int, ...]):
    size = 1
    for extent in shape:
        size *= extent
    if size > region.size:
        raise ValueError(f"view exceeds region {region.name}")
    # Basic slicing is a free view; reshape is deliberately a billed operation.
    return fnp.reshape(arena[region.start : region.start + size], shape)


def bind_arena_views(arena, layout: ArenaLayout) -> ArenaViews:
    """Create all 26 persistent max-shape views; call under a binding meter."""
    if tuple(arena.shape) != (layout.total_elements,):
        raise ValueError("arena shape does not match layout")
    if str(arena.dtype) != "float32":
        raise TypeError("arena must be float32")

    m, k, n = layout.max_m, layout.max_k, layout.max_n
    a_levels = []
    b_levels = []
    dependency_a = []
    dependency_b = []
    dep_region = layout.region("dependency")
    for level in range(1, DEPTH + 1):
        prefix = (RADIX,) * level
        a_levels.append(
            _bound_view(
                arena,
                layout.region(f"a_level_{level}"),
                prefix + (m // 2**level, k // 2**level),
            )
        )
        b_levels.append(
            _bound_view(
                arena,
                layout.region(f"b_level_{level}"),
                prefix + (k // 2**level, n // 2**level),
            )
        )
        parent_prefix = (RADIX,) * (level - 1)
        dependency_a.append(
            _bound_view(
                arena,
                dep_region,
                (4,) + parent_prefix + (m // 2**level, k // 2**level),
            )
        )
        dependency_b.append(
            _bound_view(
                arena,
                dep_region,
                (4,) + parent_prefix + (k // 2**level, n // 2**level),
            )
        )

    leaf_product = _bound_view(
        arena,
        layout.region("leaf_product"),
        (RADIX,) * DEPTH + (m // 16, n // 16),
    )
    product_bank0_l3 = _bound_view(
        arena,
        layout.region("product_bank_0"),
        (RADIX,) * 3 + (m // 8, n // 8),
    )
    product_bank0_l1 = _bound_view(
        arena,
        layout.region("product_bank_0"),
        (RADIX,) + (m // 2, n // 2),
    )
    product_bank1_l2 = _bound_view(
        arena,
        layout.region("product_bank_1"),
        (RADIX,) * 2 + (m // 4, n // 4),
    )
    u_scratch = []
    for level in range(DEPTH, 0, -1):
        u_scratch.append(
            _bound_view(
                arena,
                layout.region("u_scratch"),
                (RADIX,) * (level - 1)
                + (m // 2**level, n // 2**level),
            )
        )
    output_banks = (
        _bound_view(arena, layout.region("output_bank_0"), (m, n)),
        _bound_view(arena, layout.region("output_bank_1"), (m, n)),
    )
    return ArenaViews(
        arena=arena,
        layout=layout,
        a_levels=tuple(a_levels),
        b_levels=tuple(b_levels),
        dependency_a=tuple(dependency_a),
        dependency_b=tuple(dependency_b),
        leaf_product=leaf_product,
        product_bank0_l3=product_bank0_l3,
        product_bank0_l1=product_bank0_l1,
        product_bank1_l2=product_bank1_l2,
        u_scratch=tuple(u_scratch),
        output_banks=output_banks,
    )


def _validate_active(a, b, views: ArenaViews, out) -> tuple[int, int, int]:
    if len(a.shape) != 2 or len(b.shape) != 2 or len(out.shape) != 2:
        raise ValueError("prototype operands and destination must be matrices")
    m, k = int(a.shape[0]), int(a.shape[1])
    if int(b.shape[0]) != k:
        raise ValueError("contracted dimensions do not match")
    n = int(b.shape[1])
    if tuple(out.shape) != (m, n):
        raise ValueError("destination shape mismatch")
    if str(a.dtype) != "float32" or str(b.dtype) != "float32":
        raise TypeError("prototype operands must be float32")
    if str(out.dtype) != "float32":
        raise TypeError("prototype destination must be float32")
    if m != views.layout.max_m or k > views.layout.max_k or n > views.layout.max_n:
        raise ValueError("active shape exceeds or changes the frozen maximum")
    for value in (m, k, n):
        if value <= 0 or value % 16:
            raise ValueError("active dimensions must be positive multiples of 16")
    return m, k, n


def _forward_depth4(a, b, views: ArenaViews, active: tuple[int, int, int]):
    """Transform operands in D5's exact dependency and branch order."""
    m, k, n = active
    current_a = a
    current_b = b
    for index in range(DEPTH):
        level = index + 1
        m2 = m // 2**level
        k2 = k // 2**level
        n2 = n // 2**level

        a11, a12 = current_a[..., :m2, :k2], current_a[..., :m2, k2 : 2 * k2]
        a21, a22 = current_a[..., m2 : 2 * m2, :k2], current_a[..., m2 : 2 * m2, k2 : 2 * k2]
        b11, b12 = current_b[..., :k2, :n2], current_b[..., :k2, n2 : 2 * n2]
        b21, b22 = current_b[..., k2 : 2 * k2, :n2], current_b[..., k2 : 2 * k2, n2 : 2 * n2]

        dep_a = views.dependency_a[index][..., :m2, :k2]
        fnp.add(a21, a22, out=dep_a[0])
        fnp.subtract(dep_a[0], a11, out=dep_a[1])
        fnp.subtract(a11, a21, out=dep_a[2])
        fnp.subtract(a12, dep_a[1], out=dep_a[3])
        next_a = views.a_levels[index][..., :m2, :k2]
        fnp.copyto(next_a[0], a11)
        fnp.copyto(next_a[1], a12)
        fnp.copyto(next_a[2], dep_a[3])
        fnp.copyto(next_a[3], a22)
        fnp.copyto(next_a[4], dep_a[0])
        fnp.copyto(next_a[5], dep_a[1])
        fnp.copyto(next_a[6], dep_a[2])

        dep_b = views.dependency_b[index][..., :k2, :n2]
        fnp.subtract(b12, b11, out=dep_b[0])
        fnp.subtract(b22, dep_b[0], out=dep_b[1])
        fnp.subtract(b22, b12, out=dep_b[2])
        fnp.subtract(dep_b[1], b21, out=dep_b[3])
        next_b = views.b_levels[index][..., :k2, :n2]
        fnp.copyto(next_b[0], b11)
        fnp.copyto(next_b[1], b21)
        fnp.copyto(next_b[2], b22)
        fnp.copyto(next_b[3], dep_b[3])
        fnp.copyto(next_b[4], dep_b[0])
        fnp.copyto(next_b[5], dep_b[1])
        fnp.copyto(next_b[6], dep_b[2])
        current_a = next_a
        current_b = next_b
    return current_a, current_b


def _reconstruct_depth4(leaf, views: ArenaViews, active: tuple[int, int, int], out):
    """Reconstruct u1..u7 in frozen order directly into quadrants."""
    m, _k, n = active
    current = leaf
    for unwind_index, level in enumerate(range(DEPTH, 0, -1)):
        rows = m // 2**level
        cols = n // 2**level
        if level == 4:
            destination = views.product_bank0_l3[..., : 2 * rows, : 2 * cols]
        elif level == 3:
            destination = views.product_bank1_l2[..., : 2 * rows, : 2 * cols]
        elif level == 2:
            destination = views.product_bank0_l1[..., : 2 * rows, : 2 * cols]
        else:
            destination = out
        p0, p1, p2, p3, p4, p5, p6 = (
            current[0],
            current[1],
            current[2],
            current[3],
            current[4],
            current[5],
            current[6],
        )
        top_left = destination[..., :rows, :cols]
        top_right = destination[..., :rows, cols : 2 * cols]
        bottom_left = destination[..., rows : 2 * rows, :cols]
        bottom_right = destination[..., rows : 2 * rows, cols : 2 * cols]
        scratch = views.u_scratch[unwind_index][..., :rows, :cols]
        fnp.add(p0, p1, out=top_left)                 # u1
        fnp.add(p0, p5, out=scratch)                  # u2
        fnp.add(scratch, p6, out=bottom_right)         # u3 (temporary quadrant)
        fnp.add(scratch, p4, out=top_right)            # u4
        fnp.add(top_right, p2, out=top_right)           # u5
        fnp.subtract(bottom_right, p3, out=bottom_left) # u6
        fnp.add(bottom_right, p4, out=bottom_right)     # u7
        current = destination
    return out


def winograd_depth4(a, b, views: ArenaViews, out):
    """Compute one active matrix product in the prebound fixed arena."""
    active = _validate_active(a, b, views, out)
    leaf_a, leaf_b = _forward_depth4(a, b, views, active)
    m, k, n = active
    leaf = views.leaf_product[
        ..., : m // 16, : n // 16
    ]
    fnp.matmul(leaf_a, leaf_b, out=leaf)
    return _reconstruct_depth4(leaf, views, active, out)


def active_leaf_views(views: ArenaViews, active: tuple[int, int, int]):
    """Expose read-only semantic leaf views for the isolated verifier."""
    m, k, n = active
    return (
        views.a_levels[-1][..., : m // 16, : k // 16],
        views.b_levels[-1][..., : k // 16, : n // 16],
        views.leaf_product[..., : m // 16, : n // 16],
    )
