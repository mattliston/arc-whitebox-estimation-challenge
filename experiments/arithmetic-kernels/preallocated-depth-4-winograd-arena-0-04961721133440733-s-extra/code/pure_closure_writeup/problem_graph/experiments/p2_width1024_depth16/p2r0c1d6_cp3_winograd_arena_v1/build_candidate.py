#!/usr/bin/env python3
"""Mechanically build the preparation-only P2R0C1D6 arena descendant.

This builder reads only the exact public no-eigh incumbent source and the two
sealed D6 design authorities.  It never imports or calls an estimator and has
no target, score, board, production, submission, or network path.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from fractions import Fraction
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
D5 = P2 / "p2r0c1d5_cp3_winograd_hybrid_v1"
SOURCE = P2 / "public_noeigh_card_v1" / "estimator.py"
DESTINATION = HERE / "estimator.py"
SOURCE_DIFF = HERE / "SOURCE_DIFF.patch"
MANIFEST = HERE / "ACCOUNTING_AND_LIVENESS_MANIFEST.json"
RECEIPT = HERE / "BUILD_RECEIPT.json"

EXPECTED_SOURCE_SHA256 = (
    "373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7"
)
EXPECTED_D5_DESIGN_SHA256 = (
    "808a41b8fa1429e1efaf978cd3f17f18ecf7ae7f21c20a9359546922b3696da7"
)
EXPECTED_INDEPENDENT_DESIGN_SHA256 = (
    "df76169237166cf685ca9e0474643d1d668dcc2ae762f5e9e57af0e6b8afd6f2"
)
EXPECTED_INDEPENDENT_DESIGN_VERIFIER_SHA256 = (
    "58f28d61bcb7047f84d254d5785e7c10ffd6707e6a8e3d61f634d53546933855"
)

WIDTH = 1_024
LEVELS = 4
MAX_PUSH_RANK = 15_360
MAX_READ_RANK = 14_336
ARENA_ELEMENTS = 844_165_120
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


def _integer(value: Fraction) -> int:
    if value.denominator != 1:
        raise AssertionError(value)
    return value.numerator


def _shape_elements(shape: tuple[int, ...]) -> int:
    result = 1
    for extent in shape:
        result *= extent
    return result


def make_layout(mode: str, m: int, k: int, n: int) -> dict:
    """Return exact physical intervals plus the persistent metered views."""
    a = m * k
    b = k * n
    c = m * n
    cursor = 0
    physical: list[dict] = []

    def region(name: str, elements: int) -> dict:
        nonlocal cursor
        item = {
            "name": name,
            "start": cursor,
            "stop": cursor + elements,
            "elements": elements,
        }
        cursor += elements
        physical.append(item)
        return item

    operand_regions: list[tuple[dict, dict]] = []
    for level in range(1, LEVELS + 1):
        operand_regions.append((
            region(f"operand_a_level{level}", _integer(Q**level * a)),
            region(f"operand_b_level{level}", _integer(Q**level * b)),
        ))
    dependency = region(
        "dependency_s_t_reused", _integer(Q**3 * max(a, b))
    )
    leaf = region("leaf_products", _integer(Q**4 * c))
    product0 = region("reconstruction_product_bank0", _integer(Q**3 * c))
    product1 = region("reconstruction_product_bank1", _integer(Q**3 * c))
    u_scratch = region("reconstruction_u_scratch", _integer(Q**3 * c))
    output0 = region("output_bank0", c)
    output1 = region("output_bank1", c)

    expected = (
        sum(Q**level * (a + b) for level in range(1, LEVELS + 1))
        + Q**3 * max(a, b)
        + Q**4 * c
        + 3 * Q**3 * c
        + 2 * c
    )
    assert cursor == _integer(expected)

    views: list[dict] = []

    def view(
        name: str,
        owner: dict,
        shape: tuple[int, ...],
        *,
        offset: int = 0,
        purpose: str,
    ) -> None:
        elements = _shape_elements(shape)
        start = owner["start"] + offset
        stop = start + elements
        assert owner["start"] <= start <= stop <= owner["stop"]
        views.append({
            "name": name,
            "physical_region": owner["name"],
            "start": start,
            "stop": stop,
            "elements": elements,
            "shape": list(shape),
            "purpose": purpose,
        })

    # Indices 0..7: four transformed A/B levels, in alternating A/B order.
    for level, (areg, breg) in enumerate(operand_regions, start=1):
        branches = (7,) * level
        view(
            f"operand_a_level{level}",
            areg,
            branches + (m // 2**level, k // 2**level),
            purpose="transformed_operand_a",
        )
        view(
            f"operand_b_level{level}",
            breg,
            branches + (k // 2**level, n // 2**level),
            purpose="transformed_operand_b",
        )

    # Indices 8..15: the same one physical dependency region is reused first
    # for all four S arrays and then for all four T arrays at each level.
    for level in range(1, LEVELS + 1):
        parents = (7,) * (level - 1)
        view(
            f"dependency_s_level{level}",
            dependency,
            (4,) + parents + (m // 2**level, k // 2**level),
            purpose="four_S_dependencies_sequential_before_T",
        )
    for level in range(1, LEVELS + 1):
        parents = (7,) * (level - 1)
        view(
            f"dependency_t_level{level}",
            dependency,
            (4,) + parents + (k // 2**level, n // 2**level),
            purpose="four_T_dependencies_after_A_destination_is_filled",
        )

    # Index 16: the only leaf-batch matmul destination.
    view(
        "leaf_products",
        leaf,
        (7,) * LEVELS + (m // 16, n // 16),
        purpose="single_2401_leaf_batch",
    )

    # Indices 17..19: alternating destination banks for levels 3,2,1.
    view(
        "product_level3_bank0",
        product0,
        (7,) * 3 + (m // 8, n // 8),
        purpose="reconstruct_level4_to_level3",
    )
    view(
        "product_level2_bank1",
        product1,
        (7,) * 2 + (m // 4, n // 4),
        purpose="reconstruct_level3_to_level2",
    )
    view(
        "product_level1_bank0",
        product0,
        (7,) + (m // 2, n // 2),
        purpose="reconstruct_level2_to_level1",
    )

    # Indices 20..23: three dependency arrays U2/U3/U4.  U1/U5/U6/U7
    # are written directly to disjoint destination quadrants.
    for level in range(LEVELS, 0, -1):
        parents = (7,) * (level - 1)
        view(
            f"u234_reconstruct_level{level}",
            u_scratch,
            (3,) + parents + (m // 2**level, n // 2**level),
            purpose="reused_U2_U3_U4_dependencies",
        )

    # Indices 24..25: sequential CP-read results.  Push writes directly to
    # its existing CP state destinations and leaves these conservative slices
    # untouched.
    view(
        "output_bank0",
        output0,
        (m, n),
        purpose="first_sequential_channel_result_or_reserved_push_output",
    )
    view(
        "output_bank1",
        output1,
        (m, n),
        purpose="second_sequential_channel_result_or_reserved_push_output",
    )

    assert len(views) == 26
    reshape_bill = sum(item["elements"] for item in views)
    return {
        "mode": mode,
        "matrix_shape": [m, k, n],
        "a_elements": a,
        "b_elements": b,
        "c_elements": c,
        "physical_regions": physical,
        "physical_elements": cursor,
        "physical_bytes_float32": 4 * cursor,
        "view_specs": views,
        "view_binding_reshape_bill": reshape_bill,
        "physical_regions_contiguous_and_nonoverlapping": True,
        "alias_overlap_policy": (
            "Only views naming the same physical_region may overlap; they "
            "are temporally disjoint by the frozen schedule."
        ),
    }


def source_specs(layout: dict) -> tuple:
    return tuple(
        (
            int(item["start"]),
            int(item["stop"]),
            tuple(int(extent) for extent in item["shape"]),
        )
        for item in layout["view_specs"]
    )


def arena_constants_text(push_layout: dict, read_layout: dict) -> str:
    push_specs = repr(source_specs(push_layout))
    read_specs = repr(source_specs(read_layout))
    return (
        "_CP3_CAP = 16384\n"
        "_CP3_WINOGRAD_LEVELS = 4\n"
        "_CP3_WINOGRAD_ARENA_ELEMENTS = 844165120\n"
        f"_CP3_PUSH_ARENA_VIEW_SPECS = {push_specs}\n"
        f"_CP3_READ_ARENA_VIEW_SPECS = {read_specs}\n"
        "_CP4_CAP = 2048\n"
    )


NEW_FUNCTIONS = '''def _cp3_bind_arena_views(arena, view_specs):
    """Bind every frozen physical slice/view once in the cold call."""
    return tuple(
        fnp.reshape(arena[start:stop], shape)
        for start, stop, shape in view_specs
    )


def _cp3_transform_level(source_a, source_b, destination_a, destination_b,
                         scratch_s, scratch_t):
    """One allocation-free Winograd operand transform level."""
    m2 = int(source_a.shape[-2]) // 2
    k2 = int(source_a.shape[-1]) // 2
    n2 = int(source_b.shape[-1]) // 2
    a11 = source_a[..., :m2, :k2]
    a12 = source_a[..., :m2, k2:]
    a21 = source_a[..., m2:, :k2]
    a22 = source_a[..., m2:, k2:]
    b11 = source_b[..., :k2, :n2]
    b12 = source_b[..., :k2, n2:]
    b21 = source_b[..., k2:, :n2]
    b22 = source_b[..., k2:, n2:]

    # Frozen D5 S law: S1,S2,S3,S4, then explicit seven-way destination.
    fnp.add(a21, a22, out=scratch_s[0])
    fnp.subtract(scratch_s[0], a11, out=scratch_s[1])
    fnp.subtract(a11, a21, out=scratch_s[2])
    fnp.subtract(a12, scratch_s[1], out=scratch_s[3])
    fnp.copyto(destination_a[0], a11)
    fnp.copyto(destination_a[1], a12)
    fnp.copyto(destination_a[2], scratch_s[3])
    fnp.copyto(destination_a[3], a22)
    fnp.copyto(destination_a[4], scratch_s[0])
    fnp.copyto(destination_a[5], scratch_s[1])
    fnp.copyto(destination_a[6], scratch_s[2])

    # Reuse the same physical dependency interval only after A is complete.
    fnp.subtract(b12, b11, out=scratch_t[0])
    fnp.subtract(b22, scratch_t[0], out=scratch_t[1])
    fnp.subtract(b22, b12, out=scratch_t[2])
    fnp.subtract(scratch_t[1], b21, out=scratch_t[3])
    fnp.copyto(destination_b[0], b11)
    fnp.copyto(destination_b[1], b21)
    fnp.copyto(destination_b[2], b22)
    fnp.copyto(destination_b[3], scratch_t[3])
    fnp.copyto(destination_b[4], scratch_t[0])
    fnp.copyto(destination_b[5], scratch_t[1])
    fnp.copyto(destination_b[6], scratch_t[2])


def _cp3_reconstruct_level(products, destination, u_scratch):
    """Frozen seven-add D5 U law, directly into destination quadrants."""
    m2 = int(destination.shape[-2]) // 2
    n2 = int(destination.shape[-1]) // 2
    c11 = destination[..., :m2, :n2]
    c12 = destination[..., :m2, n2:]
    c21 = destination[..., m2:, :n2]
    c22 = destination[..., m2:, n2:]
    fnp.add(products[0], products[1], out=c11)
    fnp.add(products[0], products[5], out=u_scratch[0])
    fnp.add(u_scratch[0], products[6], out=u_scratch[1])
    fnp.add(u_scratch[0], products[4], out=u_scratch[2])
    fnp.add(u_scratch[2], products[2], out=c12)
    fnp.subtract(u_scratch[1], products[3], out=c21)
    fnp.add(u_scratch[1], products[4], out=c22)


def _cp3_winograd_arena(a, b, out, arena_views):
    """Iterative depth-4 Winograd product with no dynamic array assembly."""
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
        raise ValueError("D6 arena kernel requires compatible 16-divisible matrices")

    source_a = a
    source_b = b
    for level in range(_CP3_WINOGRAD_LEVELS):
        scale = 2 ** (level + 1)
        destination_a = arena_views[2 * level][
            ..., :m // scale, :k // scale
        ]
        destination_b = arena_views[2 * level + 1][
            ..., :k // scale, :n // scale
        ]
        scratch_s = arena_views[8 + level][
            ..., :m // scale, :k // scale
        ]
        scratch_t = arena_views[12 + level][
            ..., :k // scale, :n // scale
        ]
        _cp3_transform_level(
            source_a,
            source_b,
            destination_a,
            destination_b,
            scratch_s,
            scratch_t,
        )
        source_a = destination_a
        source_b = destination_b

    leaf = arena_views[16][..., :m // 16, :n // 16]
    fnp.matmul(source_a, source_b, out=leaf)
    products = leaf
    for step in range(_CP3_WINOGRAD_LEVELS):
        level = _CP3_WINOGRAD_LEVELS - step
        if step < 3:
            destination = arena_views[17 + step][
                ..., :m // 2 ** (level - 1), :n // 2 ** (level - 1)
            ]
        else:
            destination = out
        u_scratch = arena_views[20 + step][
            ..., :m // 2**level, :n // 2**level
        ]
        _cp3_reconstruct_level(products, destination, u_scratch)
        products = destination
    return out


def _cp_read(cp_lambda, cp_p, cp_q, products, arena_views):
    rank = int(cp_lambda.shape[0])
    if rank == 0:
        width = int(cp_p.shape[0])
        return fnp.zeros((width, width), dtype=fnp.float64)
    active_a = products[0, :, :rank]
    active_b = products[1, :, :rank]
    first = arena_views[24]
    second = arena_views[25]

    # Channel 0, then overwrite the same two incumbent product planes.
    fnp.multiply(cp_p, cp_p, out=active_a)
    fnp.multiply(cp_q, cp_lambda[None, :], out=active_b)
    _cp3_winograd_arena(
        active_a,
        fnp.transpose(active_b),
        first,
        arena_views,
    )

    # Channel 1 is sequential; no paired-channel operand or result tree lives.
    fnp.multiply(cp_p, cp_q, out=active_a)
    fnp.multiply(cp_p, cp_lambda[None, :], out=active_b)
    _cp3_winograd_arena(
        active_a,
        fnp.transpose(active_b),
        second,
        arena_views,
    )
    fnp.multiply(2.0, second, out=second)
    fnp.add(first, second, out=first)
    return fnp.asarray(first, dtype=fnp.float64)


def _cp_k3(cp_lambda, cp_p, cp_q, product_a):
    """Marginal-only final CP read; no n-by-n K21 plane is materialized."""
    rank = int(cp_lambda.shape[0])
    active_a = product_a[:, :rank]
    fnp.multiply(cp_p, cp_p, out=active_a)
    fnp.multiply(active_a, cp_q, out=active_a)
    return fnp.asarray(
        3.0 * (active_a @ cp_lambda),
        dtype=fnp.float64,
    )


def _cp_push(cp_lambda, weight_context, source_state, destination_state,
             arena_views):
    rank = int(cp_lambda.shape[0])
    if rank == 0:
        return (
            cp_lambda,
            destination_state[0, :, :0],
            destination_state[1, :, :0],
        )
    _weight32, weight32_t, _weight2, _weight2_t, _weight3, _weight3_t = (
        weight_context
    )
    source = source_state[:, :, :rank]
    destination = destination_state[:, :, :rank]
    _cp3_winograd_arena(
        weight32_t,
        source[0],
        destination[0],
        arena_views,
    )
    _cp3_winograd_arena(
        weight32_t,
        source[1],
        destination[1],
        arena_views,
    )
    return cp_lambda, destination[0], destination[1]


'''


def build_candidate(push_layout: dict, read_layout: dict) -> bytes:
    source_bytes = SOURCE.read_bytes()
    source_sha = sha256_bytes(source_bytes)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"source hash mismatch: {source_sha}")
    text = source_bytes.decode("utf-8")

    text = replace_once(
        text,
        "_CP3_CAP = 16384\n_CP4_CAP = 2048\n",
        arena_constants_text(push_layout, read_layout),
        "frozen CP3 arena constants",
    )

    function_start = text.index("def _cp_read(")
    function_end = text.index("def _cp_post_read(", function_start)
    if text.count("def _cp_read(") != 1 or text.count("def _cp_post_read(") != 1:
        raise RuntimeError("CP3 function boundary is not unique")
    text = text[:function_start] + NEW_FUNCTIONS + text[function_end:]

    text = replace_once(
        text,
        """        cp3_context[0][0],
        cp3_context[1][0],
        cp3_context[2],
""",
        """        cp3_context[0][0, 0],
        cp3_context[0][0, 1],
        cp3_context[1],
""",
        "initial CP3 grow destinations",
    )

    text = replace_once(
        text,
        """        cp3_context = (
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty(_CP3_CAP, dtype=fnp.float32),
            fnp.empty(width * _CP3_CAP, dtype=fnp.float32),
            fnp.empty(width * _CP3_CAP, dtype=fnp.float32),
        )
""",
        """        cp3_arena = fnp.empty(
            _CP3_WINOGRAD_ARENA_ELEMENTS, dtype=fnp.float32
        )
        cp3_context = (
            fnp.empty((2, 2, width, _CP3_CAP), dtype=fnp.float32),
            fnp.empty(_CP3_CAP, dtype=fnp.float32),
            fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32),
            cp3_arena,
            _cp3_bind_arena_views(cp3_arena, _CP3_PUSH_ARENA_VIEW_SPECS),
            _cp3_bind_arena_views(cp3_arena, _CP3_READ_ARENA_VIEW_SPECS),
        )
""",
        "persistent CP3 context and arena",
    )

    text = replace_once(
        text,
        """        cp_lambda, pushed_p, pushed_q = _cp_push(
            cp_lambda,
            cp_p,
            cp_q,
            weight_context,
            cp3_context[0][cp_bank_index],
            cp3_context[1][cp_bank_index],
        )
""",
        """        cp_lambda, pushed_p, pushed_q = _cp_push(
            cp_lambda,
            weight_context,
            cp3_context[0][1 - cp_bank_index],
            cp3_context[0][cp_bank_index],
            cp3_context[4],
        )
""",
        "loop sequential CP3 push",
    )

    text = replace_once(
        text,
        "_cp_k3(cp_lambda, pushed_p, pushed_q, cp3_context[3])",
        "_cp_k3(cp_lambda, pushed_p, pushed_q, cp3_context[2][0])",
        "final CP3 marginal scratch",
    )

    text = replace_once(
        text,
        """            cp3_context[3],
            cp3_context[4],
""",
        """            cp3_context[2],
            cp3_context[5],
""",
        "sequential CP3 read products and read arena views",
    )

    text = replace_once(
        text,
        """            cp3_context[0][cp_bank_index],
            cp3_context[1][cp_bank_index],
            cp3_context[2],
""",
        """            cp3_context[0][cp_bank_index, 0],
            cp3_context[0][cp_bank_index, 1],
            cp3_context[1],
""",
        "loop CP3 grow destinations",
    )
    return text.encode("utf-8")


def accounting(push_layout: dict, read_layout: dict) -> dict:
    roster = (
        [(WIDTH, WIDTH, WIDTH * layer) for layer in range(1, 16) for _ in range(2)]
        + [(WIDTH, WIDTH * layer, WIDTH) for layer in range(1, 15) for _ in range(2)]
    )

    def mm(m: int, k: int, n: int) -> int:
        return m * n * (2 * k - 1)

    def d5_kernel(m: int, k: int, n: int) -> int:
        leaf = 7**4 * mm(m // 16, k // 16, n // 16)
        overhead = sum(
            7**level * 11 * (m * k + k * n + m * n) // 4 ** (level + 1)
            for level in range(4)
        )
        return leaf + overhead

    def d6_kernel(m: int, k: int, n: int) -> int:
        leaf = 7**4 * mm(m // 16, k // 16, n // 16)
        transform = sum(
            7**level * 11 * (m * k + k * n) // 4 ** (level + 1)
            for level in range(4)
        )
        reconstruct = sum(
            7**level * 7 * (m * n) // 4 ** (level + 1)
            for level in range(4)
        )
        return leaf + transform + reconstruct

    dense = sum(mm(*shape) for shape in roster)
    d5_total = sum(d5_kernel(*shape) for shape in roster)
    d6_total = sum(d6_kernel(*shape) for shape in roster)
    assembly_saving = d5_total - d6_total
    paired_left_stack_saving = 10 * 2 * WIDTH * WIDTH
    cold_bind = (
        push_layout["view_binding_reshape_bill"]
        + read_layout["view_binding_reshape_bill"]
    )
    warm_saving = assembly_saving + paired_left_stack_saving
    cold_saving = warm_saving - cold_bind
    d5_calls = [1_320_876_655_991, 1_359_519_832_584, 1_320_873_503_232]
    incumbent_calls = [1_692_484_904_311, 1_731_128_080_904, 1_692_481_751_552]
    candidate_calls = [
        d5_calls[0] - cold_saving,
        d5_calls[1] - warm_saving,
        d5_calls[2] - warm_saving,
    ]
    deltas = [
        incumbent - candidate
        for incumbent, candidate in zip(incumbent_calls, candidate_calls)
    ]
    assert len(roster) == 58
    assert dense == 966_086_623_232
    assert d5_total == 594_693_332_992
    assert d6_total == 591_553_830_912
    assert assembly_saving == 3_139_502_080
    assert paired_left_stack_saving == 20_971_520
    assert cold_bind == 1_919_598_592
    assert warm_saving == 3_160_473_600
    assert cold_saving == 1_240_875_008
    assert candidate_calls == [
        1_319_635_780_983,
        1_356_359_358_984,
        1_317_713_029_632,
    ]
    assert deltas == [372_849_123_328, 374_768_721_920, 374_768_721_920]
    return {
        "roster": {
            "push_channel_contractions": 30,
            "read_channel_contractions": 28,
            "total_channel_contractions": 58,
            "leaf_batch_matmul_dispatches_per_prediction": 58,
            "leaf_products_per_dispatch": 2_401,
            "final_k3_read_excluded": True,
            "shapes": [list(shape) for shape in roster],
        },
        "formula": {
            "dense_matmul": "m*n*(2*k-1)",
            "d5_depth4": (
                "7^4*mm(m/16,k/16,n/16) + "
                "sum(l=0..3) 7^l*11*(m*k+k*n+m*n)/4^(l+1)"
            ),
            "d6_depth4": (
                "7^4*mm(m/16,k/16,n/16) + "
                "sum(l=0..3) 7^l*(11*(m*k+k*n)+7*m*n)/4^(l+1)"
            ),
            "cold_call": (
                "D5 bill - reconstruction assembly copies - paired-left "
                "stack + both persistent arena-view binding reshapes"
            ),
            "warm_call": (
                "D5 bill - reconstruction assembly copies - paired-left stack"
            ),
        },
        "dense_roster_bill": dense,
        "d5_materialized_depth4_roster_bill": d5_total,
        "d6_arena_depth4_roster_bill": d6_total,
        "reconstruction_assembly_copy_saving_per_prediction": assembly_saving,
        "paired_common_left_stack_saving_per_prediction": paired_left_stack_saving,
        "cold_view_binding_reshape_bill": cold_bind,
        "net_saving_vs_d5": {
            "cold_call0": cold_saving,
            "warm_calls1_and2": warm_saving,
        },
        "d5_exact_calls_35_279_35": d5_calls,
        "incumbent_exact_calls_35_279_35": incumbent_calls,
        "projected_d6_calls_35_279_35": candidate_calls,
        "projected_incumbent_minus_d6": deltas,
        "projection_disclaimer": (
            "Static literal formula only. A separately sealed meter smoke must "
            "measure these bills; no execution is authorized here."
        ),
    }


def main() -> None:
    authorities = {
        SOURCE: EXPECTED_SOURCE_SHA256,
        D5 / "D5R1_RESOURCE_CAUSAL_AUDIT_AND_D6_SUCCESSOR.md":
            EXPECTED_D5_DESIGN_SHA256,
        D5 / "INDEPENDENT_REDTEAM_D6_SUCCESSOR_AUDIT.md":
            EXPECTED_INDEPENDENT_DESIGN_SHA256,
        D5 / "verify_independent_d6_successor.py":
            EXPECTED_INDEPENDENT_DESIGN_VERIFIER_SHA256,
    }
    for path, expected in authorities.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"authority hash mismatch for {path}: {observed}")

    push_layout = make_layout("push", WIDTH, WIDTH, MAX_PUSH_RANK)
    read_layout = make_layout("read", WIDTH, MAX_READ_RANK, WIDTH)
    assert push_layout["physical_elements"] == ARENA_ELEMENTS
    assert read_layout["physical_elements"] == 681_480_192

    candidate_bytes = build_candidate(push_layout, read_layout)
    candidate_sha = sha256_bytes(candidate_bytes)
    DESTINATION.write_bytes(candidate_bytes)

    source_lines = SOURCE.read_text().splitlines(keepends=True)
    candidate_lines = candidate_bytes.decode("utf-8").splitlines(keepends=True)
    diff_text = "".join(difflib.unified_diff(
        source_lines,
        candidate_lines,
        fromfile="public_noeigh_card_v1/estimator.py",
        tofile="p2r0c1d6_cp3_winograd_arena_v1/estimator.py",
        n=5,
    ))
    SOURCE_DIFF.write_text(diff_text)
    diff_sha = sha256(SOURCE_DIFF)

    accounting_record = accounting(push_layout, read_layout)
    manifest = {
        "schema": "p2r0c1d6-accounting-liveness-v1",
        "status": "FROZEN_PRE_STATIC_REVIEW_NO_EXECUTION",
        "date": "2026-08-26",
        "node": "P2R0C1D6",
        "label": "sequential-channel fully preallocated depth-4 Winograd arena",
        "authority": {
            "sealed_design_sha256": EXPECTED_D5_DESIGN_SHA256,
            "independent_design_audit_sha256": EXPECTED_INDEPENDENT_DESIGN_SHA256,
            "independent_design_verifier_sha256":
                EXPECTED_INDEPENDENT_DESIGN_VERIFIER_SHA256,
            "incumbent_source_sha256": EXPECTED_SOURCE_SHA256,
        },
        "candidate": {
            "source": DESTINATION.name,
            "source_sha256": candidate_sha,
            "mechanical_builder": Path(__file__).name,
            "mechanical_builder_sha256": sha256(Path(__file__)),
            "source_diff": SOURCE_DIFF.name,
            "source_diff_sha256": diff_sha,
            "transform": "exact unique anchored replacements only",
        },
        "arena": {
            "dtype": "float32",
            "levels": LEVELS,
            "q": "7/4",
            "single_flat_arena_elements": ARENA_ELEMENTS,
            "single_flat_arena_bytes": 4 * ARENA_ELEMENTS,
            "allocation_count": 1,
            "layouts_share_same_flat_arena": True,
            "push": push_layout,
            "read": read_layout,
            "sealed_formula": (
                "sum(j=1..4)(7/4)^j(A+B) + (7/4)^3 max(A,B) + "
                "(7/4)^4 C + 3(7/4)^3 C + 2C"
            ),
            "incumbent_anchor_bytes": 3_421_503_488,
            "anchor_plus_push_arena_bytes": 6_798_163_968,
            "static_margin_below_8gib_bytes": 1_791_770_624,
            "future_smoke_increment_cap_bytes": 3_913_531_392,
            "future_smoke_absolute_peak_strictly_below_bytes": 7 * 1024**3,
        },
        "liveness": {
            "channel_schedule": "strictly sequential channel0 then channel1",
            "transform_schedule": (
                "level1..4; S dependencies then A destination; reuse same "
                "physical scratch for T then B destination"
            ),
            "leaf_schedule": "one 2401-way batched matmul per channel contraction",
            "reconstruction_schedule": (
                "leaf -> bank0(level3) -> bank1(level2) -> bank0(level1) -> out"
            ),
            "u_schedule": "U1,U2,U3,U4,U5,U6,U7 in frozen D5 law order",
            "large_dynamic_allocations": [],
            "forbidden_absent": [
                "recursive kernel call",
                "fnp.stack in CP3 arena functions",
                "fnp.block in CP3 arena functions",
                "fnp.concatenate in CP3 arena functions",
                "implicit full-size CP3 expression temporary",
                "paired-channel CP3 tree",
                "extra persistent read plane",
                "tensordot",
                "einsum in CP3 arena functions",
            ],
            "top_output_written_directly": True,
            "reserved_push_output_slices_are_unwritten": True,
            "dummy_billed_work": False,
        },
        "accounting": accounting_record,
        "serialization_guard": {
            "required": True,
            "filename": "verify_serialization_guard.py",
            "must_import_or_call_estimator": False,
            "must_create_runtime_result": False,
        },
        "execution_authority": {
            "source_build": True,
            "static_verification": True,
            "tiny_synthetic_kernel_smoke": False,
            "exact_shape_smoke": False,
            "fixture_call": False,
            "common_driver_call": False,
            "targets_scores_names_board": False,
            "production_submission_network": False,
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    receipt = {
        "schema": "p2r0c1d6-build-v1",
        "status": "BUILT_PRE_STATIC_REVIEW_NO_EXECUTION",
        "node": "P2R0C1D6",
        "builder_sha256": sha256(Path(__file__)),
        "incumbent_source_sha256": EXPECTED_SOURCE_SHA256,
        "candidate_source_sha256": candidate_sha,
        "source_diff_sha256": diff_sha,
        "accounting_liveness_manifest_sha256": sha256(MANIFEST),
        "arena_elements": ARENA_ELEMENTS,
        "push_layout_elements": push_layout["physical_elements"],
        "read_layout_elements": read_layout["physical_elements"],
        "view_binding_reshape_bill": (
            push_layout["view_binding_reshape_bill"]
            + read_layout["view_binding_reshape_bill"]
        ),
        "estimator_imported": False,
        "estimator_calls": 0,
        "exact_shape_calls": 0,
        "fixture_calls": 0,
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": False,
        "submission": False,
        "network": False,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
