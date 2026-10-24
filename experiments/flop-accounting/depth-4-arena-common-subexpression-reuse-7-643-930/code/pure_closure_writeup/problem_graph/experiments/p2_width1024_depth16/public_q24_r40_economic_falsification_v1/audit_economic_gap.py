#!/usr/bin/env python3
"""Static, target-free economic audit of fast products in the Q24+D1 host.

This script executes no estimator, fixture, target, package, or submission.  It
reconciles the frozen D1 FlopScope receipt, separates matrix-eligible work from
the non-matrix Q24 contractions, prices CP4 caps, and searches arithmetic
spellings made from the released <6,3,3:40> tensor and Winograd B7.

The strongest falsification intentionally grants impossible advantages: each
product may select its own recursion, operands may be tiled and lawfully
stacked, and all input transforms may be free.  Leaf products, inverse output
transforms, exact fringes, and the non-product host remain charged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import functools
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent

D1_RESULT = (
    EXPERIMENTS
    / "p2r0c1d9_cp3_arena_depth1_full_estimator_v1"
    / "D1_CANDIDATE_RESULT.json"
)
D1_SOURCE = D1_RESULT.parent / "estimator.py"
REFERENCE_RESULT = (
    EXPERIMENTS
    / "p2r0c1d5_cp3_winograd_hybrid_v1"
    / "RECOVERED_REFERENCE_EVIDENCE.json"
)
CAP_RESULTS = {
    1792: EXPERIMENTS
    / "public_cp4_cap1792_card_v1"
    / "FULL_ESTIMATOR_GATE_RESULT.json",
    1024: EXPERIMENTS
    / "public_cp4_cap1024_card_v1"
    / "FULL_ESTIMATOR_GATE_RESULT.json",
}
RAW_FRONTIER = (
    EXPERIMENTS / "p2r0g6_native_j_physical_local_oracle_v1" / "RESULT.json"
)
TINY_R40 = (
    EXPERIMENTS / "public_q24_r40_physical_kernel_audit_v1" / "RESULT.json"
)

PUBLIC = Path("/private/tmp/r40_public_extract_20260826")
R40_SOURCE = PUBLIC / "FMMa_6_3_3.m"
R40_SCHEDULE = PUBLIC / "tichavsky_r40_schedule_c142_v1.py"
M124 = PUBLIC / "audit_m124_m7b_r40r40w1_banked_physical_v090_subagent_20260723.py"
M185 = PUBLIC / "audit_m185_r40_transform_packing_v090_subagent_20260724.py"

WIDTH = 1024
TARGET_FLOPS = 573_816_383_391
CAP_2POW41 = 2**41
RANK_RATIO = Fraction(8575, 23328)

# Exact call-0 D1 decomposition.  Each row is independently reconciled below.
EINSUM_COMPONENTS = {
    "initial_three_gaussian_pair_calls": 125_872_128,
    "four_pair_edgeworth_14_layers": 39_943_421_952,
    "cp_post_read_15x3": 220_323_840,
    "cp4_pre_planes_28": 115_934_756_864,
    "cp4_drop_planes_26": 55_807_311_872,
    "reset_stars_15x2": 31_549_440,
}
CP4_PLANE_EINSUM = (
    EINSUM_COMPONENTS["cp4_pre_planes_28"]
    + EINSUM_COMPONENTS["cp4_drop_planes_26"]
)
NON_GEMM_EINSUM = (
    sum(EINSUM_COMPONENTS.values()) - CP4_PLANE_EINSUM
)

# Savings are incumbent minus candidate.  Negative rows are extra candidate
# work.  The 1792/1024 rows are checked against full-estimator receipts; the
# 768/512 rows are the exact same rank/cardinality formulas continued below
# the measured cards.
CAP_DELTAS = {
    1792: {
        "matmul": 7_519_862_784,
        "einsum": 12_886_999_040,
        "other": -7_258_880,
        "total": 20_399_602_944,
        "evidence": "measured full-estimator target-free gate",
    },
    1024: {
        "matmul": 30_079_451_136,
        "einsum": 51_541_704_704,
        "other": 134_974_464,
        "total": 81_756_130_304,
        "evidence": "measured full-estimator target-free gate",
    },
    768: {
        "matmul": 38_136_446_976,
        "einsum": 65_502_445_568,
        "other": 137_734_656,
        "total": 103_776_627_200,
        "evidence": "exact static continuation; output drift/raw unmeasured",
    },
    512: {
        "matmul": 46_193_442_816,
        "einsum": 79_461_089_280,
        "other": 188_780_544,
        "total": 125_843_312_640,
        "evidence": "exact static continuation; output drift/raw unmeasured",
    },
}

CAP_STATIC_OPERATION_DELTAS = {
    768: {
        "matmul": 38_136_446_976,
        "einsum": 65_502_445_568,
        "argsort": 772_096,
        "asarray": -8_388_608,
        "concatenate": 17_318_400,
        "getitem": 56_678_400,
        "multiply": 132_700_160,
        "negative": 13_824,
        "subtract": -8_392_704,
        "sum": 14_141_952,
        "where": -67_108_864,
    },
    512: {
        "matmul": 46_193_442_816,
        "einsum": 79_461_089_280,
        "argsort": 929_792,
        "asarray": -8_388_608,
        "concatenate": 20_992_000,
        "getitem": 71_372_800,
        "multiply": 161_550_336,
        "negative": 17_408,
        "subtract": -8_392_704,
        "sum": 17_808_384,
        "where": -67_108_864,
    },
}

# Exact CSE inventory.  "strict_slots" materializes transpose counterparts
# back into the original 68-slot layout, so its copy bill cancels the removed
# weight multiplies.  Later tiers are useful lower bounds but require a drift
# discriminator or a FlopScope symmetry/layout certificate.
G4_PAIRED_EINSUM_SAVING = 8_223_645_696
G4_MISSING_WEIGHT_MULTIPLIES = 411_041_792
RESET_REUSE_COV2_AND_VARIANCE_OUTER = 62_914_560
RESET_REUSE_SCALED_VARIANCE = 30_720
G4_ASSOCIATION_ACCUMULATOR = 234_881_024
G4_SELF_CHANNEL_SYMMETRY_TAG = 1_143_926_784

CSE_TIERS = {
    "strict_slots": (
        G4_PAIRED_EINSUM_SAVING
        + RESET_REUSE_COV2_AND_VARIANCE_OUTER
        + RESET_REUSE_SCALED_VARIANCE
    ),
    "transpose_views_algebraic": (
        G4_PAIRED_EINSUM_SAVING
        + G4_MISSING_WEIGHT_MULTIPLIES
        + RESET_REUSE_COV2_AND_VARIANCE_OUTER
        + RESET_REUSE_SCALED_VARIANCE
    ),
    "association_accumulator_conditional": (
        G4_PAIRED_EINSUM_SAVING
        + G4_MISSING_WEIGHT_MULTIPLIES
        + RESET_REUSE_COV2_AND_VARIANCE_OUTER
        + RESET_REUSE_SCALED_VARIANCE
        + G4_ASSOCIATION_ACCUMULATOR
    ),
    "symmetry_tag_conditional": (
        G4_PAIRED_EINSUM_SAVING
        + G4_MISSING_WEIGHT_MULTIPLIES
        + RESET_REUSE_COV2_AND_VARIANCE_OUTER
        + RESET_REUSE_SCALED_VARIANCE
        + G4_ASSOCIATION_ACCUMULATOR
        + G4_SELF_CHANNEL_SYMMETRY_TAG
    ),
}

SMALL_CURRENT_PRODUCT_BASE = 756_025_344


@dataclass(frozen=True)
class Product:
    label: str
    m: int
    k: int
    n: int
    rate: int = 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def direct_cost(product: Product) -> int:
    if not product.m or not product.k or not product.n:
        return 0
    return product.rate * product.m * product.n * (2 * product.k - 1)


def cp4_ranks(cap: int) -> tuple[list[int], list[int], list[int]]:
    """Return push, pre-read, and dropped ranks for depth 16."""
    if cap >= 2048:
        return [1024] + [2048] * 14, [1024] + [2048] * 13, [1024] * 13
    if cap == 1792:
        return [1024] + [1792] * 14, [1024] + [1792] * 13, [256] + [1024] * 13
    if 0 < cap <= 1024:
        return [cap] * 15, [cap] * 14, [1024 - cap] + [1024] * 14
    if cap == 0:
        return [], [], []
    raise ValueError(cap)


def major_roster(cap: int) -> list[Product]:
    """Unstacked conceptual products represented by the large D1 calls."""
    result: list[Product] = []
    for layer in range(1, 16):
        rank = WIDTH * layer
        result.extend(
            Product(f"cp3_push_{layer}_{side}", WIDTH, WIDTH, rank)
            for side in ("p", "q")
        )
        if layer <= 14:
            result.extend(
                Product(f"cp3_read_{layer}_{side}", WIDTH, rank, WIDTH)
                for side in ("first", "second")
            )
    result.extend(Product(f"square64_{i}", WIDTH, WIDTH, WIDTH, 2) for i in range(31))
    result.extend(Product(f"square32_{i}", WIDTH, WIDTH, WIDTH) for i in range(142))
    pushes, pre_reads, drops = cp4_ranks(cap)
    result.extend(
        Product(f"cp4_push_{i}", WIDTH, WIDTH, rank)
        for i, rank in enumerate(pushes)
    )
    for i, rank in enumerate(pre_reads):
        result.extend(
            Product(f"cp4_pre_{i}_{side}", WIDTH, rank, WIDTH)
            for side in ("directed", "paired")
        )
    for i, rank in enumerate(drops):
        if rank:
            result.extend(
                Product(f"cp4_drop_{i}_{side}", WIDTH, rank, WIDTH)
                for side in ("directed", "paired")
            )
    return result


def stacked_roster(cap: int, *, fuse_cp3_reads: bool) -> list[Product]:
    """Maximally stack the repeated weight operands without cross terms.

    P/Q/CP4 pushes concatenate RHS columns.  The four right-W star products
    stack LHS rows, while W2/W3 products concatenate RHS columns.  CP3 read
    fusion is a separate algebraic lower bound: it concatenates the two inner
    dimensions and folds the factor two into the second bank.  That changes
    floating reduction order and is never labeled byte preserving.
    """
    result: list[Product] = []
    pushes, pre_reads, drops = cp4_ranks(cap)
    for layer in range(1, 16):
        rank = WIDTH * layer
        cp4_rank = pushes[layer - 1] if pushes else 0
        result.append(
            Product(
                f"stacked_push_{layer}", WIDTH, WIDTH, 2 * rank + cp4_rank
            )
        )
        if layer <= 14:
            if fuse_cp3_reads:
                result.append(
                    Product(f"fused_cp3_read_{layer}", WIDTH, 2 * rank, WIDTH)
                )
            else:
                result.extend(
                    Product(f"cp3_read_{layer}_{side}", WIDTH, rank, WIDTH)
                    for side in ("first", "second")
                )

    result.extend(Product(f"square64_{i}", WIDTH, WIDTH, WIDTH, 2) for i in range(31))
    for layer in range(1, 15):
        result.extend(
            (
                Product(f"star_right_w_{layer}", 4 * WIDTH, WIDTH, WIDTH),
                Product(f"star_left_w2_{layer}", WIDTH, WIDTH, 4 * WIDTH),
                Product(f"star_left_w3_{layer}", WIDTH, WIDTH, 2 * WIDTH),
            )
        )
    result.append(Product("star_right_w_final", 2 * WIDTH, WIDTH, WIDTH))

    for i, rank in enumerate(pre_reads):
        result.extend(
            Product(f"cp4_pre_{i}_{side}", WIDTH, rank, WIDTH)
            for side in ("directed", "paired")
        )
    for i, rank in enumerate(drops):
        if rank:
            result.extend(
                Product(f"cp4_drop_{i}_{side}", WIDTH, rank, WIDTH)
                for side in ("directed", "paired")
            )
    return result


def macro_units(d: int, w: int) -> tuple[int, int]:
    return 6**d * 2**w, 3**d * 2**w


def exact_macro_parts(
    product: Product, d: int, w: int, *, charge_forward: bool
) -> dict[str, int]:
    """Arithmetic of an exactly divisible R40^d/Winograd^w product."""
    m, k, n, rate = product.m, product.k, product.n, product.rate
    unit_m, unit_kn = macro_units(d, w)
    if m % unit_m or k % unit_kn or n % unit_kn:
        raise ValueError((product, d, w))
    batch = 40**d * 7**w
    leaf = rate * batch * (m // unit_m) * (n // unit_kn) * (
        2 * (k // unit_kn) - 1
    )
    inverse_r40 = rate * sum(
        142 * 40**level * (m // 6 ** (level + 1)) * (n // 3 ** (level + 1))
        for level in range(d)
    )
    mr, kr, nr = m // 6**d, k // 3**d, n // 3**d
    inverse_b7 = rate * sum(
        7
        * 40**d
        * 7**level
        * (mr // 2 ** (level + 1))
        * (nr // 2 ** (level + 1))
        for level in range(w)
    )
    forward_a = 0
    forward_b = 0
    if charge_forward:
        # 74/53 binary c142 nodes plus literal 4/11 terminal signs.
        forward_a = rate * sum(
            78
            * 40**level
            * (m // 6 ** (level + 1))
            * (k // 3 ** (level + 1))
            for level in range(d)
        )
        forward_b = rate * sum(
            64
            * 40**level
            * (k // 3 ** (level + 1))
            * (n // 3 ** (level + 1))
            for level in range(d)
        )
        forward_a += rate * sum(
            4
            * 40**d
            * 7**level
            * (mr // 2 ** (level + 1))
            * (kr // 2 ** (level + 1))
            for level in range(w)
        )
        forward_b += rate * sum(
            4
            * 40**d
            * 7**level
            * (kr // 2 ** (level + 1))
            * (nr // 2 ** (level + 1))
            for level in range(w)
        )
    return {
        "leaf": leaf,
        "inverse_r40": inverse_r40,
        "inverse_b7": inverse_b7,
        "forward_a": forward_a,
        "forward_b": forward_b,
        "total": leaf + inverse_r40 + inverse_b7 + forward_a + forward_b,
    }


def tiled_once_parts(
    product: Product, d: int, w: int, *, charge_forward: bool
) -> dict[str, int]:
    """One maximal aligned core, exact direct fringes, and k-block adds."""
    unit_m, unit_kn = macro_units(d, w)
    core_m = product.m // unit_m * unit_m
    core_k = product.k // unit_kn * unit_kn
    core_n = product.n // unit_kn * unit_kn
    if not core_m or not core_k or not core_n:
        cost = direct_cost(product)
        return {
            "leaf": 0,
            "inverse_r40": 0,
            "inverse_b7": 0,
            "forward_a": 0,
            "forward_b": 0,
            "fringes_and_accumulation": cost,
            "total": cost,
        }
    core = Product("core", core_m, core_k, core_n, product.rate)
    parts = exact_macro_parts(core, d, w, charge_forward=charge_forward)
    ms = (core_m, product.m - core_m)
    ks = (core_k, product.k - core_k)
    ns = (core_n, product.n - core_n)
    fringe = 0
    for mi, mpart in enumerate(ms):
        for ki, kpart in enumerate(ks):
            for ni, npart in enumerate(ns):
                if mi == ki == ni == 0 or not (mpart and kpart and npart):
                    continue
                fringe += direct_cost(
                    Product("fringe", mpart, kpart, npart, product.rate)
                )
    if core_k and product.k - core_k:
        fringe += product.rate * sum(
            mpart * npart for mpart in ms for npart in ns if mpart and npart
        )
    parts["fringes_and_accumulation"] = fringe
    parts["total"] += fringe
    return parts


ALGORITHMS = tuple(
    (d, w)
    for d in range(4)
    for w in range(11)
    if (d or w) and 6**d * 2**w <= 4096
)


def make_recursive_tiler(*, charge_forward: bool):
    """Return an exact dynamic tiled/fringe arithmetic optimizer."""

    @functools.lru_cache(maxsize=None)
    def best(m: int, k: int, n: int, rate: int) -> int:
        product = Product("dynamic", m, k, n, rate)
        winner = direct_cost(product)
        for d, w in ALGORITHMS:
            unit_m, unit_kn = macro_units(d, w)
            core_m = m // unit_m * unit_m
            core_k = k // unit_kn * unit_kn
            core_n = n // unit_kn * unit_kn
            if not core_m or not core_k or not core_n:
                continue
            core = Product("core", core_m, core_k, core_n, rate)
            candidate = exact_macro_parts(
                core, d, w, charge_forward=charge_forward
            )["total"]
            if (core_m, core_k, core_n) != (m, k, n):
                ms = (core_m, m - core_m)
                ks = (core_k, k - core_k)
                ns = (core_n, n - core_n)
                for mi, mpart in enumerate(ms):
                    for ki, kpart in enumerate(ks):
                        for ni, npart in enumerate(ns):
                            if mi == ki == ni == 0 or not (
                                mpart and kpart and npart
                            ):
                                continue
                            candidate += best(mpart, kpart, npart, rate)
                if core_k and k - core_k:
                    candidate += rate * sum(
                        mpart * npart
                        for mpart in ms
                        for npart in ns
                        if mpart and npart
                    )
            winner = min(winner, candidate)
        return winner

    return best


def cap_eligible_saving(cap: int) -> int:
    return CAP_DELTAS[cap]["matmul"] + CAP_DELTAS[cap]["einsum"]


def cap_other_saving(cap: int) -> int:
    return CAP_DELTAS[cap]["other"]


def small_product_after_cap(cap: int) -> tuple[int, int]:
    baseline = sum(map(direct_cost, major_roster(2048)))
    candidate = sum(map(direct_cost, major_roster(cap)))
    major_saving = baseline - candidate
    small_saving = cap_eligible_saving(cap) - major_saving
    return SMALL_CURRENT_PRODUCT_BASE - small_saving, small_saving


def fixed_nonmacro(cap: int, cse_tier: str) -> int:
    d1 = load(D1_RESULT)
    operations = d1["trials"][0]["literal_meter"]["operations"]
    matmul = operations["matmul"]["flop_cost"]
    einsum = operations["einsum"]["flop_cost"]
    total = d1["trials"][0]["candidate_flops"]
    other = total - matmul - einsum
    small, _saving = small_product_after_cap(cap)
    return (
        other
        - cap_other_saving(cap)
        + NON_GEMM_EINSUM
        + small
        - CSE_TIERS[cse_tier]
    )


def floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


def operation_delta(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, int]:
    names = set(reference) | set(candidate)
    return {
        name: int(reference.get(name, {}).get("flop_cost", 0))
        - int(candidate.get(name, {}).get("flop_cost", 0))
        for name in names
        if int(reference.get(name, {}).get("flop_cost", 0))
        != int(candidate.get(name, {}).get("flop_cost", 0))
    }


def build_report() -> dict[str, Any]:
    d1 = load(D1_RESULT)
    trial = d1["trials"][0]
    meter = trial["literal_meter"]
    operations = meter["operations"]
    total = int(trial["candidate_flops"])
    operation_sum = sum(int(row["flop_cost"]) for row in operations.values())
    matmul = int(operations["matmul"]["flop_cost"])
    einsum = int(operations["einsum"]["flop_cost"])
    other = total - matmul - einsum
    assert total == 1_574_162_015_607 == operation_sum
    assert matmul == 1_345_955_301_376
    assert einsum == 212_063_236_096 == sum(EINSUM_COMPONENTS.values())
    assert other == 16_143_478_135
    assert NON_GEMM_EINSUM == 40_321_167_360
    eligible = matmul + CP4_PLANE_EINSUM
    assert eligible == 1_517_697_370_112

    # Verify measured cap rows against the common reference receipt.
    reference = load(REFERENCE_RESULT)
    ref_operations = reference["trials"][0]["literal_meter"]["operations"]
    measured_cap_checks = {}
    for cap, path in CAP_RESULTS.items():
        candidate = load(path)
        candidate_ops = candidate["trials"][0]["literal_meter"]["operations"]
        delta = operation_delta(ref_operations, candidate_ops)
        observed_total = sum(delta.values())
        assert observed_total == CAP_DELTAS[cap]["total"]
        assert delta["matmul"] == CAP_DELTAS[cap]["matmul"]
        assert delta["einsum"] == CAP_DELTAS[cap]["einsum"]
        assert sum(v for k, v in delta.items() if k not in ("matmul", "einsum")) == CAP_DELTAS[cap]["other"]
        measured_cap_checks[str(cap)] = {
            "operation_delta": delta,
            "sum": observed_total,
        }
    for cap, rows in CAP_STATIC_OPERATION_DELTAS.items():
        assert sum(rows.values()) == CAP_DELTAS[cap]["total"]
        assert rows["matmul"] == CAP_DELTAS[cap]["matmul"]
        assert rows["einsum"] == CAP_DELTAS[cap]["einsum"]

    optimistic_old = Fraction(other) + Fraction(matmul + einsum) * RANK_RATIO
    corrected_base = Fraction(other + NON_GEMM_EINSUM) + Fraction(eligible) * RANK_RATIO

    raw_frontier = load(RAW_FRONTIER)
    cap_science: dict[str, Any] = {}
    for cap in (1792, 1024):
        gate = load(CAP_RESULTS[cap])
        agg = gate["aggregate"]
        cap_science[str(cap)] = {
            "evidence": CAP_DELTAS[cap]["evidence"],
            "measured_flop_saving": agg["measured_incumbent_minus_candidate_flops"][0],
            "pooled_unique_final_rms_drift": agg["pooled_unique_fixture_final_layer_rms_delta"],
            "maximum_final_absolute_drift": agg["maximum_final_layer_absolute_delta"],
            "maximum_residual_s": agg["maximum_residual_wall_seconds"],
            "raw_mse_measured": False,
        }
    for cap in (768, 512):
        cap_science[str(cap)] = {
            "evidence": CAP_DELTAS[cap]["evidence"],
            "raw_mse_measured": False,
            "drift_measured": False,
        }

    rank_only: dict[str, Any] = {}
    for cap in (1792, 1024, 768, 512):
        delta = CAP_DELTAS[cap]
        value = (
            Fraction(other - delta["other"] + NON_GEMM_EINSUM)
            + Fraction(eligible - cap_eligible_saving(cap)) * RANK_RATIO
        )
        tiers = {}
        for label, saving in CSE_TIERS.items():
            tier_value = value - saving
            tiers[label] = {
                "exact_fraction": str(tier_value),
                "favorable_integer_floor": floor_fraction(tier_value),
                "margin_vs_current_target": floor_fraction(tier_value) - TARGET_FLOPS,
            }
        rank_only[str(cap)] = {
            "without_new_cse_exact_fraction": str(value),
            "without_new_cse_favorable_floor": floor_fraction(value),
            "without_new_cse_margin_vs_target": floor_fraction(value) - TARGET_FLOPS,
            "cse_tiers": tiers,
        }

    free_tiler = make_recursive_tiler(charge_forward=False)
    physical_tiler = make_recursive_tiler(charge_forward=True)
    physical_boundary: dict[str, Any] = {}
    for cap in (1024, 768, 512):
        separated = stacked_roster(cap, fuse_cp3_reads=False)
        fused = stacked_roster(cap, fuse_cp3_reads=True)
        product_costs = {
            "separated_free_forward": sum(
                free_tiler(p.m, p.k, p.n, p.rate) for p in separated
            ),
            "fused_free_forward": sum(
                free_tiler(p.m, p.k, p.n, p.rate) for p in fused
            ),
            "separated_physical_arithmetic": sum(
                physical_tiler(p.m, p.k, p.n, p.rate) for p in separated
            ),
            "fused_physical_arithmetic": sum(
                physical_tiler(p.m, p.k, p.n, p.rate) for p in fused
            ),
        }
        totals: dict[str, Any] = {}
        for cse_tier in CSE_TIERS:
            fixed = fixed_nonmacro(cap, cse_tier)
            totals[cse_tier] = {
                name: {
                    "flops": value + fixed,
                    "margin_vs_current_target": value + fixed - TARGET_FLOPS,
                }
                for name, value in product_costs.items()
            }

        # Compare the requested R40^2 B7^3 padded and one-level tiled forms.
        requested = {}
        for spelling in ("zero_pad", "tiled_direct_fringe"):
            rows = {key: 0 for key in (
                "leaf", "inverse_r40", "inverse_b7", "forward_a", "forward_b",
                "fringes_and_accumulation", "total"
            )}
            for product in separated:
                if spelling == "zero_pad":
                    unit_m, unit_kn = macro_units(2, 3)
                    padded = Product(
                        product.label,
                        -(-product.m // unit_m) * unit_m,
                        -(-product.k // unit_kn) * unit_kn,
                        -(-product.n // unit_kn) * unit_kn,
                        product.rate,
                    )
                    parts = exact_macro_parts(padded, 2, 3, charge_forward=True)
                    parts["fringes_and_accumulation"] = 0
                else:
                    parts = tiled_once_parts(product, 2, 3, charge_forward=True)
                for key in rows:
                    rows[key] += parts.get(key, 0)
            fixed = fixed_nonmacro(cap, "strict_slots")
            requested[spelling] = {
                "components": rows,
                "strict_total_before_movement": rows["total"] + fixed,
                "margin_vs_current_target": rows["total"] + fixed - TARGET_FLOPS,
                "omitted_positive_taxes": [
                    "padding/materialization copies",
                    "stack/gather/layout movement",
                    "final scale/crop",
                    "coefficient materialization",
                ],
            }

        physical_boundary[str(cap)] = {
            "stacked_product_calls_separated": len(separated),
            "stacked_product_calls_fused": len(fused),
            "product_costs_before_fixed_host": product_costs,
            "totals": totals,
            "requested_r40x2_b7x3": requested,
        }

    tiny = load(TINY_R40)
    biggest_m, biggest_k, biggest_n = 1024, 1024, 15 * 1024
    padded_m = -(-biggest_m // 288) * 288
    padded_k = -(-biggest_k // 72) * 72
    padded_n = -(-biggest_n // 72) * 72
    leaf_batch = 40**2 * 7**3
    memory = {
        "shape": [biggest_m, biggest_k, biggest_n],
        "padded_shape_r40x2_b7x3": [padded_m, padded_k, padded_n],
        "float32_leaf_bank_gib": {
            "A": leaf_batch * (padded_m // 288) * (padded_k // 72) * 4 / 2**30,
            "B": leaf_batch * (padded_k // 72) * (padded_n // 72) * 4 / 2**30,
            "C": leaf_batch * (padded_m // 288) * (padded_n // 72) * 4 / 2**30,
        },
        "current_d1_maximum_measured_rss_bytes": max(
            int(row["maximum_process_rss_bytes"]) for row in d1["trials"]
        ),
        "verdict": "A/B/C leaf banks cannot coexist with the host under 8 GiB; a streamed spelling is mandatory.",
    }

    # Logical-wave dispatch for requested d=2,w=3, cap512 separated roster.
    cap512_products = len(stacked_roster(512, fuse_cp3_reads=False))
    dispatch_calls_per_product = 2 * (8 + 6 + 14) + 3 * (4 + 4 + 7) + 1
    naive_dispatch = cap512_products * dispatch_calls_per_product
    residual_per_call = 0.002003595 / 675
    dispatch = {
        "requested_r40x2_b7x3_stacked_products": cap512_products,
        "logical_wave_calls_per_product_without_cross_product_reuse": dispatch_calls_per_product,
        "upper_wrapper_count_before_pack_splits": naive_dispatch,
        "m139_seconds_per_wrapper_model": residual_per_call,
        "modeled_wrapper_residual_s": naive_dispatch * residual_per_call,
        "d1_observed_residual_s": [
            row["literal_meter"]["residual_wall_time_s"] for row in d1["trials"]
        ],
        "note": "Cross-product wave batching lowers wrappers only by materializing/packing banks; it does not lower the arithmetic falsification.",
    }

    strict_cap512 = physical_boundary["512"]["totals"]["strict_slots"]
    strongest_cap512 = physical_boundary["512"]["totals"]["symmetry_tag_conditional"]

    bindings = {}
    for path in (
        D1_RESULT,
        D1_SOURCE,
        REFERENCE_RESULT,
        CAP_RESULTS[1792],
        CAP_RESULTS[1024],
        RAW_FRONTIER,
        TINY_R40,
        R40_SOURCE,
        R40_SCHEDULE,
        M124,
        M185,
        Path(__file__).resolve(),
    ):
        bindings[str(path)] = sha256(path)

    return {
        "schema": "public-q24-r40-economic-falsification-v1",
        "status": "FAIL_PHYSICAL_COMPUTE_ROUTE_AT_CURRENT_UNCHANGED_RAW_THRESHOLD",
        "current_board_boundary": {
            "unchanged_raw_flop_threshold": TARGET_FLOPS,
            "supersedes": 582_395_000_000,
            "reason": "G11B projection invalid; bank #328586 remains authority",
        },
        "bindings": bindings,
        "d1_reconciliation": {
            "total": total,
            "operation_sum": operation_sum,
            "matmul": matmul,
            "einsum": einsum,
            "other": other,
            "einsum_components": EINSUM_COMPONENTS,
            "matrix_eligible": eligible,
            "non_gemm_einsum": NON_GEMM_EINSUM,
            "cp4_plane_einsum": CP4_PLANE_EINSUM,
        },
        "optimistic_floor_correction": {
            "old_uniform_discount_exact_fraction": str(optimistic_old),
            "old_uniform_discount_floor": floor_fraction(optimistic_old),
            "error": "It discounts all 40.321167360B non-GEMM einsums and applies a rank ratio to an already-fast D1 leaf inventory.",
            "corrected_zero_tax_rank_only_exact_fraction": str(corrected_base),
            "corrected_zero_tax_rank_only_floor": floor_fraction(corrected_base),
            "corrected_margin_vs_current_target": floor_fraction(corrected_base) - TARGET_FLOPS,
        },
        "cap_deltas": CAP_DELTAS,
        "cap_static_operation_deltas": CAP_STATIC_OPERATION_DELTAS,
        "measured_cap_checks": measured_cap_checks,
        "cap_science": cap_science,
        "raw_context_only": {
            "cap1024_projected_raw_ceiling_not_a_cap_score": raw_frontier["objective"]["cap1024_projected_raw_ceiling_from_root_cost_frontier"],
            "warning": "No cap1024/768/512 raw MSE has been measured in these receipts.",
        },
        "cse": {
            "tiers": CSE_TIERS,
            "four_pair_reason": "M11 and M22 have 3 self derivative channels and 7 transpose-paired channel orbits; M21/M31 retain all 17 channels.",
            "strict_slot_copy_cancels_missing_weight_multiply": G4_MISSING_WEIGHT_MULTIPLIES,
            "reset_reuse": {
                "covariance2_and_variance_outer": RESET_REUSE_COV2_AND_VARIANCE_OUTER,
                "shared_scaled_variance": RESET_REUSE_SCALED_VARIANCE,
            },
            "remaining_exact_factorization_constraint": "The 54 required carrier-channel contractions (34 asymmetric plus 20 symmetric orbit representatives) have no established exact cross-family CSE in the current Q24 representation.",
        },
        "rank_only_cap_boundary": rank_only,
        "physical_tiled_boundary": physical_boundary,
        "memory": memory,
        "dispatch": dispatch,
        "terminal": {
            "cap512_strict_separated_free_forward": strict_cap512["separated_free_forward"],
            "cap512_strict_fused_free_forward": strict_cap512["fused_free_forward"],
            "cap512_strongest_conditional_fused_free_forward": strongest_cap512["fused_free_forward"],
            "cap512_strict_fused_physical_arithmetic": strict_cap512["fused_physical_arithmetic"],
            "conclusion": "No static plan reaches the current unchanged-raw threshold. Even the strongest conditional fused/free-forward lower bound remains above it, while charging actual forward transforms moves the plan far away.",
            "smallest_scientific_discriminator_if_boundary_reopens": "Measure raw MSE for cap1024 first (existing full-size drift is already bound); do not spend a cap768/512 raw run until a literal physical plan crosses the live threshold with margin.",
            "external_constraint": "A new exact carrier factorization and/or a benchmark primitive that amortizes both forward and inverse transforms is required; operand reuse alone cannot affect the charged inverse-output lower bound.",
        },
        "tiny_r40_authority": {
            "tensor_exact": tiny["tensor_certificate"]["exact"],
            "tiny_flops": tiny["tiny_physical"]["flops"],
            "tiny_wrapper_calls": tiny["tiny_physical"]["wrapper_calls"],
            "tiny_r40_vs_direct32_max_abs": tiny["tiny_physical"]["numerical"]["r40_vs_direct32_max_abs"],
        },
        "scope": {
            "estimator_executed": False,
            "fixture_opened": False,
            "target_opened": False,
            "package_changed": False,
            "submission_attempted": False,
            "only_static_integer_arithmetic_executed": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "ECONOMIC_GAP.json")
    args = parser.parse_args()
    report = build_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["terminal"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
