"""Conservative target-free source projection for width 1024/depth 16.

The measured-base carryover and enumerated source counts are not a literal
full-shape meter.  They qualify budget plausibility only.
"""

from __future__ import annotations

import json


WIDTH = 1024
DEPTH = 16
FULL_BUDGET = 1 << 41
CP3_CAP = 16 * WIDTH
CP4_CAP = 2 * WIDTH
PAIR_Q = 32
FULL_RESETS = 14

# This is the measured incumbent envelope inherited intact by the design.  It
# includes the exact first-two-layer foothold, CP factor transport/birth, and
# incumbent marginal/Hermite host work; it is not re-derived from a guessed
# GEMM count.
MEASURED_INCUMBENT_BASE = 620_090_560_514
FIXED_ASSEMBLY_RESERVE = 6_000_000_000
FINAL_CP_K3_RESERVE = 200_000_000


def gemm_flops(width: int) -> int:
    return width * width * (2 * width - 1)


def cp_k21_flops(last_full_layer: int) -> int:
    rank_sum = WIDTH * last_full_layer * (last_full_layer + 1) // 2
    return WIDTH * WIDTH * (4 * rank_sum - last_full_layer)


def projected_ledger() -> dict[str, object]:
    n = WIDTH
    gemm = gemm_flops(n)
    conservative_cp_k21 = cp_k21_flops(15)
    specialized_cp_k21 = cp_k21_flops(14)
    conservative_h3 = 15 * 4 * gemm
    specialized_h3 = (14 * 4 + 1) * gemm
    conservative_h4 = 15 * 6 * gemm
    specialized_h4 = (14 * 6 + 1) * gemm
    psd = FULL_RESETS * (2 * 9 * n ** 3 + 2 * gemm)
    q32_pairs = FULL_RESETS * 4 * 17 * 98 * n * n * 2
    q32_h1 = 4 * 98 * n * n * 2

    common = (
        MEASURED_INCUMBENT_BASE
        + psd
        + q32_pairs
        + q32_h1
        + FIXED_ASSEMBLY_RESERVE
        + FINAL_CP_K3_RESERVE
    )
    conservative_total = (
        common + conservative_cp_k21 + conservative_h3 + conservative_h4
    )
    specialized_total = common + specialized_cp_k21 + specialized_h3 + specialized_h4

    cp4_push = n * (2 * n - 1) * (n + 14 * 2 * n)
    cp4_full_plane_reads = 2 * n * n * (
        (2 * n - 1) + 13 * (4 * n - 1)
    )
    cp4_dropped_plane_reads = 13 * 2 * n * n * (2 * n - 1)
    cp4_w16_marginal_and_add = 8 * n * n
    cp4_dominant_subtotal = (
        cp4_push
        + cp4_full_plane_reads
        + cp4_dropped_plane_reads
        + cp4_w16_marginal_and_add
    )
    cp4_lower_order_enumerated = 397_488_128
    inherited_rank_sum = n + 13 * 2 * n
    cp4_inherited_diagonal_rereads = (
        2 * n * inherited_rank_sum
        + n * (2 * n - 1)
        + 13 * n * (4 * n - 1)
    )
    cp4_subtotal = (
        cp4_dominant_subtotal
        + cp4_lower_order_enumerated
        + cp4_inherited_diagonal_rereads
    )
    full_architecture_total = specialized_total + cp4_subtotal

    float32 = 4
    float64 = 8
    cp_payload = CP3_CAP * float32 + 2 * n * CP3_CAP * float32
    cp4_payload = CP4_CAP * float32 + n * CP4_CAP * float32
    cp4_candidate = 3 * n * float32 + n * 3 * n * float32
    dense_high_state = 2 * n * n * float32
    mean_covariance = (n + n * n) * float64
    rho_stack = (PAIR_Q + 1) * n * n * float64
    q32_group3_bound = 3 * rho_stack

    return {
        "schema": "fb-ec4-32-cp4r2048-source-projection-ledger-v2",
        "shape": {"width": n, "depth": DEPTH, "full_resets": FULL_RESETS},
        "constants": {
            "dense_gemm_flops": gemm,
            "cp3_cap": CP3_CAP,
            "cp4_cap": CP4_CAP,
            "q32_terms": PAIR_Q + 1,
            "derivative_channels": 17,
            "pair_moments_per_reset": 4,
        },
        "common_flops": {
            "measured_incumbent_base": MEASURED_INCUMBENT_BASE,
            "diagonal_preserving_psd_14": psd,
            "q32_pair_kernels_14x4x17": q32_pairs,
            "gaussian_h1_q32": q32_h1,
            "fixed_array_activation_recenter_reserve": FIXED_ASSEMBLY_RESERVE,
            "final_cp_k3_reserve": FINAL_CP_K3_RESERVE,
            "total": common,
        },
        "conservative_unspecialized": {
            "cp_k21_z2_through_z16": conservative_cp_k21,
            "h3_15x4_gemm": conservative_h3,
            "h4_15x6_gemm": conservative_h4,
            "total": conservative_total,
            "rounded_headline": "<1.992e12",
            "headroom": FULL_BUDGET - conservative_total,
        },
        "final_mean_specialized_target": {
            "cp_k21_z2_through_z15": specialized_cp_k21,
            "h3_14x4_plus_final_diagonal": specialized_h3,
            "h4_14x6_plus_final_diagonal": specialized_h4,
            "total": specialized_total,
            "fraction_of_2pow41": specialized_total / FULL_BUDGET,
            "headroom": FULL_BUDGET - specialized_total,
        },
        "coherent_cp4_addon": {
            "rank_schedule": {
                "w2": n,
                "w3_through_w16": 2 * n,
                "full_plane_reads": "w2_through_w15",
                "drop_rank1024_events": 13,
            },
            "factor_push": cp4_push,
            "full_d_a_b_reads": cp4_full_plane_reads,
            "dropped_d_a_b_reads": cp4_dropped_plane_reads,
            "w16_marginal_and_host_add": cp4_w16_marginal_and_add,
            "dominant_subtotal": cp4_dominant_subtotal,
            "enumerated_lower_order": cp4_lower_order_enumerated,
            "stored_gated_inherited_d_rereads": cp4_inherited_diagonal_rereads,
            "subtotal": cp4_subtotal,
        },
        "executed_specialized_architecture": {
            "base_fb_ec4_32": specialized_total,
            "coherent_cp4_addon": cp4_subtotal,
            "total": full_architecture_total,
            "fraction_of_2pow41": full_architecture_total / FULL_BUDGET,
            "headroom": FULL_BUDGET - full_architecture_total,
        },
        "unspecialized_with_cp4_negative_control": {
            "total": conservative_total + cp4_subtotal,
            "over_budget_by": conservative_total + cp4_subtotal - FULL_BUDGET,
            "eligible": False,
        },
        "inactive_cap2560_successor_policy": {
            "active": False,
            "q_above_32": "HOLD",
            "projected_addon": 272_151_468_032,
            "projected_total": 2_181_353_793_026,
            "projected_headroom": 17_669_462_526,
            "activation_gate": (
                "Only an additive separately audited source after the exact "
                "cap2048 literal meter proves sufficient measured headroom."
            ),
        },
        "memory_bytes": {
            "cp3_persistent_payload": cp_payload,
            "cp4_persistent_payload": cp4_payload,
            "cp4_append_candidate_payload": cp4_candidate,
            "h3_h4_persistent_payload": dense_high_state,
            "mean_covariance_persistent_payload": mean_covariance,
            "rho_q0_through_q32_stack": rho_stack,
            "q32_group3_working_bound": q32_group3_bound,
            "phase2_limit": 8 * (1 << 30),
        },
        "qualification": "SOURCE_ONLY_FINAL_BYTE_FREEZE__LITERAL_METER_REQUIRED",
    }


def verify() -> dict[str, object]:
    ledger = projected_ledger()
    assert ledger["constants"]["dense_gemm_flops"] == 2_146_435_072
    assert ledger["common_flops"] == {
        "measured_incumbent_base": 620_090_560_514,
        "diagonal_preserving_psd_14": 330_683_121_664,
        "q32_pair_kernels_14x4x17": 195_655_892_992,
        "gaussian_h1_q32": 822_083_584,
        "fixed_array_activation_recenter_reserve": 6_000_000_000,
        "final_cp_k3_reserve": 200_000_000,
        "total": 1_153_451_658_754,
    }
    assert ledger["conservative_unspecialized"] == {
        "cp_k21_z2_through_z16": 515_380_346_880,
        "h3_15x4_gemm": 128_786_104_320,
        "h4_15x6_gemm": 193_179_156_480,
        "total": 1_990_797_266_434,
        "rounded_headline": "<1.992e12",
        "headroom": 208_225_989_118,
    }
    specialized = ledger["final_mean_specialized_target"]
    assert specialized["cp_k21_z2_through_z15"] == 450_956_886_016
    assert specialized["h3_14x4_plus_final_diagonal"] == 122_346_799_104
    assert specialized["h4_14x6_plus_final_diagonal"] == 182_446_981_120
    assert specialized["total"] == 1_909_202_324_994
    assert specialized["headroom"] == 289_820_930_558
    assert ledger["coherent_cp4_addon"] == {
        "rank_schedule": {
            "w2": 1_024,
            "w3_through_w16": 2_048,
            "full_plane_reads": "w2_through_w15",
            "drop_rank1024_events": 13,
        },
        "factor_push": 62_246_617_088,
        "full_d_a_b_reads": 115_934_756_864,
        "dropped_d_a_b_reads": 55_807_311_872,
        "w16_marginal_and_host_add": 8_388_608,
        "dominant_subtotal": 233_997_074_432,
        "enumerated_lower_order": 397_488_128,
        "stored_gated_inherited_d_rereads": 113_231_872,
        "subtotal": 234_507_794_432,
    }
    assert ledger["executed_specialized_architecture"]["total"] == 2_143_710_119_426
    assert ledger["executed_specialized_architecture"]["headroom"] == 55_313_136_126
    assert ledger["unspecialized_with_cp4_negative_control"] == {
        "total": 2_225_305_060_866,
        "over_budget_by": 26_281_805_314,
        "eligible": False,
    }
    assert ledger["inactive_cap2560_successor_policy"] == {
        "active": False,
        "q_above_32": "HOLD",
        "projected_addon": 272_151_468_032,
        "projected_total": 2_181_353_793_026,
        "projected_headroom": 17_669_462_526,
        "activation_gate": (
            "Only an additive separately audited source after the exact "
            "cap2048 literal meter proves sufficient measured headroom."
        ),
    }
    assert ledger["conservative_unspecialized"]["total"] < FULL_BUDGET
    assert specialized["total"] < FULL_BUDGET
    assert ledger["executed_specialized_architecture"]["total"] < FULL_BUDGET
    assert ledger["memory_bytes"]["q32_group3_working_bound"] < ledger["memory_bytes"]["phase2_limit"]
    return ledger


def main() -> None:
    print(json.dumps(verify(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
