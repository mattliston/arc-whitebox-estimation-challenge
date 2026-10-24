#!/usr/bin/env python3
"""Deterministic Phase-1-style Kerdock prefix cost and constraint ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
RESULT = HERE / "COST_LEDGER.json"

BUDGET = 2**41
WIDTH = 1024
LOG2_WIDTH = 10
WINOGRAD_LEVELS = 6
ROW_TILE = 2048
D1_FLOPS = (1_574_162_015_607, 1_612_602_817_032, 1_573_956_487_680)

DEPENDENCIES = {
    "d1_candidate": (
        P2 / "p2r0c1d9_cp3_arena_depth1_full_estimator_v1/D1_CANDIDATE_RESULT.json",
        "56212fc97101fd1d6054899c57393fbc5dc0ecaa6bbe57decf91d233c0b2594d",
    ),
    "single_h7_ceiling": (
        P2 / "p2r0g1_kerdock_h7_anchor_ceiling_v1/RESULT.json",
        "4f4e2798021796ab735f3ae631b61cd67494e2bb279fc8630ef9e7de905f661e",
    ),
    "multilayer_ceiling": (
        P2 / "p2r0g1_kerdock_multilayer_anchor_ceiling_v1/RESULT.json",
        "a96828887522e8e412cfb53a9d39d2dd39d3ca561a43c1e66d52f27f73f78ca5",
    ),
    "complete_action_interface": (
        P2 / "p2r0f3b2b6_fused_directional_innovation_action_v1/RECEIPT.json",
        "04c2c18ddf6281cfdaa14d0cfaea0730415b8f9410f3341400e20bbc788bb6d6",
    ),
    "quadrature_verdict": (
        P2 / "public_quadrature_derivation_v1/QUADRATURE_VERDICT.json",
        "fd99ba5597293a2e10c98602acc884c498bf761faf8c8100ad44ad2743dbd41f",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def winograd_arena(m: int, k: int, n: int, levels: int = WINOGRAD_LEVELS) -> int:
    """Exact arithmetic count for the Phase-1 arena Winograd circuit.

    At each level the seven half-size products require four A transforms,
    four B transforms, and seven C reconstructions.  Pre-bound arena views
    remove stack/concatenate/copy arithmetic.  Leaves use M*N*(2*K-1).
    """
    total = 0
    nodes = 1
    for _ in range(levels):
        if m % 2 or k % 2 or n % 2:
            raise ValueError((m, k, n, levels))
        total += nodes * (m * k + k * n + 7 * m * n // 4)
        m //= 2
        k //= 2
        n //= 2
        nodes *= 7
    return total + nodes * m * n * (2 * k - 1)


def prefix_core(paths: int, endpoint: int) -> dict[str, int]:
    if paths % (2 * WIDTH):
        raise ValueError("an antipodal MUB union has 2*b*n paths")
    bases = paths // (2 * WIDTH)
    positive = paths // 2
    # Two independently oriented equal basis blocks are required for the
    # frozen split-block error check.  A take costs four per element.
    orientation = 2 * 4 * WIDTH * WIDTH
    layer1 = (
        orientation
        + bases * WIDTH * WIDTH
        + bases * WIDTH * WIDTH * LOG2_WIDTH
        + positive * WIDTH
    )
    layer2 = (
        winograd_arena(positive, WIDTH, WIDTH)
        + 2 * winograd_arena(WIDTH, WIDTH, WIDTH)
        + bases * WIDTH * WIDTH
        + bases * WIDTH * WIDTH * LOG2_WIDTH
        + positive * WIDTH
        + paths * WIDTH
    )
    later_layer = winograd_arena(paths, WIDTH, WIDTH) + paths * WIDTH
    total = layer1 + layer2 + (endpoint - 2) * later_layer
    return {
        "paths": paths,
        "bases": bases,
        "endpoint_one_based": endpoint,
        "independent_orientation_take": orientation,
        "layer1_sign_fwht_relu": layer1,
        "layer2_positive_antipodal_two_compositions_relu": layer2,
        "each_full_layer_after_h2": later_layer,
        "total": total,
    }


def dense_projection(paths: int, rank: int, layers: int) -> int:
    return layers * paths * rank * (2 * WIDTH - 1)


def marginal_readout(paths: int, layers: int) -> int:
    # mean, center, powers 2:5, four moment reductions = 10*N*n;
    # connected k4/k5 conversions = six vector operations.
    return layers * (10 * paths * WIDTH + 6 * WIDTH)


def layerwise_schedule(prefix_paths: int, prefix_endpoint: int, suffix_paths: int) -> dict[str, int]:
    prefix = prefix_core(prefix_paths, prefix_endpoint)["total"]
    suffix_layers = 16 - prefix_endpoint
    suffix_layer = winograd_arena(suffix_paths, WIDTH, WIDTH) + suffix_paths * WIDTH
    return {
        "prefix_paths": prefix_paths,
        "prefix_endpoint_one_based": prefix_endpoint,
        "suffix_paths": suffix_paths,
        "suffix_layers": suffix_layers,
        "prefix_flops": prefix,
        "each_suffix_layer_flops": suffix_layer,
        "total": prefix + suffix_layers * suffix_layer,
    }


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    for name, (path, expected) in DEPENDENCIES.items():
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"{name} changed: {actual} != {expected}")

    prefixes = {f"N65536_h{h}": prefix_core(65_536, h) for h in (7, 8, 9)}
    full32768 = prefix_core(32_768, 16)
    projection = {
        str(paths): {
            str(rank): dense_projection(paths, rank, 6) for rank in (6, 8, 13)
        }
        for paths in (65_536, 32_768)
    }
    marginals = {
        "N65536_h2_h7": marginal_readout(65_536, 6),
        "N65536_h2_h9": marginal_readout(65_536, 8),
        "N32768_h2_h7": marginal_readout(32_768, 6),
        "N32768_h2_h16": marginal_readout(32_768, 15),
    }
    action_receipt = json.loads(DEPENDENCIES["complete_action_interface"][0].read_text())
    action_cost = action_receipt["frozen_cost_N65536_rank13"]

    combined: dict[str, object] = {}
    for label, prefix in prefixes.items():
        rows = []
        for index, host in enumerate(D1_FLOPS):
            used = host + prefix["total"]
            rows.append(
                {
                    "fixture_index": index,
                    "d1_flops": host,
                    "combined_core_flops": used,
                    "remaining_to_2pow41": BUDGET - used,
                    "hard_budget_pass": used < BUDGET,
                }
            )
        combined[label] = rows

    vbar = 0.066794
    iid_mse_n32768 = vbar / 32_768
    required_ratio = 1.0e-8 / iid_mse_n32768
    result = {
        "schema": "p2r0g1-kerdock-prefix-cost-ledger-v1",
        "status": "complete",
        "meter_law": {
            "budget": BUDGET,
            "width": WIDTH,
            "winograd_levels": WINOGRAD_LEVELS,
            "row_tile": ROW_TILE,
            "recurrence": "T_d(M,K,N)=7*T_(d-1)(M/2,K/2,N/2)+M*K+K*N+7*M*N/4; T_0=M*N*(2*K-1)",
            "implementation": "Phase-1-style pre-bound float32 arenas; four A transforms, four B transforms, seven C reconstructions; row tiles cap memory; no stack/concatenate/copy arithmetic",
            "full_N65536_width1024_product": winograd_arena(65_536, WIDTH, WIDTH),
            "dense_comparator_same_product": 65_536 * WIDTH * (2 * WIDTH - 1),
            "ratio_to_dense": winograd_arena(65_536, WIDTH, WIDTH) / (65_536 * WIDTH * (2 * WIDTH - 1)),
        },
        "design": {
            "N65536": "two independent randomized blocks, each 16 real Kerdock bases plus antipodes",
            "N32768": "two independent randomized blocks, each 8 real Kerdock bases plus antipodes",
            "degree_exactness": "every antipodal union of orthonormal bases is exact through degree 3; these proper subsets are not spherical 5-designs",
            "fourth_moment_ratio_full_b32": WIDTH / (3 * 32) + 1 / 3,
            "fourth_moment_ratio_each_b16_split_block": WIDTH / (3 * 16) + 1 / 3,
            "bias_boundary": "split-block disagreement estimates orientation variance but cannot certify common degree>=4 cubature bias; any winning query must pass the frozen 2e-5 action-weighted error gate on allowed development data",
        },
        "prefix_core": prefixes,
        "full_depth_N32768_core": full32768,
        "combined_with_d1_fixtures": combined,
        "incremental_readouts": {
            "literal_dense_projection_h2_h7": projection,
            "marginal_mean_cov_k3_k4_k5": marginals,
            "complete_symmetric_rank13_action_receipt": action_cost,
            "shared_sample_reuse": "one propagated point cloud supplies every registered layer/ray; covariance, k3, k4, k5 and k3^2/k3*k4/k3^3 are accumulated immediately after each ReLU and need no additional network propagation",
        },
        "layerwise_frontiers": {
            "uniform_N32768_through_h16": layerwise_schedule(32_768, 16, 32_768),
            "N65536_through_h7_then_N16384_through_h16": layerwise_schedule(65_536, 7, 16_384),
            "N65536_through_h9_then_N8192_through_h16": layerwise_schedule(65_536, 9, 8_192),
            "allocation_rule": "choose nested antipodal basis blocks; endpoint-action variance and downstream Duhamel gain must determine which blocks survive, so no accuracy-optimal schedule is asserted before the action receipt",
        },
        "no_double_count_law": "At layer l estimate only the teacher-forced local connected defect: sampled post-ReLU action coordinate minus the incumbent one-step prediction evaluated from that same empirical pre-state. The endpoint lane applies its full-state future Duhamel action once. Never sum repeated accumulated-state resets.",
        "full_depth_closure_anchor_cv": {
            "status": "closed_under_current_host",
            "feasible_N32768_full_depth_core_flops": full32768["total"],
            "exact_affine_lift": "exists, but an antipodal/Kerdock rule integrates it exactly, so E[g]+Q[f-g]=Q[f] identically",
            "measured_affine_capture": 0.2050490326254727,
            "nonlinear_exact_mean_lift": "absent",
            "required_supplier_interface": "a cheap pointwise nonlinear g(x;W), an analytic proof that E[g] equals the deployed closure output coordinatewise, and residual-variance capture high enough to pass the target under the chosen structured rule",
            "row35_plain_iid_mse_at_N32768": iid_mse_n32768,
            "required_residual_variance_ratio_for_raw_1e_8_at_N32768": required_ratio,
            "required_capture_for_raw_1e_8_at_N32768": 1.0 - required_ratio,
            "degree64_oracle_at_N65536_mse": 2.850276942500284e-8,
            "conclusion": "direct final-output sampling and closure-plus-constant anchoring remain closed; only internal connected innovations into the complete-action suffix survive",
        },
        "old_scalar_suffix_ceiling": {
            "single_exact_h7_mean_residual_reset_mse": 3.3897455206496917e-8,
            "exact_h2_through_h7_mean_residual_resets_mse": 3.3897455206496917e-8,
            "exact_h2_through_h9_mean_residual_resets_mse": 3.16038750480864e-8,
            "scope": "full width mean-residual state only, not a marginal or joint cumulant re-anchor",
            "decision": "closed for the old scalar recurrence; a sampled source is useful only through the new complete-action/resummed suffix",
        },
        "access": {
            "targets_opened": 0,
            "observer_selection_opened": 0,
            "network_names_opened": 0,
            "board_or_deployment_truth_opened": 0,
            "fixture_calls": 0,
            "estimators_run": 0,
            "submissions": 0,
        },
        "provenance": {
            "program_sha256": sha256(Path(__file__)),
            "phase1_pdf_sha256": "c64ed2e42aff61c16fe95b615911499109bded2f70d00b7510095ddc2b7846c3",
            "dependencies": {name: {"path": str(path.relative_to(P2)), "sha256": digest} for name, (path, digest) in DEPENDENCIES.items()},
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(RESULT), "prefix_totals": {k: v["total"] for k, v in prefixes.items()}, "full_N32768": full32768["total"]}, sort_keys=True))


if __name__ == "__main__":
    main()
