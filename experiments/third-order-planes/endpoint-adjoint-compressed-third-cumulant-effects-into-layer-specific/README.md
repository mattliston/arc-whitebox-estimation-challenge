# Endpoint adjoint compressed third-cumulant effects into layer-specific scalar actions

2026-09-04 · experiment · informational · measured

S = sum_l <dK21_l, M_l> with closed-form t0 adjoint M_l = diag(a2_l)[sum_{L>l} B_{l->L} diag(S_v g_L) B_{l->L}^T] diag(e1_l); sanity 15.7% linear capture; value-rank r95 per layer L7 104, L8 135, L10 45, L11 14, L12 93, L13 11, L14 8 (L9 >200). Shelved 08-31 because the reads seemed to need the tensor (C3 randomized-read FAIL). Re-found 2026-09-01 (T-21).

Recorded measurement:

    S = sum_l <dK21_l, M_l> with closed-form t0 adjoint M_l = diag(a2_l)[sum_{L>l} B_{l->L} diag(S_v g_L) B_{l->L}^T] diag(e1_l); sanity 15.7% linear capture; value-rank r95 per layer L7 104, L8 135, L10 45, L11 14, L12 93, L13 11, L14 8 (L9 >200). Shelved 08-31 because the reads seemed to need the tensor (C3 randomized-read FAIL). Re-found 2026-09-01 (T-21). The DIRECT emission-delta supply for the reads is dead (T-28: engine and card emissions agree to 1e-5); the reads must be fed by the carried residual (N1.2b) — the adjoint remains the cheapest way to CONSUME whatever carrier wins. T-43: the linear consumption frame is retired for value; the adjoint compression remains a cheap SCREEN for candidate supplies, not the deploy path.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g19_attribution_v1/run_attribution.py](code/public_g19_attribution_v1/run_attribution.py)
- [code/public_g19_attribution_v1/run_c1_adjoint_rank.py](code/public_g19_attribution_v1/run_c1_adjoint_rank.py)
- [code/public_g19_attribution_v1/run_tangent_v2.py](code/public_g19_attribution_v1/run_tangent_v2.py)
- [code/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

## Evidence

- [C1_ADJOINT_RANK_rank0035.json#L1](C1_ADJOINT_RANK_rank0035.json#L1)
- [LANE_g3_tracker.md#L1](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L1)

<details>
<summary>All result files and records (2)</summary>

- [C1_ADJOINT_RANK_rank0035.json](C1_ADJOINT_RANK_rank0035.json) · result
- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-endpoint-adjoint-compressed-third-cumulant-effects-into-layer-specific) · [Copy provenance](../../../PROVENANCE.json)

