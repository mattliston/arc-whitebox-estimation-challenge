# Final-layer outer-half old-bank reconstruction cut: bill -0.70% but fresh-16 adjusted +36.5%, lost in the skew proxy

2026-10-14T18:13:06Z · experiment · rejected · measured

The 'terminal central hole' arm replaces the final-layer old-bank reconstruction by its outer half (64-row upward rounding) with an effective first-chaos central skew trace and a retained-row regression proxy for the diagonal term. Its exact price was a 0.70% bill cut, but the 16-network fresh screen lost about 44% of adjusted score. A one-network counterfactual attributed the loss to the skew proxy (the diagonal proxy alone was neutral), and a rank-8 third-Hermite repair of the skew removed only 5% of the excess error.

Recorded measurement:

    price: whole bill -0.7398917288141238% on 7100136 (736 rows to 384), -0.6677598475695157% on 7100137 (800 to 448), geometric -0.703832338%; fresh 16: mean log net +36.486051% (about 44% worse adjusted); counterfactuals on 7100136: skew-only MSE ratio 1.434103406, diagonal-proxy-only 0.994398181, both 1.426506776; q8 third-Hermite repair: adjusted +39.520106%, removes 5.132% of the excess MSE, +179,232,295 FLOPs

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_terminal_central_hole_source_v2/finite_tch_qualification_HELD_v2.py](code/lead/runs/astra_terminal_central_hole_source_v2/finite_tch_qualification_HELD_v2.py)
- [code/lead/runs/astra_tch_diagnostic_v1/terminal_diagnostic_fragment_v1.py](code/lead/runs/astra_tch_diagnostic_v1/terminal_diagnostic_fragment_v1.py)
- [code/lead/runs/astra_tch_q8_smoke_v1/terminal_q8_fragment_v1.py](code/lead/runs/astra_tch_q8_smoke_v1/terminal_q8_fragment_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/matched_c_tch_OFF_v2/candidate.py](code/lead/runs/astra_terminal_central_hole_source_v2/matched_c_tch_OFF_v2/candidate.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/matched_c_tch_v2/candidate.py](code/lead/runs/astra_terminal_central_hole_source_v2/matched_c_tch_v2/candidate.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/native_tch_qualification_judge_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/native_tch_qualification_judge_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/terminal_central_hole_fragment_v2.py](code/lead/runs/astra_terminal_central_hole_source_v2/terminal_central_hole_fragment_v2.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)

<details>
<summary>Additional programs and dependencies (3)</summary>

- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/k4liab_s51k2tw/candidate.py](code/scout_native/k4liab_s51k2tw/candidate.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2913](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2913)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2920](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2920)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2929](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2929)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2931](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2931)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2943](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2943)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2953](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2953)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2955](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2955)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2959](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2959)

<details>
<summary>All result files and records (5)</summary>

- [ASTRA_FRESH_RETIREMENT_v1.json](ASTRA_FRESH_RETIREMENT_v1.json) · record
- [ASTRA_QUALIFICATION_RETIREMENT_v1.json](ASTRA_QUALIFICATION_RETIREMENT_v1.json) · record
- [ROOT_DIAGNOSTIC_RETIREMENT_v1.json](ROOT_DIAGNOSTIC_RETIREMENT_v1.json) · result
- [ROOT_SMOKE_RETIREMENT_v1.json](ROOT_SMOKE_RETIREMENT_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-final-layer-outer-half-old-bank-reconstruction-cut-bill) · [Copy provenance](../../../PROVENANCE.json)

