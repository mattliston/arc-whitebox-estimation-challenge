# Squared right-coefficient weighting of rank-32 common tail: pilot-8 -0.21%, all-24 adjusted +0.14%

2026-10-15T03:29:04Z · experiment · rejected · measured

Weighting the weak-32 coordinate energy and the 96-wide Gram by the right lambda-squared (same 224+32 budget, +52 requests) passed its pilot against the fired file but failed the full 24 like the unweighted version.

Recorded measurement:

    pilot 8: -0.208820% vs d401, -0.012394% vs UC32; full 24: fresh 16 +0.019916%, OOS 8 +0.373081%, all 24 +0.137499% vs d401 (MSE -0.078591%, bill +0.216260%); vs UC32 -0.034920%; price +5.033/5.045M FLOPs, +52 requests

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_uc32_right_weight32_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_uc32_right_weight32_HELD_v1.py](code/candidate_uc32_right_weight32_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3285](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3285)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3289](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3289)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3293](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3293)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3295](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3295)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3297](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3297)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3299](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3299)

<details>
<summary>All result files and records (2)</summary>

- [FINAL_RESULT_v1.json](FINAL_RESULT_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Common-coordinate tail rank 32: two-network -0.84%, pilot-8 -0.20%, all-24 adjusted +0.17%](../common-coordinate-tail-rank-32-two-network-0-84/README.md)

[Research graph](../../../graph/index.html#node=2026-10-15-squared-right-coefficient-weighting-rank-32-common-tail-pilot) · [Copy provenance](../../../PROVENANCE.json)

