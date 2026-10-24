# Exact peeling with a 600-request cap did not transfer historical coverage to the current union products

2026-10-18T05:24:06Z · experiment · rejected · measured

A conservative bookkeeping reduction funded a fixed exact-peeling subset excluding known slow routes. The current native products hit only one selected push per network and no reads, so historical bill forecasts substantially overstated the saving.

Recorded measurement:

    Historical metadata: 0.468738% bill; native 35/150: raw +0.038608%, bill -0.093079%, adjusted -0.054485% → 1.45071441891e-09.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct18_bill_v1/check_peel600_rollout_v1.py](code/astra_oct18_bill_v1/check_peel600_rollout_v1.py)
- [code/astra_oct18_lead_v1/build_peel600_v2.py](code/astra_oct18_lead_v1/build_peel600_v2.py)
- [code/astra_oct18_lead_v1/candidate_bookkeeping_peel600_HELD_v3.py](code/astra_oct18_lead_v1/candidate_bookkeeping_peel600_HELD_v3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5233](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5233)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5239](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5239)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5243](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5243)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5255](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5255)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-exact-peeling-600-request-cap-did-not-transfer-historical) · [Copy provenance](../../../PROVENANCE.json)

