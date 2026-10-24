# Common-u basis reordered by within-span eigen-energy at layer 10: two-network -0.96% but fresh-16 net +0.37%, closed

2026-10-14T19:13:23Z · experiment · rejected · measured

Reordering the common-u coordinates at layer 10 by within-existing-span eigen-energy (the total-lambda-squared Nystrom core as a proxy for common-u energy) looked strong on the two fixed networks at about 0.3% extra bill, but on 16 untouched fresh networks the accuracy gain vanished. A zero-cost diagonal-core permutation passed its smoke and was held but never confirmed.

Recorded measurement:

    eigen-ordering smoke: adjusted -0.853271% / -1.061101%, geometric -0.957241%, bill ~+0.30%; fresh 16 (7103432-3447): log MSE +0.060269%, log bill +0.305358%, net +0.365627% (5/16 lower); diagonal-core permutation smoke: adjusted -0.101175% / -0.364783%, geometric -0.233066%, bill ~+0.000757%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_common_u_order_v1/common_u_order_fragment_v1.py](code/astra_common_u_order_v1/common_u_order_fragment_v1.py)
- [code/astra_common_u_diag_order_v1/common_u_diag_order_fragment_v1.py](code/astra_common_u_diag_order_v1/common_u_diag_order_fragment_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2970](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2970)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2976](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2976)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2982](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2982)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2995](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2995)

<details>
<summary>All result files and records (3)</summary>

- [CUO_NATIVE_FRESH_DECISION_v1.json](CUO_NATIVE_FRESH_DECISION_v1.json) · result
- [ROOT_SMOKE_RETIREMENT_v1.json](ROOT_SMOKE_RETIREMENT_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-common-u-basis-reordered-within-span-eigen-energy-layer) · [Copy provenance](../../../PROVENANCE.json)

