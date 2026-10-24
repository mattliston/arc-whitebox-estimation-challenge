# One-shot residual-diagonal pivot refinement: adjusted +0.162% on base and +1.520% on rank-384 stack

2026-10-16T02:52:29Z · experiment · rejected · measured

Replacing 32 pivots required a second QR/core/diagonal computation whose cost overwhelmed the accuracy change. Reducing the replacement block did not remove that fixed cost.

Recorded measurement:

    baseMSE-1.083%,SE.94,bill+1.245%,net+.162%;rank384MSE+.211%,SE.65,bill+1.309%,net+1.520%;ops+283..381;k16bill+1.055%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_join_v1/build_oracle_v1.py](code/opus_join_v1/build_oracle_v1.py)
- [code/opus_join_v1/build_pivot_v1.py](code/opus_join_v1/build_pivot_v1.py)
- [code/opus_join_v1/build_power_v1.py](code/opus_join_v1/build_power_v1.py)
- [code/opus_join_v1/candidate_2b108_join_oracle_v1.py](code/opus_join_v1/candidate_2b108_join_oracle_v1.py)
- [code/opus_join_v1/candidate_2b108_join_pivot_v1.py](code/opus_join_v1/candidate_2b108_join_pivot_v1.py)
- [code/opus_join_v1/candidate_2b108_join_power_v1.py](code/opus_join_v1/candidate_2b108_join_power_v1.py)
- [code/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py](code/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3745](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3745)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [JOIN_ORACLE_RESULT_v1.json](../../../records/research/2026-10-16-exact-join-eigenspace-oracle-all-three-join-noise-free/JOIN_ORACLE_RESULT_v1.json) · result
- [JOIN_POWER_RESULT_v1.json](../../../records/research/2026-10-16-exact-join-eigenspace-oracle-all-three-join-noise-free/JOIN_POWER_RESULT_v1.json) · result
- [diag_pj_1013_7103248.json](../../../records/research/2026-10-16-exact-join-eigenspace-oracle-all-three-join-noise-free/diag_pj_1013_7103248.json) · result
- [diag_pj_all_7103248.json](../../../records/research/2026-10-16-exact-join-eigenspace-oracle-all-three-join-noise-free/diag_pj_all_7103248.json) · result
- [diag_pr16_7103248.json](../../../records/research/2026-10-16-exact-join-eigenspace-oracle-all-three-join-noise-free/diag_pr16_7103248.json) · result
- [diag_pr32_7103248.json](../../../records/research/2026-10-16-exact-join-eigenspace-oracle-all-three-join-noise-free/diag_pr32_7103248.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-one-shot-residual-diagonal-pivot-refinement-adjusted-0-162) · [Copy provenance](../../../PROVENANCE.json)

