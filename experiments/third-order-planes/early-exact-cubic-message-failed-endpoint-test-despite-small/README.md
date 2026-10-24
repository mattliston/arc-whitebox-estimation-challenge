# Early exact cubic message failed the endpoint test despite a small arithmetic ceiling

2026-10-18T13:08:40Z · experiment · rejected · measured

The exact cubic weight message was restored only at layers 1–3 on the unchanged compact estimator. The first network lost adjusted score and the second was stopped by the memory guard, so the causal endpoint test failed.

Recorded measurement:

    First rank 35: adjusted +.081%; estimated bill ceiling +.032%; second row stopped at 7.5 GB.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/w3_early_restore_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/analyze_v1.py](code/analyze_v1.py)
- [code/candidate_95c_w3early_HELD_v1.py](code/candidate_95c_w3early_HELD_v1.py)
- [code/fragment_v1.py](code/fragment_v1.py)
- [code/analyze_partial_v1.py](code/analyze_partial_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5682](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5682)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5684](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5684)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [PARTIAL_RESULT_v1.json](PARTIAL_RESULT_v1.json) · result
- [GUARD_v1.json](GUARD_v1.json) · result
- [PLAN_v1.json](PLAN_v1.json) · result
- [GRADE_STORE_v1.json](GRADE_STORE_v1.json) · result
- [manifest.json](manifest.json) · result
- [server.json](server.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-early-exact-cubic-message-failed-endpoint-test-despite-small) · [Copy provenance](../../../PROVENANCE.json)

