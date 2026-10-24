# Four learned projections via Winograd kernels: 405,837,584 FLOPs saved, displacement RMS 0.0326 exceeds the 0.001 cap

2026-10-14T03:53:20Z · experiment · rejected · measured

Replacing only four learned linear projections by the existing Winograd kernels (with per-predict owned right-factor transforms) matched its forecast bill credit, but the all-layer displacement cap of 0.001 RMS, which every exact compute change had to satisfy, failed by a factor of 30. No retuning was allowed.

Recorded measurement:

    bill credit 405,837,584 FLOPs matched the source estimate; all-16-layer RMS displacement ratio max 0.032614579, final 0.023518435 (cap 0.001)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_learned_projection_v1/fragment_learned_projection_UNVALIDATED_v1.py](code/lead/runs/astra_learned_projection_v1/fragment_learned_projection_UNVALIDATED_v1.py)
- [code/lead/runs/astra_learned_projection_v1/finite_alp_qualification_v1.py](code/lead/runs/astra_learned_projection_v1/finite_alp_qualification_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py](code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py)
- [code/lead/runs/astra_learned_projection_v1/native_alp_qualification_judge_v1.py](code/lead/runs/astra_learned_projection_v1/native_alp_qualification_judge_v1.py)
- [code/lead/runs/astra_learned_projection_v1/owned_native_runner_v1.py](code/lead/runs/astra_learned_projection_v1/owned_native_runner_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2597](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2597)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2601](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2601)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2609](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2609)

<details>
<summary>All result files and records (2)</summary>

- [CLOSED_SOURCE_FOLLOWUPS_v1.json](CLOSED_SOURCE_FOLLOWUPS_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-four-learned-projections-via-winograd-kernels-405-837-584) · [Copy provenance](../../../PROVENANCE.json)

