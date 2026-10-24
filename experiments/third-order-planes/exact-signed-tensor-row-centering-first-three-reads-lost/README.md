# Exact signed-tensor row centering in the first three reads lost 0.482815% adjusted

2026-10-18T16:10:23Z · experiment · rejected · measured

A fixed exact-real row-centering transformation was applied to the first three signed-tensor reads. Both native networks worsened raw error and bill, so no extension was made.

Recorded measurement:

    Raw +.30127/+.61062%, bill +.02786/+.02564%, adjusted mean +.482815% → 1.453842199e-9; requests +21.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cp_affine_center_first3_v1/run_fixed2_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cp_affine_center_first3_v1/run_fixed2_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cp_affine_center_first3_v1/fragment_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cp_affine_center_first3_v1/fragment_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cp_affine_center_first3_v1/build_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cp_affine_center_first3_v1/build_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cp_affine_center_first3_v1/candidate_95c_cpcenter_first3_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cp_affine_center_first3_v1/candidate_95c_cpcenter_first3_HELD_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cp_affine_center_first3_v1/grade_fixed2_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cp_affine_center_first3_v1/grade_fixed2_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cp_affine_center_first3_v1/harness/boundary_client_fusion_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cp_affine_center_first3_v1/harness/boundary_client_fusion_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cp_affine_center_first3_v1/harness/boundary_run_fusion_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cp_affine_center_first3_v1/harness/boundary_run_fusion_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cp_affine_center_first3_v1/harness/run_guarded_fusion_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cp_affine_center_first3_v1/harness/run_guarded_fusion_v1.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/opus_reexam_20261001_v1/release/common.py](code/opus_reexam_20261001_v1/release/common.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/release/setup_clock.py](code/opus_reexam_20261001_v1/release/setup_clock.py)
- [code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5712](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5712)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5714](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5714)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [RESULT_v1.json](RESULT_v1.json) · result
- [COMPLETE_v1.json](COMPLETE_v1.json) · result
- [CONTROLLER_v1.json](CONTROLLER_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-exact-signed-tensor-row-centering-first-three-reads-lost) · [Copy provenance](../../../PROVENANCE.json)

