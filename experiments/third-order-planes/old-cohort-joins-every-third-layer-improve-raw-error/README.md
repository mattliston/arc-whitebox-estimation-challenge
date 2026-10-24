# Old-cohort joins every third layer improve raw error 2.2% on twelve training networks at neutral work

2026-09-23T04:44Z · experiment · adopted · measured

The bank joins one old cohort per layer into the rank-384 frame and every join re-truncates all old content; joining less often keeps cohorts exact longer and is more accurate. The every3 schedule (three cohorts at layers 7, 10, 13) was ported natively with the lazy factor rotation re-timed (the builder reproduces the bank bit-exactly for the every-layer schedule) and shipped in S2; every3b (8/11/14) is more accurate but +2-3.5% bill because seven young cohorts are dear.

Recorded measurement:

    torch final-MSE ratio, 4 nets (35/100/37/39) | 12 TRAIN | native bill r35 cold / r100 warm: even (2 cohorts @ L6,8,10,12,14) +2.0/-2.7/-1.9/-1.0 (mean -0.9%) | -2.0% (12/12) | -0.78% / -0.44%; every3 (L7,10,13) +0.4/-4.2/-2.1/-1.5 (mean -1.9%) | -2.2% (10/12) | -0.76% / 0.0%; every3b (L8,11,14) mean -1.4% | -2.9% (10/12) | +2.1% / +3.5%; every3 at rank 352 / 320 eats the gain; torch VALID40 raw pooled every3 0.9789 (37/40), even 0.9859 (33/40)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_frame_v1/join_schedule_test.py](code/fable_frame_v1/join_schedule_test.py)
- [code/fable_join_v1/build_join_schedule_v1_fired_S2.py](code/fable_join_v1/build_join_schedule_v1_fired_S2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)

<details>
<summary>Additional programs and dependencies (11)</summary>

- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L42](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L42)
- [TEAM_LOG_2026-09-22_to_09-26.md#L54](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L54)
- [NOTES.md#L236](../removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L236)

<details>
<summary>All result files and records (9)</summary>

- [join_schedule_test_train12.log](join_schedule_test_train12.log) · log
- [join_schedule_test_every3_train12.log](join_schedule_test_every3_train12.log) · log
- [join_schedule_test_valid40.log](join_schedule_test_valid40.log) · log
- [join_schedule_test_rank.log](join_schedule_test_rank.log) · log
- [score_every3_r35_r100.log](score_every3_r35_r100.log) · log
- [score_even_r35_r100.log](score_even_r35_r100.log) · log
- [score_every3b_r35_r100.log](score_every3b_r35_r100.log) · log
- [NOTES.md](../removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record
- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · record

</details>

## Related work

- shipped in: [Submission #331947: rowwise (3,1) correction, gain correction, and joins every third layer](../../../submissions/331947/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-old-cohort-joins-every-third-layer-improve-raw-error) · [Copy provenance](../../../PROVENANCE.json)

