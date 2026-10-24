# Third-cumulant carrier variants: best fresh-16 adjusted -0.019%, confirmation +0.006%

2026-10-14T00:54:14Z · experiment · rejected · measured

The hunter supervisor's experiment 151 screened five carrier-form arms on 16 fresh networks with a strictly-negative per-dial rule; only kp was negative and was sent to a 16-network confirmation, where it came out positive, so the pooled negative did not count. The NFS family (sp, sm) was positive on all 48 and selected nothing.

Recorded measurement:

    fresh 16 mean log %: sp +0.1179, sm +0.0375, dp +0.1946, dm +0.1985, kp -0.01943 (SE 0.0626, 7/16 lower); kp confirmation 16: +0.006320060%, all 32 -0.006555504% (strict negative sign FAIL); NFS all 48: sp +0.117899585%, sm +0.037450909%, selected none

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)

<details>
<summary>Additional programs and dependencies (30)</summary>

- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)
- [code/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/t151_kp_family_astra_v1/prepare_kp_confirmation_v1.py](code/opus_reexam_20261001_v1/lead/runs/t151_kp_family_astra_v1/prepare_kp_confirmation_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/t151_kp_family_astra_v1/table_kp_family_v1.py](code/opus_reexam_20261001_v1/lead/runs/t151_kp_family_astra_v1/table_kp_family_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/t151_nfd_family_astra_v1/prepare_nfd_confirmation_v1.py](code/opus_reexam_20261001_v1/lead/runs/t151_nfd_family_astra_v1/prepare_nfd_confirmation_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/t151_nfd_family_astra_v1/table_nfd_family_v1.py](code/opus_reexam_20261001_v1/lead/runs/t151_nfd_family_astra_v1/table_nfd_family_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/t151_nfs_family_astra_v1/prepare_nfs_confirmation_v1.py](code/opus_reexam_20261001_v1/lead/runs/t151_nfs_family_astra_v1/prepare_nfs_confirmation_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/t151_nfs_family_astra_v1/table_nfs_family_v1.py](code/opus_reexam_20261001_v1/lead/runs/t151_nfs_family_astra_v1/table_nfs_family_v1.py)
- [code/opus_reexam_20261001_v1/lead/tools/t151_supervisor_astra_v10.py](code/opus_reexam_20261001_v1/lead/tools/t151_supervisor_astra_v10.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/otr_cut_runtime_ph153.py](code/opus_reexam_20261001_v1/physicist/otr_cut_runtime_ph153.py)
- [code/opus_reexam_20261001_v1/physicist/t151_make.py](code/opus_reexam_20261001_v1/physicist/t151_make.py)
- [code/opus_reexam_20261001_v1/physicist/t151_table.py](code/opus_reexam_20261001_v1/physicist/t151_table.py)
- [code/opus_reexam_20261001_v1/physicist/worker_ph153.py](code/opus_reexam_20261001_v1/physicist/worker_ph153.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_cut_runtime_sc4.py](code/opus_reexam_20261001_v1/scout_card/otr_cut_runtime_sc4.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py](code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py)
- [code/opus_reexam_20261001_v1/scout_card/worker_sc8.py](code/opus_reexam_20261001_v1/scout_card/worker_sc8.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2440](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2440)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2444](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2444)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2457](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2457)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2461](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2461)

<details>
<summary>All result files and records (5)</summary>

- [screen151.json](screen151.json) · result
- [complete151.json](complete151.json) · result
- [FAMILY_SCREEN_DECISION_v1.json](FAMILY_SCREEN_DECISION_v1.json) · result
- [FAMILY_SCREEN_DECISION_v1-8bc1d72c.json](FAMILY_SCREEN_DECISION_v1-8bc1d72c.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-third-cumulant-carrier-variants-best-fresh-16-adjusted-0) · [Copy provenance](../../../PROVENANCE.json)

