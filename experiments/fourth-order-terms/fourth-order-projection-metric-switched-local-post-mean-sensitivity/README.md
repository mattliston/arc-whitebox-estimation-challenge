# Fourth-order projection metric switched to local post-mean sensitivity: fresh half-arm -0.14%, OOS +0.11%, rejected

2026-10-14T08:46:09Z · experiment · rejected · measured

The K4L family removes the isotropic projection in a q^2-weighted raw fourth-cumulant metric; the arm replaced it with the actual local post-mean sensitivity metric to reduce scalar-level shifts after the age 3/4 addition. The half-strength arm won the 48-network fresh screen and lost the out-of-sample confirmation.

Recorded measurement:

    fresh 16 original grader: half -0.141305424% log, full -0.124706711% log, selected half; OOS 16: meanlog +0.112000720%, all 32 -0.014652352%, passed=False

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
<summary>Additional programs and dependencies (32)</summary>

- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)
- [code/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/cutsim/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/cutsim/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/cutsim/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/cutsim/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/cutsim/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/cutsim/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/cutsim/torch_push_native_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/cutsim/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/cutsim/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/cutsim/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/fragment_k4metric_full_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/fragment_k4metric_full_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/fragment_k4metric_half_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/fragment_k4metric_half_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/fragment_k4metric_zero_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/fragment_k4metric_zero_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/launch_phase_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/launch_phase_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/runtime_k4metric_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/runtime_k4metric_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/table_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/table_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/table_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/table_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/worker_k4metric_finite_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_20261013_v1/worker_k4metric_finite_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_confirm_20261014_v1/finite_confirmation_HELD_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_k4metric_confirm_20261014_v1/finite_confirmation_HELD_v2.py)
- [code/opus_reexam_20261001_v1/physicist/b6cols.py](code/opus_reexam_20261001_v1/physicist/b6cols.py)
- [code/opus_reexam_20261001_v1/physicist/b7cols.py](code/opus_reexam_20261001_v1/physicist/b7cols.py)
- [code/opus_reexam_20261001_v1/physicist/f4b.py](code/opus_reexam_20261001_v1/physicist/f4b.py)
- [code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py](code/opus_reexam_20261001_v1/physicist/ncar_bank_frozen_s2s.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py](code/opus_reexam_20261001_v1/scout_card/otr_m_sc4.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)
- [code/opus_reexam_20261001_v1/trainer/otr_engine.py](code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v2.py)
- [code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py](code/opus_reexam_20261001_v1/trainer/otr_generated_v3.py)
- [code/opus_reexam_20261001_v1/trainer/otr_pl.py](code/opus_reexam_20261001_v1/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2770](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2770)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2815](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2815)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2818](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2818)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2841](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2841)

<details>
<summary>All result files and records (3)</summary>

- [ASTRA_RECOVERY_RETIREMENT_v1.json](ASTRA_RECOVERY_RETIREMENT_v1.json) · record
- [CONFIRM_PROTOCOL_HELD_v3.json](CONFIRM_PROTOCOL_HELD_v3.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-fourth-order-projection-metric-switched-local-post-mean-sensitivity) · [Copy provenance](../../../PROVENANCE.json)

