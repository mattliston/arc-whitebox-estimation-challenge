# Weighted scheduling variants: fresh-48 adjusted -0.007%/-0.028%, confirmation +0.009%

2026-10-14T02:04:19Z · experiment · rejected · measured

Two weighted scheduling-only variants (arms q and h, changing only when plane work is scheduled, same physics) were screened on 48 fresh networks; the h arm won by a hair and failed its strictly-negative 16-network confirmation.

Recorded measurement:

    fresh 48 net log: q -0.0070649733%, h -0.0282454381%; h confirmation 16: +0.009393771646338788% (SE 0.033514838412679826%, 8 lower); all 32: -0.009670940820245251% (SE 0.021308785954849432%, 18 lower), passed=false

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
<summary>Additional programs and dependencies (29)</summary>

- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)
- [code/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py](code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/n6y_cutsim_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/n6y_cutsim_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/pbase_cutsim_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/pbase_cutsim_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/torch_cutsim_s46_v11.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/torch_cutsim_s46_v11.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/torch_push_native_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/torch_push_native_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/torch_rowcuts_v15.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/cutsim/torch_rowcuts_v15.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/launch_weightfit_oos_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/launch_weightfit_oos_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/launch_weightfit_v5.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/launch_weightfit_v5.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/prepare_weightfit_v3.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/prepare_weightfit_v3.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/runtime_weightfit_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/runtime_weightfit_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/table_weightfit_v3.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/table_weightfit_v3.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/worker_weightfit_finite_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_newideas_v1/worker_weightfit_finite_v1.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L2446](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2446)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2466](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2466)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2470](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2470)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2517](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2517)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-weighted-scheduling-variants-fresh-48-adjusted-0-007-0) · [Copy provenance](../../../PROVENANCE.json)

