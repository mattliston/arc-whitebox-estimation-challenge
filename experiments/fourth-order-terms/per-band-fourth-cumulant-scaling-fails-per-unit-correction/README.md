# Per-band fourth-cumulant scaling fails; per-unit correction helps only with learned corrections off

2026-09-25T01:11Z · experiment · rejected · measured

The fourth-cumulant norm already matched truth reasonably well. Missing per-unit variation could not be supplied by scalar bands, while a one-product per-unit model conflicted with learned corrections.

Recorded measurement:

    - 2026-09-24 18:11 [physicist] LEVER 1 CLOSED (no-refit). Per-band k4 scalar, VALID16 learned ON vs B2 ep1: L1-7 x1.2/1.5/2.0 = +4.3/+30/+140%; L8-11 +7.3/+47/+230%; L12-15 +9.7/+59/+330% (0-1/16 better). Today's card k4 norm already matches the 32M truth within 1-10% (the old cap defect is gone); the missing piece is per-unit variation, which a band scalar cannot add. One-GEMM per-unit k4 learned ON VALID16 1.0083 (+0.83%, 4/16), learned OFF 0.9703: PARKED for a future refit bundle. j05-card learned-OFF VALID16 control 2.02997e-8 (carry/j05f_off_ctrl.json) sent to covhunter.
    - 2026-09-24 18:30 lead: PER-UNIT k4 CLOSED for no-refit use (physicist). Per-band k4 scalars all hurt (x1.2: +4.3/+7.3/+9.7%; larger is worse). Today's card k4 NORM already matches the 32M truth to 1-10%; what is missing is per-unit VARIATION (truth t4 spread .004-.016, card 0), which no band scalar can add. The one-GEMM per-unit k4 is +0.83% learned ON (4/16) / -3.0% learned OFF: parked for a refit bundle. j05-card learned-OFF control = 2.02997e-8 (sent to covhunter). The second j05 epoch's VALID16 is ~19:20. Star-plane torch VALID40 at 12 nets: B2 r32 +0.26%, r64 +0.45%, j05 r32 +0.14%. S17p (S16 + trim + j05 + prebind v2b, 9a4ccc1a): VALID40 ~19:00, staging ~19:25.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

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
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)

<details>
<summary>Additional programs and dependencies (17)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)
- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)
- [code/fable_covhunter_v1/covstep_s2.py](code/fable_covhunter_v1/covstep_s2.py)
- [code/fable_covhunter_v1/gn3_knobs.py](code/fable_covhunter_v1/gn3_knobs.py)
- [code/fable_physicist_v1/refit6b/carry/band_k4.py](code/fable_physicist_v1/refit6b/carry/band_k4.py)
- [code/fable_physicist_v1/refit6b/carry/kurt_plugin.py](code/fable_physicist_v1/refit6b/carry/kurt_plugin.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L537](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L537)
- [TEAM_LOG_2026-09-22_to_09-26.md#L538](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L538)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [j05f_off_ctrl.json](j05f_off_ctrl.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-per-band-fourth-cumulant-scaling-fails-per-unit-correction) · [Copy provenance](../../../PROVENANCE.json)

