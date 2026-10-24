# Native three-dial joint fit: predicted gain 0.14259635011146798% misses the 0.5% gate

2026-09-30 · experiment · rejected · measured

Six fixed ±2% variants measured trace scale, newborn coefficient and three-one coefficient on four fitting networks; a bounded ridge fit produced no admitted endpoint.

Recorded measurement:

    Predicted FIT gain0.14259635011146798%<.5%;24 native perturbation forwards;ridge.1 diagonal;box x∈[-1,1].

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_combo_native_joint3_v1/build.py](code/codex_resume_20260926_v2/hill_climb_combo_native_joint3_v1/build.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py](code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)

<details>
<summary>Additional programs and dependencies (28)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/codex_resume_20260926_v2/hill_climb_combo_native_joint3_v1/bscale_minus.py](code/codex_resume_20260926_v2/hill_climb_combo_native_joint3_v1/bscale_minus.py)
- [code/codex_resume_20260926_v2/hill_climb_combo_native_joint3_v1/bscale_plus.py](code/codex_resume_20260926_v2/hill_climb_combo_native_joint3_v1/bscale_plus.py)
- [code/codex_resume_20260926_v2/hill_climb_combo_native_joint3_v1/c1_minus.py](code/codex_resume_20260926_v2/hill_climb_combo_native_joint3_v1/c1_minus.py)
- [code/codex_resume_20260926_v2/hill_climb_combo_native_joint3_v1/c1_plus.py](code/codex_resume_20260926_v2/hill_climb_combo_native_joint3_v1/c1_plus.py)
- [code/codex_resume_20260926_v2/hill_climb_combo_native_joint3_v1/fit.py](code/codex_resume_20260926_v2/hill_climb_combo_native_joint3_v1/fit.py)
- [code/codex_resume_20260926_v2/hill_climb_combo_native_joint3_v1/newborn_minus.py](code/codex_resume_20260926_v2/hill_climb_combo_native_joint3_v1/newborn_minus.py)
- [code/codex_resume_20260926_v2/hill_climb_combo_native_joint3_v1/newborn_plus.py](code/codex_resume_20260926_v2/hill_climb_combo_native_joint3_v1/newborn_plus.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_pair_read7_scan_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_s34_pair_read7_scan_v1/candidate.py)
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)
- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)
- [code/fable_covhunter_v1/covstep_s2.py](code/fable_covhunter_v1/covstep_s2.py)
- [code/fable_covhunter_v1/gn3_knobs.py](code/fable_covhunter_v1/gn3_knobs.py)
- [code/fable_covhunter_v1/gn_joint_fit.py](code/fable_covhunter_v1/gn_joint_fit.py)
- [code/fable_covhunter_v1/s6bcard_j03.py](code/fable_covhunter_v1/s6bcard_j03.py)
- [code/fable_covhunter_v1/s6bcard_joint.py](code/fable_covhunter_v1/s6bcard_joint.py)
- [code/fable_covhunter_v1/s6bcard_joint2.py](code/fable_covhunter_v1/s6bcard_joint2.py)
- [code/fable_engineer_v1/gn_linear_hook.py](code/fable_engineer_v1/gn_linear_hook.py)
- [code/fable_engineer_v1/join_nystrom_torch.py](code/fable_engineer_v1/join_nystrom_torch.py)
- [code/fable_engineer_v1/lrstar_torch_hook.py](code/fable_engineer_v1/lrstar_torch_hook.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/carry/variant_bd.py](code/fable_physicist_v1/refit6b/carry/variant_bd.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z#L50](../../setup-budget/all-setup-keys-fused-eagerly-degraded-setup-drops-bindings/HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z#L50)

<details>
<summary>All result files and records (3)</summary>

- [HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z](../../setup-budget/all-setup-keys-fused-eagerly-degraded-setup-drops-bindings/HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z) · record
- [build.json](build.json) · result
- [protocol.json](protocol.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-native-three-dial-joint-fit-predicted-gain-0-14259635011146798) · [Copy provenance](../../../PROVENANCE.json)

