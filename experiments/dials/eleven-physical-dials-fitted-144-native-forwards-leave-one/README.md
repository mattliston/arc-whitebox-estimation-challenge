# Eleven physical dials fitted from 144 native forwards: leave-one-out gain is negative

2026-09-28 · experiment · rejected · measured

Twelve baselines and eleven sets of perturbations measured current-native physical dial responses. Fixed damping selected by grouped leave-one-network-out fitting failed all admission gates, so no candidate was exported.

Recorded measurement:

    144 forwards;selected damping1;predicted gain0.203670845%;leave-one-out gain-0.286933568%,4/12wins;required gains>=.25% and8/12wins.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/build.py](code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/build.py)
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

<details>
<summary>Additional programs and dependencies (35)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_X0.py](code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_X0.py)
- [code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_X1.py](code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_X1.py)
- [code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_X2.py](code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_X2.py)
- [code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_bscale.py](code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_bscale.py)
- [code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_c0.py](code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_c0.py)
- [code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_c1.py](code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_c1.py)
- [code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_k4a.py](code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_k4a.py)
- [code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_k4b.py](code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_k4b.py)
- [code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_k4c.py](code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_k4c.py)
- [code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_s_carry.py](code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_s_carry.py)
- [code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_s_new.py](code/codex_resume_20260926_v2/hill_climb_native_physical11_v1/perturb_s_new.py)
- [code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py)
- [code/codex_resume_20260926_v2/native_eval.py](code/codex_resume_20260926_v2/native_eval.py)
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
- [code/fable_engineer_v1/mps_card.py](code/fable_engineer_v1/mps_card.py)
- [code/fable_engineer_v1/mps_lock.py](code/fable_engineer_v1/mps_lock.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/carry/variant_bd.py](code/fable_physicist_v1/refit6b/carry/variant_bd.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_physicist_v1/refit6b/refit_mps_v3j2.py](code/fable_physicist_v1/refit6b/refit_mps_v3j2.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [HILL_CLIMB_before_hk_train8_closeout_v1.md#L9](../../flop-accounting/base-factor-cache-36-setup-bindings-combined-eight-network/HILL_CLIMB_before_hk_train8_closeout_v1.md#L9)

<details>
<summary>All result files and records (3)</summary>

- [HILL_CLIMB_before_hk_train8_closeout_v1.md](../../flop-accounting/base-factor-cache-36-setup-bindings-combined-eight-network/HILL_CLIMB_before_hk_train8_closeout_v1.md) · record
- [RESULT.md](RESULT.md) · record
- [protocol.json](protocol.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-eleven-physical-dials-fitted-144-native-forwards-leave-one) · [Copy provenance](../../../PROVENANCE.json)

