# Further refit after removing the residual-plane cross term gains only 0.331% and fails the 0.5% gate

2026-09-25T06:29Z · experiment · rejected · measured

The next learned-parameter epoch did not meet its predeclared improvement threshold. The previous parameters remained deployed.

Recorded measurement:

    - 2026-09-24 23:29 lead: loopR2 FAILS its bar: VALID16 1.642373e-8 vs valid_before 1.647834e-8 = -0.33% (bar -0.5%). With the dials converged at round 1, BOTH halves of the loop have saturated on this structure. loopR1 stays deployed (S22/S23).
      Re-pointed: covhunter's round-3 dials on the S23 structure with loopR1 params (dials_r3_loopR1.json if VALID16 <= -0.3%); porter's S24 = S23 + round-3 dials only if that file is written. No loopR3 by default: refits pay only after a structural or physics change (ablation audit survivors, a k3/k4 source). Physicist priority: the k3/k4 source page.
    2026-09-24 23:29 porter: S24 STANDS DOWN (lead rule): the physicist's loopR2 (s6b_loopR2_fckpt0_psloop_s1zero) VALID16 1.642373e-8 vs valid_before 1.647834e-8 = -0.33% > the 0.995x bar (1.639592e-8) -> FAIL (23:27; covhunter round3_chain.sh stopped on the same check). No S24 build; prebuild_S23struct.sh stays ready for a later loop round.
    - 2026-09-24 23:45 [physicist] loopR2 FAILS its pre-registered bar (VALID16 <= 0.995 x its own start). Card s6bcard_loop3 (jfitR1 dials + gn_linear_hook + E-67 fine_ckpt=False + pair_series loop form + s1=0 star), 1 epoch full+P from loopR1, CPU float64: VALID16 1.64783e-8 -> 1.64237e-8 = -0.331% pooled, 12/16 better (median -0.28%, range -1.03..+0.24%); bar 1.63960e-8 (-0.5%). Params kept, not the next base: s6b_loopR2_fckpt0_psloop_s1zero/params_epoch0.pt (sha256 b836044243b92373...). Reading: the dial<->refit loop is converging (round 1 -0.95%, round 2 -0.33% on VALID16); loopR1 stays the base. loopR3 only makes sense on a changed physics (the S23 Nystrom join), from loopR1, same bar.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_covhunter_v1/round3_chain.sh](code/fable_covhunter_v1/round3_chain.sh)
- [code/fable_porter_v1/prebuild_S23struct.sh](code/fable_porter_v1/prebuild_S23struct.sh)
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
<summary>Additional programs and dependencies (31)</summary>

- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
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
- [code/fable_covhunter_v1/gn_joint_fit.py](code/fable_covhunter_v1/gn_joint_fit.py)
- [code/fable_covhunter_v1/s6bcard_j03.py](code/fable_covhunter_v1/s6bcard_j03.py)
- [code/fable_covhunter_v1/s6bcard_jfitR1.py](code/fable_covhunter_v1/s6bcard_jfitR1.py)
- [code/fable_covhunter_v1/s6bcard_joint.py](code/fable_covhunter_v1/s6bcard_joint.py)
- [code/fable_covhunter_v1/s6bcard_joint2.py](code/fable_covhunter_v1/s6bcard_joint2.py)
- [code/fable_covhunter_v1/valid40_dials.py](code/fable_covhunter_v1/valid40_dials.py)
- [code/fable_engineer_v1/gn_linear_hook.py](code/fable_engineer_v1/gn_linear_hook.py)
- [code/fable_engineer_v1/join_nystrom_torch.py](code/fable_engineer_v1/join_nystrom_torch.py)
- [code/fable_engineer_v1/lrstar_torch_hook.py](code/fable_engineer_v1/lrstar_torch_hook.py)
- [code/fable_engineer_v1/pair_series_torch.py](code/fable_engineer_v1/pair_series_torch.py)
- [code/fable_physicist_v1/refit6b/carry/variant_bd.py](code/fable_physicist_v1/refit6b/carry/variant_bd.py)
- [code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [code/fable_physicist_v1/refit6b/refit_s6b.py](code/fable_physicist_v1/refit6b/refit_s6b.py)
- [code/fable_physicist_v1/refit6b/s6bcard_loop3.py](code/fable_physicist_v1/refit6b/s6bcard_loop3.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L758](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L758)
- [TEAM_LOG_2026-09-22_to_09-26.md#L759](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L759)
- [TEAM_LOG_2026-09-22_to_09-26.md#L760](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L760)
- [TEAM_LOG_2026-09-22_to_09-26.md#L762](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L762)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-further-refit-after-removing-residual-plane-cross-term-gains) · [Copy provenance](../../../PROVENANCE.json)

