# Auxiliary intermediate-mean supervision driver matches the final-only control at zero weight

2026-09-23T14:39Z · tool · informational · measured

The refit objective can add normalized intermediate-layer mean losses. Zero auxiliary weight and training-order checks establish a matched final-only control.

Recorded measurement:

    2026-09-23 07:39 covhunter: AUX-MEAN REFIT ready (RESULT.md sec 20; fable_covhunter_v1/refit_s6b_aux.py = analyst refit_s6b.py + --aux_w/--aux_layers/--aux_calib: loss = MSE_15 + w sum_l MSE_l s15/s_l vs truth688 all-layer 1e9 means). Smoke (TRAIN640 nets 0,1): w=0 loss == original bit-exact; w=.3 grads finite (norms .19-42.6), one Adam step moves heads 1e-4. S6b init per-layer MSE on VALID 37/39/42/48: L10 55-64% of the final, L12 68-89%, L14 85-97% -> big unused supervision signal. Full run waits for CPU.
    2026-09-23 07:49 covhunter: AUX RUN LAUNCHED (lead GO) fable_covhunter_v1/aux_w03_A (log aux_w03_A.log), 3 threads nice 10. A-matched: TRAIN160, full, lr 1e-4, warmup 3, clip 1, batch 4, 2 epochs, VALID16, init S6b params_epoch2; + aux_w .3 on L10-14 (scales from TRAIN160[:8]). Same net order as A by construction (same list, seed, one permutation per epoch; truth688[15] == public_g22 truthpost on all 200 nets) -> A is the exact w=0 control. Certificates in-log: valid_before must equal A's 1.77402e-8, update-1 batch_final A's 1.77516e-8. Driver v2 (evaluate grades FINAL MSE only + L10-14 beside it; v1 kept). CORRECTION to my 07:39 line: that VALID4 table carried one smoke Adam step on A/cstar/edges (only heads restored); true init on r37 differs by +0.27% at L15, <0.1% at L10-14; ratios unchanged; exact init table follows from this run's valid_before.
    2026-09-23 08:19 covhunter: aux run certificates vs A = rounding-level, not bit-exact: valid_before 1.773985e-8 vs A 1.774024e-8 (-2.2e-5 rel; per-net random sign, sd 4.8e-5, max 1.2e-4; net 37 at 2 threads matches A to 2e-9, at 3 threads -8e-5 -> BLAS thread order; the MSE is a 1e-4 residual on ~0.3 values); update-1 batch_final 1.775109e-8 vs A 1.775160e-8 (same 4 nets). So A's epoch-0 1.77082e-8 is the w=0 control to ~1e-5. True-init VALID16: L10-14 = 54/62/72/79/88% of the final MSE (RESULT sec 20 table). ~290 s/update at 3 threads -> epoch-0 VALID16 ~11:20. Follow-up driver READY (not run): fable_covhunter_v1/refit6b_aux.py = physicist refit6b/refit_s6b.py + the aux block (same trunk set without gain, warmup-trunk 5), so B2 is its exact w=0 control; needs a 3-min end-to-end test before launch.

## Programs

- [code/fable_covhunter_v1/refit_s6b_aux.py](code/fable_covhunter_v1/refit_s6b_aux.py)
- [code/fable_covhunter_v1/refit6b_aux.py](code/fable_covhunter_v1/refit6b_aux.py)
- [code/fable_physicist_v1/refit6b/refit_s6b.py](code/fable_physicist_v1/refit6b/refit_s6b.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)

<details>
<summary>Additional programs and dependencies (19)</summary>

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
- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/s6bcard.py](code/fable_analyst_v1/s6bcard.py)
- [code/fable_analyst_v1/s6bknobs.py](code/fable_analyst_v1/s6bknobs.py)
- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)
- [code/fable_covhunter_v1/covstep_s2.py](code/fable_covhunter_v1/covstep_s2.py)
- [code/fable_covhunter_v1/gn3_knobs.py](code/fable_covhunter_v1/gn3_knobs.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L253](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L253)
- [TEAM_LOG_2026-09-22_to_09-26.md#L257](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L257)
- [TEAM_LOG_2026-09-22_to_09-26.md#L267](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L267)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [aux_w03_A.log](aux_w03_A.log) · log

</details>

[Research graph](../../graph/index.html#node=2026-09-23-auxiliary-intermediate-mean-supervision-driver-matches-final-only-control) · [Copy provenance](../../PROVENANCE.json)

