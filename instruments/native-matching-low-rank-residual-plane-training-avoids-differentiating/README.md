# Native-matching low-rank residual-plane training avoids differentiating through QR and singular vectors

2026-09-25T01:57Z · tool · informational · measured

The hook uses the same seeded factorization as the native estimator and differentiates through the plane with a detached basis. Function-level comparisons match within float32 rounding.

Recorded measurement:

    - 2026-09-24 18:57 [engineer] TORCH LOW-RANK STAR HOOK for refits on the S18 physics: fable_engineer_v1/lrstar_torch_hook.py (lh.install(card, torch) after the card is built; replaces env _flms_star / _flms_star_diagonal). Same randomized factorization as native S18 (Omega = numpy default_rng(20260924) 1024x48, 1 power step, rank 32); basis on a DETACHED plane, the plane enters as Q_r(Q_r^T A): forward = native, gradients through A without QR/SVD backward. Certificate (lrstar_torch_cert.log, j05 torch card, S17 literals): hook vs base 37 +0.80% / 39 +0.11% (porter's exact r32 +1.34 / +0.48%, native S18 +1.45 / +0.08%); base reproduces the porter's torch base exactly; backward on 37: loss and all 27 grads finite (max 2.0e-4); forward ~11 s/net at 3 threads either way.
    - 2026-09-24 19:00 [engineer] TORCH HOOK CERTIFIED AGAINST NATIVE (function level): on all 14 star planes of a native S18-physics run of net 37 (lrstar_planes_r37.npz), the torch hook's _flms_star / _flms_star_diagonal vs the native fragment's on identical inputs (net 37 weights): max relative difference 1.6e-6 (star term) and 1.3e-6 (diagonal) = float32 rounding, even at singular-value gaps of 0.05% at the rank-32 cutoff (same Omega, same algorithm). Files: lrstar_native_outputs.py (native half), lrstar_factor_cert.py (torch half), lrstar_factor_cert.log. Now running the covariance diag+low-rank test (cov_lowrank_s18.py; S18 physics torch card, 8 VALID nets, r = 128/256/512).

## Programs

- [code/fable_engineer_v1/lrstar_torch_hook.py](code/fable_engineer_v1/lrstar_torch_hook.py)
- [code/fable_engineer_v1/lrstar_native_outputs.py](code/fable_engineer_v1/lrstar_native_outputs.py)
- [code/fable_engineer_v1/lrstar_factor_cert.py](code/fable_engineer_v1/lrstar_factor_cert.py)
- [code/fable_engineer_v1/cov_lowrank_s18.py](code/fable_engineer_v1/cov_lowrank_s18.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)

<details>
<summary>Additional programs and dependencies (22)</summary>

- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)
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
- [code/fable_covhunter_v1/covstep.py](code/fable_covhunter_v1/covstep.py)
- [code/fable_covhunter_v1/covstep_s2.py](code/fable_covhunter_v1/covstep_s2.py)
- [code/fable_covhunter_v1/gn3_knobs.py](code/fable_covhunter_v1/gn3_knobs.py)
- [code/fable_engineer_v1/lrstar2_k48_q1_r32.py](code/fable_engineer_v1/lrstar2_k48_q1_r32.py)
- [code/fable_physicist_v1/refit6b/carry/variant_bd.py](code/fable_physicist_v1/refit6b/carry/variant_bd.py)
- [code/fable_physicist_v1/refit6b/s6bcard_j05.py](code/fable_physicist_v1/refit6b/s6bcard_j05.py)
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L559](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L559)
- [TEAM_LOG_2026-09-22_to_09-26.md#L568](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L568)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [lrstar_torch_cert.log](lrstar_torch_cert.log) · log
- [lrstar_factor_cert.log](lrstar_factor_cert.log) · log

</details>

[Research graph](../../graph/index.html#node=2026-09-25-native-matching-low-rank-residual-plane-training-avoids-differentiating) · [Copy provenance](../../PROVENANCE.json)

