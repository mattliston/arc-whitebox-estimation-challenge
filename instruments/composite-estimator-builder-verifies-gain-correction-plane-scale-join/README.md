# Composite estimator builder verifies gain correction, plane scale and join schedule against native outputs

2026-09-23T05:52Z · tool · informational · measured

The builder composes the adopted mathematical changes and optional kernels. The research implementation matches the native estimator closely, and native kernel-only changes preserve all layer predictions.

Recorded measurement:

    2026-09-22 22:52 analyst: S2 torch card (fable_analyst_v1/s2card.py = model_bank_bfix gamma .1 partition + K31 row .85/.5 L1-14 + every3 push) parity vs native S2: r35 1.61200e-8 vs 1.61237e-8 (-0.02%), r100 1.57727e-8 vs 1.57638e-8 (+0.06%). Heads-only refit on S2 physics running (refit_s2.py, TRAIN[:32], VALID[:16], 2 epochs, lr 1e-3, 3 threads; nested checkpoints keep fwd+bwd at 6.9 GB): fable_analyst_v1/refit_s2_heads.log
    2026-09-22 22:53 porter: S3 chain ready: fable_porter_v1/build_S3.sh <out> [build_eng flags] (bank -> build_e2 v8 pinned -> physics K31 -> build_bfix_v2 0.1 -> build_eng -> every3). Verified r35: no-kernel and --holdonly == S2 exactly (flops 408,474,919,414, ops, MSE); --grouped --read5 MSE identical, flops -4.43% (r100 -4.29%).
    - 2026-09-22 23:12 [physicist] S2/S3 native review CLEAN (fable_physicist_v1/review/): every3 joins + lazy re-timing + row tiling correct; b-fix uses post-hook m, incoming radial, reaches trace_post/births; torch composite (K31 row L1-14 + b-fix .1 + every3) vs native S2 final MSE r35 +0.02%, r100 -0.04% (bank gap -0.03/-0.18%); S3 builds = S2 to 16 digits; joined blocks 4608-6144 rows = the max, so the tiled path is fully exercised.

## Programs

- [code/fable_analyst_v1/s2card.py](code/fable_analyst_v1/s2card.py)
- [code/fable_analyst_v1/refit_s2.py](code/fable_analyst_v1/refit_s2.py)
- [code/fable_porter_v1/build_S3.sh](code/fable_porter_v1/build_S3.sh)
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
<summary>Additional programs and dependencies (12)</summary>

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
- [code/fable_refit_v1/common.py](code/fable_refit_v1/common.py)
- [code/fable_refit_v1/model_bank_bfix.py](code/fable_refit_v1/model_bank_bfix.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L76](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L76)
- [TEAM_LOG_2026-09-22_to_09-26.md#L77](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L77)
- [TEAM_LOG_2026-09-22_to_09-26.md#L82](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L82)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [refit_s2_heads.log](refit_s2_heads.log) · log

</details>

[Research graph](../../graph/index.html#node=2026-09-23-composite-estimator-builder-verifies-gain-correction-plane-scale-join) · [Copy provenance](../../PROVENANCE.json)

