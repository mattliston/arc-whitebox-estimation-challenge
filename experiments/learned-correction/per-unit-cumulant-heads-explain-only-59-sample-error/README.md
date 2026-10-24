# Per-unit cumulant heads explain only 5–9% of in-sample error variance and fail transfer

2026-09-23T00:16Z · experiment · rejected · measured

Zero-initialised 64-vector heads on the causal GNN state giving multiplicative corrections to the pair map's per-unit k3, k4 (and K22, variance) were trained jointly or heads-only on the torch card, with a native hook builder whose zero-head parity is exact. A probe showed the card's per-unit k3/k4 errors are not predictable from the GNN state (ridge R^2 0.05-0.09 in-sample, negative on transfer; a tanh-MLP overfits and transfers negatively), joint lines jolted the closure +16%, and the heads-only control gained nothing, so the line was killed.

Recorded measurement:

    head_info_probe (r35 fit / r100 transfer): e3 rms err 0.145-0.194, ridge in-sample 0.047-0.082, transfer -0.069..-0.19; e4 rms 0.181-0.223, ridge 0.052-0.073, transfer -0.062..-0.133; heads-only G0 VALID40 epoch 0 1.9440e-8 vs 1.9415e-8 before; zero-head native parity r35 1.648017257382719e-08 util .18831

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_refit_v1/refit_cumhead.py](code/fable_refit_v1/refit_cumhead.py)
- [code/fable_refit_v1/build_cumhead_variant.py](code/fable_refit_v1/build_cumhead_variant.py)
- [code/fable_refit_v1/head_info_probe.py](code/fable_refit_v1/head_info_probe.py)
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
<summary>Additional programs and dependencies (13)</summary>

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
- [code/fable_refit_v1/export_params.py](code/fable_refit_v1/export_params.py)
- [code/fable_refit_v1/model_bank_research.py](code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [NOTES.md#L140](../../third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L140)
- [NOTES.md#L125](../../third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L125)

<details>
<summary>All result files and records (6)</summary>

- [head_info_probe.log](head_info_probe.log) · log
- [cumhead_headsonly_lr1e3_w5.log](cumhead_headsonly_lr1e3_w5.log) · log
- [cumhead_joint_lr1e4_w20.log](cumhead_joint_lr1e4_w20.log) · log
- [cumhead_v_joint_lr1e4_w20.log](cumhead_v_joint_lr1e4_w20.log) · log
- [score_cumhead_zero_r35.log](score_cumhead_zero_r35.log) · log
- [NOTES.md](../../third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-per-unit-cumulant-heads-explain-only-59-sample-error) · [Copy provenance](../../../PROVENANCE.json)

