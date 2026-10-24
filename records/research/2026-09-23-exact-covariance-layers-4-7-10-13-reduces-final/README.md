# Exact covariance at layers 4, 7, 10, and 13 reduces final-error ratio to 0.765–0.785

2026-09-23T03:31Z · finding · informational · measured

The exact-state oracle prices the carried covariance error at about -22%: truth post covariance at four layers gives x0.765/0.785, the part orthogonal to the card mean carries all of it while the parallel part slightly hurts, and the off-diagonal error of P dC P (0.44-0.86% of the norm, real beyond the 32M noise from layer 7) is broad (top-64 SVD holds 60-79%) with no scalar multiple of C, C^2, C o C or C^3 explaining more than 1%.

Recorded measurement:

    C@13 .904/.862 (off-diagonal only .927/.850, diagonal only .996/1.013); C@10 .918/.792; C@7 .868/.947; C@4 .960/.994; C@4+7+10+13 .765/.785; Cperp@4+7+10+13 .740; Cperp@13 .888/.844; Cpar 1.015-1.019; m@13 .517/.568; mC@13 .403/.424; P dC P off-diag rel error .44% (L4) .55% (L7) .69% (L10) .86% (L13), 32M noise .31/.24/.20/.18%

## Programs

- [cov_oracle_closedloop.py](cov_oracle_closedloop.py)
- [dcperp_structure.py](dcperp_structure.py)
- [../2026-09-23-one-step-truth-covariance-supply-removes-4253-next-layer/mc_second_order_mps.py](../2026-09-23-one-step-truth-covariance-supply-removes-4253-next-layer/mc_second_order_mps.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/reference_numpy.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/reference_numpy.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native_v2.py)
- [../../../experiments/third-order-planes/aligning-pair-precision-covariance-arithmetic-still-fails-fidelity-maximum/code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](../../../experiments/third-order-planes/aligning-pair-precision-covariance-arithmetic-still-fails-fidelity-maximum/code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py)

<details>
<summary>Additional programs and dependencies (12)</summary>

- [../../../experiments/learned-correction/covariance-edge-refit-passes-native40-but-fails-independent-public8/code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](../../../experiments/learned-correction/covariance-edge-refit-passes-native40-but-fails-independent-public8/code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [../../../submissions/331824/estimator.py](../../../submissions/331824/estimator.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/common.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/model_bank_research.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [NOTES.md#L225](../../../experiments/third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L225)
- [NOTES.md#L229](../../../experiments/third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L229)

<details>
<summary>All result files and records (4)</summary>

- [cov_oracle_closedloop.log](cov_oracle_closedloop.log) · log
- [cov_oracle_perp.log](cov_oracle_perp.log) · log
- [dcperp_structure_r35.log](dcperp_structure_r35.log) · log
- [NOTES.md](../../../experiments/third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-exact-covariance-layers-4-7-10-13-reduces-final) · [Copy provenance](../../../PROVENANCE.json)

