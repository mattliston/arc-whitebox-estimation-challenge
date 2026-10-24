# Partial truth-plane correction outperforms full replacement: kink-row half strength improves raw error 8.7-9.6%

2026-09-23T04:18Z · finding · informational · measured

The covariance consumer overshoots when given the full true kink-row plane correction. The measured rule became to evaluate every correction at quarter, half and full strength.

Recorded measurement:

    2026-09-22 21:18 covhunter: PARADOX RESOLVED (torch, learned ON, no refit; truth fed to the pair map's COVARIANCE output only, input = card + eps(truth-card)): K21 off-diag KINK rows @L6-14 eps .25/.5/1 = -7.7/-9.6/+4.1% (r35), -7.3/-8.7/+6.2% (r100) -> overshoot, optimum eps ~.45 (quadratic fit exact: linear price -42%, curvature .45-.48); K31 @L2-14 eps .5/1 = -13.2/-13.0% (r35), -14.2/-15.5% (r100); all pair inputs @L6-14 eps .5/1 = -25.6/-23.8% (r35), -27.0/-20.6% (r100). One-step injection (truth C at l-1) is ~100% pair-map inputs by adjoint price at L7+: kink-row K21 read first, K31 second, K22/k4 small, mean 0, edge helps, recentering 0. clpair_v1.jsonl, attrib_v1.jsonl, knob_price_rank*.json.
    - 2026-09-22 21:29 lead: OVERSHOOT LAW recorded in TEAM_BRIEF (covhunter): plane fixes pay at partial strength (kink K21 eps .5 -9%, all pair inputs eps .5 -26%); grade every fix at eps .25/.5/1.
    2026-09-22 23:42 covhunter: DONE (fable_covhunter_v1/RESULT.md). Zero-bill covariance-channel knob set on top of the K31 row factor (.85/.5 L1-14), Gauss-Newton fit on 18 TRAIN nets: kink-row K21 += -0.094 k3 C/v, non-kink K21 x0.972, T31 x0.788 (all three in the covariance output only, L2-14), message edge x1.164, recentering x0.92 -> VALID40 torch vs row factor 0.9834 (CI .9789-.9883) 32/40 wins, worst +2.3%; vs bank 0.9694 (38/40). Native form = closed-form delta on the pair map's covariance output (certified vs second call to 1e-4, cert_delta.py), ~30 n^2 ops/layer, no second pair call. Learned-OFF check: truth K21 kink rows still overshoots at eps 1 (+10%/+6%), K31 truth -20%: the overshoot is physics-level, not co-adaptation.

## Programs

- [cert_delta.py](cert_delta.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/build_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/core_v2.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/reference_numpy.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/fnp/reference_numpy.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native_v2.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/birth_direction_fit_v1/native_v2.py)
- [../../../experiments/third-order-planes/aligning-pair-precision-covariance-arithmetic-still-fails-fidelity-maximum/code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](../../../experiments/third-order-planes/aligning-pair-precision-covariance-arithmetic-still-fails-fidelity-maximum/code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py](../../../experiments/birth-terms/four-fitted-birth-directions-pass-native40-independent14-adjusted-ratios/code/unused_materializations_v1/estimator.py)
- [../../../experiments/learned-correction/covariance-edge-refit-passes-native40-but-fails-independent-public8/code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](../../../experiments/learned-correction/covariance-edge-refit-passes-native40-but-fails-independent-public8/code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [../../../submissions/331824/estimator.py](../../../submissions/331824/estimator.py)

<details>
<summary>Additional programs and dependencies (12)</summary>

- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [covstep.py](covstep.py)
- [gn2_knobs.py](gn2_knobs.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/common.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/model_bank_research.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L46](../../TEAM_LOG_2026-09-22_to_09-26.md#L46)
- [TEAM_LOG_2026-09-22_to_09-26.md#L49](../../TEAM_LOG_2026-09-22_to_09-26.md#L49)
- [TEAM_LOG_2026-09-22_to_09-26.md#L94](../../TEAM_LOG_2026-09-22_to_09-26.md#L94)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [clpair_v1.jsonl](clpair_v1.jsonl) · result
- [attrib_v1.jsonl](attrib_v1.jsonl) · result
- [RESULT.md](RESULT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-partial-truth-plane-correction-outperforms-full-replacement-kink-row) · [Copy provenance](../../../PROVENANCE.json)

