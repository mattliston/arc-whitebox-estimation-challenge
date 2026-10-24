# Skew error has little correlation with either gain or bulk components

2026-10-01T18:00Z · finding · informational · measured

Splitting the card's third-cumulant error against Monte Carlo truth on network 35 showed it is uncorrelated with both the collective gain part and the bulk part (R^2 at most 0.10). The card's b carrier holds only 20-30% of the gain part, the bank and star hold the rest (correlation 0.995), and the variance error is not a gain-variance amplitude error either (R^2 on (mu/sigma)^2 at most 0.1). No per-unit carrier transports the network-specific kurtosis or skew (R^2 <= 0.16 for the true kurtosis pushed through variance-share, coincident-index or elasticity-dressed operators).

Recorded measurement:

    R^2 <= 0.10 (skew error vs gain and bulk parts); b carrier 20-30% of the gain part, bank + star the rest (corr 0.995); variance error vs (mu/sigma)^2 R^2 <= 0.1; per-unit transport of true kurtosis R^2 <= 0.16, skew about 0

## Programs

- [e3_card_vs_truth_split.py](e3_card_vs_truth_split.py)
- [e2_dump_card_cumulants.py](e2_dump_card_cumulants.py)
- [e4_kurt_transport_test.py](e4_kurt_transport_test.py)
- [../../../submissions/331824/estimator.py](../../../submissions/331824/estimator.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [../../../instruments/composite-estimator-builder-verifies-gain-correction-plane-scale-join/code/fable_analyst_v1/s2card.py](../../../instruments/composite-estimator-builder-verifies-gain-correction-plane-scale-join/code/fable_analyst_v1/s2card.py)
- [../2026-09-23-training-normalization-constants-caused-70-loss-spike-network-weight/s6bcard.py](../2026-09-23-training-normalization-constants-caused-70-loss-spike-network-weight/s6bcard.py)
- [../2026-09-23-partial-truth-plane-correction-outperforms-full-replacement-kink-row/covstep.py](../2026-09-23-partial-truth-plane-correction-outperforms-full-replacement-kink-row/covstep.py)
- [../../../experiments/dials/extra-kink-row-carrier-scales-add-no-useful-gain/code/fable_covhunter_v1/covstep_s2.py](../../../experiments/dials/extra-kink-row-carrier-scales-add-no-useful-gain/code/fable_covhunter_v1/covstep_s2.py)
- [../../../experiments/late-layer-exactness/local-final-layer-error-prediction-null-held-out-r/code/fable_covhunter_v1/gn3_knobs.py](../../../experiments/late-layer-exactness/local-final-layer-error-prediction-null-held-out-r/code/fable_covhunter_v1/gn3_knobs.py)

<details>
<summary>Additional programs and dependencies (3)</summary>

- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/common.py)
- [../2026-09-23-rank-one-covariancemean-gain-mixture-explains-6368-2-1/model_bank_bfix.py](../2026-09-23-rank-one-covariancemean-gain-mixture-explains-6368-2-1/model_bank_bfix.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/model_bank_research.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/model_bank_research.py)

</details>

## Evidence

- [SESSION_NOTES_2026-10-01_to_10-03.md#L39](../../SESSION_NOTES_2026-10-01_to_10-03.md#L39)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L50](../../SESSION_NOTES_2026-10-01_to_10-03.md#L50)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L56](../../SESSION_NOTES_2026-10-01_to_10-03.md#L56)

<details>
<summary>All result files and records (1)</summary>

- [SESSION_NOTES_2026-10-01_to_10-03.md](../../SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-skew-error-has-little-correlation-either-gain-or-bulk) · [Copy provenance](../../../PROVENANCE.json)

