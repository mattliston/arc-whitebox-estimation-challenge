# Norm-coupling contraction explains 36-46% of the per-unit fourth-cumulant residual

2026-10-02T08:45Z · finding · informational · measured

Monte Carlo on network 35 (2 x 1M samples per unit, then 16M) shows that Q_i = sum_k kappa4(z_i, z_i, z_k, z_k) (how a unit's variance moves with the layer's squared norm, a matrix contraction) explains 36-46% of the card's per-unit kurtosis residual at layers 5-15 with slope 5.0-5.7/n where the chaos expansion for Gaussian weights predicts 6/n; it is not a function of any free per-unit quantity (those explain 13-18%) and the hub-local terms are contained in it. About 40% of what the card's per-unit kurtosis misses at every depth is a matrix-level (second-chaos) object; its one-layer source from the third cumulant explains only 0.2-10% of it.

Recorded measurement:

    corr of unit-specific parts of k4 and Q 0.60 / 0.67 / 0.63 / 0.58 (layers 5 / 9 / 13 / 15); slope x n 5.0 / 5.7 / 5.1 / 5.0 (predicted 6); residual explained by isotropic + v + Q 0.39 / 0.46 / 0.40 / 0.36; K4L hub terms 0.14-0.31; free per-unit forms 0.13-0.18; T19c: Q at 6/n with nothing fitted 39 / 42 / 39 / 36 / 28% (layers 5 / 9 / 11 / 13 / 15); hub-local + Q 40 / 43 / 39 / 29%

## Programs

- [e31_second_chaos_mc.py](e31_second_chaos_mc.py)
- [e31_analyze.py](e31_analyze.py)
- [e31b_alpha_forms.py](e31b_alpha_forms.py)
- [../../../experiments/fourth-order-terms/carried-third-fourth-conversion-accurate-early-but-worsens-final/code/e31c_carrier_vs_truth.py](../../../experiments/fourth-order-terms/carried-third-fourth-conversion-accurate-early-but-worsens-final/code/e31c_carrier_vs_truth.py)
- [t19_second_chaos_mc.py](t19_second_chaos_mc.py)
- [../../../submissions/331824/estimator.py](../../../submissions/331824/estimator.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../experiments/learned-correction/eight-updates-detached-star-bases-fail-40-network-component/code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py](../../../experiments/learned-correction/eight-updates-detached-star-bases-fail-40-network-component/code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py)
- [../2026-09-30-detaching-star-basis-restores-historical-training-gradient-without-changing/train_segment.py](../2026-09-30-detaching-star-basis-restores-historical-training-gradient-without-changing/train_segment.py)

<details>
<summary>Additional programs and dependencies (17)</summary>

- [../../../instruments/composite-estimator-builder-verifies-gain-correction-plane-scale-join/code/fable_analyst_v1/s2card.py](../../../instruments/composite-estimator-builder-verifies-gain-correction-plane-scale-join/code/fable_analyst_v1/s2card.py)
- [../2026-09-23-training-normalization-constants-caused-70-loss-spike-network-weight/s6bcard.py](../2026-09-23-training-normalization-constants-caused-70-loss-spike-network-weight/s6bcard.py)
- [../2026-09-23-partial-truth-plane-correction-outperforms-full-replacement-kink-row/covstep.py](../2026-09-23-partial-truth-plane-correction-outperforms-full-replacement-kink-row/covstep.py)
- [../../../experiments/dials/extra-kink-row-carrier-scales-add-no-useful-gain/code/fable_covhunter_v1/covstep_s2.py](../../../experiments/dials/extra-kink-row-carrier-scales-add-no-useful-gain/code/fable_covhunter_v1/covstep_s2.py)
- [../../../experiments/late-layer-exactness/local-final-layer-error-prediction-null-held-out-r/code/fable_covhunter_v1/gn3_knobs.py](../../../experiments/late-layer-exactness/local-final-layer-error-prediction-null-held-out-r/code/fable_covhunter_v1/gn3_knobs.py)
- [../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_covhunter_v1/s6bcard_joint2.py](../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_covhunter_v1/s6bcard_joint2.py)
- [../../../experiments/dials/third-dial-fit-round-nystrom-structure-null-sixteen-network/code/fable_covhunter_v1/s6bcard_joint2t.py](../../../experiments/dials/third-dial-fit-round-nystrom-structure-null-sixteen-network/code/fable_covhunter_v1/s6bcard_joint2t.py)
- [../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_engineer_v1/gn_linear_hook.py](../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_engineer_v1/gn_linear_hook.py)
- [../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_engineer_v1/join_nystrom_torch.py](../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_engineer_v1/join_nystrom_torch.py)
- [../../../instruments/termwise-hermite-pair-series-implementation-reduces-training-intermediates-small/code/pair_series_torch.py](../../../instruments/termwise-hermite-pair-series-implementation-reduces-training-intermediates-small/code/pair_series_torch.py)
- [../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [../../../experiments/learned-correction/training-reduced-join-rank-fails-show-specific-co-adaptation/code/fable_physicist_v1/refit6b/s6bcard_loop5js.py](../../../experiments/learned-correction/training-reduced-join-rank-fails-show-specific-co-adaptation/code/fable_physicist_v1/refit6b/s6bcard_loop5js.py)
- [../../../experiments/third-order-planes/nystrom-residual-plane-range-finder-needs-power-iteration-without/code/fable_porter_v1/star_nys_test.py](../../../experiments/third-order-planes/nystrom-residual-plane-range-finder-needs-power-iteration-without/code/fable_porter_v1/star_nys_test.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/common.py)
- [../2026-09-23-rank-one-covariancemean-gain-mixture-explains-6368-2-1/model_bank_bfix.py](../2026-09-23-rank-one-covariancemean-gain-mixture-explains-6368-2-1/model_bank_bfix.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/model_bank_research.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/model_bank_research.py)
- [../2026-10-01-kink-row-third-order-terms-account-6374-one-step/pcard.py](../2026-10-01-kink-row-third-order-terms-account-6374-one-step/pcard.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L203](../../TEAM_LOG_2026-10-01_to_10-22.md#L203)
- [TEAM_LOG_2026-10-01_to_10-22.md#L209](../../TEAM_LOG_2026-10-01_to_10-22.md#L209)
- [TEAM_LOG_2026-10-01_to_10-22.md#L211](../../TEAM_LOG_2026-10-01_to_10-22.md#L211)
- [TEAM_LOG_2026-10-01_to_10-22.md#L218](../../TEAM_LOG_2026-10-01_to_10-22.md#L218)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L481](../../SESSION_NOTES_2026-10-01_to_10-03.md#L481)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L497](../../SESSION_NOTES_2026-10-01_to_10-03.md#L497)

<details>
<summary>All result files and records (3)</summary>

- [e31_rank0035.log](e31_rank0035.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-norm-coupling-contraction-explains-36-46-per-unit-fourth) · [Copy provenance](../../../PROVENANCE.json)

