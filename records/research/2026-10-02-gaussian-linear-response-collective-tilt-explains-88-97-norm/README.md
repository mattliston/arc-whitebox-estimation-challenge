# Gaussian linear response to a collective tilt explains 88-97% of the norm-coupling matrix per step

2026-10-02T09:50Z · finding · informational · derived

Where the one-layer law of the norm-coupling matrix leaks was located at layer 12 of network 35: pre pairs with the post norm equal gammabar times the carried matrix (98-99.6%), the push is exact (100%), and the ReLU step is the Gaussian linear response to the layer's tilt by its own post norm: the gate term, Gaussian births, the units' own (3,1) term, a mean-tilt row form and a variance-tilt row form reach 84-85% (80-82% with every coefficient at its predicted value 1). The complete list (with the (2,2) row sums on the diagonal and a homogeneity term for the pure-gain part of the tilt) holds 99.4% of the whole matrix at layer 0 to 1 and 88-95% of the network-specific part one step from truth at depth, with fitted coefficients 0.92-1.08; the terms cancel each other, a subset is worse than nothing.

Recorded measurement:

    E35: (A) 98-99.6%, coefficient 0.429 = gammabar; (C) 100%, coefficient 0.96; (B) gate term 70-71% (coefficient 0.82), + births 72-73%, + own (3,1) 73-74%, + mean tilt 77%, + variance tilt 84-85% (fitted 0.82 / 1.22 / 0.60 / 1.16 / 0.88), 80-82% at 1; E35b naive pair-map derivative 28% at 0.33 (own-unit rows left in); T19k layer 0->1: 99.4% whole, 96.9% network-specific (651: 99.3% / 96.3%), cumulative births 5%, + row forms 35%, + own (3,1) -78%, + S4 28%, + last diagonal term 97%; T19n0 transitions 5->6 / 9->10 / 11->12 / 13->14: 0.88 / 0.93 / 0.95 / 0.92 predicted, 0.89 / 0.94 / 0.95 / 0.93 fitted; variance tilt +10 to +26 points, homogeneity +13 to +25

## Programs

- [e35_post_matrix_mc.py](e35_post_matrix_mc.py)
- [e35_analyze.py](e35_analyze.py)
- [e35b_gaussian_jvp.py](e35b_gaussian_jvp.py)
- [e35b_debug.py](e35b_debug.py)
- [t19k_law_cert.py](t19k_law_cert.py)
- [t19n0_law_depth_partial.py](t19n0_law_depth_partial.py)
- [nlaw.py](nlaw.py)
- [nlaw_block.py](nlaw_block.py)
- [../../../submissions/331824/estimator.py](../../../submissions/331824/estimator.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)

<details>
<summary>Additional programs and dependencies (20)</summary>

- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../experiments/learned-correction/eight-updates-detached-star-bases-fail-40-network-component/code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py](../../../experiments/learned-correction/eight-updates-detached-star-bases-fail-40-network-component/code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py)
- [../2026-09-30-detaching-star-basis-restores-historical-training-gradient-without-changing/train_segment.py](../2026-09-30-detaching-star-basis-restores-historical-training-gradient-without-changing/train_segment.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L231](../../TEAM_LOG_2026-10-01_to_10-22.md#L231)
- [TEAM_LOG_2026-10-01_to_10-22.md#L234](../../TEAM_LOG_2026-10-01_to_10-22.md#L234)
- [TEAM_LOG_2026-10-01_to_10-22.md#L235](../../TEAM_LOG_2026-10-01_to_10-22.md#L235)
- [TEAM_LOG_2026-10-01_to_10-22.md#L236](../../TEAM_LOG_2026-10-01_to_10-22.md#L236)
- [TEAM_LOG_2026-10-01_to_10-22.md#L237](../../TEAM_LOG_2026-10-01_to_10-22.md#L237)
- [TEAM_LOG_2026-10-01_to_10-22.md#L238](../../TEAM_LOG_2026-10-01_to_10-22.md#L238)
- [TEAM_LOG_2026-10-01_to_10-22.md#L248](../../TEAM_LOG_2026-10-01_to_10-22.md#L248)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L505](../../SESSION_NOTES_2026-10-01_to_10-03.md#L505)

<details>
<summary>All result files and records (3)</summary>

- [e35_rank0035_L12.log](e35_rank0035_L12.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-gaussian-linear-response-collective-tilt-explains-88-97-norm) · [Copy provenance](../../../PROVENANCE.json)

