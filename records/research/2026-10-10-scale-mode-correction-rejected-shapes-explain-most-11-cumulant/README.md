# Scale-mode correction rejected: its shapes explain at most 11% of cumulant-error variance

2026-10-10T05:32Z · finding · rejected · measured

Tested on 7 networks with paired per-unit truths at layers 6-15 (t73_channel_interaction), the scale mode's shapes hold only 0-4% of dv/v, 1-11% of dk3 and 3-5% of dk4, and the three input-error channels compensate only mildly (sum of energies over energy of the sum 1.13-1.22), so a scale-mode consistency correction (option a) cannot reach 3%. The physicist's honest read, accepted by the lead: no 3% physics lever exists by 10-16; fourth-cumulant plane with index multiplicities (3,1) (K31) conversions at layers 10-12 are worth about -0.5..-1% for +5-6% metered FLOPs (a net loss) and the planes at 8-14 are a real ceiling without a cheap carrier; what remained was the adjoint screen for layer 1-14 columns and the mean-correction covariance-fold coefficient (kappa_C) dial inside the refit.

Recorded measurement:

    scale-mode shapes: 0-4% of dv/v, 1-11% of dk3, 3-5% of dk4; channel compensation 1.13-1.22; K31 conversions at layers 10-12 about -0.5..-1% for +5-6% bill; none of options (a)-(c) reaches 3%

## Programs

- [t74_kappa_c.py](t74_kappa_c.py)
- [../../../experiments/third-order-planes/aligning-pair-precision-covariance-arithmetic-still-fails-fidelity-maximum/code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](../../../experiments/third-order-planes/aligning-pair-precision-covariance-arithmetic-still-fails-fidelity-maximum/code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
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

<details>
<summary>Additional programs and dependencies (25)</summary>

- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
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
- [../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b6cols.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b6cols.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b7cols.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b7cols.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/f4b.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/f4b.py)
- [../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py](../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py)
- [../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/trainer/otr_engine.py](../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_m.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_m.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_pl.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L558](../../TEAM_LOG_2026-10-01_to_10-22.md#L558)
- [TEAM_LOG_2026-10-01_to_10-22.md#L559](../../TEAM_LOG_2026-10-01_to_10-22.md#L559)

<details>
<summary>All result files and records (2)</summary>

- [t73_channel_interaction.txt](t73_channel_interaction.txt) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-10-scale-mode-correction-rejected-shapes-explain-most-11-cumulant) · [Copy provenance](../../../PROVENANCE.json)

