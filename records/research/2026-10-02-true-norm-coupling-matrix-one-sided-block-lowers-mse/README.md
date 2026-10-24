# True norm-coupling matrix: one-sided block lowers MSE 2.2% / 9.9%; bulk complement 3.8% / 3.4%

2026-10-02T10:39Z · finding · informational · measured

The true network-specific matrix was cut by the card's own top-16 covariance eigenvectors (projector P): whole -5.9% / -14.3% (35 / 651); the one-sided block (I-P) R P + transpose -2.2% / -9.9%; the two-sided block -2.0% / -4.8%; the complement (I-P) R (I-P) -3.8% / -3.4%; rank 32 by eigenvalue -4.6% / -14.3%. The networks differ (on 651 the collective-mode block is two thirds of the gain, on 35 the bulk complement is) and the mean-direction mode never matters, so a vector carrier can reach the one-sided number at best and the complement needs a generic low-rank state.

Recorded measurement:

    whole -5.9% / -14.3%; one-sided -2.2% / -9.9% (without mode 1 -2.7% / -10.1%; mode 1 alone -0.2% / +0.1%); two-sided -2.0% / -4.8%; complement -3.8% / -3.4%; one-sided with 64 eigenvectors -2.2% / -6.4%, its complement -1.1% / -1.5%; rank 32 by eigenvalue -4.6% / -14.3%

## Programs

- [e41_oracle_split.sh](e41_oracle_split.sh)
- [../../../submissions/331824/estimator.py](../../../submissions/331824/estimator.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../experiments/learned-correction/eight-updates-detached-star-bases-fail-40-network-component/code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py](../../../experiments/learned-correction/eight-updates-detached-star-bases-fail-40-network-component/code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py)
- [../2026-09-30-detaching-star-basis-restores-historical-training-gradient-without-changing/train_segment.py](../2026-09-30-detaching-star-basis-restores-historical-training-gradient-without-changing/train_segment.py)
- [../../../instruments/composite-estimator-builder-verifies-gain-correction-plane-scale-join/code/fable_analyst_v1/s2card.py](../../../instruments/composite-estimator-builder-verifies-gain-correction-plane-scale-join/code/fable_analyst_v1/s2card.py)
- [../2026-09-23-training-normalization-constants-caused-70-loss-spike-network-weight/s6bcard.py](../2026-09-23-training-normalization-constants-caused-70-loss-spike-network-weight/s6bcard.py)
- [../2026-09-23-partial-truth-plane-correction-outperforms-full-replacement-kink-row/covstep.py](../2026-09-23-partial-truth-plane-correction-outperforms-full-replacement-kink-row/covstep.py)
- [../../../experiments/dials/extra-kink-row-carrier-scales-add-no-useful-gain/code/fable_covhunter_v1/covstep_s2.py](../../../experiments/dials/extra-kink-row-carrier-scales-add-no-useful-gain/code/fable_covhunter_v1/covstep_s2.py)

<details>
<summary>Additional programs and dependencies (27)</summary>

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
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/e16b_k4_closed_loop.py](../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/e16b_k4_closed_loop.py)
- [../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/k31c.py](../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/k31c.py)
- [../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/k31diag.py](../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/k31diag.py)
- [../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/k31l.py](../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/k31l.py)
- [../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/k31m.py](../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/k31m.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar.py)
- [../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/ncar3.py](../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/ncar3.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar4.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar4.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar5.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar5.py)
- [../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/ncar6.py](../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/ncar6.py)
- [../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/physicist/f4.py](../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/physicist/f4.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncarry.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncarry.py)
- [../2026-10-01-kink-row-third-order-terms-account-6374-one-step/pcard.py](../2026-10-01-kink-row-third-order-terms-account-6374-one-step/pcard.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L255](../../TEAM_LOG_2026-10-01_to_10-22.md#L255)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-true-norm-coupling-matrix-one-sided-block-lowers-mse) · [Copy provenance](../../../PROVENANCE.json)

