# Fourth-cumulant shift strength 0.6 and per-unit kurtosis through age two win held-out selection

2026-10-02T02:05Z · decision · adopted · measured

After the fresh-rank runs the lead fixed the S37 physics constants: the fourth-order column shift (F4) at strength 0.6 rather than the 0.4 the screen networks preferred, and the per-unit kurtosis term (K4L) restricted to pairs at most two layers old with coefficient 1. The engineer rebuilt the file with one changed literal; the generated networks nobody tuned on confirmed F4 0.6 + K4L at -5.93% (8 of 8).

Recorded measurement:

    theta 0.6 vs 0.4 -0.52% (se 0.22, 11 of 17); 0.8 vs 0.4 -0.06% (se 0.39); generated seeds 7100000-7100007: F4 0.4 -3.22% (8 of 8), F4 0.6 + K4L -5.93% (8 of 8), K4L on top of F4 0.6 mean -1.8% (5 of 7)

## Programs

- [../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/e16e_fresh24_theta.sh](../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/e16e_fresh24_theta.sh)
- [e16g_generated.sh](e16g_generated.sh)
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

<details>
<summary>Additional programs and dependencies (28)</summary>

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L77](../../TEAM_LOG_2026-10-01_to_10-22.md#L77)
- [TEAM_LOG_2026-10-01_to_10-22.md#L78](../../TEAM_LOG_2026-10-01_to_10-22.md#L78)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L349](../../SESSION_NOTES_2026-10-01_to_10-03.md#L349)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L351](../../SESSION_NOTES_2026-10-01_to_10-03.md#L351)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

## Related work

- shipped in: [Submission #333435: fourth-cumulant column shifts and per-unit kurtosis for pairs up to age two](../../../submissions/333435/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-fourth-cumulant-shift-strength-0-6-per-unit-kurtosis) · [Copy provenance](../../../PROVENANCE.json)

