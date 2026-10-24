# Truth planes at layers 10-14 improve the older research estimator -17.8% and -19.4% on two networks

2026-10-20T02:43:00Z · finding · informational · measured

In the physicist's torch harness the truth K21/K31/K22 planes (8M-sample planes of networks 35 and 100) replace the card's in the covariance output only. On the record's base, all three truth planes at layers 10-14 alone give -17.8% (net 35) / -19.4% (net 100) of the final MSE, while truth planes at all layers give less (early truth planes hurt on net 35) and layers 12-14 only give -6.1% / -14.2%; K31 alone at layers 10-14 is -11.6%. Exact plane transport at five late layers costs about +10% bill (the record's +27% was over 14 layers): rate about 1.7, the favorable trade the owner asked for; the cheapest knobs on those layers (join ranks at layers 10/13/7) were queued first.

Recorded measurement:

    pass 1 (bare torch base c=0, f4=0; net 35): K31 truth all layers -11.9%, L12-14 only -5.9%; K21 kink rows L12-14 -2.8%; all three planes L12-14 -8.6%; all three all layers -14.5%; pass 2 (record's base c=0.75, age=2, f4=0.6): net 35 all planes L10-14 -17.8% (L12-14 -6.1%, all layers -12.3%), K31 alone L10-14 -11.6% (L12-14 -5.9%, all -14.8%), K21 kink rows L12-14 -0.4%; net 100 all planes L10-14 -19.4% (L12-14 -14.2%, all -19.3%), K31 L12-14 -6.6%; exact late transport ~+10% bill vs +27% over 14 layers (rate ~1.7)

## Programs

- [e16b_f4b.py](e16b_f4b.py)
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

<details>
<summary>Additional programs and dependencies (21)</summary>

- [../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_engineer_v1/join_nystrom_torch.py](../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_engineer_v1/join_nystrom_torch.py)
- [../../../instruments/termwise-hermite-pair-series-implementation-reduces-training-intermediates-small/code/pair_series_torch.py](../../../instruments/termwise-hermite-pair-series-implementation-reduces-training-intermediates-small/code/pair_series_torch.py)
- [../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py](../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_physicist_v1/refit6b/lrstar_s1zero_hook.py)
- [../../../experiments/learned-correction/training-reduced-join-rank-fails-show-specific-co-adaptation/code/fable_physicist_v1/refit6b/s6bcard_loop5js.py](../../../experiments/learned-correction/training-reduced-join-rank-fails-show-specific-co-adaptation/code/fable_physicist_v1/refit6b/s6bcard_loop5js.py)
- [../../../experiments/third-order-planes/nystrom-residual-plane-range-finder-needs-power-iteration-without/code/fable_porter_v1/star_nys_test.py](../../../experiments/third-order-planes/nystrom-residual-plane-range-finder-needs-power-iteration-without/code/fable_porter_v1/star_nys_test.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/common.py)
- [../2026-09-23-rank-one-covariancemean-gain-mixture-explains-6368-2-1/model_bank_bfix.py](../2026-09-23-rank-one-covariancemean-gain-mixture-explains-6368-2-1/model_bank_bfix.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/model_bank_research.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/fable_refit_v1/model_bank_research.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/physicist/f4.py](../../../experiments/fourth-order-terms/young-column-fourth-cumulant-shifts-reduce-held-out-error/code/opus_reexam_20261001_v1/physicist/f4.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/f4b.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/f4b.py)
- [k31l_audit.py](k31l_audit.py)
- [ncar6_audit.py](ncar6_audit.py)
- [../../../experiments/fourth-order-terms/second-weight-per-plane-column-folded-into-2-1/code/ncar_bank.py](../../../experiments/fourth-order-terms/second-weight-per-plane-column-folded-into-2-1/code/ncar_bank.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py)
- [../../../experiments/fourth-order-terms/second-weight-per-plane-column-folded-into-2-1/code/ncar_fold.py](../../../experiments/fourth-order-terms/second-weight-per-plane-column-folded-into-2-1/code/ncar_fold.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncarry.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncarry.py)
- [ntor.py](ntor.py)
- [../2026-10-10-rejected-analytic-terms-become-zero-initialised-per-unit-learned/pbase.py](../2026-10-10-rejected-analytic-terms-become-zero-initialised-per-unit-learned/pbase.py)
- [../2026-10-01-kink-row-third-order-terms-account-6374-one-step/pcard.py](../2026-10-01-kink-row-third-order-terms-account-6374-one-step/pcard.py)
- [../2026-10-02-lean-second-order-third-cumulant-target-costs-0-29/so2.py](../2026-10-02-lean-second-order-third-cumulant-target-costs-0-29/so2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6119](../../TEAM_LOG_2026-10-01_to_10-22.md#L6119)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6121](../../TEAM_LOG_2026-10-01_to_10-22.md#L6121)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6123](../../TEAM_LOG_2026-10-01_to_10-22.md#L6123)

<details>
<summary>All result files and records (4)</summary>

- [t4_late31_rank35.log](t4_late31_rank35.log) · log
- [t4_late31b_rank35.log](t4_late31b_rank35.log) · log
- [t4_late31b_rank100.log](t4_late31b_rank100.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-truth-planes-layers-10-14-improve-older-research-estimator) · [Copy provenance](../../../PROVENANCE.json)

