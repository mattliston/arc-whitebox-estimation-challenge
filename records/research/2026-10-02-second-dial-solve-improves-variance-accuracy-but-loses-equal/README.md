# Second dial solve improves variance accuracy but loses equal cancellation in final mean error

2026-10-02T18:04Z · finding · informational · measured

On networks 35 and 100 the second dial solve makes late-layer variance errors 3-12% smaller but reduces their cancellation with skew and kurtosis errors. Final MSE stays nearly unchanged, while exact planes at pair layers 11-14 still lower it 10.7% / 14.6%.

Recorded measurement:

    network 35 final 11.720e-9 -> 11.783e-9 (+0.54%); network 100 13.756e-9 -> 13.754e-9 (-0.01%); variance-caused power 8.59 -> 8.05e-9 and 9.41 -> 9.09e-9; cross terms -5.29 -> -4.68e-9 and -3.38 -> -3.09e-9

## Programs

- [../2026-10-02-norm-coupling-law-improves-cancellation-mean-error-sources-late/a20_budget.py](../2026-10-02-norm-coupling-law-improves-cancellation-mean-error-sources-late/a20_budget.py)
- [../../../submissions/331824/estimator.py](../../../submissions/331824/estimator.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [../../../instruments/composite-estimator-builder-verifies-gain-correction-plane-scale-join/code/fable_analyst_v1/s2card.py](../../../instruments/composite-estimator-builder-verifies-gain-correction-plane-scale-join/code/fable_analyst_v1/s2card.py)
- [../2026-09-23-training-normalization-constants-caused-70-loss-spike-network-weight/s6bcard.py](../2026-09-23-training-normalization-constants-caused-70-loss-spike-network-weight/s6bcard.py)
- [../2026-09-23-partial-truth-plane-correction-outperforms-full-replacement-kink-row/covstep.py](../2026-09-23-partial-truth-plane-correction-outperforms-full-replacement-kink-row/covstep.py)
- [../../../experiments/dials/extra-kink-row-carrier-scales-add-no-useful-gain/code/fable_covhunter_v1/covstep_s2.py](../../../experiments/dials/extra-kink-row-carrier-scales-add-no-useful-gain/code/fable_covhunter_v1/covstep_s2.py)
- [../../../experiments/late-layer-exactness/local-final-layer-error-prediction-null-held-out-r/code/fable_covhunter_v1/gn3_knobs.py](../../../experiments/late-layer-exactness/local-final-layer-error-prediction-null-held-out-r/code/fable_covhunter_v1/gn3_knobs.py)
- [../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_covhunter_v1/s6bcard_joint2.py](../../../experiments/dials/consistent-semantics-physics-refit-improves-raw-error-1-98/code/fable_covhunter_v1/s6bcard_joint2.py)
- [../../../experiments/dials/third-dial-fit-round-nystrom-structure-null-sixteen-network/code/fable_covhunter_v1/s6bcard_joint2t.py](../../../experiments/dials/third-dial-fit-round-nystrom-structure-null-sixteen-network/code/fable_covhunter_v1/s6bcard_joint2t.py)

<details>
<summary>Additional programs and dependencies (31)</summary>

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
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar4.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar4.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar5.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/ncar5.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b6cols.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b6cols.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b7cols.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b7cols.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/f4b.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/f4b.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar6_audit_frozen_s3d.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar6_audit_frozen_s3d.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_fold_frozen_s2w.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_fold_frozen_s2w.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncarry.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncarry.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/pbase_frozen_s4a.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/pbase_frozen_s4a.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/k31l_t1508.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/k31l_t1508.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/k31l_t1508y.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/k31l_t1508y.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/ncar6_t1339.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/ncar6_t1339.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/ncar6_t1402.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/ncar6_t1402.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/ncar6_t1402x.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/ncar6_t1402x.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/ncar6_t1402y.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/frozen/ncar6_t1402y.py)
- [../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py](../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py)
- [../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/trainer/otr_engine.py](../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_m.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_m.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_pl.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L409](../../TEAM_LOG_2026-10-01_to_10-22.md#L409)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [a20_card_s42.json](a20_card_s42.json) · config

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-second-dial-solve-improves-variance-accuracy-but-loses-equal) · [Copy provenance](../../../PROVENANCE.json)

