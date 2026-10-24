# Native response-row cuts account for 1.20% error relative to the Torch estimator

2026-10-02T05:17Z · finding · informational · derived

Native and torch predictions agree to 2e-6 rms through layer 6 and separate at layer 7, the first layer with native row cuts; applying the native cut rules to the torch card reproduces the difference (cosine 0.92-0.98, +3.55% simulated vs +3.79% actual on network 35). The young columns of dead rows (alpha < -2.5 at layers 7-14) cost +2.10% / +0.77% on networks 35 / 100 and all cuts +2.66% / +1.55% (40-network native excess +1.20%, se 0.26). Raising the threshold to 3.0 looked like -2.0% for +0.8% bill on network 35 but on three fresh ranks the 2.5-3.0 band costs 0.63% of MSE and 0.78% of bill: break-even, thresholds unchanged.

Recorded measurement:

    separation 2.0 -> 10.1e-6 rms at layer 7, 19.6e-6 at layer 15 (network 35); native MSE +3.79% / +0.52% over torch (35 / 100), +1.20% (se 0.26) over 40 validation nets; cut costs 35 / 100: young dead +2.10% / +0.77%, old-tier dead +0.38% / +0.14%, saturated +0.27% / +0.20%, dead-side 3.5 +0.06% / +0.11%; three fresh ranks: threshold 2.5 +1.34%, 2.75 +0.82%, 3.0 +0.71%, 3.5 +0.69%; band 2.5-3.0 = 0.63% MSE for 0.78% bill

## Programs

- [torch_cutsim_v2.py](torch_cutsim_v2.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L110](../../TEAM_LOG_2026-10-01_to_10-22.md#L110)
- [TEAM_LOG_2026-10-01_to_10-22.md#L123](../../TEAM_LOG_2026-10-01_to_10-22.md#L123)
- [TEAM_LOG_2026-10-01_to_10-22.md#L125](../../TEAM_LOG_2026-10-01_to_10-22.md#L125)
- [TEAM_LOG_2026-10-01_to_10-22.md#L128](../../TEAM_LOG_2026-10-01_to_10-22.md#L128)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L406](../../SESSION_NOTES_2026-10-01_to_10-03.md#L406)
- [RESULT.md#L714](../../../experiments/arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md#L714)
- [RESULT.md#L814](../../../experiments/arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md#L814)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](../../../experiments/arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-native-response-row-cuts-account-1-20-error-relative) · [Copy provenance](../../../PROVENANCE.json)

