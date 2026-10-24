# Ridge re-sizing on 32 networks adopted: pooled independent 64-network MSE change -0.323%

2026-10-11T07:25Z · decision · adopted · measured

The same 96 gradients and 12 directions plus a layer-15 correction-scale arm, sized on 32 networks with ridge and the arm set chosen by leave-one-out (kappa out), passed a pre-registered two-stage judge and went into the safe file.

Recorded measurement:

    stage 1 (A30 7100232-263) -0.482% (se 0.205, 19/32); stage 2 (sealed A27 7100168-199) -0.155% (se 0.172, 19/32); pool 64 -0.323% (se 0.135, -2.40 se, 38/64); native 651/378 -1.04 / -2.10% vs S46

## Programs

- [solve_ridge32.py](solve_ridge32.py)
- [../../../submissions/335340/candidate-b0d9b738.py](../../../submissions/335340/candidate-b0d9b738.py)
- [t92_table.py](t92_table.py)
- [s47_build.sh](s47_build.sh)
- [verify_handover.py](verify_handover.py)
- [../../../experiments/third-order-planes/aligning-pair-precision-covariance-arithmetic-still-fails-fidelity-maximum/code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](../../../experiments/third-order-planes/aligning-pair-precision-covariance-arithmetic-still-fails-fidelity-maximum/code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [../../../experiments/learned-correction/covariance-edge-refit-passes-native40-but-fails-independent-public8/code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](../../../experiments/learned-correction/covariance-edge-refit-passes-native40-but-fails-independent-public8/code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [../../../submissions/331824/estimator.py](../../../submissions/331824/estimator.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)

<details>
<summary>Additional programs and dependencies (46)</summary>

- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
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
- [../../../instruments/paired-bootstrap-promotion-scorer-pins-all-truths-predictions-provenance/code/opus_reexam_20261001_v1/analyst/a28_promotion_scorer_20261011T0033Z/score_v2.py](../../../instruments/paired-bootstrap-promotion-scorer-pins-all-truths-predictions-provenance/code/opus_reexam_20261001_v1/analyst/a28_promotion_scorer_20261011T0033Z/score_v2.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_s46_v11.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_s46_v11.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_push_native_v2.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_push_native_v2.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py)
- [../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [dly_from_pt.py](dly_from_pt.py)
- [../2026-10-10-per-layer-fourth-order-isotropic-dial-refit-adopted-layers/read_pt.py](../2026-10-10-per-layer-fourth-order-isotropic-dial-refit-adopted-layers/read_pt.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b6cols.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b6cols.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b7cols.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b7cols.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/f4b.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/f4b.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/pbase_frozen_s4a.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/pbase_frozen_s4a.py)
- [../../../instruments/paired-bootstrap-promotion-scorer-pins-all-truths-predictions-provenance/code/opus_reexam_20261001_v1/physicist/t85_pinned_judge_v7.py](../../../instruments/paired-bootstrap-promotion-scorer-pins-all-truths-predictions-provenance/code/opus_reexam_20261001_v1/physicist/t85_pinned_judge_v7.py)
- [../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py](../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py)
- [../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/trainer/otr_engine.py](../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_m.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_m.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_pl.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_pl.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/worker_generated_v4.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/worker_generated_v4.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L992](../../TEAM_LOG_2026-10-01_to_10-22.md#L992)
- [TEAM_LOG_2026-10-01_to_10-22.md#L997](../../TEAM_LOG_2026-10-01_to_10-22.md#L997)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1001](../../TEAM_LOG_2026-10-01_to_10-22.md#L1001)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1019](../../TEAM_LOG_2026-10-01_to_10-22.md#L1019)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1020](../../TEAM_LOG_2026-10-01_to_10-22.md#L1020)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1025](../../TEAM_LOG_2026-10-01_to_10-22.md#L1025)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1026](../../TEAM_LOG_2026-10-01_to_10-22.md#L1026)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1027](../../TEAM_LOG_2026-10-01_to_10-22.md#L1027)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1028](../../TEAM_LOG_2026-10-01_to_10-22.md#L1028)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1030](../../TEAM_LOG_2026-10-01_to_10-22.md#L1030)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1035](../../TEAM_LOG_2026-10-01_to_10-22.md#L1035)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1036](../../TEAM_LOG_2026-10-01_to_10-22.md#L1036)

<details>
<summary>All result files and records (4)</summary>

- [FROZEN_POINT_v1.json](FROZEN_POINT_v1.json) · record
- [CHECK_VERDICT_v1.json](CHECK_VERDICT_v1.json) · record
- [manifest.json](manifest.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Untouched generated networks become the primary judge because reused ranks overstate refit gains](../2026-10-10-untouched-generated-networks-become-primary-judge-because-reused-ranks/README.md)
- ← enables: [First cut-simulated refit rejected: independent MSE change -0.235%, standard error 0.340](../../../experiments/learned-correction/first-cut-simulated-refit-rejected-independent-mse-change-0/README.md)
- enables: [Repeated public and generated gradient refits converge to the existing point](../2026-10-11-repeated-public-generated-gradient-refits-converge-existing-point/README.md)
- shipped in: [Submission #335340: ridge-re-sized refit, request caches and seven-level full-packet strip pushes](../../../submissions/335340/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-ridge-re-sizing-32-networks-adopted-pooled-independent-64) · [Copy provenance](../../../PROVENANCE.json)

