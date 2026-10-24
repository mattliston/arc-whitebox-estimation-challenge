# Refit-round rules remove re-sized arms and second-stage sizing, retain one aggregate arm

2026-10-12T16:49:54Z · decision · adopted · measured

After radlab showed candidate generation 50 (S50)'s eight strength arms re-sized on fresh networks carry no signal (chi2 6.1 on 8 df) and that the -SE acceptance rule passes a true -0.05% round only 31% of the time (50% at -0.10%), the round rules were rewritten before any candidate generation 52 (S52) data: rule (c) (re-sizing the previous round's strength arms) off from candidate generation 53 (S53) and escal out, phase 2 off unless a PHASE2_ON file exists (freeze on 128 networks), a CONTINUE_RULE 'plus1se' under which the line continues from a frozen point whose pooled check is not worse than +1 SE while the -SE 'accepted' verdict stays the only trigger of port / stage 2 / release and stage 2 from S52 judges against the LAST FIRED point, a stop/slow rule (two DRY rounds in a row, leave-one-out (LOO) above -0.03% or NULL, set WORKER_CAP 2), rule (c) once by hand after an accepted structural moment-closure estimator change, gradient rotation declined, and from S53 one aggregate arm 'fagg' built by radlab's frozen builder (sha-pinned) from the weak bands rule (b) did not take.

Recorded measurement:

    re-sized arms chi2 6.1 on 8 df, single-arm LOO >= -0.001%, |t| <= 1.22; new arms arrp + diso + dk21n + drsy chi2 19.6 on 4 df (-0.046 / -0.002 / -0.031 / -0.022%); -SE rule passes a true -0.05% round 31%, -0.10% 50%; S52 v2 expected -0.03..-0.06%; supply estimate 0.1-0.3% left after S51; step efficiency eta(n) = 1 - 20/n; aggregate test on S51's gradients: bscale all, iso 7-10, k21n 1-6, kc 7-10, s_carry 11-14 (A36 t 5.13 in quadrature, fresh 2.3-2.8)

## Programs

- [yield_v1.py](yield_v1.py)
- [make_s52_v2.py](make_s52_v2.py)
- [make_rule_c_off_v1.py](make_rule_c_off_v1.py)
- [make_continue_rule_v1.py](make_continue_rule_v1.py)
- [stop_slow_v1.sh](stop_slow_v1.sh)
- [make_agg_arm_v1.py](make_agg_arm_v1.py)
- [make_agg_arm_v2.py](make_agg_arm_v2.py)
- [aggregate_spec_v1.py](aggregate_spec_v1.py)
- [ptload.py](ptload.py)
- [t127_stage2_loop_v6.py](t127_stage2_loop_v6.py)
- [../../../experiments/third-order-planes/aligning-pair-precision-covariance-arithmetic-still-fails-fidelity-maximum/code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](../../../experiments/third-order-planes/aligning-pair-precision-covariance-arithmetic-still-fails-fidelity-maximum/code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [../../../experiments/learned-correction/covariance-edge-refit-passes-native40-but-fails-independent-public8/code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](../../../experiments/learned-correction/covariance-edge-refit-passes-native40-but-fails-independent-public8/code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)

<details>
<summary>Additional programs and dependencies (52)</summary>

- [../../../submissions/331824/estimator.py](../../../submissions/331824/estimator.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
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
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [../2026-10-11-one-direction-covariance-star-adopted-0-26-mse-0/star_ss_torch_v2.py](../2026-10-11-one-direction-covariance-star-adopted-0-26-mse-0/star_ss_torch_v2.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b6cols.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b6cols.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b7cols.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b7cols.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/f4b.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/f4b.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/pbase_frozen_s4a.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/pbase_frozen_s4a.py)
- [../../../instruments/generated-refit-after-drop-rounds-connect-sizing-porting-sealed/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/t112_chain_v17.sh](../../../instruments/generated-refit-after-drop-rounds-connect-sizing-porting-sealed/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/t112_chain_v17.sh)
- [../2026-10-12-acceptance-trigger-misses-marker-manual-restart-also-fails-because/t112_table.py](../2026-10-12-acceptance-trigger-misses-marker-manual-restart-also-fails-because/t112_table.py)
- [../../../instruments/generated-refit-after-drop-rounds-connect-sizing-porting-sealed/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/t85_pinned_judge_v14.py](../../../instruments/generated-refit-after-drop-rounds-connect-sizing-porting-sealed/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/physicist/t85_pinned_judge_v14.py)
- [../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/release/guard_footprint_v2.py](../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/release/guard_footprint_v2.py)
- [../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)
- [../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/trainer/cert/RAD/rad_v6.py](../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/trainer/cert/RAD/rad_v6.py)
- [../2026-10-11-first-refit-after-drop-strength-round-adopted-0-223/jmix_generated_v7.py](../2026-10-11-first-refit-after-drop-strength-round-adopted-0-223/jmix_generated_v7.py)
- [../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py](../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_cut_runtime_v2.py)
- [../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/trainer/otr_engine.py](../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [../2026-10-11-research-native-parity-column-cut-covariance-star-strength-literals/otr_generated_v2.py](../2026-10-11-research-native-parity-column-cut-covariance-star-strength-literals/otr_generated_v2.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_m.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_m.py)
- [../2026-10-12-queue-claim-race-kills-three-four-refit-workers-vanished/otr_m_v5.py](../2026-10-12-queue-claim-race-kills-three-four-refit-workers-vanished/otr_m_v5.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_pl.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_pl.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py)
- [../2026-10-12-queue-claim-race-kills-three-four-refit-workers-vanished/worker_generated_v12.py](../2026-10-12-queue-claim-race-kills-three-four-refit-workers-vanished/worker_generated_v12.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1521](../../TEAM_LOG_2026-10-01_to_10-22.md#L1521)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1534](../../TEAM_LOG_2026-10-01_to_10-22.md#L1534)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1538](../../TEAM_LOG_2026-10-01_to_10-22.md#L1538)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1543](../../TEAM_LOG_2026-10-01_to_10-22.md#L1543)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1544](../../TEAM_LOG_2026-10-01_to_10-22.md#L1544)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1546](../../TEAM_LOG_2026-10-01_to_10-22.md#L1546)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1550](../../TEAM_LOG_2026-10-01_to_10-22.md#L1550)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1551](../../TEAM_LOG_2026-10-01_to_10-22.md#L1551)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1560](../../TEAM_LOG_2026-10-01_to_10-22.md#L1560)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1563](../../TEAM_LOG_2026-10-01_to_10-22.md#L1563)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1564](../../TEAM_LOG_2026-10-01_to_10-22.md#L1564)

<details>
<summary>All result files and records (3)</summary>

- [CONTINUE_RULE](../../../instruments/generated-refit-after-drop-rounds-connect-sizing-porting-sealed/CONTINUE_RULE) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [REPORT_v1_lead_copy.md](REPORT_v1_lead_copy.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-refit-round-rules-remove-re-sized-arms-second-stage) · [Copy provenance](../../../PROVENANCE.json)

