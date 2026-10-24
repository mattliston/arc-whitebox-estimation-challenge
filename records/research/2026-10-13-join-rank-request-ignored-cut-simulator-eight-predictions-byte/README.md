# Join-rank request ignored by cut simulator: eight predictions are byte-identical to baseline

2026-10-13T17:54Z · incident · failed · derived

A capacity re-screen on C ran the old-tier rank 384 through the request field sched, but the simulator's native push replaces the moment-closure estimator's push with fixed ranks {7: 384, 10: 320, 13: 384}, so the arm was bit for bit C; pair-mean read rank (r_u) 320 on C was null and a parser bug had read its table as n = 0.

Recorded measurement:

    c_sched384 = C bit for bit on 8 of 8; r_u 320 on C -0.041% (se 0.236); the 10-10 384-384-384 test ran without the simulator

## Programs

- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_push_native_v2.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_push_native_v2.py)
- [k_line_v2.py](k_line_v2.py)
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

<details>
<summary>Additional programs and dependencies (40)</summary>

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
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/n6y_cutsim_v2.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/pbase_cutsim_v1.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_s46_v11.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_cutsim_s46_v11.py)
- [../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py](../../../instruments/cut-simulated-research-estimator-removes-95-native-prediction-discrepancy/code/opus_reexam_20261001_v1/engineer/a22_cutsim/torch_rowcuts_v15.py)
- [../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py](../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/engineer/a33_starsum/star_ss_torch.py)
- [../2026-10-11-one-direction-covariance-star-adopted-0-26-mse-0/star_ss_torch_v2.py](../2026-10-11-one-direction-covariance-star-adopted-0-26-mse-0/star_ss_torch_v2.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b6cols.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b6cols.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b7cols.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/b7cols.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/f4b.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/f4b.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/physicist/ncar_bank_frozen_s2s.py)
- [../2026-10-13-repeated-swap-guard-stops-lose-about-45-minutes-experiments/otr_cut_runtime_sc4.py](../2026-10-13-repeated-swap-guard-stops-lose-about-45-minutes-experiments/otr_cut_runtime_sc4.py)
- [../../../experiments/fourth-order-terms/inner-quartic-re-screens-reject-per-unit-covariance-squared/code/opus_reexam_20261001_v1/scout_card/otr_m_sc7.py](../../../experiments/fourth-order-terms/inner-quartic-re-screens-reject-per-unit-covariance-squared/code/opus_reexam_20261001_v1/scout_card/otr_m_sc7.py)
- [../../../experiments/fourth-order-terms/inner-quartic-re-screens-reject-per-unit-covariance-squared/code/opus_reexam_20261001_v1/scout_card/worker_sc11.py](../../../experiments/fourth-order-terms/inner-quartic-re-screens-reject-per-unit-covariance-squared/code/opus_reexam_20261001_v1/scout_card/worker_sc11.py)
- [../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py](../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py)
- [../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/trainer/otr_engine.py](../../../experiments/learned-correction/least-squares-correction-refits-improve-unseen-error-0-170/code/opus_reexam_20261001_v1/trainer/otr_engine.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [../2026-10-11-research-native-parity-column-cut-covariance-star-strength-literals/otr_generated_v2.py](../2026-10-11-research-native-parity-column-cut-covariance-star-strength-literals/otr_generated_v2.py)
- [../2026-10-12-weight-cache-growth-stops-regenerated-seed-weights-byte-identical/otr_generated_v3.py](../2026-10-12-weight-cache-growth-stops-regenerated-seed-weights-byte-identical/otr_generated_v3.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_m.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_m.py)
- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_pl.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_pl.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1060](../../TEAM_LOG_2026-10-01_to_10-22.md#L1060)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2177](../../TEAM_LOG_2026-10-01_to_10-22.md#L2177)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2200](../../TEAM_LOG_2026-10-01_to_10-22.md#L2200)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2201](../../TEAM_LOG_2026-10-01_to_10-22.md#L2201)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2204](../../TEAM_LOG_2026-10-01_to_10-22.md#L2204)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2207](../../TEAM_LOG_2026-10-01_to_10-22.md#L2207)

<details>
<summary>All result files and records (78)</summary>

- [sc_k25x_r7100601_c_ru320.json](sc_k25x_r7100601_c_ru320.json) · result
- [sc_k25x_r7100602_c_ru320.json](sc_k25x_r7100602_c_ru320.json) · result
- [sc_k25x_r7100603_c_ru320.json](sc_k25x_r7100603_c_ru320.json) · result
- [sc_k25x_r7100604_c_ru320.json](sc_k25x_r7100604_c_ru320.json) · result
- [sc_k25x_r7100605_c_ru320.json](sc_k25x_r7100605_c_ru320.json) · result
- [sc_k25x_r7100606_c_ru320.json](sc_k25x_r7100606_c_ru320.json) · result
- [sc_k25x_r7100607_c_ru320.json](sc_k25x_r7100607_c_ru320.json) · result
- [sc_k250_r7100584_cbase.w0.json](sc_k250_r7100584_cbase.w0.json) · result
- [sc_k251_r7100584_c_cap2.w1.json](sc_k251_r7100584_c_cap2.w1.json) · result
- [sc_k251_r7100584_c_ru320.w0.json](sc_k251_r7100584_c_ru320.w0.json) · result
- [sc_k251_r7100584_c_sched384.w1.json](sc_k251_r7100584_c_sched384.w1.json) · result
- [sc_k251_r7100585_c_cap2.w0.json](sc_k251_r7100585_c_cap2.w0.json) · result
- [sc_k251_r7100585_c_ru320.w1.json](sc_k251_r7100585_c_ru320.w1.json) · result
- [sc_k251_r7100585_c_sched384.w0.json](sc_k251_r7100585_c_sched384.w0.json) · result
- [sc_k251_r7100586_c_cap2.w1.json](sc_k251_r7100586_c_cap2.w1.json) · result
- [sc_k251_r7100586_c_ru320.w0.json](sc_k251_r7100586_c_ru320.w0.json) · result
- [sc_k251_r7100586_c_sched384.w1.json](sc_k251_r7100586_c_sched384.w1.json) · result
- [sc_k251_r7100587_c_cap2.w0.json](sc_k251_r7100587_c_cap2.w0.json) · result
- [sc_k251_r7100587_c_ru320.w1.json](sc_k251_r7100587_c_ru320.w1.json) · result
- [sc_k251_r7100587_c_sched384.w0.json](sc_k251_r7100587_c_sched384.w0.json) · result
- [sc_k251_r7100588_c_cap2.w1.json](sc_k251_r7100588_c_cap2.w1.json) · result
- [sc_k251_r7100588_c_ru320.w0.json](sc_k251_r7100588_c_ru320.w0.json) · result
- [sc_k251_r7100588_c_sched384.w1.json](sc_k251_r7100588_c_sched384.w1.json) · result
- [sc_k251_r7100589_c_cap2.w0.json](sc_k251_r7100589_c_cap2.w0.json) · result
- [sc_k251_r7100589_c_ru320.w1.json](sc_k251_r7100589_c_ru320.w1.json) · result
- [sc_k251_r7100589_c_sched384.w0.json](sc_k251_r7100589_c_sched384.w0.json) · result
- [sc_k251_r7100590_c_cap2.w1.json](sc_k251_r7100590_c_cap2.w1.json) · result
- [sc_k251_r7100590_c_ru320.w0.json](sc_k251_r7100590_c_ru320.w0.json) · result
- [sc_k251_r7100590_c_sched384.w1.json](sc_k251_r7100590_c_sched384.w1.json) · result
- [sc_k251_r7100591_c_cap2.w0.json](sc_k251_r7100591_c_cap2.w0.json) · result
- [sc_k251_r7100591_c_ru320.w1.json](sc_k251_r7100591_c_ru320.w1.json) · result
- [sc_k251_r7100591_c_sched384.w0.json](sc_k251_r7100591_c_sched384.w0.json) · result
- [sc_k25x_r7100592_c_ru320.w1.json](sc_k25x_r7100592_c_ru320.w1.json) · result
- [sc_k25x_r7100593_c_ru320.w0.json](sc_k25x_r7100593_c_ru320.w0.json) · result
- [sc_k25x_r7100594_c_ru320.w1.json](sc_k25x_r7100594_c_ru320.w1.json) · result
- [sc_k25x_r7100595_c_ru320.w0.json](sc_k25x_r7100595_c_ru320.w0.json) · result
- [sc_k25x_r7100596_c_ru320.w1.json](sc_k25x_r7100596_c_ru320.w1.json) · result
- [sc_k25x_r7100597_c_ru320.w0.json](sc_k25x_r7100597_c_ru320.w0.json) · result
- [sc_k25x_r7100598_c_ru320.w1.json](sc_k25x_r7100598_c_ru320.w1.json) · result
- [sc_k25x_r7100599_c_ru320.w0.json](sc_k25x_r7100599_c_ru320.w0.json) · result
- [sc_k25x_r7100600_c_ru320.w1.json](sc_k25x_r7100600_c_ru320.w1.json) · result
- [guard_v3_w0_20261013T170736Z.json](guard_v3_w0_20261013T170736Z.json) · result
- [guard_v3_w1_20261013T170757Z.json](guard_v3_w1_20261013T170757Z.json) · result
- [sc_k250_r7100584_cbase.json](sc_k250_r7100584_cbase.json) · result
- [sc_k251_r7100584_c_cap2.json](sc_k251_r7100584_c_cap2.json) · result
- [sc_k251_r7100584_c_ru320.json](sc_k251_r7100584_c_ru320.json) · result
- [sc_k251_r7100584_c_sched384.json](sc_k251_r7100584_c_sched384.json) · result
- [sc_k251_r7100585_c_cap2.json](sc_k251_r7100585_c_cap2.json) · result
- [sc_k251_r7100585_c_ru320.json](sc_k251_r7100585_c_ru320.json) · result
- [sc_k251_r7100585_c_sched384.json](sc_k251_r7100585_c_sched384.json) · result
- [sc_k251_r7100586_c_cap2.json](sc_k251_r7100586_c_cap2.json) · result
- [sc_k251_r7100586_c_ru320.json](sc_k251_r7100586_c_ru320.json) · result
- [sc_k251_r7100586_c_sched384.json](sc_k251_r7100586_c_sched384.json) · result
- [sc_k251_r7100587_c_cap2.json](sc_k251_r7100587_c_cap2.json) · result
- [sc_k251_r7100587_c_ru320.json](sc_k251_r7100587_c_ru320.json) · result
- [sc_k251_r7100587_c_sched384.json](sc_k251_r7100587_c_sched384.json) · result
- [sc_k251_r7100588_c_cap2.json](sc_k251_r7100588_c_cap2.json) · result
- [sc_k251_r7100588_c_ru320.json](sc_k251_r7100588_c_ru320.json) · result
- [sc_k251_r7100588_c_sched384.json](sc_k251_r7100588_c_sched384.json) · result
- [sc_k251_r7100589_c_cap2.json](sc_k251_r7100589_c_cap2.json) · result
- [sc_k251_r7100589_c_ru320.json](sc_k251_r7100589_c_ru320.json) · result
- [sc_k251_r7100589_c_sched384.json](sc_k251_r7100589_c_sched384.json) · result
- [sc_k251_r7100590_c_cap2.json](sc_k251_r7100590_c_cap2.json) · result
- [sc_k251_r7100590_c_ru320.json](sc_k251_r7100590_c_ru320.json) · result
- [sc_k251_r7100590_c_sched384.json](sc_k251_r7100590_c_sched384.json) · result
- [sc_k251_r7100591_c_cap2.json](sc_k251_r7100591_c_cap2.json) · result
- [sc_k251_r7100591_c_ru320.json](sc_k251_r7100591_c_ru320.json) · result
- [sc_k251_r7100591_c_sched384.json](sc_k251_r7100591_c_sched384.json) · result
- [sc_k25x_r7100592_c_ru320.json](sc_k25x_r7100592_c_ru320.json) · result
- [sc_k25x_r7100593_c_ru320.json](sc_k25x_r7100593_c_ru320.json) · result
- [sc_k25x_r7100594_c_ru320.json](sc_k25x_r7100594_c_ru320.json) · result
- [sc_k25x_r7100595_c_ru320.json](sc_k25x_r7100595_c_ru320.json) · result
- [sc_k25x_r7100596_c_ru320.json](sc_k25x_r7100596_c_ru320.json) · result
- [sc_k25x_r7100597_c_ru320.json](sc_k25x_r7100597_c_ru320.json) · result
- [sc_k25x_r7100598_c_ru320.json](sc_k25x_r7100598_c_ru320.json) · result
- [worker_w0.log](worker_w0.log) · log
- [worker_w1.log](worker_w1.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-13-join-rank-request-ignored-cut-simulator-eight-predictions-byte) · [Copy provenance](../../../PROVENANCE.json)

