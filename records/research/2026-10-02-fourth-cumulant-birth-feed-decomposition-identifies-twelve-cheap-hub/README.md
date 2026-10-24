# Fourth-cumulant birth-feed decomposition identifies twelve cheap hub-local contributions

2026-10-02T00:25Z · finding · informational · derived

Open-loop regression of the Monte Carlo fourth-cumulant feed (pushed distinct content on the kink block, network 35, layers 9 and 13) on first-principles terms: isotropic only explains 0.214 / 0.212 of its energy; the four exact hub coefficients S1..S4 on every pair (LOCAL) 0.335 / 0.341; adding the collective drive 0.355 / 0.346; adding the s-channel exchange terms TB1 and TB2 0.432 / 0.415, every coefficient 0.9-1.0 with nothing fitted. The feed is many comparable diagrams; only the hub-local part (12 points) is free and F4's rule A already holds 10 of them, each further 3-4 points costs a push-sized product (+10..45% bill). A perfect feed would be worth roughly -20..-35% of the final error.

Recorded measurement:

    share of feed energy explained L9 / L13: isotropic 0.214 / 0.212; + rule A (fitted 0.73) 0.312; + LOCAL S1..S4 0.335 / 0.341 (fitted 0.90); + RIDE 0.333; + COLLX 0.355 / 0.346; + TB1 + TB2 0.432 / 0.415; LOCAL coefficients per age 0.92 / 0.94 / 0.83 / 0.76; TB1 coefficient 1.004 / 1.03 (+3.5..3.9 points, costs a second push); TB2 0.86 (+3 points, plane-sized product); COLL +0.8..1.5 points free; 57% unexplained

## Programs

- [e15_local_feed.py](e15_local_feed.py)
- [e15b_collective_feed.py](e15b_collective_feed.py)
- [e15c_exchange_feed.py](e15c_exchange_feed.py)
- [e15d_gate_feed.py](e15d_gate_feed.py)
- [e15_capture_cols.py](e15_capture_cols.py)
- [../../../submissions/331824/estimator.py](../../../submissions/331824/estimator.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](../../../instruments/differentiable-estimator-reproduces-native-layer-15-output-4-2e/code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../experiments/learned-correction/eight-updates-detached-star-bases-fail-40-network-component/code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py](../../../experiments/learned-correction/eight-updates-detached-star-bases-fail-40-network-component/code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star_v1/native_parameters.py)
- [../2026-09-30-detaching-star-basis-restores-historical-training-gradient-without-changing/train_segment.py](../2026-09-30-detaching-star-basis-restores-historical-training-gradient-without-changing/train_segment.py)

<details>
<summary>Additional programs and dependencies (19)</summary>

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
- [../2026-10-01-carried-content-accounts-5079-third-order-kink-plane-error/t2_lib.py](../2026-10-01-carried-content-accounts-5079-third-order-kink-plane-error/t2_lib.py)
- [t4_lib.py](t4_lib.py)

</details>

## Evidence

- [SESSION_NOTES_2026-10-01_to_10-03.md#L293](../../SESSION_NOTES_2026-10-01_to_10-03.md#L293)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L319](../../SESSION_NOTES_2026-10-01_to_10-03.md#L319)
- [TEAM_LOG_2026-10-01_to_10-22.md#L67](../../TEAM_LOG_2026-10-01_to_10-22.md#L67)

<details>
<summary>All result files and records (5)</summary>

- [e15_rank0035_L9.log](e15_rank0035_L9.log) · log
- [e15c_rank0035_L13.log](e15c_rank0035_L13.log) · log
- [e15d_rank0035_L13.log](e15d_rank0035_L13.log) · log
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../SESSION_NOTES_2026-10-01_to_10-03.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-fourth-cumulant-birth-feed-decomposition-identifies-twelve-cheap-hub) · [Copy provenance](../../../PROVENANCE.json)

