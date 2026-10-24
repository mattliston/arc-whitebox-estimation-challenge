# Endpoint-action correction head: local projection failed to transfer to the public grade

2026-08-27 · experiment · rejected · measured

The frozen alpha=.75 q-head integration passed its target-free gate with projected raw 2.897601102e-8, but board #328616 returned raw 3.065038698e-8 and adjusted 2.215580652e-8. The full688 refit repaired that head but is 3.1113% worse than the native model on the identical all688 population. depth-4 third-cumulant arena (D4)+full688 failed residual; depth-3 third-cumulant arena (D3)+full688 passed, packaged, and is held because compute-only D4 #328622 is already better.

Recorded measurement:

    The frozen alpha=.75 q-head integration passed its target-free gate with projected raw 2.897601102e-8, but board #328616 returned raw 3.065038698e-8 and adjusted 2.215580652e-8. The full688 refit repaired that head but is 3.1113% worse than the native model on the identical all688 population. D4+full688 failed residual; D3+full688 passed, packaged, and is held because compute-only D4 #328622 is already better. Strict held-fold global shrink finds a reproducible 0.07894% native gain but misses the stricter raw gate and actual D4 score; this repair program is terminal and non-submit.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1024_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1024_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1792_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1792_card_v1/estimator.py)

<details>
<summary>Additional programs and dependencies (58)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp3_action_gap_screen_v1/build_source.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp3_action_gap_screen_v1/build_source.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp3_action_gap_screen_v1/score_once.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp3_action_gap_screen_v1/score_once.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp3_nested_ht_quotient_discriminator_v1/build_native_replay.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp3_nested_ht_quotient_discriminator_v1/build_native_replay.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp3_nested_ht_quotient_discriminator_v1/build_source_manifest.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp3_nested_ht_quotient_discriminator_v1/build_source_manifest.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp3_nested_ht_quotient_discriminator_v1/cost_audit.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp3_nested_ht_quotient_discriminator_v1/cost_audit.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp3_nested_ht_quotient_discriminator_v1/nested_ht_state.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp3_nested_ht_quotient_discriminator_v1/nested_ht_state.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_co_design_sign_screen_v1/estimator_cap768.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_co_design_sign_screen_v1/estimator_cap768.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_co_design_sign_screen_v1/run_cap_arm.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_co_design_sign_screen_v1/run_cap_arm.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/build_sources.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/build_sources.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/estimator_cap512.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/estimator_cap512.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/estimator_cap768.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/estimator_cap768.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/run_all_target_free.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/run_all_target_free.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/run_arm.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/run_arm.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/score_panel.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/score_panel.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/generate_expected_outputs.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/generate_expected_outputs.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/generate_torch_authority.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/generate_torch_authority.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/run_full_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/run_full_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/verify_transport_parity.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/verify_transport_parity.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/build_sources.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/build_sources.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/estimator_cap512.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/estimator_cap512.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/estimator_cap768.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/estimator_cap768.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/run_all_target_free.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/run_all_target_free.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/run_arm.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/run_arm.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/score_panel.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/score_panel.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d3_alpha075_full688_qhead_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d3_alpha075_full688_qhead_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d3_alpha075_full688_qhead_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d3_alpha075_full688_qhead_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d3_alpha075_full688_qhead_v1/generate_expected_outputs.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d3_alpha075_full688_qhead_v1/generate_expected_outputs.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d3_alpha075_full688_qhead_v1/run_full_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d3_alpha075_full688_qhead_v1/run_full_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d3_alpha075_full688_qhead_v1/verify_transport_parity.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d3_alpha075_full688_qhead_v1/verify_transport_parity.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_alpha075_full688_qhead_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_alpha075_full688_qhead_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_alpha075_full688_qhead_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_alpha075_full688_qhead_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_alpha075_full688_qhead_v1/generate_expected_outputs.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_alpha075_full688_qhead_v1/generate_expected_outputs.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_alpha075_full688_qhead_v1/generate_torch_authority.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_alpha075_full688_qhead_v1/generate_torch_authority.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_alpha075_full688_qhead_v1/run_full_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_alpha075_full688_qhead_v1/run_full_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_alpha075_full688_qhead_v1/verify_transport_parity.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_alpha075_full688_qhead_v1/verify_transport_parity.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_full688_vs_native_bank_shrink_diagnostic_v1/run_diagnostic.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_full688_vs_native_bank_shrink_diagnostic_v1/run_diagnostic.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_g11b_all688_shrink_diagnostic_v1/run_diagnostic.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_g11b_all688_shrink_diagnostic_v1/run_diagnostic.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_gaussian_higher_path_shells_v1/run_source_replay.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_gaussian_higher_path_shells_v1/run_source_replay.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_gaussian_higher_path_shells_v1/score_once.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_gaussian_higher_path_shells_v1/score_once.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_gaussian_higher_path_shells_v1/verify_formula.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_gaussian_higher_path_shells_v1/verify_formula.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_gaussian_mixed_d6_shell_v1/run_source_replay.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_gaussian_mixed_d6_shell_v1/run_source_replay.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_gaussian_mixed_d6_shell_v1/score_once.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_gaussian_mixed_d6_shell_v1/score_once.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_gaussian_mixed_d6_shell_v1/verify_formula.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_gaussian_mixed_d6_shell_v1/verify_formula.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_autonomous_continuation_v1/run_discriminator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_autonomous_continuation_v1/run_discriminator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_checkpoint_autonomy_discriminator_v1/run_discriminator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_checkpoint_autonomy_discriminator_v1/run_discriminator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_checkpoint_interpolation_v1/run_control.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_checkpoint_interpolation_v1/run_control.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_qhead_action_finetune_v1/run_discriminator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_qhead_action_finetune_v1/run_discriminator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_transported_raw20_bank_v1/run_screen.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_transported_raw20_bank_v1/run_screen.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/deployment_prep/export_literal_payload.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/deployment_prep/export_literal_payload.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/instrumented_estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/instrumented_estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/run_transfer_pilot.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/run_transfer_pilot.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_qhead_full688_refit_v1/run_refit.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_qhead_full688_refit_v1/run_refit.py)

</details>

## Evidence

- [RESULT.json#L1](../../series-expansions/kerdock-quadrature-empirical-gate-control-variates-adapted-phase-2/RESULT.json#L1)
- [RESULT.json#L1](../../flop-accounting/continued-training-native-transport-energy-observer-protected-error-improved/RESULT.json#L1)
- [RESULT.json#L1](../../moment-propagation/early-checkpoint-exposure-autonomy-screen-measured-rejected/RESULT.json#L1)
- [TARGET_SCORE.json#L1](../../flop-accounting/cheaper-pure-closure-third-cumulant-canonical-polyadic-carrier-state/TARGET_SCORE.json#L1)
- [TARGET_SCORE.json#L1](TARGET_SCORE.json#L1)
- [RESULT.json#L1](../../wall-and-residual/depth-1-arena-correction-strength-0-75-frozen-endpoint/RESULT.json#L1)
- [RESULT-97a7f1b9.json#L1](../../wall-and-residual/depth-1-arena-correction-strength-0-75-frozen-endpoint/RESULT-97a7f1b9.json#L1)
- [RESULT.json#L1](RESULT.json#L1)
- [RESULT.json#L1](../../flop-accounting/endpoint-fine-tuning-same-correction-head-features-improvement-missed/RESULT.json#L1)
- [RESULT-de944704.json#L1](RESULT-de944704.json#L1)
- [TARGET_SCORE-1f23eb49.json#L1](TARGET_SCORE-1f23eb49.json#L1)
- [AUDIT.md#L1](AUDIT.md#L1)
- [FULL_GATE_RESULT.json#L1](../../wall-and-residual/depth-1-arena-correction-strength-0-75-frozen-endpoint/FULL_GATE_RESULT.json#L1)
- [SUBMIT_WATCH.log#L1](../688-network-endpoint-action-head-refit-accepted-relative-parent/SUBMIT_WATCH.log#L1)
- [ABORTED_PRELAUNCH.json#L1](../../covariance-structure/depth-1-arena-caps-768-512-disjoint-network-validation/ABORTED_PRELAUNCH.json#L1)
- [PREREGISTRATION.json#L1](../688-network-global-shrinkage-diagnostic-0-07894-gain-missed/PREREGISTRATION.json#L1)
- [PREREGISTRATION.json#L1](../688-network-endpoint-action-head-refit-accepted-relative-parent/PREREGISTRATION.json#L1)
- [RUN_SEAL.json#L1](../688-network-endpoint-action-head-refit-accepted-relative-parent/RUN_SEAL.json#L1)
- [RESULT.json#L4914](../../wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/RESULT.json#L4914)
- [RESULT-3d8c7fd1.json#L1](../../covariance-structure/depth-1-arena-caps-768-512-disjoint-network-validation/RESULT-3d8c7fd1.json#L1)
- [FULL_GATE_RESULT.json#L1](FULL_GATE_RESULT.json#L1)
- [FULL_GATE_RESULT.json#L1](../../wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/FULL_GATE_RESULT.json#L1)
- [PREREGISTRATION-57882c7f.json#L1](../688-network-global-shrinkage-diagnostic-0-07894-gain-missed/PREREGISTRATION-57882c7f.json#L1)
- [RESULT.json#L1](../688-network-global-shrinkage-diagnostic-0-07894-gain-missed/RESULT.json#L1)
- [SUBMIT_WATCH.log#L1](../../wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/SUBMIT_WATCH.log#L1)

<details>
<summary>All result files and records (25)</summary>

- [RESULT.json](../../series-expansions/kerdock-quadrature-empirical-gate-control-variates-adapted-phase-2/RESULT.json) · result
- [RESULT.json](../../flop-accounting/continued-training-native-transport-energy-observer-protected-error-improved/RESULT.json) · result
- [RESULT.json](../../moment-propagation/early-checkpoint-exposure-autonomy-screen-measured-rejected/RESULT.json) · result
- [TARGET_SCORE.json](../../flop-accounting/cheaper-pure-closure-third-cumulant-canonical-polyadic-carrier-state/TARGET_SCORE.json) · result
- [TARGET_SCORE.json](TARGET_SCORE.json) · result
- [RESULT.json](../../wall-and-residual/depth-1-arena-correction-strength-0-75-frozen-endpoint/RESULT.json) · result
- [RESULT-97a7f1b9.json](../../wall-and-residual/depth-1-arena-correction-strength-0-75-frozen-endpoint/RESULT-97a7f1b9.json) · result
- [RESULT.json](RESULT.json) · result
- [RESULT.json](../../flop-accounting/endpoint-fine-tuning-same-correction-head-features-improvement-missed/RESULT.json) · result
- [RESULT-de944704.json](RESULT-de944704.json) · result
- [TARGET_SCORE-1f23eb49.json](TARGET_SCORE-1f23eb49.json) · result
- [AUDIT.md](AUDIT.md) · record
- [FULL_GATE_RESULT.json](../../wall-and-residual/depth-1-arena-correction-strength-0-75-frozen-endpoint/FULL_GATE_RESULT.json) · result
- [SUBMIT_WATCH.log](../688-network-endpoint-action-head-refit-accepted-relative-parent/SUBMIT_WATCH.log) · log
- [ABORTED_PRELAUNCH.json](../../covariance-structure/depth-1-arena-caps-768-512-disjoint-network-validation/ABORTED_PRELAUNCH.json) · result
- [PREREGISTRATION.json](../688-network-global-shrinkage-diagnostic-0-07894-gain-missed/PREREGISTRATION.json) · result
- [PREREGISTRATION.json](../688-network-endpoint-action-head-refit-accepted-relative-parent/PREREGISTRATION.json) · result
- [RUN_SEAL.json](../688-network-endpoint-action-head-refit-accepted-relative-parent/RUN_SEAL.json) · result
- [RESULT.json](../../wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/RESULT.json) · result
- [RESULT-3d8c7fd1.json](../../covariance-structure/depth-1-arena-caps-768-512-disjoint-network-validation/RESULT-3d8c7fd1.json) · result
- [FULL_GATE_RESULT.json](FULL_GATE_RESULT.json) · result
- [FULL_GATE_RESULT.json](../../wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/FULL_GATE_RESULT.json) · result
- [PREREGISTRATION-57882c7f.json](../688-network-global-shrinkage-diagnostic-0-07894-gain-missed/PREREGISTRATION-57882c7f.json) · result
- [RESULT.json](../688-network-global-shrinkage-diagnostic-0-07894-gain-missed/RESULT.json) · result
- [SUBMIT_WATCH.log](../../wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/SUBMIT_WATCH.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-endpoint-action-correction-head-local-projection-failed-transfer-public) · [Copy provenance](../../../PROVENANCE.json)

