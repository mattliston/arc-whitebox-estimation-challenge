# Depth-1 arena caps 768 and 512: disjoint-network validation rejected promotion

2026-08-26 · experiment · rejected · measured

All 48 target-free calls and bindings passed before the sole target read, but neither direct-depth-1 third-cumulant arena (D1) cap transfers. Baseline mean mean squared error (MSE) is 3.026499284e-8; cap768 is 3.058098629e-8 (+1.0441%, 3/16 wins, paired t=-2.851) and cap512 is 3.072255205e-8 (+1.5118%, 3/16 wins, paired t=-3.326). Every literal candidate bill is still lower.

Recorded measurement:

    All 48 target-free calls and bindings passed before the sole target read, but neither direct-D1 cap transfers. Baseline mean MSE is 3.026499284e-8; cap768 is 3.058098629e-8 (+1.0441%, 3/16 wins, paired t=-2.851) and cap512 is 3.072255205e-8 (+1.5118%, 3/16 wins, paired t=-3.326). Every literal candidate bill is still lower.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1024_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1024_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1792_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1792_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_adaptive_bracket_v1/build_sources.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_adaptive_bracket_v1/build_sources.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_adaptive_bracket_v1/estimator_cap256.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_adaptive_bracket_v1/estimator_cap256.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_adaptive_bracket_v1/estimator_cap384.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_adaptive_bracket_v1/estimator_cap384.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_adaptive_bracket_v1/estimator_cap512.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_adaptive_bracket_v1/estimator_cap512.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_adaptive_bracket_v1/estimator_cap640.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_adaptive_bracket_v1/estimator_cap640.py)

<details>
<summary>Additional programs and dependencies (19)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_adaptive_bracket_v1/run_arm.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_adaptive_bracket_v1/run_arm.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_adaptive_bracket_v1/score_arm.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_adaptive_bracket_v1/score_arm.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_co_design_sign_screen_v1/build_cap768.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_co_design_sign_screen_v1/build_cap768.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_co_design_sign_screen_v1/estimator_cap768.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_co_design_sign_screen_v1/estimator_cap768.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_co_design_sign_screen_v1/run_cap_arm.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_co_design_sign_screen_v1/run_cap_arm.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_co_design_sign_screen_v1/score_screen.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_co_design_sign_screen_v1/score_screen.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/build_sources.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/build_sources.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/estimator_cap512.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/estimator_cap512.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/estimator_cap768.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/estimator_cap768.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/run_all_target_free.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/run_all_target_free.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/run_arm.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/run_arm.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/score_panel.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_cp4_cap_disjoint16_validation_v1/score_panel.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/build_sources.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/build_sources.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/estimator_cap512.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/estimator_cap512.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/estimator_cap768.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/estimator_cap768.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/run_all_target_free.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/run_all_target_free.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/run_arm.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/run_arm.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/score_panel.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_cp4_cap_disjoint16_validation_v1/score_panel.py)

</details>

## Evidence

- [SUBMIT_WATCH.log#L1](../../flop-accounting/deployed-estimator-score-economics-minimum-charged-utilisation-verified/SUBMIT_WATCH.log#L1)
- [RESULT.json#L1](RESULT.json#L1)
- [RESULT-25638345.json#L1](RESULT-25638345.json#L1)
- [ABORTED_PRELAUNCH.json#L1](ABORTED_PRELAUNCH.json#L1)
- [PREREGISTRATION.json#L1](PREREGISTRATION.json#L1)
- [TARGET_FREE_RUN_RESULT.json#L1](TARGET_FREE_RUN_RESULT.json#L1)
- [RESULT-3d8c7fd1.json#L57](RESULT-3d8c7fd1.json#L57)

<details>
<summary>All result files and records (7)</summary>

- [SUBMIT_WATCH.log](../../flop-accounting/deployed-estimator-score-economics-minimum-charged-utilisation-verified/SUBMIT_WATCH.log) · log
- [RESULT.json](RESULT.json) · result
- [RESULT-25638345.json](RESULT-25638345.json) · result
- [ABORTED_PRELAUNCH.json](ABORTED_PRELAUNCH.json) · result
- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [TARGET_FREE_RUN_RESULT.json](TARGET_FREE_RUN_RESULT.json) · result
- [RESULT-3d8c7fd1.json](RESULT-3d8c7fd1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-26-depth-1-arena-caps-768-512-disjoint-network-validation) · [Copy provenance](../../../PROVENANCE.json)

