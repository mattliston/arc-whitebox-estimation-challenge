# Fourth-cumulant cap validation cancelled after its parent failed public transfer

2026-08-27 · experiment · rejected · measured

The larger G11B cap panel was invalidated before launch by board #328616. The immutable abort records zero target-free calls, no target columns, no output directories, and no submission. Cap science moved to an independently preregistered plain-depth-1 third-cumulant arena (D1) panel.

Recorded measurement:

    The larger G11B cap panel was invalidated before launch by board #328616. The immutable abort records zero target-free calls, no target columns, no output directories, and no submission. Cap science moved to an independently preregistered plain-D1 panel.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py](code/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py)
- [code/public_cp4_cap1024_card_v1/estimator.py](code/public_cp4_cap1024_card_v1/estimator.py)
- [code/public_cp4_cap1792_card_v1/estimator.py](code/public_cp4_cap1792_card_v1/estimator.py)
- [code/public_q24_cp4_cap_adaptive_bracket_v1/build_sources.py](code/public_q24_cp4_cap_adaptive_bracket_v1/build_sources.py)
- [code/public_q24_cp4_cap_adaptive_bracket_v1/estimator_cap256.py](code/public_q24_cp4_cap_adaptive_bracket_v1/estimator_cap256.py)
- [code/public_q24_cp4_cap_adaptive_bracket_v1/estimator_cap384.py](code/public_q24_cp4_cap_adaptive_bracket_v1/estimator_cap384.py)
- [code/public_q24_cp4_cap_adaptive_bracket_v1/estimator_cap512.py](code/public_q24_cp4_cap_adaptive_bracket_v1/estimator_cap512.py)
- [code/public_q24_cp4_cap_adaptive_bracket_v1/estimator_cap640.py](code/public_q24_cp4_cap_adaptive_bracket_v1/estimator_cap640.py)
- [code/public_q24_cp4_cap_adaptive_bracket_v1/run_arm.py](code/public_q24_cp4_cap_adaptive_bracket_v1/run_arm.py)
- [code/public_q24_cp4_cap_adaptive_bracket_v1/score_arm.py](code/public_q24_cp4_cap_adaptive_bracket_v1/score_arm.py)
- [code/public_q24_cp4_cap_co_design_sign_screen_v1/build_cap768.py](code/public_q24_cp4_cap_co_design_sign_screen_v1/build_cap768.py)
- [code/public_q24_cp4_cap_co_design_sign_screen_v1/estimator_cap768.py](code/public_q24_cp4_cap_co_design_sign_screen_v1/estimator_cap768.py)

<details>
<summary>Additional programs and dependencies (15)</summary>

- [code/public_q24_cp4_cap_co_design_sign_screen_v1/run_cap_arm.py](code/public_q24_cp4_cap_co_design_sign_screen_v1/run_cap_arm.py)
- [code/public_q24_cp4_cap_co_design_sign_screen_v1/score_screen.py](code/public_q24_cp4_cap_co_design_sign_screen_v1/score_screen.py)
- [code/public_q24_cp4_cap_disjoint16_validation_v1/build_sources.py](code/public_q24_cp4_cap_disjoint16_validation_v1/build_sources.py)
- [code/public_q24_cp4_cap_disjoint16_validation_v1/estimator_cap512.py](code/public_q24_cp4_cap_disjoint16_validation_v1/estimator_cap512.py)
- [code/public_q24_cp4_cap_disjoint16_validation_v1/estimator_cap768.py](code/public_q24_cp4_cap_disjoint16_validation_v1/estimator_cap768.py)
- [code/public_q24_cp4_cap_disjoint16_validation_v1/run_all_target_free.py](code/public_q24_cp4_cap_disjoint16_validation_v1/run_all_target_free.py)
- [code/public_q24_cp4_cap_disjoint16_validation_v1/run_arm.py](code/public_q24_cp4_cap_disjoint16_validation_v1/run_arm.py)
- [code/public_q24_cp4_cap_disjoint16_validation_v1/score_panel.py](code/public_q24_cp4_cap_disjoint16_validation_v1/score_panel.py)
- [code/public_q24_d1_alpha075_qhead_bank_v1/estimator.py](code/public_q24_d1_alpha075_qhead_bank_v1/estimator.py)
- [code/public_q24_d1_cp4_cap_disjoint16_validation_v1/build_sources.py](code/public_q24_d1_cp4_cap_disjoint16_validation_v1/build_sources.py)
- [code/public_q24_d1_cp4_cap_disjoint16_validation_v1/estimator_cap512.py](code/public_q24_d1_cp4_cap_disjoint16_validation_v1/estimator_cap512.py)
- [code/public_q24_d1_cp4_cap_disjoint16_validation_v1/estimator_cap768.py](code/public_q24_d1_cp4_cap_disjoint16_validation_v1/estimator_cap768.py)
- [code/public_q24_d1_cp4_cap_disjoint16_validation_v1/run_all_target_free.py](code/public_q24_d1_cp4_cap_disjoint16_validation_v1/run_all_target_free.py)
- [code/public_q24_d1_cp4_cap_disjoint16_validation_v1/run_arm.py](code/public_q24_d1_cp4_cap_disjoint16_validation_v1/run_arm.py)
- [code/public_q24_d1_cp4_cap_disjoint16_validation_v1/score_panel.py](code/public_q24_d1_cp4_cap_disjoint16_validation_v1/score_panel.py)

</details>

## Evidence

- [RESULT.json#L1](../../covariance-structure/depth-1-arena-caps-768-512-disjoint-network-validation/RESULT.json#L1)
- [RESULT-25638345.json#L1](../../covariance-structure/depth-1-arena-caps-768-512-disjoint-network-validation/RESULT-25638345.json#L1)
- [ABORTED_PRELAUNCH.json#L1](../../covariance-structure/depth-1-arena-caps-768-512-disjoint-network-validation/ABORTED_PRELAUNCH.json#L1)
- [PREREGISTRATION.json#L1](../../covariance-structure/depth-1-arena-caps-768-512-disjoint-network-validation/PREREGISTRATION.json#L1)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.json](../../covariance-structure/depth-1-arena-caps-768-512-disjoint-network-validation/RESULT.json) · result
- [RESULT-25638345.json](../../covariance-structure/depth-1-arena-caps-768-512-disjoint-network-validation/RESULT-25638345.json) · result
- [ABORTED_PRELAUNCH.json](../../covariance-structure/depth-1-arena-caps-768-512-disjoint-network-validation/ABORTED_PRELAUNCH.json) · result
- [PREREGISTRATION.json](../../covariance-structure/depth-1-arena-caps-768-512-disjoint-network-validation/PREREGISTRATION.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-fourth-cumulant-cap-validation-cancelled-after-parent-failed-public) · [Copy provenance](../../../PROVENANCE.json)

