# Known arithmetic reductions and fourth-cumulant rank cuts: rejected by measured results

2026-08-24 · experiment · rejected · measured

Known arithmetic trims remain insufficient, and the two direct fourth-cumulant canonical-polyadic carrier (CP4) cap descendants are now measured. Cap1024 saves 81,756,130,304 FLOPs per call but produces pooled unique-fixture final drift 9.999932286193823e-6. Cap1792 saves 20,399,602,944 FLOPs per call but still produces pooled unique drift 2.503792400835152e-6.

Recorded measurement:

    Known arithmetic trims remain insufficient, and the two direct CP4 cap descendants are now measured. Cap1024 saves 81,756,130,304 FLOPs per call but produces pooled unique-fixture final drift 9.999932286193823e-6. Cap1792 saves 20,399,602,944 FLOPs per call but still produces pooled unique drift 2.503792400835152e-6. Both fail their frozen accuracy gates.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_a1_no_cp4_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_a1_no_cp4_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_a1_no_cp4_v1/verify_a1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_a1_no_cp4_v1/verify_a1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_a1_no_cp4_v1/verify_exact_static_delta.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_a1_no_cp4_v1/verify_exact_static_delta.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_floor_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_floor_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_v1/verify_tiny.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_v1/verify_tiny.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_raw_diag_load_rescale_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_raw_diag_load_rescale_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/build_q24_from_v4.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/build_q24_from_v4.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/compare_target_free_v4_q24.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/compare_target_free_v4_q24.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/run_additional_fullwidth_seeds.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/run_additional_fullwidth_seeds.py)

<details>
<summary>Additional programs and dependencies (9)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/verify_parity.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/verify_parity.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/verify_resources.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/verify_resources.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/verify_static.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/verify_static.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1024_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1024_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1024_card_v1/run_full_estimator_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1024_card_v1/run_full_estimator_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1792_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1792_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1792_card_v1/run_full_estimator_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp4_cap1792_card_v1/run_full_estimator_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py)

</details>

## Evidence

- [INDEPENDENT_SOURCE_AND_METER_AUDIT_Q24.json#L1](INDEPENDENT_SOURCE_AND_METER_AUDIT_Q24.json#L1)
- [EXACT_STATIC_DELTA.json#L1](EXACT_STATIC_DELTA.json#L1)
- [FREEZE_RECEIPT.json#L1](FREEZE_RECEIPT.json#L1)
- [FULL_ESTIMATOR_GATE_RESULT.json#L1](FULL_ESTIMATOR_GATE_RESULT.json#L1)
- [FULL_ESTIMATOR_GATE_RESULT-3722d52d.json#L1](FULL_ESTIMATOR_GATE_RESULT-3722d52d.json#L1)

<details>
<summary>All result files and records (5)</summary>

- [INDEPENDENT_SOURCE_AND_METER_AUDIT_Q24.json](INDEPENDENT_SOURCE_AND_METER_AUDIT_Q24.json) · result
- [EXACT_STATIC_DELTA.json](EXACT_STATIC_DELTA.json) · result
- [FREEZE_RECEIPT.json](FREEZE_RECEIPT.json) · result
- [FULL_ESTIMATOR_GATE_RESULT.json](FULL_ESTIMATOR_GATE_RESULT.json) · result
- [FULL_ESTIMATOR_GATE_RESULT-3722d52d.json](FULL_ESTIMATOR_GATE_RESULT-3722d52d.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-24-known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured) · [Copy provenance](../../../PROVENANCE.json)

