# Pair-series order 32 reduced to 24: final prediction disagreement at most 9.45e-17 on three full-width seeds

2026-08-24 · experiment · adopted · measured

The frozen order-24 Hermite pair-series closure (Q24) arm changes only the Hermite/rho term axis from q=0..32 to q=0..24 while retaining third-cumulant canonical-polyadic carrier (CP3), third-order Hermite state (H3), fourth-cumulant canonical-polyadic carrier (CP4), coherent fourth-order Hermite state (H4)+alpha, fourth-cumulant Edgeworth operator (L4), one-half third-cumulant Edgeworth operator (L3)-squared, and all 17 channels. Its literal meter passed at 2,131,249,329,109 FLOPs (96.9180%), 29.222195 seconds wall, 0.294491 seconds residual, and 4,507,058,176 bytes sampled resident-set size (RSS). Three fixed production-shape official-He seeds were finite and differed from V4 by 2.78e-17, 9.45e-17, and 3.46e-17 final prediction mean squared error (MSE), satisfying this node's one-arm activation gate without establishing global Q24/Q32 science equivalence.

Recorded measurement:

    The frozen Q24 arm changes only the Hermite/rho term axis from q=0..32 to q=0..24 while retaining CP3, H3, CP4, coherent H4+alpha, L4, one-half L3-squared, and all 17 channels. Its literal meter passed at 2,131,249,329,109 FLOPs (96.9180%), 29.222195 seconds wall, 0.294491 seconds residual, and 4,507,058,176 bytes sampled RSS. Three fixed production-shape official-He seeds were finite and differed from V4 by 2.78e-17, 9.45e-17, and 3.46e-17 final prediction MSE, satisfying this node's one-arm activation gate without establishing global Q24/Q32 science equivalence. Its single authorized board arm terminated as submission 328210 with a 100/100 operational pass, zero failures, and exact raw final MSE 5.765125013112993e-7; this did not meet the 1e-8 target. A Q32 fp32 pair-kernel alternative was separately rejected target-free for eigensolve failure, nonfinite recursion, and catastrophic official-He disagreement; it created no meter, package, target access, or submission.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_accuracy_frontier_v1/verify_design.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_accuracy_frontier_v1/verify_design.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/build_q24_from_v4.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/build_q24_from_v4.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/compare_target_free_v4_q24.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/compare_target_free_v4_q24.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/run_additional_fullwidth_seeds.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/run_additional_fullwidth_seeds.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/verify_parity.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/verify_parity.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/verify_resources.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1/verify_resources.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1_board_promotion_v1/verify_board_promotion_scaffold.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_v1_board_promotion_v1/verify_board_promotion_scaffold.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q32_fp32_pair_v1/build_from_v4.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q32_fp32_pair_v1/build_from_v4.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q32_fp32_pair_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q32_fp32_pair_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q32_fp32_pair_v1/run_target_free_compare.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q32_fp32_pair_v1/run_target_free_compare.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q32_fp32_pair_v1/verify_results.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q32_fp32_pair_v1/verify_results.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v2/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v2/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v3/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v3/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_v4/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py)

</details>

## Evidence

- [DESIGN_MANIFEST.json#L1](DESIGN_MANIFEST.json#L1)
- [PREREGISTER.md#L1](PREREGISTER.md#L1)
- [VARIANT_MATRIX.json#L1](VARIANT_MATRIX.json#L1)
- [SOURCE_BINDINGS.json#L1](SOURCE_BINDINGS.json#L1)
- [GRAPH_CHILDREN.json#L1](GRAPH_CHILDREN.json#L1)
- [PREREGISTER-a7b54086.md#L1](../order-32-high-order-closure-literal-run-exceeded-wall/PREREGISTER-a7b54086.md#L1)
- [SOURCE_MANIFEST-1627632a.json#L101](../order-32-high-order-closure-literal-run-exceeded-wall/SOURCE_MANIFEST-1627632a.json#L101)
- [FREEZE_RECEIPT_Q24.json#L85](../order-32-high-order-closure-literal-run-exceeded-wall/FREEZE_RECEIPT_Q24.json#L85)
- [INDEPENDENT_SOURCE_AND_METER_AUDIT_Q24.json#L149](../../flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/INDEPENDENT_SOURCE_AND_METER_AUDIT_Q24.json#L149)
- [receipt-d69fd338.json#L603](../order-32-high-order-closure-literal-run-exceeded-wall/receipt-d69fd338.json#L603)
- [receipt-70c8bbc1.json#L1](../order-32-high-order-closure-literal-run-exceeded-wall/receipt-70c8bbc1.json#L1)
- [receipt-777dfdf5.json#L1](../order-32-high-order-closure-literal-run-exceeded-wall/receipt-777dfdf5.json#L1)
- [INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_Q24.json#L163](../order-32-high-order-closure-literal-run-exceeded-wall/INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_Q24.json#L163)
- [PACKAGE_MANIFEST.json#L1](../order-32-high-order-closure-literal-run-exceeded-wall/PACKAGE_MANIFEST.json#L1)
- [PHASE2_WEIGHT_GENERATION_LAW_AUDIT.json#L1](../order-32-high-order-closure-literal-run-exceeded-wall/PHASE2_WEIGHT_GENERATION_LAW_AUDIT.json#L1)
- [PROMOTION_MANIFEST-850be6d2.json#L109](../order-32-high-order-closure-literal-run-exceeded-wall/PROMOTION_MANIFEST-850be6d2.json#L109)
- [INDEPENDENT_FINAL_READY_AUDIT.json#L149](../order-32-high-order-closure-literal-run-exceeded-wall/INDEPENDENT_FINAL_READY_AUDIT.json#L149)
- [BOARD_ATTEMPT_328210_IN_FLIGHT.json#L1](../order-32-high-order-closure-literal-run-exceeded-wall/BOARD_ATTEMPT_328210_IN_FLIGHT.json#L1)
- [BOARD_ATTEMPT_328210_PROGRESS_65.json#L1](../order-32-high-order-closure-literal-run-exceeded-wall/BOARD_ATTEMPT_328210_PROGRESS_65.json#L1)
- [BOARD_ATTEMPT_328210_RESULT.json#L1](../order-32-high-order-closure-literal-run-exceeded-wall/BOARD_ATTEMPT_328210_RESULT.json#L1)
- [INDEPENDENT_BOARD_RESULT_AUDIT_328210.json#L1](../order-32-high-order-closure-literal-run-exceeded-wall/INDEPENDENT_BOARD_RESULT_AUDIT_328210.json#L1)
- [REJECTION_RECEIPT.json#L1](../order-32-high-order-closure-literal-run-exceeded-wall/REJECTION_RECEIPT.json#L1)

<details>
<summary>All result files and records (22)</summary>

- [DESIGN_MANIFEST.json](DESIGN_MANIFEST.json) · result
- [PREREGISTER.md](PREREGISTER.md) · record
- [VARIANT_MATRIX.json](VARIANT_MATRIX.json) · result
- [SOURCE_BINDINGS.json](SOURCE_BINDINGS.json) · result
- [GRAPH_CHILDREN.json](GRAPH_CHILDREN.json) · result
- [PREREGISTER-a7b54086.md](../order-32-high-order-closure-literal-run-exceeded-wall/PREREGISTER-a7b54086.md) · record
- [SOURCE_MANIFEST-1627632a.json](../order-32-high-order-closure-literal-run-exceeded-wall/SOURCE_MANIFEST-1627632a.json) · result
- [FREEZE_RECEIPT_Q24.json](../order-32-high-order-closure-literal-run-exceeded-wall/FREEZE_RECEIPT_Q24.json) · result
- [INDEPENDENT_SOURCE_AND_METER_AUDIT_Q24.json](../../flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/INDEPENDENT_SOURCE_AND_METER_AUDIT_Q24.json) · result
- [receipt-d69fd338.json](../order-32-high-order-closure-literal-run-exceeded-wall/receipt-d69fd338.json) · result
- [receipt-70c8bbc1.json](../order-32-high-order-closure-literal-run-exceeded-wall/receipt-70c8bbc1.json) · result
- [receipt-777dfdf5.json](../order-32-high-order-closure-literal-run-exceeded-wall/receipt-777dfdf5.json) · result
- [INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_Q24.json](../order-32-high-order-closure-literal-run-exceeded-wall/INDEPENDENT_FULLWIDTH_COMPARATOR_AUDIT_Q24.json) · result
- [PACKAGE_MANIFEST.json](../order-32-high-order-closure-literal-run-exceeded-wall/PACKAGE_MANIFEST.json) · result
- [PHASE2_WEIGHT_GENERATION_LAW_AUDIT.json](../order-32-high-order-closure-literal-run-exceeded-wall/PHASE2_WEIGHT_GENERATION_LAW_AUDIT.json) · result
- [PROMOTION_MANIFEST-850be6d2.json](../order-32-high-order-closure-literal-run-exceeded-wall/PROMOTION_MANIFEST-850be6d2.json) · result
- [INDEPENDENT_FINAL_READY_AUDIT.json](../order-32-high-order-closure-literal-run-exceeded-wall/INDEPENDENT_FINAL_READY_AUDIT.json) · result
- [BOARD_ATTEMPT_328210_IN_FLIGHT.json](../order-32-high-order-closure-literal-run-exceeded-wall/BOARD_ATTEMPT_328210_IN_FLIGHT.json) · result
- [BOARD_ATTEMPT_328210_PROGRESS_65.json](../order-32-high-order-closure-literal-run-exceeded-wall/BOARD_ATTEMPT_328210_PROGRESS_65.json) · result
- [BOARD_ATTEMPT_328210_RESULT.json](../order-32-high-order-closure-literal-run-exceeded-wall/BOARD_ATTEMPT_328210_RESULT.json) · result
- [INDEPENDENT_BOARD_RESULT_AUDIT_328210.json](../order-32-high-order-closure-literal-run-exceeded-wall/INDEPENDENT_BOARD_RESULT_AUDIT_328210.json) · result
- [REJECTION_RECEIPT.json](../order-32-high-order-closure-literal-run-exceeded-wall/REJECTION_RECEIPT.json) · result

</details>

## Related work

- ← shipped in: [Submission #328210: full-budget closure with order-24 pair series, raw 5.765125013112993e-7](../../../submissions/328210/README.md)

[Research graph](../../../graph/index.html#node=2026-08-24-pair-series-order-32-reduced-24-final-prediction-disagreement) · [Copy provenance](../../../PROVENANCE.json)

