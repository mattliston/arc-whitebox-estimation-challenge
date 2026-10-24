# Submission #328277: variance-floor trigger widened to twice the floor, raw 8.758639268080515e-8

2026-08-24T19:20:38Z · submission · adopted · measured

The scalar-minimal repair of the order-24 Hermite pair-series closure (Q24) closure (the floor equality test changed to variance <= 2 x floor, source 26aca6ed) passed its literal meter and package audit and graded successfully. It improved #328273 by about 5.13% in both raw and adjusted terms and became the pure-closure incumbent on which the learned residual observer was then fitted.

Recorded measurement:

    score_adjusted_mse 8.475125442816834e-08, score_raw_mse 8.758639268080515e-08 (about 5.133% adjusted and 5.135% raw better than #328273); literal meter 2,126,977,459,133 FLOPs, residual 0.3976605720818043 s, zero failures; package SHA-256 af7e6282...

Public score: 8.475125442816834e-08. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py](../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py)
- [../328273/estimator.py](../328273/estimator.py)
- [../../experiments/flop-accounting/twice-floor-variance-repair-legal-meter-2-126-977/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/verify_tiny.py](../../experiments/flop-accounting/twice-floor-variance-repair-legal-meter-2-126-977/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/verify_tiny.py)
- [../../experiments/flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_v1/estimator.py](../../experiments/flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_v1/estimator.py)
- [../../experiments/flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_raw_diag_load_rescale_v1/estimator.py](../../experiments/flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_raw_diag_load_rescale_v1/estimator.py)
- [../../instruments/literal-meter-harness-subprocess-evaluation-run-records-verification-resource/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py](../../instruments/literal-meter-harness-subprocess-evaluation-run-records-verification-resource/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/run_synthetic_meter.py)
- [../../instruments/literal-meter-harness-subprocess-evaluation-run-records-verification-resource/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py](../../instruments/literal-meter-harness-subprocess-evaluation-run-records-verification-resource/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/meter_protocol/verify_meter_run.py)

## Evidence

- [BOARD_RESULT_328277.json#L1](BOARD_RESULT_328277.json#L1)
- [FREEZE_RECEIPT.json#L1](../../experiments/flop-accounting/twice-floor-variance-repair-legal-meter-2-126-977/FREEZE_RECEIPT.json#L1)

<details>
<summary>All result files and records (4)</summary>

- [BOARD_RESULT_328277.json](BOARD_RESULT_328277.json) · record
- [FREEZE_RECEIPT.json](../../experiments/flop-accounting/twice-floor-variance-repair-legal-meter-2-126-977/FREEZE_RECEIPT.json) · record
- [receipt.json](../../experiments/flop-accounting/twice-floor-variance-repair-legal-meter-2-126-977/receipt.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record

</details>

## Related work

- shipped in: [Variance-floor equality missed three tiny positive excursions; a twice-floor threshold repaired them](../../experiments/covariance-structure/variance-floor-equality-missed-three-tiny-positive-excursions-twice/README.md)
- supersedes: [Submission #328273: order-24 closure with hybrid variance-floor repair, raw 9.232750883825246e-8](../328273/README.md)
- ← supersedes: [Submission #328317: native-trajectory residual observer on the order-24 closure](../328317/README.md)

[Research graph](../../graph/index.html#node=submission-328277) · [Copy provenance](../../PROVENANCE.json)

