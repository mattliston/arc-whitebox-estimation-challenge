# Submission #328273: order-24 closure with hybrid variance-floor repair, raw 9.232750883825246e-8

2026-08-24T18:01:07Z · submission · adopted · measured

The order-24 Hermite pair-series closure (Q24) closure with the left-associated rho recurrence and the first ('hybrid floor') variance-floor repair of the late-layer blow-ups graded successfully on the 50 public networks. It cut the raw final mean squared error (MSE) of #328210 by about six times by removing the tail rows and became the pre-repair incumbent, still far above the raw < 1e-8 goal.

Recorded measurement:

    score_adjusted_mse 8.93370759014933e-08, score_raw_mse 9.232750883825246e-08, 50/50 public MLPs graded

Public score: 8.93370759014933e-08. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [verify_candidate.py](verify_candidate.py)
- [../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py](../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py)
- [../../experiments/wall-and-residual/complete-connected-state-shutdown-residual-0-4026352697983384-s-exceeded/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_v1/estimator.py](../../experiments/wall-and-residual/complete-connected-state-shutdown-residual-0-4026352697983384-s-exceeded/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_v1/estimator.py)
- [../328210/estimator.py](../328210/estimator.py)

## Evidence

- [BOARD_RESULT_328273.json#L1](BOARD_RESULT_328273.json#L1)

<details>
<summary>All result files and records (2)</summary>

- [BOARD_RESULT_328273.json](BOARD_RESULT_328273.json) · record
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record

</details>

## Related work

- supersedes: [Submission #328210: full-budget closure with order-24 pair series, raw 5.765125013112993e-7](../328210/README.md)
- ← supersedes: [Submission #328277: variance-floor trigger widened to twice the floor, raw 8.758639268080515e-8](../328277/README.md)

[Research graph](../../graph/index.html#node=submission-328273) · [Copy provenance](../../PROVENANCE.json)

