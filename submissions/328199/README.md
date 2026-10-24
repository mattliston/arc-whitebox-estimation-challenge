# Submission #328199: full-budget closure with order-32 pair series exceeded the 120 s wall

2026-08-24T09:10:02Z · submission · failed · measured

The first board attempt of the full-budget third-cumulant repeated-index plane (K21) plus coherent-quartic closure (the V4 estimator with the Q=32 Hermite/rho pair series, archive fb-ec4-32-v4-3ddddd87) passed the grader's smoke test but was reported as 'Error while scoring your submission'. The independent audit of the board telemetry reconciled the counts: 2 networks completed, 12 failed with explicit TIME_EXHAUSTED and 86 failed without telemetry; the two completed rows ran at a participant predict wall of about 119 s against the 120 s cap. The attempt was consumed and no aggregate score exists; the Q=24 variant (about 14% less wall) was fired 1.5 hours later as #328210.

Recorded measurement:

    completed rows: natasha-heath final_layer_mse 7.661e-08 (predict wall 119.41478967799998 s, backend 116.965546986 s, flops_used 2,179,759,731,685), david-webb 7.071e-08 (118.92704962300002 s); explicit time exhaustions e.g. alicia-levy roundtrip 125.30999673600002 s; counts completed 2 / failed 98 / TIME_EXHAUSTED 12 / no telemetry 86

Public score: None. Status: FAILED. Failed networks: 12.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [BOARD_ATTEMPT_328199_RESULT.json#L1](../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/BOARD_ATTEMPT_328199_RESULT.json#L1)
- [INDEPENDENT_BOARD_RESULT_AUDIT.json#L1](../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/INDEPENDENT_BOARD_RESULT_AUDIT.json#L1)

<details>
<summary>All result files and records (4)</summary>

- [BOARD_ATTEMPT_328199_RESULT.json](../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/BOARD_ATTEMPT_328199_RESULT.json) · record
- [INDEPENDENT_BOARD_RESULT_AUDIT.json](../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/INDEPENDENT_BOARD_RESULT_AUDIT.json) · record
- [BOARD_EXPLORATORY_PREREG_V1.json](../../experiments/wall-and-residual/order-32-high-order-closure-literal-run-exceeded-wall/BOARD_EXPLORATORY_PREREG_V1.json) · record
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record

</details>

## Related work

- enables: [Submission #328210: full-budget closure with order-24 pair series, raw 5.765125013112993e-7](../328210/README.md)

[Research graph](../../graph/index.html#node=submission-328199) · [Copy provenance](../../PROVENANCE.json)

