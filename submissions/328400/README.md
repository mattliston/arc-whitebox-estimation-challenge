# Submission #328400: conditional positive-semidefinite repair timed out to zero output

2026-08-25T13:41:37Z · submission · failed · measured

The W2 card differed from the graded #328326 by exactly a 16-line Cholesky certificate block that was meant to skip the unconditional positive-semidefinite (PSD) eigensolve when the covariance was already positive definite. On the board one private row timed out and entered the zero-output fallback; its truth-power mean squared error (MSE) of about 1.32 divided by 50 rows reproduces both displayed scores, so the submission was graded but useless. The likely cause recorded is repeated failed-certificate attempts (the Cholesky wall on top of the full eigensolve wall) on an already slow hard row. The planned W3 refire was superseded by removing the eigensolve outright (#328439).

Recorded measurement:

    score_adjusted 0.026391339219659114, score_secondary_raw_final 0.02639134407304059; 1.32/50 = 0.0264; attempts authorized 1, consumed 1

Public score: 0.026391339219659114. Status: GRADED. Failed networks: 1.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [run_w2_meter.py](run_w2_meter.py)
- [../../experiments/learned-correction/carrier-host-refit-ceiling-measured-rejected/code/public_condrepair_deployment_card_v3/estimator.py](../../experiments/learned-correction/carrier-host-refit-ceiling-measured-rejected/code/public_condrepair_deployment_card_v3/estimator.py)

## Evidence

- [BOARD_ATTEMPT_328400_RESULT.json#L1](BOARD_ATTEMPT_328400_RESULT.json#L1)
- [SUBMISSION_LEDGER.md#L678](../../records/SUBMISSION_LEDGER.md#L678)

<details>
<summary>All result files and records (3)</summary>

- [BOARD_ATTEMPT_328400_RESULT.json](BOARD_ATTEMPT_328400_RESULT.json) · record
- [W_CLOSURE.json](../../experiments/flop-accounting/conditional-positive-semidefinite-repair-seven-129-rows-missed-frozen/W_CLOSURE.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record

</details>

## Related work

- supersedes: [Submission #328326: transport-energy residual observer refitted on all 688 development networks](../328326/README.md)
- enables: [Submission #328439: unconditional positive-semidefinite eigensolve removed](../328439/README.md)

[Research graph](../../graph/index.html#node=submission-328400) · [Copy provenance](../../PROVENANCE.json)

