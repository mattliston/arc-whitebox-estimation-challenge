# Submission #328433: pair-series trim to order 7 and output clamp

2026-08-25T18:00:14Z · submission · adopted · measured

The first utilisation cut on the banked card: the Mehler tail of the pair series was trimmed at term 7 (the 't7 trim', measured output effect 5e-5) and an output nonnegativity clamp added, with the raw prediction otherwise unchanged. It graded 5.13% better adjusted than #328326 at the same raw mean squared error (MSE) and was fired under standing authority plus the explicit budget-cut sanction.

Recorded measurement:

    score 2.7012413010271763e-08 (adjusted; -5.13% vs 2.8472e-8), score_secondary raw 2.940918140126314e-08 (unchanged); trim -109.1e9 FLOPs; 4 of 10 daily submissions used

Public score: 2.7012413010271763e-08. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [build_estimator_trim.py](build_estimator_trim.py)
- [run_trim_meter.py](run_trim_meter.py)
- [../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py](../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py)
- [../328326/estimator.py](../328326/estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L682](../../records/SUBMISSION_LEDGER.md#L682)
- [SUBMIT_WATCH.log#L1](SUBMIT_WATCH.log#L1)
- [SUBMISSION_LEDGER.md#L677](../../records/SUBMISSION_LEDGER.md#L677)

<details>
<summary>All result files and records (2)</summary>

- [SUBMIT_WATCH.log](SUBMIT_WATCH.log) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record

</details>

## Related work

- supersedes: [Submission #328326: transport-energy residual observer refitted on all 688 development networks](../328326/README.md)
- ← supersedes: [Submission #328439: unconditional positive-semidefinite eigensolve removed](../328439/README.md)

[Research graph](../../graph/index.html#node=submission-328433) · [Copy provenance](../../PROVENANCE.json)

