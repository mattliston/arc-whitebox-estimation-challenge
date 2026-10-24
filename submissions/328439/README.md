# Submission #328439: unconditional positive-semidefinite eigensolve removed

2026-08-25T18:31:27Z · submission · adopted · measured

The unconditional positive-semidefinite (PSD)-repair eigensolve block of the covariance update was removed from the trim card. The target-free attribution had shown that the eigensolve was a numerical identity on about 92% of calls and that even on rows with 2 to 12 clipped eigenvalues its output effect was 1.7e-5 to 3.3e-5 of the error scale, with nothing downstream factorising on it and the floor arm intact. The bill fell by about 300e9 FLOPs per row at unchanged raw, giving -15.4% adjusted versus #328433 and -19.8% for the day.

Recorded measurement:

    score 2.2842888692472472e-08 adjusted (-15.4% vs 328433), score_secondary raw 2.9409159800763974e-08 (unchanged); eigensolve identity on ~92% of calls, median output effect 1.861434569e-5 of error scale; day total 2.847e-8 to 2.284e-8 (-19.8%); 5 of 10 daily submissions used

Public score: 2.2842888692472472e-08. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L686](../../records/SUBMISSION_LEDGER.md#L686)
- [SUBMIT_WATCH.log#L1](SUBMIT_WATCH.log#L1)
- [V_ATTRIBUTION_RECEIPT.json#L1](V_ATTRIBUTION_RECEIPT.json#L1)
- [SUBMISSION_LEDGER.md#L677](../../records/SUBMISSION_LEDGER.md#L677)

<details>
<summary>All result files and records (3)</summary>

- [SUBMIT_WATCH.log](SUBMIT_WATCH.log) · log
- [V_ATTRIBUTION_RECEIPT.json](V_ATTRIBUTION_RECEIPT.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record

</details>

## Related work

- ← enables: [Submission #328400: conditional positive-semidefinite repair timed out to zero output](../328400/README.md)
- supersedes: [Submission #328433: pair-series trim to order 7 and output clamp](../328433/README.md)

[Research graph](../../graph/index.html#node=submission-328439) · [Copy provenance](../../PROVENANCE.json)

