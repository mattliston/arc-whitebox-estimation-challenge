# Submission #330924: four-level factor propagation and square depth cap 4 receive a failure-penalized score

2026-09-13T21:45:43Z · submission · failed · measured

The shared rank-384 estimator replaced serial six-level Winograd push products with packed four-level products and capped square-product depth at four. Local production walls of 35.6-36.4s did not prevent hosted failures, and the ledger recorded adjusted 0.17235479 with the previous best unchanged. The failure motivated a census of synchronous remote requests.

Recorded measurement:

    Local final MSE 1.8389e-8 versus 1.8410e-8; bill +6.3%; local production wall35.6-36.4s; graded 0.17235479 failure-penalized.

Public score: 0.17235479462369463. Status: GRADED. Failed networks: 31.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L705](../../records/SUBMISSION_LEDGER.md#L705)
- [SUBMISSION_LEDGER.md#L706](../../records/SUBMISSION_LEDGER.md#L706)

<details>
<summary>All result files and records (2)</summary>

- [RECEIPTS.md](../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md) · record
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record

</details>

[Research graph](../../graph/index.html#node=submission-330924) · [Copy provenance](../../PROVENANCE.json)

