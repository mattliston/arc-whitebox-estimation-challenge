# Submission #330931: batched six-level products, split frame caches and size-based depth selection passed

2026-09-13T23:13:32Z · submission · adopted · measured

The shared rank-384 estimator used batched six-level push products, separate frame caches, fused pair-polynomial sums and a deterministic size-based Winograd depth schedule. It preserved the moment-closure model while reducing remote requests and graded successfully without a failure penalty. The exact source SHA256 is dd64dfc367fbaf154fe956f335c21aa51c1589e7f19a731be74c29ca41bc29b3.

Recorded measurement:

    Official adjusted 4.188358e-9, raw 2.129570e-8, utilisation0.1967; cold45,118 requests /29.5s /util0.2010; later networks36.9k-39.0k requests; prior best4.539e-9.

Public score: 4.188358306539703e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L707](../../records/SUBMISSION_LEDGER.md#L707)
- [SUBMISSION_LEDGER.md#L708](../../records/SUBMISSION_LEDGER.md#L708)
- [RECEIPTS.md#L59](../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md#L59)

<details>
<summary>All result files and records (2)</summary>

- [RECEIPTS.md](../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md) · record
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record

</details>

## Related work

- ← measured by: [Leaderboard on 2026-09-19: submission #330931 scored 4.188358306539703e-9, ranked 12th; leader about 2.1e-9](../../records/research/2026-09-19-leaderboard-2026-09-19-submission-330931-scored-4-188358306539703e/README.md)
- ← supersedes: [Submission #331584: exact-cost Winograd depth selection and a scalar final readout on the wall-repaired estimator](../331584/README.md)

[Research graph](../../graph/index.html#node=submission-330931) · [Copy provenance](../../PROVENANCE.json)

