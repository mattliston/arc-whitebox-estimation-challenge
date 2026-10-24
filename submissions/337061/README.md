# Submission #337061: level fusion, prebinding refit and two operand box views, graded 1.4427189e-9

2026-10-19T06:26:25Z · submission · adopted · measured

Fired as a verified improvement, not a bank step, to measure the platform transfer of the lazy-bind, prebind and level-fusion cuts and to move the lineage reference: locally -0.11% adjusted on 32 paired ranks vs #337028 (bill -0.10%, raw -0.01%); hosted VALID 100/100 at 1.4427188587106575e-9 with util -0.053%, so the binding cuts transferred at about half.

Recorded measurement:

    local 32 ranks vs #337028: adjusted -0.11% (se .03, 25/32 lower; bill -0.10, raw -0.01); hosted 1.443e-9 displayed (exact 1.4427188587106575e-9), MSE 1.315e-8, util .10968286 (-0.053% vs .1097413); residual public median .3484 / p90 .3595 / max .3654 (private max .3882); predict wall public max 113.8 s, private max 115.4 s; setup walls 4.14-4.26 s; gates setup5 3.30 s, stage A 23 signatures none new; new valid best by ~0.05%

Public score: 1.4427188587106575e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5917](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5917)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5918](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5918)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5921](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5921)
- [SUBMISSION_LEDGER.md#L1102](../../records/SUBMISSION_LEDGER.md#L1102)
- [SUBMISSION_LEDGER.md#L1103](../../records/SUBMISSION_LEDGER.md#L1103)

<details>
<summary>All result files and records (5)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [receipt.json](receipt.json) · record
- [eval_337061.json](eval_337061.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← shipped in: [Kernel-tree box views: one operand route corrupts predictions; the other two save -0.011% FLOPs](../../experiments/arithmetic-kernels/kernel-tree-box-views-one-operand-route-corrupts-predictions/README.md)
- ← shipped in: [Level fusion and count-only prebinding refit: -0.102% adjusted versus #337028 on 32 paired networks](../../experiments/flop-accounting/level-fusion-count-only-prebinding-refit-0-102-adjusted/README.md)
- supersedes: [Submission #337028: landmark-symmetrized Nystrom join sketch, graded 1.4435396e-9](../337028/README.md)
- ← supersedes: [Submission #337065: join sketch columns 1536/1024/1024 on #337061, graded 1.4390249e-9 (-0.252%, full bill transfer)](../337065/README.md)

[Research graph](../../graph/index.html#node=submission-337061) · [Copy provenance](../../PROVENANCE.json)

