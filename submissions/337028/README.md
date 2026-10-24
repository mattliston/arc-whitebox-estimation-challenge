# Submission #337028: landmark-symmetrized Nystrom join sketch, graded 1.4435396e-9

2026-10-19T02:19:59Z · submission · adopted · measured

The held bundle 6e4c5b87 (the #336935 physics plus gate33, centered22, Qcore reuse, pad18 and a Nystrom landmark symmetrization of the join sketch, JSK) was fired unchanged, without prebind refit or governor, on the owner's instruction. It graded VALID 100/100 at exactly 1.4435396311622371e-9, -0.2292% vs #336935, the first 0.005e-9 bank step; the local seven-rank forecast had been -0.2486% adjusted.

Recorded measurement:

    exact 1.4435396311622371e-9 (-0.2292% vs #336935 1.4468565569762618e-9); public final MSE 1.315e-8, util .1097413 (vs #336935 1.316e-8, .1099204); paired telemetry vs #336935 flops med x0.998, residual med x0.991 (max x1.052), kernel med x0.998; residual median .3484 / p90 .3575 / max .3693; roundtrip max 110.9 s (vs 114.9); local 7 ranks vs 95c: adjusted -0.2486% (se .12, 5/7 lower; raw -0.109%, bill -0.140%); gates setup5 ~3.34 s of 5, stage A archive 4c2ddc441dde (23 einsum signatures, none new)

Public score: 1.4435396311622371e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5887](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5887)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5893](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5893)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5894](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5894)
- [SUBMISSION_LEDGER.md#L1099](../../records/SUBMISSION_LEDGER.md#L1099)
- [SUBMISSION_LEDGER.md#L1100](../../records/SUBMISSION_LEDGER.md#L1100)
- [SUBMISSION_LEDGER.md#L1101](../../records/SUBMISSION_LEDGER.md#L1101)

<details>
<summary>All result files and records (5)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [receipt.json](receipt.json) · record
- [eval_337028.json](eval_337028.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- supersedes: [Submission #336935: compact setup repair: 100 networks passed at 1.4468565569762618e-09](../336935/README.md)
- ← supersedes: [Submission #337061: level fusion, prebinding refit and two operand box views, graded 1.4427189e-9](../337061/README.md)

[Research graph](../../graph/index.html#node=submission-337028) · [Copy provenance](../../PROVENANCE.json)

