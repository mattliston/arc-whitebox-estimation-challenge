# Submission #336119: same-physics safe fallback for #336101 with two-strip covariance request geometry, +0.64% vs primary

2026-10-14T22:51:52Z · submission · adopted · measured

The same-physics safe variant of #336101 uses the two-strip covariance request geometry and the original safe runtime gates (loss risk under 10%, warm requests under 42,700). It graded 0.64% worse than the primary and was recorded as its designation fallback; it was later accepted as the fallback for #336125 too, avoiding a duplicate safe fire.

Recorded measurement:

    official 1.5148461508077626e-9, +0.644884% vs #336101; paired 50 meanlog +0.638145% (SE 0.050043%); 100/100, 0 failed; safe gates: risk 3.40%, warm 42,418 requests, 3 PB keys coverage approved

Public score: 1.5148461508077626e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3148](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3148)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3163](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3163)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3165](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3165)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3176](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3176)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3204](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3204)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3213](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3213)
- [SUBMISSION_LEDGER.md#L977](../../records/SUBMISSION_LEDGER.md#L977)
- [SUBMISSION_LEDGER.md#L979](../../records/SUBMISSION_LEDGER.md#L979)

<details>
<summary>All result files and records (9)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [receipt.json](receipt.json) · record
- [attempt.json](attempt.json) · record
- [eval_336119.json](eval_336119.json) · result
- [stage_a.json](stage_a.json) · record
- [ROOT_PUBLIC_GRADE_v1.json](ROOT_PUBLIC_GRADE_v1.json) · result
- [FINAL_CONFIRMED_RECEIPT_v1.json](FINAL_CONFIRMED_RECEIPT_v1.json) · record
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Submission #336101: rank-two learned output projection with planner request repair, public score 1.50514e-9](../336101/README.md)
- shipped in: [Submission #336101: rank-two learned output projection with planner request repair, public score 1.50514e-9](../336101/README.md)

[Research graph](../../graph/index.html#node=submission-336119) · [Copy provenance](../../PROVENANCE.json)

