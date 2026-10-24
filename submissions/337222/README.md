# Submission #337222: combined dials and gated kurtosis; one network exceeds residual limit at 0.411 s

2026-10-19T23:28:56Z · submission · failed · measured

Micro-stack v3 (v2 plus the GK4 coefficient x1.04; local -0.336% adjusted on 32 ranks, 21/32 lower, held-out -0.32%) lost public network 4 (a cold, generation-0 network) to a residual of 0.411 s against the 0.4 s cap on a slow worker pool, so its public score is the penalized 0.0204. Its request and flop profile is identical to #337212 (which had max .367 the same day): the margin law (about 11% margin, per-network noise sd 5%, p95 +10%) means every fire of the lineage carried a 20-30% public-failure risk.

Recorded measurement:

    public score 0.020395828697101034, 1 failed network; net 4 residual 0.411 s > 0.4; 12/100 networks > .38, p90 .381 (vs #337212 max .367); local 32 ranks -0.336% adjusted (21/32 lower, held-out -0.32%); wall max 117.331 s

Public score: 0.020395828697101034. Status: GRADED. Failed networks: 1.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6063](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6063)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6071](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6071)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6079](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6079)
- [SUBMISSION_LEDGER.md#L1116](../../records/SUBMISSION_LEDGER.md#L1116)
- [SUBMISSION_LEDGER.md#L1119](../../records/SUBMISSION_LEDGER.md#L1119)

<details>
<summary>All result files and records (6)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [receipt.json](receipt.json) · record
- [eval_337222.json](eval_337222.json) · result
- [RESULT32_v1.json](RESULT32_v1.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=submission-337222) · [Copy provenance](../../PROVENANCE.json)

