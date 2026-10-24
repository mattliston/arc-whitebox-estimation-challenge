# Submission #336506: uncapped strided reads with residual routing: 11 networks failed

2026-10-16T13:38:29Z · submission · failed · measured

The uncapped grouped strided-read grid and residual-time routing used thresholds 0.40/0.42/0.46 s. The controller did not rescue any network because its FULL threshold had no margin; 11 networks failed.

Recorded measurement:

    Official public score 0.16383566261855953; failed networks 11; utilization 0.11180469280993748; wall mean/max 101.736/116.586 s; residual mean/max 0.374/0.471 s.

Public score: 0.16383566261855953. Status: GRADED. Failed networks: 11.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4228](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4228)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4236](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4236)
- [SUBMISSION_LEDGER.md#L1029](../../records/SUBMISSION_LEDGER.md#L1029)

<details>
<summary>All result files and records (10)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · result
- [eval_336506.json](eval_336506.json) · result
- [eval_336506-71a8dfce.json](eval_336506-71a8dfce.json) · result
- [response.json](response.json) · result
- [attempt.json](attempt.json) · result
- [eval_summary.txt](eval_summary.txt) · record
- [eval_336506_partial_1421Z.json](eval_336506_partial_1421Z.json) · result

</details>

## Related work

- ← shipped in: [Public-fire runtime thresholds optimized for expected leaderboard improvement per slot](../../records/research/2026-10-16-public-fire-runtime-thresholds-optimized-expected-leaderboard-improvement-per/README.md)

[Research graph](../../graph/index.html#node=submission-336506) · [Copy provenance](../../PROVENANCE.json)

