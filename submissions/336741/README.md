# Submission #336741: native residual accessor with inactive fallback: two late-session residual failures

2026-10-17T14:54:11Z · submission · failed · measured

The controller read the native SDK budget accessor and treated unavailable or invalid metering as no rescue. Rescue stayed inactive and the public score improved, but hidden networks 85 and 89 failed at .404/.413 s; end-of-prediction collection growth was investigated.

Recorded measurement:

    Official public score 1.4725444779344095e-09; failed networks 2; utilization 0.11212354285476067; wall mean/max 97.309/108.626 s; residual mean/max 0.362/0.413 s.

Public score: 1.4725444779344095e-09. Status: GRADED. Failed networks: 2.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4874](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4874)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4875](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4875)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4876](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4876)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4877](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4877)
- [SUBMISSION_LEDGER.md#L1065](../../records/SUBMISSION_LEDGER.md#L1065)
- [SUBMISSION_LEDGER.md#L1066](../../records/SUBMISSION_LEDGER.md#L1066)
- [SUBMISSION_LEDGER.md#L1067](../../records/SUBMISSION_LEDGER.md#L1067)
- [SUBMISSION_LEDGER.md#L1068](../../records/SUBMISSION_LEDGER.md#L1068)

<details>
<summary>All result files and records (10)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · result
- [eval_336741.json](eval_336741.json) · result
- [response.json](response.json) · result
- [attempt.json](attempt.json) · result
- [eval_summary.txt](eval_summary.txt) · record
- [stage_a.json](stage_a.json) · record
- [package.json](package.json) · record

</details>

## Related work

- ← shipped in: [Charged end scans grew with worker age; per-prediction freezing kept scans at 0.8–2.9 ms](../../records/research/2026-10-17-charged-end-scans-grew-worker-age-per-prediction-freezing/README.md)

[Research graph](../../graph/index.html#node=submission-336741) · [Copy provenance](../../PROVENANCE.json)

