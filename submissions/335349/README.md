# Submission #335349: covariance scratch reuse and cheaper strip and sandwich products

2026-10-11T08:50:06Z · submission · adopted · measured

The same physics as #335340 with persistent covariance-sandwich scratch (XRQS) + direct strip-load implementation (HZD3) + true-inner-size covariance sandwich (XSDM); a second single-network platform spike of the day (residual max 0.3790 s, max/mean 1.212) survived.

Recorded measurement:

    1.567560e-9 (-0.275% vs #335340); final MSE 1.348e-8 (+0.009%); share 0.116245 (-0.290%); 100/100; residual 0.3126 / 0.3790 s; wall max 100.8 s; requests 43,596 / 41,898; official public score 1.567533489997822e-09; official utilisation 0.11624511016007091; failed networks 0

Public score: 1.567533489997822e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [candidate.py](candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1058](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1058)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1063](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1063)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1064](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1064)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1065](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1065)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1070](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1070)
- [SUBMISSION_LEDGER.md#L937](../../records/SUBMISSION_LEDGER.md#L937)
- [SUBMISSION_LEDGER.md#L938](../../records/SUBMISSION_LEDGER.md#L938)
- [SUBMISSION_LEDGER.md#L939](../../records/SUBMISSION_LEDGER.md#L939)
- [SUBMISSION_LEDGER.md#L940](../../records/SUBMISSION_LEDGER.md#L940)

<details>
<summary>All result files and records (6)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [stage_a.json](stage_a.json) · record
- [receipt.json](receipt.json) · record
- [eval_335349.json](eval_335349.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← shipped in: [Persistent sandwich scratch, direct strip loads and true-inner-size products adopted: -0.200/-0.284% FLOPs](../../records/research/2026-10-11-persistent-sandwich-scratch-direct-strip-loads-true-inner-size/README.md)
- ← supersedes: [Submission #335340: ridge-re-sized refit, request caches and seven-level full-packet strip pushes](../335340/README.md)
- supersedes: [Submission #335411: carrier-strength refit, one-direction covariance star and plane-read column cut](../335411/README.md)

[Research graph](../../graph/index.html#node=submission-335349) · [Copy provenance](../../PROVENANCE.json)

