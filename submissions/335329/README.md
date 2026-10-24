# Submission #335329 fails: network 14 exceeds the residual cap at 0.4083 s

2026-10-11T05:47:13Z · submission · failed · measured

The candidate generation 46 (S46) physics with the request diet and seven-level strips submitted as a board test of the twin; one ordinary warm network ran ~1.2-1.3x slow end to end and crossed the 0.4 s residual cap, the zero-prediction fallback ruined the score; the twin carried ~1,400 more first-network requests than the safe file.

Recorded measurement:

    residual mean 0.3171 / p90 0.3435 / max 0.4083 s (max/mean 1.288 vs 1.12-1.14 before); network 14 wall 94.6 s vs mean 81.4; requests 44,875 / 43,469; score 0.01703; official public score 0.01702924167468891; official utilisation 0.11651876239759985; failed networks 1

Public score: 0.01702924167468891. Status: GRADED. Failed networks: 1.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [candidate.py](candidate.py)
- [post_s46x.py](post_s46x.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1009](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1009)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1010](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1010)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1012](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1012)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1013](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1013)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1023](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1023)
- [SUBMISSION_LEDGER.md#L933](../../records/SUBMISSION_LEDGER.md#L933)
- [SUBMISSION_LEDGER.md#L934](../../records/SUBMISSION_LEDGER.md#L934)

<details>
<summary>All result files and records (6)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [stage_a.json](stage_a.json) · record
- [receipt.json](receipt.json) · record
- [eval_335329.json](eval_335329.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← shipped in: [Request caches and seven-level strip pushes adopted: byte-identical request savings and about -0.18% FLOPs](../../records/research/2026-10-11-request-caches-seven-level-strip-pushes-adopted-byte-identical/README.md)
- ← enables: [Submission #335259: per-layer refit and a gate-floor 1e-3 dead-side push cut](../335259/README.md)
- enables: [Residual-time model has a heavy tail; retain free request reductions without buying costly margin](../../records/research/2026-10-11-residual-time-model-has-heavy-tail-retain-free-request/README.md)

[Research graph](../../graph/index.html#node=submission-335329) · [Copy provenance](../../PROVENANCE.json)

