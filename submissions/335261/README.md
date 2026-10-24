# Submission #335261: kernel-price variant of the same estimator gives no FLOP advantage

2026-10-10T22:25:38Z · submission · adopted · measured

The engineer's s46p form (safe physics + kernel prices OVR@kp10 + push row cut) submitted as the twin after the release twin crashed; the safe file's zero planner prices already hold the kernel-price trade, so it graded level with the safe file.

Recorded measurement:

    1.581e-9; share 0.11680; 100/100; wall 79.42 / 86.60 s; residual 0.31 / 0.36 s; P(loss) 2.06% at 43,977 requests; official public score 1.581165431246144e-09; official utilisation 0.11679674392580637; failed networks 0

Public score: 1.581165431246144e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [candidate.py](candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L725](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L725)
- [TEAM_LOG_2026-10-01_to_10-22.md#L726](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L726)
- [TEAM_LOG_2026-10-01_to_10-22.md#L728](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L728)
- [SUBMISSION_LEDGER.md#L930](../../records/SUBMISSION_LEDGER.md#L930)
- [SUBMISSION_LEDGER.md#L932](../../records/SUBMISSION_LEDGER.md#L932)
- [SUBMISSION_LEDGER.md#L934](../../records/SUBMISSION_LEDGER.md#L934)

<details>
<summary>All result files and records (7)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [stage_a.json](stage_a.json) · record
- [receipt.json](receipt.json) · record
- [eval_335261.json](eval_335261.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [eval_335261-83914a81.json](eval_335261-83914a81.json) · result

</details>

## Related work

- ← enables: [Release variant fails stage C because a fourth-order implementation lacks the per-layer switch table](../../records/research/2026-10-10-release-variant-fails-stage-c-because-fourth-order-implementation/README.md)
- ← enables: [Submission #335259: per-layer refit and a gate-floor 1e-3 dead-side push cut](../335259/README.md)

[Research graph](../../graph/index.html#node=submission-335261) · [Copy provenance](../../PROVENANCE.json)

