# Submission #335213: refitted learned arrays and jointly re-solved estimator dials

2026-10-10T15:11:29Z · submission · adopted · measured

candidate generation 45 (S45) = candidate generation 44 (S44) + three refit rounds (jmD, jmG without batch 7, joint global-dial and learned-array refit (jmH)): re-fitted arrays and the physics dials re-solved after the arrays moved, mean-correction covariance-fold coefficient (kappa_C) by band; the same parts as S44 with new numbers only; submitted below the 2% bar on the lead's judgement and passed oliver_chen.

Recorded measurement:

    1.586e-9 (-2.64% vs S44); final MSE 1.357e-8; share 0.11688; 100/100; residual max 0.35 s; evidence untouched 32 -1.33% (se 0.31), native ten -1.48% (9 of 10, bill +0.000%), 48 clean -1.75%; official public score 1.586458431739398e-09; official utilisation 0.11688153429446174; failed networks 0

Public score: 1.586458431739398e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [candidate.py](candidate.py)
- [../../records/research/2026-10-10-refit-four-displaced-global-dials-adopted-after-zero-gradient/candidate.py](../../records/research/2026-10-10-refit-four-displaced-global-dials-adopted-after-zero-gradient/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L646](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L646)
- [TEAM_LOG_2026-10-01_to_10-22.md#L648](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L648)
- [TEAM_LOG_2026-10-01_to_10-22.md#L651](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L651)
- [TEAM_LOG_2026-10-01_to_10-22.md#L652](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L652)
- [TEAM_LOG_2026-10-01_to_10-22.md#L654](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L654)
- [TEAM_LOG_2026-10-01_to_10-22.md#L656](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L656)
- [TEAM_LOG_2026-10-01_to_10-22.md#L659](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L659)
- [SUBMISSION_LEDGER.md#L925](../../records/SUBMISSION_LEDGER.md#L925)
- [SUBMISSION_LEDGER.md#L926](../../records/SUBMISSION_LEDGER.md#L926)
- [SUBMISSION_LEDGER.md#L927](../../records/SUBMISSION_LEDGER.md#L927)
- [SUBMISSION_LEDGER.md#L928](../../records/SUBMISSION_LEDGER.md#L928)
- [SUBMISSION_LEDGER.md#L934](../../records/SUBMISSION_LEDGER.md#L934)

<details>
<summary>All result files and records (7)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [stage_a.json](stage_a.json) · record
- [receipt.json](receipt.json) · record
- [eval_335213.json](eval_335213.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [eval_335213-0fc23915.json](eval_335213-0fc23915.json) · result

</details>

## Related work

- ← shipped in: [Refit of four displaced global dials adopted after zero-gradient covariance screens](../../records/research/2026-10-10-refit-four-displaced-global-dials-adopted-after-zero-gradient/README.md)
- ← shipped in: [Untouched generated networks become the primary judge because reused ranks overstate refit gains](../../records/research/2026-10-10-untouched-generated-networks-become-primary-judge-because-reused-ranks/README.md)
- ← supersedes: [Submission #335147: joint correction refit, layer-15 law diagonal and padded covariance cut](../335147/README.md)
- enables: [Submission #335215: lower-FLOP implementation of the refitted estimator; predicted loss risk 16.2%](../335215/README.md)
- supersedes: [Submission #335259: per-layer refit and a gate-floor 1e-3 dead-side push cut](../335259/README.md)

[Research graph](../../graph/index.html#node=submission-335213) · [Copy provenance](../../PROVENANCE.json)

