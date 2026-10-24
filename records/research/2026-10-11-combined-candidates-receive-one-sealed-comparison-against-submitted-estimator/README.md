# Combined candidates receive one sealed comparison against the submitted estimator before promotion

2026-10-11T17:20Z · lesson · informational · measured

Parts individually validated on fresh networks get one look on a sealed block of 32 against the submitted file (score = MSE change + metered FLOPs change, pooled); negative is submitted as the safe file, since pairing with the previous file cannot lower the expected private score.

Recorded measurement:

    S48 sealed 7100328-359: score -0.580% pooled (mean log -0.550%, se 0.310, 18 of 32), MSE +0.244%, bill -0.822%

## Programs

- [judge_sealed_v1.py](judge_sealed_v1.py)
- [make_s48_v1.py](make_s48_v1.py)
- [candidate.py](candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1127](../../TEAM_LOG_2026-10-01_to_10-22.md#L1127)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1131](../../TEAM_LOG_2026-10-01_to_10-22.md#L1131)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1142](../../TEAM_LOG_2026-10-01_to_10-22.md#L1142)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1149](../../TEAM_LOG_2026-10-01_to_10-22.md#L1149)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1169](../../TEAM_LOG_2026-10-01_to_10-22.md#L1169)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1171](../../TEAM_LOG_2026-10-01_to_10-22.md#L1171)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Untouched generated networks become the primary judge because reused ranks overstate refit gains](../2026-10-10-untouched-generated-networks-become-primary-judge-because-reused-ranks/README.md)
- enables: [Combined-candidate workflow applies verified changes, seals on fresh networks and runs a fast release gate](../2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/README.md)
- shipped in: [Submission #335411: carrier-strength refit, one-direction covariance star and plane-read column cut](../../../submissions/335411/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-combined-candidates-receive-one-sealed-comparison-against-submitted-estimator) · [Copy provenance](../../../PROVENANCE.json)

