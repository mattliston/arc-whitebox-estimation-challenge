# Remaining float64 conversion: ideal 0.0373% saving before casts, no 0.15% low-risk patch

2026-10-15T17:58:20Z · finding · rejected · measured

The current source already removed float64 zero fills and dense fourth-order covariance work. Converting the remaining mean-preserving output matvec could not produce the requested gain.

Recorded measurement:

    historical debit1.448G/.5595%; already removes164.8M; output projection F64 virtual53.8M; ideal.0373%

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3455](../../TEAM_LOG_2026-10-01_to_10-22.md#L3455)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SOURCE_COST_FINDING_v1.json](SOURCE_COST_FINDING_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-15-remaining-float64-conversion-ideal-0-0373-saving-before-casts) · [Copy provenance](../../../PROVENANCE.json)

