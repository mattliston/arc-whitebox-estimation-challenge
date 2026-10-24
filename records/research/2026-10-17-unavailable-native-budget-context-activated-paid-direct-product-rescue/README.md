# Unavailable native budget context activated paid direct-product rescue on every network

2026-10-17T09:17:33Z · incident · failed · measured

The runtime controller interpreted an unavailable native SDK context as a reason to rescue every case. A native official tiny reproduced the bug and verified a dual context getter; the repair used an inactive fallback for missing or invalid meter readings.

Recorded measurement:

    100/100 with zero failures; score 1.5984331261131061e-9; all 100 bills +7.77–9.35%, median 8.55%.

## Programs

- [../../../submissions/336741/estimator.py](../../../submissions/336741/estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4861](../../TEAM_LOG_2026-10-01_to_10-22.md#L4861)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4874](../../TEAM_LOG_2026-10-01_to_10-22.md#L4874)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-unavailable-native-budget-context-activated-paid-direct-product-rescue) · [Copy provenance](../../../PROVENANCE.json)

