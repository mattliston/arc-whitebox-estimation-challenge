# Timeout model from 1,000 networks supports a worst expected roundtrip near 105 seconds

2026-09-23T13:28Z · finding · informational · measured

Observed platform excursions informed a conservative designation rule. Stalls outside predict were distinguished from ordinary execution-time variation.

Recorded measurement:

    - 2026-09-23 06:28 analyst: TIMEOUT RISK MODEL (10 eval pages, 1000 MLPs, JSON only): in-predict excursions p99 +8.7 s, p99.9 +13.5 s, max +13.8 s; stalls outside predict 3/1000, all J2W w-3 (66/71/76), roundtrips pinned 120.24-120.28 s = cut by the kill (stall >= 10.2 s, maybe a hang). P(>=1 timeout/100): S6b +0 0.03%, +3.4 (S7) 0.14%, +5 0.29%, +10 2.6% (a hang episode would add a ~10%/run design-independent floor); J2W profile 76% at +0 (observed: lost 3). Model calibrated: 0.16 predicted vs 0 observed over 9 stall-free runs. Designation rule: worst expected roundtrip <= ~105 s. RESULT.md sec 24.

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L232](../../TEAM_LOG_2026-09-22_to_09-26.md#L232)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-timeout-model-1-000-networks-supports-worst-expected-roundtrip) · [Copy provenance](../../../PROVENANCE.json)

