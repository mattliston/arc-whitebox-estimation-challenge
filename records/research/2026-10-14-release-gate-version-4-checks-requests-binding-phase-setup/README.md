# Release gate version 4 checks requests, binding phase, setup duration and worst predicted wall time

2026-10-14T02:46:27Z · tool · informational · measured

Every candidate passed release_gate_v4 before firing: it prices the first and warm per-network request counts at 7.35 us a request into a P(loss) per 100 networks (central and fat-tailed, also under a 1.5% slower worker), checks the binding phase at the platform pace (requests and share of keys bound), the projected setup wall (cap 5 s) and the projected worst-network wall, and a 1.3x slower setup. The same-physics safe designation read compares the safe file's native ten networks against the fired twin's native session for MSE, requests and bill.

Recorded measurement:

    example (#335954 safe build): first/warm 43,202/41,843 requests; P(loss) 2.8% central / 2.9% fat per 100 networks, 3.4%/3.5% under 1.5% slower; binding 17,819 requests, 594 of 649 keys bound (limit <= 18,600, >= 90%); setup 4.313 s (limit <= 4.5 s); worst wall 92.9 s (mean 86.1 s, mean residual 0.302 s; limit <= 100 s); requests under 1.3x slower setup 42,712 (+0.45%, limit <= +2%)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2541](../../TEAM_LOG_2026-10-01_to_10-22.md#L2541)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2542](../../TEAM_LOG_2026-10-01_to_10-22.md#L2542)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2555](../../TEAM_LOG_2026-10-01_to_10-22.md#L2555)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2571](../../TEAM_LOG_2026-10-01_to_10-22.md#L2571)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2716](../../TEAM_LOG_2026-10-01_to_10-22.md#L2716)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2725](../../TEAM_LOG_2026-10-01_to_10-22.md#L2725)

<details>
<summary>All result files and records (4)</summary>

- [quick_s56k1s_repair1r1_a.json](quick_s56k1s_repair1r1_a.json) · result
- [quick_s56k1s_astra_v1r1_a.json](quick_s56k1s_astra_v1r1_a.json) · result
- [validate.json](validate.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-release-gate-version-4-checks-requests-binding-phase-setup) · [Copy provenance](../../../PROVENANCE.json)

