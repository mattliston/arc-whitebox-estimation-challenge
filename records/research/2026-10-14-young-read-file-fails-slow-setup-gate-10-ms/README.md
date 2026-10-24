# Young-read file fails the slow setup gate by 10 ms (4.610 s vs 4.600 s); repaired by a setup constant before firing

2026-10-14T23:25:26Z · incident · adopted · measured

The young-read production run stopped because its projected setup under the 1.3x-slow model exceeded the internal 4.6 s gate by ten milliseconds. A setup-only repair (one constant set to 3.97) kept all prebind evidence and passed, after which the file was fired.

Recorded measurement:

    slow projected setup 4.610 s > 4.600 s (by 10 ms); normal production risk 14.22%, warm 45,691 requests, projected setup 4.469 s

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3188](../../TEAM_LOG_2026-10-01_to_10-22.md#L3188)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3196](../../TEAM_LOG_2026-10-01_to_10-22.md#L3196)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3200](../../TEAM_LOG_2026-10-01_to_10-22.md#L3200)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3207](../../TEAM_LOG_2026-10-01_to_10-22.md#L3207)

<details>
<summary>All result files and records (4)</summary>

- [ASTRA_BOUNDARY_OWNER_RETIREMENT_v1.json](ASTRA_BOUNDARY_OWNER_RETIREMENT_v1.json) · record
- [BOUNDARY_COLLECTED_slow_v2.json](BOUNDARY_COLLECTED_slow_v2.json) · result
- [FINAL_RELEASE_CONFIRMED_v1.json](FINAL_RELEASE_CONFIRMED_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Submission #336125: young plane read at contraction depth 7 with exact setup prebinding on #336101, 1.50435e-9](../../../submissions/336125/README.md)
- shipped in: [Submission #336125: young plane read at contraction depth 7 with exact setup prebinding on #336101, 1.50435e-9](../../../submissions/336125/README.md)

[Research graph](../../../graph/index.html#node=2026-10-14-young-read-file-fails-slow-setup-gate-10-ms) · [Copy provenance](../../../PROVENANCE.json)

