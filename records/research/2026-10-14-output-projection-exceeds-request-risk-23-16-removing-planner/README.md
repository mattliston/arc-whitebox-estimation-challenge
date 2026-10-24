# Output projection exceeds request risk 23.16%; removing a planner limit reduces it to 13.38%

2026-10-14T21:20:14Z · incident · adopted · measured

The first production composition of the output projection on the current file (017) failed the primary request-count gate. A canonical repair that only removed the LVFS limit (f043) kept all ten native networks byte-identical in prediction with equal or lower bill and passed the normal and slow gates, so the earlier science and seal evidence transferred to it and it was fired.

Recorded measurement:

    017 primary count gate FAIL: risk 23.16% > 20%; f043 repair: risk 13.38%, first/warm 47,403/45,630 requests, setup 4.312 s, worst 93.5 s; native ten: all 10 x 16 predictions byte-identical to 017, bill 8 equal / 2 lower

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3091](../../TEAM_LOG_2026-10-01_to_10-22.md#L3091)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3093](../../TEAM_LOG_2026-10-01_to_10-22.md#L3093)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3099](../../TEAM_LOG_2026-10-01_to_10-22.md#L3099)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3104](../../TEAM_LOG_2026-10-01_to_10-22.md#L3104)
- [SUBMISSION_LEDGER.md#L973](../../SUBMISSION_LEDGER.md#L973)

<details>
<summary>All result files and records (5)</summary>

- [BOUNDARY_COMPLETE_HELD_v2.json](BOUNDARY_COMPLETE_HELD_v2.json) · result
- [BOUNDARY_COLLECTED_normal_v2.json](BOUNDARY_COLLECTED_normal_v2.json) · result
- [BOUNDARY_COLLECTED_slow_v2.json](BOUNDARY_COLLECTED_slow_v2.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SUBMISSION_LEDGER.md](../../SUBMISSION_LEDGER.md) · record

</details>

## Related work

- enables: [Submission #336101: rank-two learned output projection with planner request repair, public score 1.50514e-9](../../../submissions/336101/README.md)
- shipped in: [Submission #336101: rank-two learned output projection with planner request repair, public score 1.50514e-9](../../../submissions/336101/README.md)

[Research graph](../../../graph/index.html#node=2026-10-14-output-projection-exceeds-request-risk-23-16-removing-planner) · [Copy provenance](../../../PROVENANCE.json)

