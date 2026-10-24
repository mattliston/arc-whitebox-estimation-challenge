# Exact fast-multiplication audits (rank-26, rank-63 3x4x7, rank-48 grouped frames): all under 0.5% of the bill

2026-10-17T02:46:15Z · finding · rejected · derived

The Hopcroft-Kerr rank-26 tensors for 2x4x4 products were parsed and verified (exact integer tensors for 244/442/424) but the current-shape zero-outer ceiling is 0.159731% of the whole bill; the rank-63 3x4x7 scheme over all six permutations saves at most 0.486740% (0.398379% mean) on the matched 8 against 0.758707% unpeeled, below the 0.5% threshold; rank-48 grouped existing frames bound only 0.0031%. No kernel was built.

Recorded measurement:

    HK26 ceiling 0.159731% whole bill; rank63 saved matched8 current 750-cap mean 0.398379%, max 0.486740%, vs 0.758707% unpeeled; rank48 .0031%.

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4572](../../TEAM_LOG_2026-10-01_to_10-22.md#L4572)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4574](../../TEAM_LOG_2026-10-01_to_10-22.md#L4574)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4760](../../TEAM_LOG_2026-10-01_to_10-22.md#L4760)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4772](../../TEAM_LOG_2026-10-01_to_10-22.md#L4772)

<details>
<summary>All result files and records (4)</summary>

- [VERIFY_v1.json](VERIFY_v1.json) · record
- [CLOSURE_v1.json](CLOSURE_v1.json) · record
- [CEILING_v1.json](CEILING_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-exact-fast-multiplication-audits-rank-26-rank-63-3x4x7) · [Copy provenance](../../../PROVENANCE.json)

