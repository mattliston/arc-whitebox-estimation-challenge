# Page-fault handling adds approximately 14,200 kernel instructions per minor fault

2026-10-01T19:41Z · finding · informational · derived

Instruction counts repeat to +-0.5% between clean runs but a run that pages reads 5-8% high because kernel-mode fault handling is counted. Runs with system/user CPU above 0.30 are rejected and repeated, and the projection laws later subtract 14,200 instructions per minor fault; this was the cause of every window-dependent projection miss.

Recorded measurement:

    paging runs: system/user CPU 0.42-0.67 instead of 0.13-0.21, instructions +5..+8%; threshold system/user 0.30; 14,200 instructions per minor fault

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L22](../../TEAM_LOG_2026-10-01_to_10-22.md#L22)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-page-fault-handling-adds-approximately-14-200-kernel-instructions) · [Copy provenance](../../../PROVENANCE.json)

