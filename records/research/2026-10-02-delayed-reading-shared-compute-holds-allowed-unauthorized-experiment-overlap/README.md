# Delayed reading of shared compute holds allowed unauthorized experiment overlap

2026-10-02T18:05Z · incident · informational · measured

A continuing attribution chain consumed 25 minutes of memory after a hold order, and six native sessions began after another lane had committed to stop. Launchers and polling rules were changed to read new coordinator instructions before each start; later a 30-minute implementation delay was traced to acknowledging an unread instruction.

Recorded measurement:

    attribution runs17:43-18:02 after17:37 hold,25minutes,2cores,3-4GB; six native sessions18:01-18:06,one killed after30s; head-input implementation delayed30minutes

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L410](../../TEAM_LOG_2026-10-01_to_10-22.md#L410)
- [TEAM_LOG_2026-10-01_to_10-22.md#L411](../../TEAM_LOG_2026-10-01_to_10-22.md#L411)
- [TEAM_LOG_2026-10-01_to_10-22.md#L433](../../TEAM_LOG_2026-10-01_to_10-22.md#L433)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [RESULT.md](../../../experiments/arithmetic-kernels/half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-delayed-reading-shared-compute-holds-allowed-unauthorized-experiment-overlap) · [Copy provenance](../../../PROVENANCE.json)

