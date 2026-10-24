# Stale swap-usage threshold blocked training despite 24.4 GB available memory

2026-09-25T03:42Z · incident · informational · measured

The launcher used historical swap occupancy rather than active memory pressure. It was replaced with available-memory and swapout-rate checks.

Recorded measurement:

    - 2026-09-24 20:42 [physicist] loopR1 launcher STALLED ~15 min on its swap<4.8GB gate (MPS-era rule): swap 7.69 of 9.2 GB used but STALE (0 swapouts/pageouts over 20 s), RAM 83% free, 24.4 GB available. Relaunched same run (s6b_loopR1) with the standing team gate (available >= 8 GB, memory_pressure free >= 35%) and a watchdog: pause if available < 3 GB or swapouts > 50k pages/min (covhunter flagged the stall).

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L618](../../TEAM_LOG_2026-09-22_to_09-26.md#L618)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-stale-swap-usage-threshold-blocked-training-despite-24-4) · [Copy provenance](../../../PROVENANCE.json)

