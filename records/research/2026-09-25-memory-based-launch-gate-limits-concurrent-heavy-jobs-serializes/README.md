# Memory-based launch gate limits concurrent heavy jobs and serializes GPU work

2026-09-25T04:50Z · decision · informational · measured

Heavy jobs require available memory above their own measured peak plus reserve and fewer than four existing heavy processes. GPU work is serialized.

Recorded measurement:

    - 2026-09-24 21:50 lead: MACHINE RULE (all agents, effective now; binding constraint = MEMORY, Matt: "use the machine, just don't OOM it"). Before ANY heavy launch (card worker, refit, VALID40, MPS), including auto-launchers:
      (1) count heavy python processes on the box (RSS > 3 GB, all agents): launch only if fewer than 4 are running;
      (2) require free+inactive >= (the job's own peak RSS + 4 GB) AND memory_pressure free >= 35%;
      (3) MPS: one job at a time (the engineer holds it for E-68(c) now).
      If blocked, wait-and-retry every 60 s; don't skip the gate. At 21:50: 4 heavy workers (covhunter gn_joint_fit 6.8 GB, analyst l11_oracle 6.0 GB, two .venv workers 5.2/5.5 GB), 80% free, free+inactive 18.8 GB. The round-2 auto-launches (physicist refit ~8.4 GB peak, S22 chain, covhunter VALID40) must pass this gate.

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L671](../../TEAM_LOG_2026-09-22_to_09-26.md#L671)
- [TEAM_LOG_2026-09-22_to_09-26.md#L672](../../TEAM_LOG_2026-09-22_to_09-26.md#L672)
- [TEAM_LOG_2026-09-22_to_09-26.md#L673](../../TEAM_LOG_2026-09-22_to_09-26.md#L673)
- [TEAM_LOG_2026-09-22_to_09-26.md#L674](../../TEAM_LOG_2026-09-22_to_09-26.md#L674)
- [TEAM_LOG_2026-09-22_to_09-26.md#L675](../../TEAM_LOG_2026-09-22_to_09-26.md#L675)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-memory-based-launch-gate-limits-concurrent-heavy-jobs-serializes) · [Copy provenance](../../../PROVENANCE.json)

