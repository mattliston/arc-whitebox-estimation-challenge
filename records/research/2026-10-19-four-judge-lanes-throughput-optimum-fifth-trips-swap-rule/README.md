# Four judge lanes are the throughput optimum: a fifth trips the swap rule at 35% free, slowing all (3.5 vs 4.4 ranks/min)

2026-10-19T18:10:00Z · lesson · informational · measured

On the 48 GB, 16-core, 96 W-capped box four 32-rank judge lanes are the maximum: a fifth lane pushes memory-pressure free to about 34% and grows swap about 300 MB, which trips the guard's swap rule (free < 35% with +256 MiB swap) even under the 15%-free guard chain and kills a lane mid-run, and five lanes deliver fewer ranks per minute than four because the box is power-bound. In-process probes count as lanes: run them only when judge lanes are at most two.

Recorded measurement:

    5 lanes: 0.7 vs 1.1 ranks/min/lane (3.5 vs 4.4 total); swap growth 302 MB at free 34%; each lane ~5.7 GB steady, 7.3-7.8 GB peaks; later four lanes died on the 1-GiB swap-growth rule beside two or three 6 GB probes

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6003](../../TEAM_LOG_2026-10-01_to_10-22.md#L6003)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-four-judge-lanes-throughput-optimum-fifth-trips-swap-rule) · [Copy provenance](../../../PROVENANCE.json)

