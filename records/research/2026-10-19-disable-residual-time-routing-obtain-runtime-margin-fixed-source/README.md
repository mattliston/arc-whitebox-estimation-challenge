# Disable residual-time routing and obtain runtime margin from fixed source profiles

2026-10-19T02:18:16Z · lesson · informational · measured

When a two-strip governor keyed on the SDK's live residual meter was proposed, the owner ruled it out: the October governor line (rescue tiers, fast rescue) lost fires and never demonstrably saved a network. Residual margin must come from static request profiles (fewer requests by construction), not runtime switching; an authorized fire runs only the mandatory gates (setup5, stage A, prefire, five starts) and fires. Orphaned python jobs from dead sessions may be killed when they hold memory.

Recorded measurement:

    #336687 rescue-all +8.5% bill; 27 orphaned processes killed on 10-19: memory_pressure free 49% -> 76%

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5887](../../TEAM_LOG_2026-10-01_to_10-22.md#L5887)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-disable-residual-time-routing-obtain-runtime-margin-fixed-source) · [Copy provenance](../../../PROVENANCE.json)

