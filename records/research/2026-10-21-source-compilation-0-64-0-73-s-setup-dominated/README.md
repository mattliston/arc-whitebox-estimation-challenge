# Source compilation is 0.64-0.73 s; setup is dominated by roughly 14.8k requests

2026-10-21T20:42:28Z · finding · informational · measured

Setup profiling attributed most time to prebinding, views and remaining request work rather than source compilation. The recorded per-submission timeout risk was a working estimate, not a measured probability or a validated trim.

Recorded measurement:

    - 2026-10-21T20:42:28Z [fable-lead] Setup anatomy: compile of the 3.1 MB source = 0.64-0.73 s on the loaded VM (not the lever); setup = ~14.8k requests (prebind 128 frames ~1.0 s local, views/peel/smallsm12/generic ~0.9 s, unlogged tail ~0.9 s, constants 0.13 s); hosted 4.1-4.3 s/worker with +20% spikes -> ~20% setup-timeout risk per fire; no trim validated. Mitigation: refire_watch_337619.sh re-fires the same gated archive once if #337619 fails at setup (Matt: 'get it graded successfully ... that is your priority'); FIRE_refire_hgfh_cap4s8e9_clean.sh prepared. Matt away ~20:50-21:35Z; last two shots HELD.
    - 2026-10-21T20:38:30Z [fable-lead] Mac32 hgfh_r0p03_s15_cap4s8e9 vs 49bd: all32 -0.423% (21/32) held25 -0.524% raw -0.632% bill +0.211% -> vs #337565 -0.32/-0.13; pooled all ~-0.37% / held ~-0.26% -> clean twin ~1.400e-9. CLEAN TWINS GATED: hgfh_r0p03_s15_cap3_clean archive 1966e86ecaf8 (tag ..._4952fa73_v1), hgfh_r0p03_s15_cap4s8e9_clean archive 55a93ef002c2 (tag fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1; local setup envelope 3.54 s). #337614 FAILED on SETUP_TIMEOUT (4 workers 4.21-4.25 s SUCCESS, one >5 s; not counted against quota); hosted setup record: 337506 4.10-4.26, 337534 4.11-4.66, 337565 4.12-4.26 -> ~20% per-fire risk at this profile; setup = ~14.8k requests (prebind + views), no safe quick trim. Matt: fire the hgfh cap4s8e9 clean twin ASAP as the 7th counted shot; fire8_clean.sh armed.

## Programs

- [FIRE_refire_hgfh_cap4s8e9_clean.sh](FIRE_refire_hgfh_cap4s8e9_clean.sh)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7550](../../TEAM_LOG_2026-10-01_to_10-22.md#L7550)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7548](../../TEAM_LOG_2026-10-01_to_10-22.md#L7548)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-source-compilation-0-64-0-73-s-setup-dominated) · [Copy provenance](../../../PROVENANCE.json)

