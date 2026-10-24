# Terminal scalar closure with exact inputs lies within 6% of the Monte Carlo noise floor

2026-10-20T04:10:05Z · finding · informational · measured

With exact terminal mean, variance and third/fourth cumulants, the first-order scalar closure produced 2.06 versus 1.94e-9 MSE on network 35. Second-order terms were 100x smaller and a terminal learned correction rescale was worth only -0.06..-0.2% per network.

Recorded measurement:

    - [fable-lead] 2026-10-20T04:10:05Z TERMINAL CLOSURE CLOSED natively: with exact mu/v/k3/k4 at l=15 the card's first-order formula is within 6% of the Monte Carlo noise floor (2.06 vs 1.94e-9 on net 35; second-order terms 100x smaller); the terminal learned correction (gnn_step l=15) is worth ~6% of MSE, its LS rescale -0.06..-0.2% per net. RUNNING: terminal readout refit (head[15]) on 36 nets (native_head_probe_v1 -> fit_head15_v1), judge lanes: capacity arms lcr_512/slr_512/uvp_384 + queued rel_v4a, pe34_late_p/m, pe3/pe4_late_p, gnng15_p/m, gnnglate_p/m (all on the v4a base; bank any verified negative per Matt 06:xxZ). Box is 48 GB: 4 lanes + 2 probes thrash; driver paused until lanes finish.

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6129](../../TEAM_LOG_2026-10-01_to_10-22.md#L6129)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-terminal-scalar-closure-exact-inputs-lies-within-6-monte) · [Copy provenance](../../../PROVENANCE.json)

