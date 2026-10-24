# Combined late learned gain, isotropic scale and terminal skew scale: +0.055% adjusted

2026-10-20T10:54:52Z · experiment · rejected · measured

The combined three small scalar changes did not preserve their individual apparent gains. The full-32 adjusted result was +0.055% and held-out 25 result +0.090% at zero FLOP change.

Recorded measurement:

    [fable-lead] 2026-10-20T10:54:52Z mstack5 DEAD (32 ranks vs science32_rel_v4a_m15): all32 adjusted +0.055% (se 0.068, lower 13/32), heldout25 +0.090% (lower 9/25), zero bill. The three remaining flat optima (late GNN gain x0.9, late iso x0.8, terminal k3 x1.007) do not add: even the sub-bar zero-bill option is gone. Nothing to fire at any bar. Held and gated: the safe twin only (private-margin file).
    [fable-lead] 2026-10-20T10:34:23Z SUB-BAR STACK mstack5 BUILT AND UNDER JUDGMENT (lead/runs/fable_oct19_kc_v1/candidate_rel_mstack5.py 168e01d9 release form, candidate_mstack5_sci_v1.py 6a0b49ec science; = #337259 physics + GNN gain x0.90 at L10-14 (gnnglate_m, -0.055% on 7 ranks) + GN iso x0.8 at L10-14 (isol_m, -0.033% on 32) + terminal k3 x1.007 (tk3s parabola, ~-0.1% on 4 nets; tail wrapper of nonlinear_scalar with the dense context re-pinned)). Native smoke net 35: +0.015% vs base (pieces cancel on this net). Lane launched 10:33Z (queue_v8). Expectation -0.1..-0.2% local = BELOW Matt's 0.005e-9 bar; prepared only so a judged zero-bill option exists if he chooses to use a slot.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_rel_mstack5.py](code/candidate_rel_mstack5.py)
- [code/candidate_mstack5_sci_v1.py](code/candidate_mstack5_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6183](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6183)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6181](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6181)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-combined-late-learned-gain-isotropic-scale-terminal-skew-scale) · [Copy provenance](../../../PROVENANCE.json)

