# 961-column readout with cap 4 and size threshold: -0.620% Linux and -0.423% Mac before accounting repair

2026-10-21T20:25:43Z · experiment · adopted · measured

The 961-column readout with grouped-read cap and matrix-size threshold improved both fixed native panels and the fresh confirmation set. These reported sources still used scatter-for-gather; the subsequent submitted sources disabled it and paid the additional measured FLOPs.

Recorded measurement:

    - 2026-10-21T20:25:43Z [fable-lead] hgfh_r0p03_s15_cap4s8e9 (hg+fh seal x1.5 + cap4 + size8e9) VM32 vs 49bd: all32 -0.620% (23/32) held25 -0.678% raw -0.826% bill +0.13; fresh12 -0.637% (7/12). vs #337565: VM32 -0.41/-0.35, fresh -0.37 -> ~-0.37% vs best = ~1.3984e-9 (AT the 1.3986 bar; clean twin +0.077% bill -> ~1.3994). hgfh_r0p03_s15_cap3: VM32 -0.438/-0.465 raw -0.804; fresh12 -0.573 -> ~-0.2% vs best (~1.401). Mac32s: hgfh cap3 done (see chain log), cap4s8e9 running, hgfh cap4s8e9 next; clean twins + gates queued.
    - 2026-10-21T20:25:56Z [fable-lead] Mac32 hgfh_r0p03_s15_cap3 vs 49bd: all32 -0.353% (18/32) held25 -0.494% raw -0.721% bill +0.371% -> vs #337565: -0.25/-0.10. hgfh cap3 pooled ~-0.25% all / -0.17% held vs best (~1.401e-9); hgfh cap4s8e9 pooled (VM32+fresh) ~-0.37% (~1.3984e-9), Mac32 pending.
    - 2026-10-21T20:38:30Z [fable-lead] Mac32 hgfh_r0p03_s15_cap4s8e9 vs 49bd: all32 -0.423% (21/32) held25 -0.524% raw -0.632% bill +0.211% -> vs #337565 -0.32/-0.13; pooled all ~-0.37% / held ~-0.26% -> clean twin ~1.400e-9. CLEAN TWINS GATED: hgfh_r0p03_s15_cap3_clean archive 1966e86ecaf8 (tag ..._4952fa73_v1), hgfh_r0p03_s15_cap4s8e9_clean archive 55a93ef002c2 (tag fable_hgfh_r0p03_s15_cap4s8e9_clean_f66cb28b_v1; local setup envelope 3.54 s). #337614 FAILED on SETUP_TIMEOUT (4 workers 4.21-4.25 s SUCCESS, one >5 s; not counted against quota); hosted setup record: 337506 4.10-4.26, 337534 4.11-4.66, 337565 4.12-4.26 -> ~20% per-fire risk at this profile; setup = ~14.8k requests (prebind + views), no safe quick trim. Matt: [instruction removed]; fire8_clean.sh armed.
    - 2026-10-21T20:29:18Z [fable-lead] Mac32 hg_r0p03_s15_cap4s8e9 vs 49bd: all32 -0.412% held25 -0.526% raw -0.617% bill +0.206% -> vs #337565 -0.31/-0.13; pooled with VM32 (-0.28/-0.23) and fresh (-0.17): ~-0.26% all / -0.18% held -> ~1.400e-9 XSG-on, ~1.4012 clean. Matt: [instruction removed]). Built + launched hgfh_c4s8_k4lc105_clean (hg+fh seal x1.5 + cap4 + 8e9 + _K4L_C x1.05 + XSG off): VM32+fresh running; Mac32+gates queued (k4lc_mac32_gates.sh) behind the clean twins.
    - 2026-10-21T20:25:03Z [fable-lead] hg_r0p03_s15_cap4s8e9 (hidden x gate seal x1.5 + cap4 + size8e9): VM32 vs 49bd all32 -0.496% (21/32) held25 -0.556% raw -0.701% bill +0.206% requests +0.765% puts -0.079%; fresh12 -0.437% (7/12, bill +0.13). vs #337565's own numbers: VM32 -0.28/-0.23, fresh -0.17 -> ~-0.25% vs best (~1.400e-9) at the cap4+8e9 margin. Mac32 running (serial v2); hgfh cap3 / cap4s8e9 VM32 scorers imminent.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/stacks/hgfh_r0p03_s15_cap3/candidate_hgfh_r0p03_s15_cap3_science_v1.py](code/stacks/hgfh_r0p03_s15_cap3/candidate_hgfh_r0p03_s15_cap3_science_v1.py)
- [code/stacks/hgfh_r0p03_s15_cap4s8e9/candidate_hgfh_r0p03_s15_cap4s8e9_science_v1.py](code/stacks/hgfh_r0p03_s15_cap4s8e9/candidate_hgfh_r0p03_s15_cap4s8e9_science_v1.py)
- [code/tails/sm40_cap4_size8e9.py](code/tails/sm40_cap4_size8e9.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7544](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7544)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7545](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7545)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7548](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7548)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7547](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7547)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7543](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7543)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337619: 961-column on-policy readout, grouped-read cap 4 and 8e9 size threshold](../../../submissions/337619/README.md)
- shipped in: [Submission #337641: Final 961-column readout at strength 1.35 with residual-meter references removed](../../../submissions/337641/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-961-column-readout-cap-4-size-threshold-0-620) · [Copy provenance](../../../PROVENANCE.json)

