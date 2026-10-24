# Hidden times gate features improve native score; ridge 0.01 is worse than 0.03

2026-10-21T18:58:53Z · experiment · adopted · measured

A hidden-times-gate feature block improved the Linux panel and slightly improved the Mac overall mean at ridge 0.03. The lower ridge did not transfer better to native predictions.

Recorded measurement:

    - 2026-10-21T18:58:53Z [fable-lead] Mac32 hg_r0p03_s15_cap3 (hidden x gate block, ridge .03, x1.5, cap3) vs 49bd: all32 -0.261% (SE 0.33, 16/32), held25 -0.403%, raw -0.626%, bill +0.367% (new step override runs clean, no bill cost). vs r003_s15_cap3 Mac32 -0.210/-0.417: +0.05% all32, held25 flat. VM32+fresh pending; Mac32 hg_r0p01 next; gates queued (gates_hg.sh).
    - 2026-10-21T18:54:56Z [fable-lead] Matt: [instruction removed]; push the hidden x gate (hg) stack; keep the VM hunting. hg TRAIN90 fits done (r.03 train -2.64%, r.01 -2.86%, physical rms .20/.22 pass); exported x1.5 via export_blocks_v1.py (new override: gate90 once, sq/gate readouts, no prev-hidden), stacks hg_r0p03_s15_cap3 / hg_r0p01_s15_cap3 staged + VM32/fresh launched 18:49Z, Mac32 chain running (chain_hg.sh). Per-layer strength brackets of the hg r.03 seal built: late2 (rows 9-15 x2.0) and late1 (rows 9-15 x1.0), VM32 launchers paused 4 min for memory then resumed. Teammate: 7e9 size pair killed (70 GB), launches held until GO; dial hunt (np/k4band/k4lc/k4band3/np_late/ynt14/ott) queued after.
    - 2026-10-21T19:07:39Z [fable-lead] VM32 hg_r0p03_s15_cap3 vs 49bd: all32 -0.351% (18/32), held25 -0.422%, raw -0.715%, bill +0.367% = vs #337565's own VM32 (-0.212/-0.330): +0.14%/+0.09% better. hgfh fits crashed on a helper name (with_prev) -> patched, relaunched 2026-10-21T19:07:39Z.
    - 2026-10-21T19:08:09Z [fable-lead] Mac32 hg_r0p01_s15_cap3 vs 49bd: all32 -0.215%, held25 -0.324% (worse than hg_r0p03 -0.261/-0.403 and r003 -0.210/-0.417): ridge .01 does not transfer natively; ridge .03 stays. GATES CLEAR hg_r0p03_s15_cap3: archive 08e372eef257 (tag fable_hg_r0p03_s15_cap3_7ba1da27_v1). Exporter self-test with the fh block passed; chain_hgfh launched (fits relaunched 19:08Z after the with_prev crash).
    Sweep result detail: - 2026-10-21T18:44:26Z [fable-lead] extgate_r003_s15_cap4 all three sets vs 49bd: Mac32 -0.292% (held25 -0.497), VM32 -0.326% (held25 -0.484, raw -0.511, bill +0.186), fresh12 -0.647% (8/12); pooled -0.36%, held25-pooled -0.52% -> expected hosted ~1.4012e-9 at cap4 margin (slowest ~112.6 s hosted = failure zone). r003_s15_cap3: pooled -0.27%, held25-pooled -0.44% -> ~1.4025e-9 at the 337565 margin. BOTH GATED: archives 44617c8011b1 (cap3, tag fable_extgate_r003_s15_cap3_2973434b_v1) and df9314b4a0ef (cap4, tag fable_extgate_r003_s15_cap4_d702ed9d_v1); FIRE_extgate_r003_s15_cap{3,4}.sh prepared, NOT fired (below the 0.005e-9 bar). BLOCKS PRICING (fold0 held-18): base -0.826, +prev-hidden -0.779 (worse), +hidden x gate -0.882 (+0.06 proxy), both -0.836: pv DEAD; hg TRAIN90 fits launched at ridge .03/.01 (fit_blocks_hg_r0p03/r0p01). Teammate: 5.5e9 size cap under matched load = wall -5.6% for bill +0.435% (worse than cap3); 8e9/7e9 pairs running.
    - 2026-10-21T18:58:53Z [fable-lead] Mac32 hg_r0p03_s15_cap3 (hidden x gate block, ridge .03, x1.5, cap3) vs 49bd: all32 -0.261% (SE 0.33, 16/32), held25 -0.403%, raw -0.626%, bill +0.367% (new step override runs clean, no bill cost). vs r003_s15_cap3 Mac32 -0.210/-0.417: +0.05% all32, held25 flat. VM32+fresh pending; Mac32 hg_r0p01 next; gates queued (gates_hg.sh).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1/on_policy_first128_head90_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/export_blocks_v1.py](code/export_blocks_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7529](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7529)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7527](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7527)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7531](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7531)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7532](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7532)
- [VM_SWEEP_RESULTS_2026-10-21.md#L71](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L71)
- [VM_SWEEP_RESULTS_2026-10-21.md#L76](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L76)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [VM_SWEEP_RESULTS_2026-10-21.md](../../../records/VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

## Related work

- shipped in: [Submission #337614: Hidden-times-gate readout and grouped-read cap 3; setup timeout](../../../submissions/337614/README.md)
- shipped in: [Submission #337617: Hidden-times-gate readout and cap 3 with direct gather](../../../submissions/337617/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-hidden-times-gate-features-improve-native-score-ridge-0) · [Copy provenance](../../../PROVENANCE.json)

