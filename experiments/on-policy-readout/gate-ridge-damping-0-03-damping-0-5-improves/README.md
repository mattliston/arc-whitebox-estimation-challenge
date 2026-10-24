# Gate ridge and damping: 0.03 with damping 0.5 improves proxy fit but misses the submission threshold

2026-10-21T17:45:49Z · experiment · rejected · measured

Lower ridge with half damping improved cross-validation, while removing damping overfit. Native cap-3 and cap-4 exports improved score but did not meet the current-best improvement requirement; higher output strength was worse.

Recorded measurement:

    - 2026-10-21T17:45:49Z [fable-lead] RIDGE: gate fit r0.03/d0.5 CV -0.912% (89/90, phys 0.189) PASS (better than r0.1 -0.836); r0.3/d0.5 CV -0.750% PASS (worse); r0.03/d1.0 CLOSED. Exporting r0.03 x1.5 + cap3 (extgate_r003_s15_cap3); iter2 refit at ridge 0.03 launched.
    - 2026-10-21T17:34:06Z [fable-lead] ridge/damping gate fit r0.03/d1.0: CLOSED (CV -0.337%, 61/90; undamped overfits). r0.03/d0.5 and r0.3/d0.5 pending. iter2 fit running on the 90 fresh captures.
    - 2026-10-21T18:24:37Z [fable-lead] LEARNING CURVE (lc_gate_fold0_v1, gate 705-col fit, fold0 held-18): n_train 18 +0.076%, 36 -0.502%, 54 -0.683%, 72 -0.826% (18/18 lower). 1/n asymptote ~-1.15%; projected -0.95% at 115 nets -> <= +0.05% hosted: held25 capture NOT worth a shot (teammate redirected to cheaper wall-margin levers: size-cap thresholds, slow-10 walls). Ridge .03 stacks VM32: s15_cap3 all32 -0.253% held25 -0.450% raw -0.615% bill +0.364%, fresh12 -0.461% (8/12); s20_cap3 all32 -0.036% held25 -0.299%, fresh12 -0.222% (7/12). Next: Mac32 r003_s15_cap3; build r003_s15_cap4 (endgame gun).
    - 2026-10-21T18:36:41Z [fable-lead] Mac32 extgate_r003_s15_cap3 (ridge .03 gate x1.5 + cap3) vs 49bd: all32 -0.210% (SE 0.30, 16/32), held25 -0.417%, raw -0.572%, bill +0.364%. With VM32 -0.253/-0.450 and fresh12 -0.461: pooled ~-0.29% (held25-pooled ~-0.44%) vs the fired file's -0.18/-0.36 -> expected hosted ~1.402e-9 at the same cap3 margin. NOT a bank (bar 1.3986). Mac32 r003_s15_cap4 running; blocks pricing (prev-hidden, hidden x gate) on VM.
    - 2026-10-21T18:44:26Z [fable-lead] extgate_r003_s15_cap4 all three sets vs 49bd: Mac32 -0.292% (held25 -0.497), VM32 -0.326% (held25 -0.484, raw -0.511, bill +0.186), fresh12 -0.647% (8/12); pooled -0.36%, held25-pooled -0.52% -> expected hosted ~1.4012e-9 at cap4 margin (slowest ~112.6 s hosted = failure zone). r003_s15_cap3: pooled -0.27%, held25-pooled -0.44% -> ~1.4025e-9 at the 337565 margin. BOTH GATED: archives 44617c8011b1 (cap3, tag fable_extgate_r003_s15_cap3_2973434b_v1) and df9314b4a0ef (cap4, tag fable_extgate_r003_s15_cap4_d702ed9d_v1); FIRE_extgate_r003_s15_cap{3,4}.sh prepared, NOT fired (below the 0.005e-9 bar). BLOCKS PRICING (fold0 held-18): base -0.826, +prev-hidden -0.779 (worse), +hidden x gate -0.882 (+0.06 proxy), both -0.836: pv DEAD; hg TRAIN90 fits launched at ridge .03/.01 (fit_blocks_hg_r0p03/r0p01). Teammate: 5.5e9 size cap under matched load = wall -5.6% for bill +0.435% (worse than cap3); 8e9/7e9 pairs running.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1/on_policy_first128_head90_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fit_extended_gate_rd_v1.py](code/fit_extended_gate_rd_v1.py)
- [code/build_parent_sources_v1.py](code/build_parent_sources_v1.py)
- [code/fit_second_step_v1.py](code/fit_second_step_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7510](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7510)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7502](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7502)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7523](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7523)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7525](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7525)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7526](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7526)

<details>
<summary>All result files and records (9)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [validate.json](validate.json) · result
- [stage_a.json](stage_a.json) · result
- [package.json](package.json) · result
- [validate_package.json](validate_package.json) · result
- [validate-beff11f4.json](validate-beff11f4.json) · result
- [stage_a-21083d17.json](stage_a-21083d17.json) · result
- [package-ac3ef52c.json](package-ac3ef52c.json) · result
- [validate_package-4432a149.json](validate_package-4432a149.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-gate-ridge-damping-0-03-damping-0-5-improves) · [Copy provenance](../../../PROVENANCE.json)

