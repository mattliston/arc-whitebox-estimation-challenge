# Hidden times gate plus first-128 times hidden: -0.995% held-fold proxy at 961 columns

2026-10-21T19:05:50Z · experiment · adopted · measured

The combined hidden-times-gate and first-128-times-hidden block outperformed the separate block and other tested products on the fixed held fold. Cubes, variance products and additional higher products did not add convincing gain; the 961-column fit became the final feature design.

Recorded measurement:

    - 2026-10-21T19:05:50Z [fable-lead] BLOCKS PRICING 2 (fold0 held-18, vs base705 -0.826): hg -0.882; hg+fh (first128 x hidden2) -0.995 (+0.17 proxy over the fired design); cubes -0.803 (worse); sigma products -0.843; hg+fh+cubes -0.906; all -0.863. Design hg+fh chosen: TRAIN90 fits launched 19:05Z at ridge .03/.01 (fit_blocks_hgfh_r0p03/r0p01); exporter extended (first90 kept, fh_prod term, tensor first_hidden_prod_head).
    - 2026-10-21T19:55:46Z [fable-lead] hg+fh TRAIN90 fit DONE 19:55:10Z (design hg,fh; width 961; ridge .03; training -3.06% 90/90; physical rms .219 max .549 PASS). chain_hgfh exporting x1.5 -> hgfh_r0p03_s15_cap3 (VM32+fresh+Mac32); chain_hgfh_cap4s8 builds hgfh_r0p03_s15_cap4s8e9 on export. Also in flight: hg_r0p03_s15_cap4s8e9 (VM32+fresh, Mac32 queued), early-heavy brackets, teammate cap4+8e9 vs cap3 fire-evidence pair (ETA 20:10Z). Matt: readout -> he decides whether to fire.
    - 2026-10-21T20:21:37Z [fable-lead] BLOCKS PRICING 3 (fold0 held-18): hg+fh -0.995; +hfg -1.011 (noise); +h2f+f2h -0.948 (worse); +state x gate -0.993 (flat); all -0.956 (worse). Readout feature search EXHAUSTED: final design = hg+fh (ridge .03). Remaining arms: strength (early-heavy bracket), cap4+8e9 margin tail, clean twins.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1/on_policy_first128_head90_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/build_parent_sources_v1.py](code/build_parent_sources_v1.py)
- [code/export_blocks_v1.py](code/export_blocks_v1.py)
- [code/fit_blocks_full90_v1.py](code/fit_blocks_full90_v1.py)
- [code/fit_extended_gate_v1.py](code/fit_extended_gate_v1.py)
- [code/fit_second_step_v1.py](code/fit_second_step_v1.py)
- [code/lc_blocks2_fold0_v1.py](code/lc_blocks2_fold0_v1.py)
- [code/lc_blocks3_fold0_v1.py](code/lc_blocks3_fold0_v1.py)
- [code/lc_blocks_fold0_v1.py](code/lc_blocks_fold0_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7530](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7530)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7535](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7535)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7541](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7541)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337619: 961-column on-policy readout, grouped-read cap 4 and 8e9 size threshold](../../../submissions/337619/README.md)
- shipped in: [Submission #337641: Final 961-column readout at strength 1.35 with residual-meter references removed](../../../submissions/337641/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-hidden-times-gate-plus-first-128-times-hidden-0) · [Copy provenance](../../../PROVENANCE.json)

