# Strength 1.28 and late-layer strength 1.0 tie the final source within measured Linux variation

2026-10-21T23:04:39Z · experiment · inconclusive · measured

The additional Linux-only strength bracket improved by amounts recorded as noise-level and lacked a completed Mac and fresh confirmation before close. Mixed early/late strengths, partially completed scalar variants and the two-dial combination did not establish a better source.

Recorded measurement:

    - 2026-10-21T23:04:39Z [fable-lead] STRENGTH BRACKET VM32 vs 49bd (clean cap4+8e9): x1.28 all32 -0.595 held25 -0.697 raw -0.881 bill +0.289; x1.42 all32 -0.528 held25 -0.549 raw -0.812. Full VM32 ladder all32: x1.05 -0.502, x1.2 -0.503, x1.28 -0.595, x1.35 -0.572, x1.42 -0.528, x1.5 -0.542, x1.65 -0.481: flat-topped 1.28-1.35 with ~0.03-0.05 scatter; x1.28 vs x1.35 = 0.023 = noise-level, NOT a confirmed improvement (no Mac32/fresh possible before close) -> NOT a fire; 2 shots held. Remaining arms (late1, gsk10, bfix10, hg074, s15e12l, teammate k4l34 up/dn, np_dn+k4lc_up) score as they land, stop 23:35Z.
    - 2026-10-21T23:17:54Z [fable-lead] hgfh_s135late1_c4s8e9_clean (x1.35 with late rows x1.0) VM32 vs 49bd: all32 -0.585 held25 -0.667 raw -0.872 bill +0.289 vs x1.35 -0.572/-0.654 raw -0.859: +0.013 = noise-level; NOT a fire. Three brackets (x1.28 -0.595, x1.42 -0.528, late1 -0.585) confirm a flat top around x1.3; #337641 remains the file. #337641 hosted: 35/100 nets clean, raw +0.066% util -0.18% vs #337619 (adj ~-0.11%), walls 99/108 s, residual .34/.36. hg074 arm retired 23:09Z (1/32). Remaining: s15e12l 19/32, gsk10 14, bfix10 12, teammate k4l34 late up 26 / dn 19, np_dn+k4lc_up 23; scorer stop 23:35Z.
    - 2026-10-21T23:27:19Z [fable-lead] BRACKET SCORER DONE (VM32 all32/held25 raw vs 49bd; x1.35 = -0.572/-0.654 raw -0.859): s15e12l (x1.5 early / x1.2 late) -0.562/-0.611 raw -0.846 (not better); gsk10 (x1.35 + _GSK 1.0) -0.569/-0.675 raw -0.855 PARTIAL (~16-20 ranks, tie); bfix10 (x1.35 + _BFIX_GAMMA 0.10) -0.193/-0.386 PARTIAL (much worse: gamma 0.10 hurts); hg074 retired. Best measured file = x1.28 (-0.595/-0.697), then late1 (-0.585/-0.667); both gated/gating for the two remaining shots, awaiting Matt.
    - 2026-10-21T23:28:42Z [vm_screamer via lead] f21 dial runs complete 32/32 (VM32 vs 49bd, bill 0): k4l34_late_up +0.060 (se .058); k4l34_late_dn -0.046 (se .075); np_dn+k4lc_up two-dial stack -0.112 (se .088, below the -0.24 sum of parts). None near the readout stack; plan unchanged. late1 arm GATED: tag fable_hgfh_s135late1_c4s8e9_clean_govstrip_27f6dc37_v1 archive f4d2be7f3bd3. PROJECTION of the two re-rolls vs #337641 1.4001e-9: x1.28 ~1.3999e-9 (VM32 -0.023/-0.043 x ~0.5 transfer), late1 ~1.4000e-9: ties; value = hosted draw only (~25% each of landing >=0.05% better).
    Sweep result detail: - 2026-10-21T23:22:57Z | f21_k4l34_late_up | _K4L34_BY_LAYER = {8: (0.5, 0.5), 9: (0.5, 0.5), 10: (0.75, 0.75), 11: (0.75, 0.75), 12: (0.75, 0.75), 13: (0.75, 0.75), 14: (0.75, 0.75), 15: (0.5, 0.5)} | ref first128_paid_nocensus_native32_v1 | all32 adj +0.060% (se 0.058) | held25 adj +0.072% (se 0.072) | wins 17/32 | raw +0.060% | bill +0.000% | requests +0.00% | puts +0.00% | put_bytes_f4 +0.00% | wall +62.11% | backend +61.77% | residual +50.09% | max-rank wall +83.70% | slow10 wall +52.52% req +0.00% adj +0.173% | slow10ctl ratio 1.525 | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_k4l34_late_up | sha256 b001584b0603bcb3265609fdbd756dd75c3a895aff8adbb7d80ac05ed57718ad | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_k4l34_late_up.py
    - 2026-10-21T23:25:48Z | f21_k4l34_late_dn | _K4L34_BY_LAYER = {8: (0.5, 0.5), 9: (0.5, 0.5), 10: (0.25, 0.25), 11: (0.25, 0.25), 12: (0.25, 0.25), 13: (0.25, 0.25), 14: (0.25, 0.25), 15: (0.5, 0.5)} | ref first128_paid_nocensus_native32_v1 | all32 adj -0.046% (se 0.075) | held25 adj +0.032% (se 0.085) | wins 17/32 | raw -0.046% | bill +0.000% | requests +0.00% | puts +0.00% | put_bytes_f4 +0.00% | wall +54.67% | backend +54.54% | residual +43.55% | max-rank wall +91.98% | slow10 wall +52.48% req +0.00% adj -0.085% | slow10ctl ratio 1.525 | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_k4l34_late_dn | sha256 a29d5b3e1149a4003256573a0ac43914d8e7ae927fd74a2ddadb76c12600997b | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_k4l34_late_dn.py
    - 2026-10-21T23:25:51Z | f21_np_dn_k4lc_up | _NP_STRENGTH = (0.0, 1.7187586508966153, 1.3864997166584256, 1.200298293039114, 1.2469612883973824, 1.1716862057997373, 1.1716862057997373, 1.1716862057997373,  | ref first128_paid_nocensus_native32_v1 | all32 adj -0.112% (se 0.088) | held25 adj -0.097% (se 0.099) | wins 15/32 | raw -0.112% | bill -0.000% | requests +0.00% | puts +0.00% | put_bytes_f4 +0.00% | wall +57.85% | backend +57.59% | residual +44.88% | max-rank wall +89.70% | slow10 wall +52.56% req +0.00% adj -0.120% | slow10ctl ratio 1.526 | sm40 calls 6.125 vs 6.125 | byte-identical n (0/32) | VM run /home/arcworker/arc-runs/f21_np_dn_k4lc_up | sha256 e3f7a2e9184da7a2c5a21e3e49c0fb8b8cb67cef5e26f28f0fbfefb11ee4e745 | /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/cand_np_dn_k4lc_up.py

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cand_k4l34_late_up.py](code/cand_k4l34_late_up.py)
- [code/cand_k4l34_late_dn.py](code/cand_k4l34_late_dn.py)
- [code/cand_np_dn_k4lc_up.py](code/cand_np_dn_k4lc_up.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7587](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7587)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7588](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7588)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7590](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7590)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7591](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7591)
- [VM_SWEEP_RESULTS_2026-10-21.md#L110](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L110)
- [VM_SWEEP_RESULTS_2026-10-21.md#L111](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L111)
- [VM_SWEEP_RESULTS_2026-10-21.md#L112](../../../records/VM_SWEEP_RESULTS_2026-10-21.md#L112)

<details>
<summary>All result files and records (6)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [validate.json](validate.json) · result
- [stage_a.json](stage_a.json) · result
- [package.json](package.json) · result
- [validate_package.json](validate_package.json) · result
- [VM_SWEEP_RESULTS_2026-10-21.md](../../../records/VM_SWEEP_RESULTS_2026-10-21.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-strength-1-28-late-layer-strength-1-0-tie) · [Copy provenance](../../../PROVENANCE.json)

