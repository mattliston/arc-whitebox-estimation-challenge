# Final readout strength 1.35 improves Mac and Linux panels; strengths 1.05 through 1.65 give a broad optimum

2026-10-21T22:33:35Z · experiment · adopted · measured

The 961-column correction with direct gather, four reads and the size threshold was evaluated across output strengths. Strength 1.35 had the strongest combined fixed-panel evidence, although its fresh-set result was weaker and the hosted score later tied the 1.5 source.

Recorded measurement:

    - 2026-10-21T22:33:35Z [fable-lead] STRENGTH LADDER complete (hg+fh seal, clean cap4+8e9, vs 49bd): VM32 x1.05 -0.502/-0.553 raw -0.787; x1.2 -0.503/-0.543 raw -0.789; x1.35 -0.572/-0.654 raw -0.859 (PEAK); x1.5 -0.542/-0.600 raw -0.826; x1.65 -0.481/-0.527 raw -0.766. Mac32: x1.2 -0.485/-0.639, x1.35 -0.538/-0.649 (peak), x1.5 -0.345/-0.446. Fresh12 (noisy): x1.05 -0.683, x1.2 -0.494, x1.35 -0.384. x1.35 = best-supported point -> #337641 fired on it. Last roll: hgfh_s135_gsk08_clean (x1.35 + _GSK 0.8) VM32+fresh+Mac32 running for a possible 9th shot (~23:15Z decision).
    - 2026-10-21T21:52:11Z [fable-lead] Mac32 hgfh_fh10_c4s8e9_clean = exact twin of #337619's clean file (block factor 1.0 = no-op): all32 -0.345% held25 -0.446% raw -0.632% bill +0.289% vs 49bd -> the clean Mac32 REFERENCE for #337619 (XSG-on twin -0.423/-0.524, bill +0.211; XSG-off costs +0.078 bill). Candidates must beat all32 -0.345 / held25 -0.446 on Mac32 and VM32 -0.542/-0.600 (fh10 VM32).
    - 2026-10-21T22:00:13Z [fable-lead] STRENGTH x1.35 (hgfh_s135_c4s8e9_clean) VM32 vs 49bd: all32 -0.572% (22/32) held25 -0.654% raw -0.859% bill +0.289% vs the clean twin -0.542/-0.600 raw -0.826: -0.03/-0.05 better (noise-level, positive). s165 VM32 complete, scoring; Mac32 s135 in window. Fresh cohorts launched for s135/s165. cap5 family closed on the full pair (+0.071%).
    - 2026-10-21T22:00:53Z [fable-lead] STRENGTH CURVE on the hgfh seal (VM32 raw vs 49bd, clean cap4+8e9): x1.65 -0.766 (adj -0.481/-0.527) < x1.5 -0.826 (-0.542/-0.600) < x1.35 -0.859 (-0.572/-0.654): the optimum is BELOW 1.5 for the 961-col seal (more columns, less shrinkage needed). Queued x1.2 and x1.05 (export+build+VM32+fresh after the Mac window); s135 fresh launched; Mac32 s135 in window; s165 CLOSED.
    - 2026-10-21T22:01:39Z [fable-lead] LEAD CANDIDATE: Mac32 hgfh_s135_c4s8e9_clean (strength x1.35) vs 49bd: all32 -0.538% (21/32) held25 -0.649% raw -0.825% bill +0.289% vs the clean twin's Mac32 -0.345/-0.446 raw -0.632: -0.19/-0.20 BETTER. With VM32 (-0.03/-0.05 better): pooled ~-0.11/-0.13% vs #337619 -> projected ~1.3987e-9 (at the 0.005 bar). Fresh12 running; gates after the wave-3 Mac32s (~22:25Z); x1.2 / x1.05 ladder queued. Fire window: s135 by ~22:35Z if the ladder is not ready; else the ladder winner by ~23:05Z (deadline 23:59Z).
    - 2026-10-21T22:18:03Z [fable-lead] Mac32 hgfh_s12_c4s8e9_clean (strength x1.2) vs 49bd: all32 -0.485% held25 -0.639% raw -0.770% bill +0.287% -> better than x1.5 (-0.345/-0.446) but slightly worse than x1.35 (-0.538/-0.649 raw -0.825): the optimum sits near x1.35. x1.35 remains the fire candidate; its fresh cohorts at 2/8 and 2/4 ranks (fire ~22:30Z).
    - 2026-10-21T22:29:46Z [fable-lead] FIRED #337641 = hgfh_s135_c4s8e9_clean governor-stripped (archive e0a79ccb4c52), pooled ~-0.06% vs #337619 (fresh12 weak at -0.384); 2 shots remain; grading poll on the VM.
    - 2026-10-21T23:35:07Z [fable-lead] #337641 GRADED: official 1.4006490e-9 (100/100, 0 failed, util 0.108659, final-layer MSE 1.289e-8, walls 98.7/109.7 s, residual .336/.363, kernel 89.5/100.3 s). vs #337619 1.4005566e-9: +0.0066% = dead tie (the x1.35 strength change is a wash on the hosted public set, as the noise-level VM32 edge implied). NOTE: the official score is the MEAN of per-net adjusted scores, not mean-MSE x mean-util (product-of-means understates by 0.03-0.04%: 1.4001e-9 for both files). BEST stays #337619 1.4005566e-9; board best unchanged; designations = #337619 + #337641.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/stacks`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hgfh_r0p03_s15_cap4s8e9_clean/candidate_hgfh_r0p03_s15_cap4s8e9_clean_science_v1.py](code/hgfh_r0p03_s15_cap4s8e9_clean/candidate_hgfh_r0p03_s15_cap4s8e9_clean_science_v1.py)
- [code/hgfh_s105_c4s8e9_clean/candidate_hgfh_s105_c4s8e9_clean_science_v1.py](code/hgfh_s105_c4s8e9_clean/candidate_hgfh_s105_c4s8e9_clean_science_v1.py)
- [code/hgfh_s12_c4s8e9_clean/candidate_hgfh_s12_c4s8e9_clean_science_v1.py](code/hgfh_s12_c4s8e9_clean/candidate_hgfh_s12_c4s8e9_clean_science_v1.py)
- [code/hgfh_s135_c4s8e9_clean/candidate_hgfh_s135_c4s8e9_clean_science_v1.py](code/hgfh_s135_c4s8e9_clean/candidate_hgfh_s135_c4s8e9_clean_science_v1.py)
- [code/hgfh_s165_c4s8e9_clean/candidate_hgfh_s165_c4s8e9_clean_science_v1.py](code/hgfh_s165_c4s8e9_clean/candidate_hgfh_s165_c4s8e9_clean_science_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7582](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7582)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7570](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7570)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7572](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7572)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7573](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7573)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7574](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7574)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7578](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7578)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7580](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7580)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7592](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7592)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337641: Final 961-column readout at strength 1.35 with residual-meter references removed](../../../submissions/337641/README.md)

[Research graph](../../../graph/index.html#node=2026-10-21-final-readout-strength-1-35-improves-mac-linux-panels) · [Copy provenance](../../../PROVENANCE.json)

