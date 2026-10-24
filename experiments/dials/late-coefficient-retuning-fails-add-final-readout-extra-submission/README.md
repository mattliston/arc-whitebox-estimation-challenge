# Late coefficient retuning fails to add to the final readout; the extra submission is withheld

2026-10-21T22:32:39Z · experiment · rejected · measured

A coefficient change that marginally improved the strength-1.5 source became worse on the strength-1.35 source on both fixed native panels. Other scalar alternatives were flat or worse, so the conditional submission was withheld.

Recorded measurement:

    - 2026-10-21T22:32:39Z [fable-lead] DIAL BATCH VM32 (vs the clean twin -0.542/-0.600 raw -0.826): _GSK 0.8 -0.574/-0.641 raw -0.858 (+0.03/+0.04 better, noise-level positive); _BFIX_GAMMA 0.14 -0.536/-0.564 (flat); _SS_BETA 0.90 -0.403/-0.496 (worse, closed). Combining GSK 0.8 with strength x1.35 (hgfh_s135_gsk08_clean) for a possible 9th shot: VM32+fresh+Mac32 now. #337641 queued at the grader.
    - 2026-10-21T22:41:06Z [fable-lead] hgfh_s135_gsk08_clean (x1.35 + _GSK 0.8) VM32 vs 49bd: all32 -0.532 held25 -0.596 raw -0.819 (vs x1.35 alone -0.572/-0.654 raw -0.859): WORSE by ~0.04 -> GSK does not add on the stack (dial non-transfer again); fresh12 -0.610 (noise). Fire condition (VM32 all32 < -0.572) NOT met -> the guarded 9th fire will not fire; Mac32 pending for the record. Final position: #337641 (x1.35) grading; #337619 1.40056 best graded; 2 shots held.
    - 2026-10-21T22:42:14Z [fable-lead] 9th fire STOOD DOWN: hgfh_s135_gsk08_clean Mac32 all32 -0.523 held25 -0.670 (vs x1.35 alone -0.538/-0.649), VM32 -0.532/-0.596 (vs -0.572/-0.654): not better on either 32-net set; fresh -0.610 (noise). GSK closed on the stack. No candidate left that beats #337641's file; 2 shots held unless the board moves. #337641 still queued (12 min).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct21_final_v1/stacks`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hgfh_bfix14_c4s8e9_clean/candidate_hgfh_bfix14_c4s8e9_clean_science_v1.py](code/hgfh_bfix14_c4s8e9_clean/candidate_hgfh_bfix14_c4s8e9_clean_science_v1.py)
- [code/hgfh_gsk08_c4s8e9_clean/candidate_hgfh_gsk08_c4s8e9_clean_science_v1.py](code/hgfh_gsk08_c4s8e9_clean/candidate_hgfh_gsk08_c4s8e9_clean_science_v1.py)
- [code/hgfh_s135_gsk08_clean/candidate_hgfh_s135_gsk08_clean_science_v1.py](code/hgfh_s135_gsk08_clean/candidate_hgfh_s135_gsk08_clean_science_v1.py)
- [code/hgfh_ssb90_c4s8e9_clean/candidate_hgfh_ssb90_c4s8e9_clean_science_v1.py](code/hgfh_ssb90_c4s8e9_clean/candidate_hgfh_ssb90_c4s8e9_clean_science_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7581](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7581)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7584](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7584)
- [TEAM_LOG_2026-10-01_to_10-22.md#L7585](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7585)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-late-coefficient-retuning-fails-add-final-readout-extra-submission) · [Copy provenance](../../../PROVENANCE.json)

