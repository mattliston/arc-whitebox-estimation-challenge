# Young fusion with early temporary release reduced memory but worsened the held symmetric-join stack

2026-10-18T19:22:56Z · experiment · rejected · measured

Young-product fusion was added with audited early temporary release under the unchanged 7.5 GB guard. Memory decreased, but the seven-network adjusted score was worse than the held symmetric-join stack, so no submission followed.

Recorded measurement:

    Raw −0.112009%, bill −0.118458%, adjusted −0.230227%, SE 0.125845 percentage points; projection 1.443525504e-9; first-group RSS 7.333 GB versus 7.463 GB.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_jsym_fusion_release_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate95c_jsym_fusion_release_HELD_v1.py](code/candidate95c_jsym_fusion_release_HELD_v1.py)
- [code/fragment_fusion_release_HELD_v1.py](code/fragment_fusion_release_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5881](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5881)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SEARCH_STATE_v35.json](SEARCH_STATE_v35.json) · record
- [SEARCH_STATE_v34.json](../../join-sketch/symmetrizing-nystrom-landmark-block-before-qr-improved-held-stack/SEARCH_STATE_v34.json) · result
- [GRADE_STORE_v1.json](GRADE_STORE_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-young-fusion-early-temporary-release-reduced-memory-but-worsened) · [Copy provenance](../../../PROVENANCE.json)

