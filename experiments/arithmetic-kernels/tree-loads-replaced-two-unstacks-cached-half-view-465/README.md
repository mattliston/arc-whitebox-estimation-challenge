# Tree loads replaced by two unstacks and a cached half view: 465/506 fewer requests at identical FLOPs

2026-10-16T01:48:47Z · experiment · adopted · measured

The exact single-tree load rewrite preserved predictions and arithmetic while lowering both cold and warm request counts. It became part of the submitted stack.

Recorded measurement:

    first-465,warm-506;first47960/warm45664;setup19007unchanged;risk19.00%to14.51%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_tree_v1/build_r1_v1.py](code/lead/runs/opus_tree_v1/build_r1_v1.py)
- [code/lead/runs/opus_tree_v1/candidate_2b108_treeR1_v1.py](code/lead/runs/opus_tree_v1/candidate_2b108_treeR1_v1.py)
- [code/lead/runs/opus_tree_v1/candidate_S2_lf_v1.py](code/lead/runs/opus_tree_v1/candidate_S2_lf_v1.py)
- [code/lead/runs/opus_tree_v1/candidate_S2_lf_v2.py](code/lead/runs/opus_tree_v1/candidate_S2_lf_v2.py)
- [code/lead/runs/opus_tree_v1/candidate_S2_lf_v3.py](code/lead/runs/opus_tree_v1/candidate_S2_lf_v3.py)
- [code/lead/runs/opus_tree_v1/diag2_cache_judge_v1.py](code/lead/runs/opus_tree_v1/diag2_cache_judge_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py](code/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3652](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3652)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3656](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3656)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [TREE_R1_RESULT_v1.json](TREE_R1_RESULT_v1.json) · result
- [fragment_tree_lf_v1.py.txt](fragment_tree_lf_v1.py.txt) · result
- [fragment_tree_lf_v2.py.txt](fragment_tree_lf_v2.py.txt) · result
- [fragment_tree_lf_v3.py.txt](fragment_tree_lf_v3.py.txt) · result
- [fragment_tree_r1_v1.py.txt](fragment_tree_r1_v1.py.txt) · result
- [diag2_2b108.json](diag2_2b108.json) · result

</details>

## Related work

- shipped in: [Submission #336367: layer-10 rank 384, compact kernels and exact request cuts; seven runtime failures](../../../submissions/336367/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-tree-loads-replaced-two-unstacks-cached-half-view-465) · [Copy provenance](../../../PROVENANCE.json)

