# Lazy fusion of tree arithmetic: identical FLOPs but 908 more first and 96 more warm requests

2026-10-16T02:23:14Z · experiment · rejected · measured

The proposed lazy fusion preserved predictions but raised production request counts and memory use instead of delivering the expected saving. Size limits did not cure the memory pressure.

Recorded measurement:

    first49126(+908),warm46093(+96);risk21.82%vs17.17%;RSS7.9GBvs6.1-6.4GB;footprint4.36GB

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L3700](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3700)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3699](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3699)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [TREE_R1_RESULT_v1.json](../tree-loads-replaced-two-unstacks-cached-half-view-465/TREE_R1_RESULT_v1.json) · result
- [fragment_tree_lf_v1.py.txt](../tree-loads-replaced-two-unstacks-cached-half-view-465/fragment_tree_lf_v1.py.txt) · result
- [fragment_tree_lf_v2.py.txt](../tree-loads-replaced-two-unstacks-cached-half-view-465/fragment_tree_lf_v2.py.txt) · result
- [fragment_tree_lf_v3.py.txt](../tree-loads-replaced-two-unstacks-cached-half-view-465/fragment_tree_lf_v3.py.txt) · result
- [fragment_tree_r1_v1.py.txt](../tree-loads-replaced-two-unstacks-cached-half-view-465/fragment_tree_r1_v1.py.txt) · result
- [diag2_2b108.json](../tree-loads-replaced-two-unstacks-cached-half-view-465/diag2_2b108.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-lazy-fusion-tree-arithmetic-identical-flops-but-908-more) · [Copy provenance](../../../PROVENANCE.json)

