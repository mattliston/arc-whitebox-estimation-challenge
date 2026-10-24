# Left-transform common-subexpression reuse: 5.978 ms median saving missed the 15 ms credibility threshold

2026-08-27 · experiment · rejected · measured

The left-transform CSE is byte exact and projects 888,898,560 FLOPs and 825 wrapper calls saved, but its median residual saving is only 5.978 ms versus the frozen 15 ms credibility threshold. Unified-code dispatch saves only 0.470 ms. No full-estimator successor is licensed.

Recorded measurement:

    The left-transform CSE is byte exact and projects 888,898,560 FLOPs and 825 wrapper calls saved, but its median residual saving is only 5.978 ms versus the frozen 15 ms credibility threshold. Unified-code dispatch saves only 0.470 ms. No full-estimator successor is licensed.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d26_cp3_d5push_left_transform_cse_v1/audit_dispatch_cse_tiny.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d26_cp3_d5push_left_transform_cse_v1/audit_dispatch_cse_tiny.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/run_guarded.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/run_guarded.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d25_cp3_d5push_d4read_hybrid_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d25_cp3_d5push_d4read_hybrid_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7_cp3_arena_depth_ladder_v1/run_depth_ladder.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7_cp3_arena_depth_ladder_v1/run_depth_ladder.py)

## Evidence

- [D26_DISPATCH_CSE_TINY_RECEIPT.json#L1](D26_DISPATCH_CSE_TINY_RECEIPT.json#L1)
- [D25_HYBRID_CANDIDATE_RESULT.json#L1](../depth-5-transport-depth-4-arena-reads-residual-time/D25_HYBRID_CANDIDATE_RESULT.json#L1)

<details>
<summary>All result files and records (2)</summary>

- [D26_DISPATCH_CSE_TINY_RECEIPT.json](D26_DISPATCH_CSE_TINY_RECEIPT.json) · result
- [D25_HYBRID_CANDIDATE_RESULT.json](../depth-5-transport-depth-4-arena-reads-residual-time/D25_HYBRID_CANDIDATE_RESULT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-left-transform-common-subexpression-reuse-5-978-ms-median) · [Copy provenance](../../../PROVENANCE.json)

