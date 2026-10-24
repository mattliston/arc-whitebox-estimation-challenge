# Splitting arena aliases saved only 0.683 ms while allocating 2,320,941,056 extra bytes

2026-08-27 · experiment · rejected · measured

Shared and split arena spellings have the exact same 58-call operation inventory and output bytes. Splitting saves only 0.683 ms median residual while allocating 2,320,941,056 extra bytes and leaving about 222 MB below the 8 GiB ceiling. It misses the frozen 15 ms credibility gate and is not board-safe.

Recorded measurement:

    Shared and split arena spellings have the exact same 58-call operation inventory and output bytes. Splitting saves only 0.683 ms median residual while allocating 2,320,941,056 extra bytes and leaving about 222 MB below the 8 GiB ceiling. It misses the frozen 15 ms credibility gate and is not board-safe.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d27_cp3_hybrid_shared_alias_discriminator_v1/audit_shared_alias_tiny.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d27_cp3_hybrid_shared_alias_discriminator_v1/audit_shared_alias_tiny.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/run_guarded.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/run_guarded.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d25_cp3_d5push_d4read_hybrid_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d25_cp3_d5push_d4read_hybrid_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d26_cp3_d5push_left_transform_cse_v1/audit_dispatch_cse_tiny.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d26_cp3_d5push_left_transform_cse_v1/audit_dispatch_cse_tiny.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7_cp3_arena_depth_ladder_v1/run_depth_ladder.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7_cp3_arena_depth_ladder_v1/run_depth_ladder.py)

## Evidence

- [D27_SHARED_ALIAS_TINY_RECEIPT.json#L1](D27_SHARED_ALIAS_TINY_RECEIPT.json#L1)
- [D25_HYBRID_CANDIDATE_RESULT.json#L1](../../flop-accounting/depth-5-transport-depth-4-arena-reads-residual-time/D25_HYBRID_CANDIDATE_RESULT.json#L1)
- [D26_DISPATCH_CSE_TINY_RECEIPT.json#L1](../../flop-accounting/left-transform-common-subexpression-reuse-5-978-ms-median/D26_DISPATCH_CSE_TINY_RECEIPT.json#L1)

<details>
<summary>All result files and records (3)</summary>

- [D27_SHARED_ALIAS_TINY_RECEIPT.json](D27_SHARED_ALIAS_TINY_RECEIPT.json) · result
- [D25_HYBRID_CANDIDATE_RESULT.json](../../flop-accounting/depth-5-transport-depth-4-arena-reads-residual-time/D25_HYBRID_CANDIDATE_RESULT.json) · result
- [D26_DISPATCH_CSE_TINY_RECEIPT.json](../../flop-accounting/left-transform-common-subexpression-reuse-5-978-ms-median/D26_DISPATCH_CSE_TINY_RECEIPT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-splitting-arena-aliases-saved-only-0-683-ms-while) · [Copy provenance](../../../PROVENANCE.json)

