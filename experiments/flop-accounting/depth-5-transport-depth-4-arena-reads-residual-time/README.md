# Depth-5 transport with depth-4 arena reads: residual-time gate failed

2026-08-27 · experiment · rejected · measured

The hybrid saves 27.332B cold FLOPs and 650.740MB versus depth-4 third-cumulant arena (D4), passes call 0 at residual 0.392053 s, but rank 279 reaches 0.412123 s and fails the strict 0.4 s gate. The one-shot stops without retry, packaging, target access, or submission.

Recorded measurement:

    The hybrid saves 27.332B cold FLOPs and 650.740MB versus D4, passes call 0 at residual 0.392053 s, but rank 279 reaches 0.412123 s and fails the strict 0.4 s gate. The one-shot stops without retry, packaging, target access, or submission.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d25_cp3_d5push_d4read_hybrid_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d25_cp3_d5push_d4read_hybrid_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d25_cp3_d5push_d4read_hybrid_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d25_cp3_d5push_d4read_hybrid_v1/run_candidate_gate.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/run_guarded.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/run_guarded.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py)

<details>
<summary>Additional programs and dependencies (14)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/packed_d5_kernel.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/packed_d5_kernel.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d17_d5_official_env_failure_repro_v1/run_rank35_official_env_repro.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d17_d5_official_env_failure_repro_v1/run_rank35_official_env_repro.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d18_d5_official_env_full_gate_v1/run_official_env_full_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d18_d5_official_env_full_gate_v1/run_official_env_full_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d25_cp3_d5push_d4read_hybrid_v1/audit_tiny_static.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d25_cp3_d5push_d4read_hybrid_v1/audit_tiny_static.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d25_cp3_d5push_d4read_hybrid_v1/packed_d5_kernel.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d25_cp3_d5push_d4read_hybrid_v1/packed_d5_kernel.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7_cp3_arena_depth_ladder_v1/run_depth_ladder.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7_cp3_arena_depth_ladder_v1/run_depth_ladder.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py)

</details>

## Evidence

- [D25_HYBRID_TINY_STATIC_RECEIPT.json#L1](D25_HYBRID_TINY_STATIC_RECEIPT.json#L1)
- [D25_HYBRID_CANDIDATE_RESULT.json#L420](D25_HYBRID_CANDIDATE_RESULT.json#L420)

<details>
<summary>All result files and records (2)</summary>

- [D25_HYBRID_TINY_STATIC_RECEIPT.json](D25_HYBRID_TINY_STATIC_RECEIPT.json) · result
- [D25_HYBRID_CANDIDATE_RESULT.json](D25_HYBRID_CANDIDATE_RESULT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-depth-5-transport-depth-4-arena-reads-residual-time) · [Copy provenance](../../../PROVENANCE.json)

