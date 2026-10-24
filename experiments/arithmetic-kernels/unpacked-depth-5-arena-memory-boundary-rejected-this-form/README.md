# Unpacked depth-5 arena memory boundary: rejected in this form

2026-08-26 · experiment · rejected · measured

The prior depth-5 implementation projects above 8.8 GB and remains closed in its unchanged allocation form. Lifetime packing later reduced the measured official peak below 7 GB, so the old receipt is not a family-wide depth-5 third-cumulant arena (D5) impossibility proof.

Recorded measurement:

    The prior depth-5 implementation projects above 8.8 GB and remains closed in its unchanged allocation form. Lifetime packing later reduced the measured official peak below 7 GB, so the old receipt is not a family-wide D5 impossibility proof.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

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
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/audit_tiny_static.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/audit_tiny_static.py)

<details>
<summary>Additional programs and dependencies (11)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/packed_d5_kernel.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/packed_d5_kernel.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d17_d5_official_env_failure_repro_v1/run_rank35_official_env_repro.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d17_d5_official_env_failure_repro_v1/run_rank35_official_env_repro.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d18_d5_official_env_full_gate_v1/run_official_env_full_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d18_d5_official_env_full_gate_v1/run_official_env_full_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7_cp3_arena_depth_ladder_v1/run_depth_ladder.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7_cp3_arena_depth_ladder_v1/run_depth_ladder.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py)

</details>

## Evidence

- [MEMORY_ERRATUM_001.json#L1](MEMORY_ERRATUM_001.json#L1)
- [PACKED_D5_STATIC_RECEIPT.json#L1](PACKED_D5_STATIC_RECEIPT.json#L1)
- [D18_OFFICIAL_ENV_FULL_GATE_RESULT.json#L1](D18_OFFICIAL_ENV_FULL_GATE_RESULT.json#L1)

<details>
<summary>All result files and records (3)</summary>

- [MEMORY_ERRATUM_001.json](MEMORY_ERRATUM_001.json) · result
- [PACKED_D5_STATIC_RECEIPT.json](PACKED_D5_STATIC_RECEIPT.json) · result
- [D18_OFFICIAL_ENV_FULL_GATE_RESULT.json](D18_OFFICIAL_ENV_FULL_GATE_RESULT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-26-unpacked-depth-5-arena-memory-boundary-rejected-this-form) · [Copy provenance](../../../PROVENANCE.json)

