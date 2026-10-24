# Depth-4 arena common-subexpression reuse: 7,643,930,112 FLOPs saved but full runtime failed

2026-08-27 · experiment · rejected · measured

The exact static CSE candidate saved 7,643,930,112 FLOPs per call and passed tiny/reset parity, but its sole official depth-4 third-cumulant arena (D4) full call failed. TERMINAL_CLOSURE fb3bef24 records FAIL_TERMINAL / NO_RETRY.

Recorded measurement:

    The exact static CSE candidate saved 7,643,930,112 FLOPs per call and passed tiny/reset parity, but its sole official D4 full call failed. TERMINAL_CLOSURE fb3bef24 records FAIL_TERMINAL / NO_RETRY.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d12_d4_non_gemm_cse_v1/audit_tiny_static.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d12_d4_non_gemm_cse_v1/audit_tiny_static.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d12_d4_non_gemm_cse_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d12_d4_non_gemm_cse_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d12_d4_non_gemm_cse_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d12_d4_non_gemm_cse_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d12_d4_non_gemm_cse_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d12_d4_non_gemm_cse_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/build_candidate.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_r40_economic_falsification_v1/audit_economic_gap.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_r40_economic_falsification_v1/audit_economic_gap.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_r40_physical_kernel_audit_v1/audit_tiny_r40.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_r40_physical_kernel_audit_v1/audit_tiny_r40.py)

</details>

## Evidence

- [RESULT.json#L1](RESULT.json#L1)
- [ECONOMIC_GAP.json#L1](ECONOMIC_GAP.json#L1)
- [TERMINAL_CLOSURE.json#L1](TERMINAL_CLOSURE.json#L1)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.json](RESULT.json) · result
- [ECONOMIC_GAP.json](ECONOMIC_GAP.json) · result
- [TERMINAL_CLOSURE.json](TERMINAL_CLOSURE.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-depth-4-arena-common-subexpression-reuse-7-643-930) · [Copy provenance](../../../PROVENANCE.json)

