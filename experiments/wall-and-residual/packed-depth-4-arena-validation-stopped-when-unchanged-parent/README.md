# Packed depth-4 arena validation stopped when the unchanged parent used 0.5711876060813665 s residual

2026-08-27 · experiment · rejected · measured

The target-free static gate proves changing-rank byte/meter parity and a lifetime-layout saving of 1,322,696,704 persistent scratch bytes plus 33,554,432 cold binding FLOPs. The frozen no-retry five-component [35,279,35] gate then stopped on its first unchanged-depth-4 third-cumulant arena (D4) parent call: residual 0.5711876060813665 seconds was not strictly below 0.4, while FLOPs, call count, resident-set size (RSS), and output were otherwise admissible. The packed candidate made zero calls.

Recorded measurement:

    The target-free static gate proves changing-rank byte/meter parity and a lifetime-layout saving of 1,322,696,704 persistent scratch bytes plus 33,554,432 cold binding FLOPs. The frozen no-retry five-component [35,279,35] gate then stopped on its first unchanged-D4 parent call: residual 0.5711876060813665 seconds was not strictly below 0.4, while FLOPs, call count, RSS, and output were otherwise admissible. The packed candidate made zero calls. A static residual-sync audit found the same parent arithmetic and 15,416-call inventory had previously measured 0.3257444277405739 seconds, so the parent-over-cap run is non-discriminating: it closes this gate without rejecting packed-D4 runtime, and it does not certify candidate runtime or full-component parity. D4 #328622 remains the scored bank.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/run_guarded.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/run_guarded.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7_cp3_arena_depth_ladder_v1/run_depth_ladder.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7_cp3_arena_depth_ladder_v1/run_depth_ladder.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_lifetime_packed_full_gate_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_lifetime_packed_full_gate_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_lifetime_packed_full_gate_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_lifetime_packed_full_gate_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_lifetime_packed_full_gate_v1/run_full_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_lifetime_packed_full_gate_v1/run_full_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_lifetime_packed_full_gate_v1/verify_static.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_lifetime_packed_full_gate_v1/verify_static.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_lifetime_packed_reinvestment_v1/audit_tiny_static.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_lifetime_packed_reinvestment_v1/audit_tiny_static.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_lifetime_packed_reinvestment_v1/packed_d4_kernel.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d4_lifetime_packed_reinvestment_v1/packed_d4_kernel.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_residual_sync_protocol_v1/verify_audit.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_residual_sync_protocol_v1/verify_audit.py)

</details>

## Evidence

- [PACKED_D4_STATIC_RECEIPT.json#L1](PACKED_D4_STATIC_RECEIPT.json#L1)
- [FULL_GATE_RESULT.json#L31](FULL_GATE_RESULT.json#L31)
- [TERMINAL_CLOSURE_RECEIPT.json#L20](TERMINAL_CLOSURE_RECEIPT.json#L20)
- [RESIDUAL_SYNC_PROTOCOL_AUDIT.json#L39](RESIDUAL_SYNC_PROTOCOL_AUDIT.json#L39)
- [STATIC_VERIFICATION.json#L8](STATIC_VERIFICATION.json#L8)

<details>
<summary>All result files and records (5)</summary>

- [PACKED_D4_STATIC_RECEIPT.json](PACKED_D4_STATIC_RECEIPT.json) · result
- [FULL_GATE_RESULT.json](FULL_GATE_RESULT.json) · result
- [TERMINAL_CLOSURE_RECEIPT.json](TERMINAL_CLOSURE_RECEIPT.json) · result
- [RESIDUAL_SYNC_PROTOCOL_AUDIT.json](RESIDUAL_SYNC_PROTOCOL_AUDIT.json) · result
- [STATIC_VERIFICATION.json](STATIC_VERIFICATION.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-packed-depth-4-arena-validation-stopped-when-unchanged-parent) · [Copy provenance](../../../PROVENANCE.json)

