# Preallocated depth-4 Winograd arena: 0.04961721133440733 s extra residual exceeded the 0.010 s gate

2026-08-26 · experiment · rejected · measured

Static arena arithmetic, aliasing, bill, and tiny byte parity pass, but the same-source exact-runtime tiny discriminator measures median D6-minus-direct residual 0.04961721133440733 s versus the frozen 0.010 s cap. The 6,786 Python hot records are the cause. Exact-shape execution was correctly skipped.

Recorded measurement:

    Static arena arithmetic, aliasing, bill, and tiny byte parity pass, but the same-source exact-runtime tiny discriminator measures median D6-minus-direct residual 0.04961721133440733 s versus the frozen 0.010 s cap. The 6,786 Python hot records are the cause. Exact-shape execution was correctly skipped.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_d5r1_resource_successor.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_d5r1_resource_successor.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_d6_successor.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_d6_successor.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/verify_d6_terminal_closure.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/verify_d6_terminal_closure.py)
- [code/autoresearch/experiments/cands/_cand_cx_strassen_s3_winograd15_batched.py](code/autoresearch/experiments/cands/_cand_cx_strassen_s3_winograd15_batched.py)
- [code/autoresearch/experiments/cands/fable_clean_strassen_20260810/estimator.py](code/autoresearch/experiments/cands/fable_clean_strassen_20260810/estimator.py)
- [code/autoresearch/experiments/cands/fable_clean_td52best_20260810/estimator.py](code/autoresearch/experiments/cands/fable_clean_td52best_20260810/estimator.py)
- [code/autoresearch/experiments/cands/fable_spend7_strassen_20260810/estimator.py](code/autoresearch/experiments/cands/fable_spend7_strassen_20260810/estimator.py)
- [code/autoresearch/experiments/cands/fable_spend_strassen_20260810/estimator.py](code/autoresearch/experiments/cands/fable_spend_strassen_20260810/estimator.py)
- [code/autoresearch/experiments/cands/util_probe_strassen/estimator.py](code/autoresearch/experiments/cands/util_probe_strassen/estimator.py)
- [code/autoresearch/flopscope_swarm/K_probe_strassen.py](code/autoresearch/flopscope_swarm/K_probe_strassen.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)

<details>
<summary>Additional programs and dependencies (34)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/run_guarded.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/run_guarded.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/build_recovered_reference_evidence.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/build_recovered_reference_evidence.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/build_recovery_descendant_runner.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/build_recovery_descendant_runner.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/recover_descendant_terminal_execution.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/recover_descendant_terminal_execution.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/recover_reference_terminal_failure.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/recover_reference_terminal_failure.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/run_common_driver_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/run_common_driver_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/run_recovery_descendant_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/run_recovery_descendant_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/run_reference_capture.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/run_reference_capture.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/test_recovery_descendant_receipt_serialization.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/test_recovery_descendant_receipt_serialization.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_descendant_terminal_execution.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_descendant_terminal_execution.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_recovered_reference.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_recovered_reference.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_recovery_descendant_prerun.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_recovery_descendant_prerun.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_recovery_descendant_terminal.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_recovery_descendant_terminal.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_reference_terminal.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_reference_terminal.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_recovered_reference_evidence.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_recovered_reference_evidence.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_recovery_descendant_launch_seal.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_recovery_descendant_launch_seal.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_recovery_descendant_preseal.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_recovery_descendant_preseal.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_recovery_descendant_static.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_recovery_descendant_static.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_reference_terminal_failure.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_reference_terminal_failure.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/run_exact_shape_kernel_resource_smoke.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/run_exact_shape_kernel_resource_smoke.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/run_independent_tiny_residual_diagnostic.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/run_independent_tiny_residual_diagnostic.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/verify_exact_shape_smoke_seal.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/verify_exact_shape_smoke_seal.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/verify_independent_d6_implementation_audit.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/verify_independent_d6_implementation_audit.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/verify_serialization_guard.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/verify_serialization_guard.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/verify_static_liveness.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_cp3_winograd_arena_v1/verify_static_liveness.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_tiny_prototype_v1/tiny_d6_kernel.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_tiny_prototype_v1/tiny_d6_kernel.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_tiny_prototype_v1/verify_tiny_prototype.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_tiny_prototype_v1/verify_tiny_prototype.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_tiny_prototype_v1/verify_tiny_prototype_final_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d6_tiny_prototype_v1/verify_tiny_prototype_final_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py)

</details>

## Evidence

- [D5R1_RESOURCE_CAUSAL_AUDIT_AND_D6_SUCCESSOR.md#L172](../../wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/D5R1_RESOURCE_CAUSAL_AUDIT_AND_D6_SUCCESSOR.md#L172)
- [INDEPENDENT_REDTEAM_D6_SUCCESSOR_AUDIT.md#L131](INDEPENDENT_REDTEAM_D6_SUCCESSOR_AUDIT.md#L131)
- [RECOVERY_DESCENDANT_RESULT.json#L352](../../wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/RECOVERY_DESCENDANT_RESULT.json#L352)
- [RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json#L389](../../wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json#L389)
- [INDEPENDENT_REDTEAM_RECOVERY_DESCENDANT_TERMINAL_AUDIT.md#L1](../../wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/INDEPENDENT_REDTEAM_RECOVERY_DESCENDANT_TERMINAL_AUDIT.md#L1)
- [D6_TERMINAL_RECEIPT.json#L67](D6_TERMINAL_RECEIPT.json#L67)
- [INDEPENDENT_REDTEAM_D6_IMPLEMENTATION_AUDIT.md#L13](INDEPENDENT_REDTEAM_D6_IMPLEMENTATION_AUDIT.md#L13)
- [DEPTH_LADDER_RECEIPT.json#L1](DEPTH_LADDER_RECEIPT.json#L1)

<details>
<summary>All result files and records (8)</summary>

- [D5R1_RESOURCE_CAUSAL_AUDIT_AND_D6_SUCCESSOR.md](../../wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/D5R1_RESOURCE_CAUSAL_AUDIT_AND_D6_SUCCESSOR.md) · record
- [INDEPENDENT_REDTEAM_D6_SUCCESSOR_AUDIT.md](INDEPENDENT_REDTEAM_D6_SUCCESSOR_AUDIT.md) · record
- [RECOVERY_DESCENDANT_RESULT.json](../../wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/RECOVERY_DESCENDANT_RESULT.json) · result
- [RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json](../../wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json) · result
- [INDEPENDENT_REDTEAM_RECOVERY_DESCENDANT_TERMINAL_AUDIT.md](../../wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/INDEPENDENT_REDTEAM_RECOVERY_DESCENDANT_TERMINAL_AUDIT.md) · record
- [D6_TERMINAL_RECEIPT.json](D6_TERMINAL_RECEIPT.json) · result
- [INDEPENDENT_REDTEAM_D6_IMPLEMENTATION_AUDIT.md](INDEPENDENT_REDTEAM_D6_IMPLEMENTATION_AUDIT.md) · record
- [DEPTH_LADDER_RECEIPT.json](DEPTH_LADDER_RECEIPT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-26-preallocated-depth-4-winograd-arena-0-04961721133440733-s-extra) · [Copy provenance](../../../PROVENANCE.json)

