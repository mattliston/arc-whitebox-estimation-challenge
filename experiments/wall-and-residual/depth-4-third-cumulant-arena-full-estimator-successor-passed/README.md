# Depth-4 third-cumulant arena full-estimator successor: passed

2026-08-27 · experiment · adopted · measured

Exact depth-4 third-cumulant arena (D4) passes all three target-free calls at literal bills [1,319,871,710,583; 1,356,595,288,584; 1,317,948,959,232], maximum residual 0.351513 s, maximum resident-set size (RSS) 6,734,069,760 bytes, exact repeat, and final drift within gate. Submission #328622 graded adjusted 1.783621478716186e-8 and raw 2.9409071835573285e-8, becoming bank.

Recorded measurement:

    Exact D4 passes all three target-free calls at literal bills [1,319,871,710,583; 1,356,595,288,584; 1,317,948,959,232], maximum residual 0.351513 s, maximum RSS 6,734,069,760 bytes, exact repeat, and final drift within gate. Submission #328622 graded adjusted 1.783621478716186e-8 and raw 2.9409071835573285e-8, becoming bank.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py)
- [code/autoresearch/experiments/cands/_cand_cx_strassen_s3_winograd15_batched.py](code/autoresearch/experiments/cands/_cand_cx_strassen_s3_winograd15_batched.py)
- [code/autoresearch/experiments/cands/fable_clean_strassen_20260810/estimator.py](code/autoresearch/experiments/cands/fable_clean_strassen_20260810/estimator.py)
- [code/autoresearch/experiments/cands/fable_clean_td52best_20260810/estimator.py](code/autoresearch/experiments/cands/fable_clean_td52best_20260810/estimator.py)
- [code/autoresearch/experiments/cands/fable_spend7_strassen_20260810/estimator.py](code/autoresearch/experiments/cands/fable_spend7_strassen_20260810/estimator.py)
- [code/autoresearch/experiments/cands/fable_spend_strassen_20260810/estimator.py](code/autoresearch/experiments/cands/fable_spend_strassen_20260810/estimator.py)
- [code/autoresearch/experiments/cands/util_probe_strassen/estimator.py](code/autoresearch/experiments/cands/util_probe_strassen/estimator.py)
- [code/autoresearch/flopscope_swarm/K_probe_strassen.py](code/autoresearch/flopscope_swarm/K_probe_strassen.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py)

<details>
<summary>Additional programs and dependencies (34)</summary>

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
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_d5r1_resource_successor.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_d5r1_resource_successor.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_descendant_terminal_execution.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_descendant_terminal_execution.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_d6_successor.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_d6_successor.py)
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

- [ACCOUNTING_AND_LIVENESS_MANIFEST.json#L1](ACCOUNTING_AND_LIVENESS_MANIFEST.json#L1)
- [D6_TERMINAL_RECEIPT.json#L1](../../arithmetic-kernels/preallocated-depth-4-winograd-arena-0-04961721133440733-s-extra/D6_TERMINAL_RECEIPT.json#L1)
- [SUBMIT_WATCH.log#L1](../../flop-accounting/deployed-estimator-score-economics-minimum-charged-utilisation-verified/SUBMIT_WATCH.log#L1)
- [D4_CANDIDATE_RESULT.json#L1051](D4_CANDIDATE_RESULT.json#L1051)
- [SUBMIT_WATCH.log#L1](../688-network-endpoint-action-head-depth-3-arena-measured/SUBMIT_WATCH.log#L1)

<details>
<summary>All result files and records (5)</summary>

- [ACCOUNTING_AND_LIVENESS_MANIFEST.json](ACCOUNTING_AND_LIVENESS_MANIFEST.json) · result
- [D6_TERMINAL_RECEIPT.json](../../arithmetic-kernels/preallocated-depth-4-winograd-arena-0-04961721133440733-s-extra/D6_TERMINAL_RECEIPT.json) · result
- [SUBMIT_WATCH.log](../../flop-accounting/deployed-estimator-score-economics-minimum-charged-utilisation-verified/SUBMIT_WATCH.log) · log
- [D4_CANDIDATE_RESULT.json](D4_CANDIDATE_RESULT.json) · result
- [SUBMIT_WATCH.log](../688-network-endpoint-action-head-depth-3-arena-measured/SUBMIT_WATCH.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-depth-4-third-cumulant-arena-full-estimator-successor-passed) · [Copy provenance](../../../PROVENANCE.json)

