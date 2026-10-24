# Depth-4 Winograd recovery: residual 1.875688566826284 s and memory 11,953,569,792 bytes failed

2026-08-26 · experiment · rejected · measured

Terminal FAIL_TERMINAL_NO_RETRY after candidate ordinal 0. Exact savings 371,608,248,320, final-layer RMS drift 4.109649235540117e-8, maximum drift 2.384185791015625e-7, schema, finiteness, and absolute bill all passed. Residual wall 1.875688566826284 s failed <0.4/<=0.385 and peak resident-set size (RSS) 11,953,569,792 bytes failed <8 GiB.

Recorded measurement:

    Terminal FAIL_TERMINAL_NO_RETRY after candidate ordinal 0. Exact savings 371,608,248,320, final-layer RMS drift 4.109649235540117e-8, maximum drift 2.384185791015625e-7, schema, finiteness, and absolute bill all passed. Residual wall 1.875688566826284 s failed <0.4/<=0.385 and peak RSS 11,953,569,792 bytes failed <8 GiB. Ordinals 1/2, repeat, and pooled gates were not run.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_descendant_terminal_execution.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_descendant_terminal_execution.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_recovery_descendant_terminal.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_recovery_descendant_terminal.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_d5r1_resource_successor.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_d5r1_resource_successor.py)
- [code/autoresearch/experiments/cands/_cand_cx_strassen_s3_winograd15_batched.py](code/autoresearch/experiments/cands/_cand_cx_strassen_s3_winograd15_batched.py)
- [code/autoresearch/experiments/cands/fable_clean_strassen_20260810/estimator.py](code/autoresearch/experiments/cands/fable_clean_strassen_20260810/estimator.py)
- [code/autoresearch/experiments/cands/fable_spend7_strassen_20260810/estimator.py](code/autoresearch/experiments/cands/fable_spend7_strassen_20260810/estimator.py)
- [code/autoresearch/experiments/cands/fable_spend_strassen_20260810/estimator.py](code/autoresearch/experiments/cands/fable_spend_strassen_20260810/estimator.py)
- [code/autoresearch/flopscope_swarm/K_probe_strassen.py](code/autoresearch/flopscope_swarm/K_probe_strassen.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py)

<details>
<summary>Additional programs and dependencies (19)</summary>

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
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_recovered_reference.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_recovered_reference.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_recovery_descendant_prerun.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_recovery_descendant_prerun.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_reference_terminal.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_reference_terminal.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_recovered_reference_evidence.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_recovered_reference_evidence.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_recovery_descendant_launch_seal.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_recovery_descendant_launch_seal.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_recovery_descendant_preseal.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_recovery_descendant_preseal.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_recovery_descendant_static.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_recovery_descendant_static.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_reference_terminal_failure.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_reference_terminal_failure.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py)

</details>

## Evidence

- [RECOVERY_DESCENDANT_RESULT.json#L367](RECOVERY_DESCENDANT_RESULT.json#L367)
- [RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json#L404](RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json#L404)
- [RECOVERY_DESCENDANT_EXECUTION_TRANSCRIPT.txt#L23](RECOVERY_DESCENDANT_EXECUTION_TRANSCRIPT.txt#L23)
- [INDEPENDENT_REDTEAM_RECOVERY_DESCENDANT_TERMINAL_AUDIT.md#L42](INDEPENDENT_REDTEAM_RECOVERY_DESCENDANT_TERMINAL_AUDIT.md#L42)
- [D5R1_RESOURCE_CAUSAL_AUDIT_AND_D6_SUCCESSOR.md#L204](D5R1_RESOURCE_CAUSAL_AUDIT_AND_D6_SUCCESSOR.md#L204)

<details>
<summary>All result files and records (5)</summary>

- [RECOVERY_DESCENDANT_RESULT.json](RECOVERY_DESCENDANT_RESULT.json) · result
- [RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json](RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json) · result
- [RECOVERY_DESCENDANT_EXECUTION_TRANSCRIPT.txt](RECOVERY_DESCENDANT_EXECUTION_TRANSCRIPT.txt) · record
- [INDEPENDENT_REDTEAM_RECOVERY_DESCENDANT_TERMINAL_AUDIT.md](INDEPENDENT_REDTEAM_RECOVERY_DESCENDANT_TERMINAL_AUDIT.md) · record
- [D5R1_RESOURCE_CAUSAL_AUDIT_AND_D6_SUCCESSOR.md](D5R1_RESOURCE_CAUSAL_AUDIT_AND_D6_SUCCESSOR.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-08-26-depth-4-winograd-recovery-residual-1-875688566826284-s-memory) · [Copy provenance](../../../PROVENANCE.json)

