# Depth-4 Winograd third-cumulant transport: original sealed runtime protocol

2026-08-26 · incident · failed · measured

The original sealed protocol is terminal FAIL_TERMINAL_NO_RETRY. Its three exact incumbent-reference calls all passed compute, residual <0.4, resident-set size (RSS), schema, finiteness, and repeat-parity checks, but final receipt construction raised NameError on lowercase `false`; the candidate was NOT_RUN. This is a harness-form closure, not evidence for or against candidate numerics.

Recorded measurement:

    The original sealed protocol is terminal FAIL_TERMINAL_NO_RETRY. Its three exact incumbent-reference calls all passed compute, residual <0.4, RSS, schema, finiteness, and repeat-parity checks, but final receipt construction raised NameError on lowercase `false`; the candidate was NOT_RUN. This is a harness-form closure, not evidence for or against candidate numerics.

## Programs

- [verify_receipt.py](verify_receipt.py)
- [../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_reference_terminal_failure.py](../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_reference_terminal_failure.py)
- [../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_reference_terminal.py](../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/verify_independent_reference_terminal.py)
- [../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/autoresearch/experiments/cands/fable_clean_strassen_20260810/estimator.py](../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/autoresearch/experiments/cands/fable_clean_strassen_20260810/estimator.py)
- [../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/autoresearch/experiments/cands/fable_spend7_strassen_20260810/estimator.py](../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/autoresearch/experiments/cands/fable_spend7_strassen_20260810/estimator.py)
- [../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/autoresearch/experiments/cands/fable_spend_strassen_20260810/estimator.py](../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/autoresearch/experiments/cands/fable_spend_strassen_20260810/estimator.py)
- [../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py](../../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py)
- [estimator.py](estimator.py)
- [../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/estimator.py](../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/estimator.py)
- [independent_redteam_check.py](independent_redteam_check.py)
- [../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/recover_reference_terminal_failure.py](../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/recover_reference_terminal_failure.py)
- [../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/run_common_driver_gate.py](../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/run_common_driver_gate.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/run_reference_capture.py](../../../experiments/wall-and-residual/depth-4-winograd-recovery-residual-1-875688566826284-s-memory/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/run_reference_capture.py)
- [../../../submissions/328439/estimator.py](../../../submissions/328439/estimator.py)

</details>

## Evidence

- [AUDIT_RECEIPT.md#L1](AUDIT_RECEIPT.md#L1)
- [PREREGISTER.md#L1](PREREGISTER.md#L1)
- [STATIC_VERIFICATION.json#L1](STATIC_VERIFICATION.json#L1)
- [INDEPENDENT_REDTEAM_AUDIT.md#L1](INDEPENDENT_REDTEAM_AUDIT.md#L1)
- [REFERENCE_EXECUTION_TRANSCRIPT.txt#L1](REFERENCE_EXECUTION_TRANSCRIPT.txt#L1)
- [REFERENCE_TERMINAL_FAILURE_RECEIPT.json#L1](REFERENCE_TERMINAL_FAILURE_RECEIPT.json#L1)
- [INDEPENDENT_REDTEAM_REFERENCE_TERMINAL_AUDIT.md#L1](INDEPENDENT_REDTEAM_REFERENCE_TERMINAL_AUDIT.md#L1)

<details>
<summary>All result files and records (7)</summary>

- [AUDIT_RECEIPT.md](AUDIT_RECEIPT.md) · record
- [PREREGISTER.md](PREREGISTER.md) · record
- [STATIC_VERIFICATION.json](STATIC_VERIFICATION.json) · result
- [INDEPENDENT_REDTEAM_AUDIT.md](INDEPENDENT_REDTEAM_AUDIT.md) · record
- [REFERENCE_EXECUTION_TRANSCRIPT.txt](REFERENCE_EXECUTION_TRANSCRIPT.txt) · record
- [REFERENCE_TERMINAL_FAILURE_RECEIPT.json](REFERENCE_TERMINAL_FAILURE_RECEIPT.json) · result
- [INDEPENDENT_REDTEAM_REFERENCE_TERMINAL_AUDIT.md](INDEPENDENT_REDTEAM_REFERENCE_TERMINAL_AUDIT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-08-26-depth-4-winograd-third-cumulant-transport-original-sealed-runtime) · [Copy provenance](../../../PROVENANCE.json)

