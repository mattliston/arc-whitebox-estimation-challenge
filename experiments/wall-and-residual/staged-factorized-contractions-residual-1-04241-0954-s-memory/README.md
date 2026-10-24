# Staged factorized contractions: residual 1.0424–1.0954 s and memory 11,619,074,048 bytes failed

2026-08-26 · experiment · rejected · measured

Tiny admission passed at 7.248 ms median excess, but full integration failed every call's residual/resident-set size (RSS) resource conjunction: residuals 1.0424-1.0954 s and maximum RSS 11,619,074,048 bytes. Exact bills, drift, schema, and repeat parity passed.

Recorded measurement:

    Tiny admission passed at 7.248 ms median excess, but full integration failed every call's residual/RSS resource conjunction: residuals 1.0424-1.0954 s and maximum RSS 11,619,074,048 bytes. Exact bills, drift, schema, and repeat parity passed.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/full_lower_covariance_innovation_supervision_v1/statistics.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/run_guarded.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1/run_guarded.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/run_kernel_smoke.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d5_cp3_winograd_hybrid_v1/run_kernel_smoke.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7r1_cp3_staged_factorized_einsum_recovery_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7r1_cp3_staged_factorized_einsum_recovery_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7r1_cp3_staged_factorized_einsum_recovery_v1/run_full_estimator_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7r1_cp3_staged_factorized_einsum_recovery_v1/run_full_estimator_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7r1_cp3_staged_factorized_einsum_recovery_v1/run_tiny_runtime_discriminator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7r1_cp3_staged_factorized_einsum_recovery_v1/run_tiny_runtime_discriminator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7r1_cp3_staged_factorized_einsum_recovery_v1/staged_winograd_kernel.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7r1_cp3_staged_factorized_einsum_recovery_v1/staged_winograd_kernel.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7r1_cp3_staged_factorized_einsum_recovery_v1/verify_d7r1_launch_seal.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7r1_cp3_staged_factorized_einsum_recovery_v1/verify_d7r1_launch_seal.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7r1_cp3_staged_factorized_einsum_recovery_v1/verify_d7r1_tiny_result.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7r1_cp3_staged_factorized_einsum_recovery_v1/verify_d7r1_tiny_result.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7r1_cp3_staged_factorized_einsum_recovery_v1/verify_design_static.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d7r1_cp3_staged_factorized_einsum_recovery_v1/verify_design_static.py)

</details>

## Evidence

- [D7R1_TINY_RUNTIME_RESULT.json#L1](D7R1_TINY_RUNTIME_RESULT.json#L1)
- [FULL_INTEGRATION_BUILD_RECEIPT.json#L1](FULL_INTEGRATION_BUILD_RECEIPT.json#L1)
- [FULL_ESTIMATOR_GATE_RESULT.json#L1052](FULL_ESTIMATOR_GATE_RESULT.json#L1052)

<details>
<summary>All result files and records (3)</summary>

- [D7R1_TINY_RUNTIME_RESULT.json](D7R1_TINY_RUNTIME_RESULT.json) · result
- [FULL_INTEGRATION_BUILD_RECEIPT.json](FULL_INTEGRATION_BUILD_RECEIPT.json) · result
- [FULL_ESTIMATOR_GATE_RESULT.json](FULL_ESTIMATOR_GATE_RESULT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-26-staged-factorized-contractions-residual-1-04241-0954-s-memory) · [Copy provenance](../../../PROVENANCE.json)

