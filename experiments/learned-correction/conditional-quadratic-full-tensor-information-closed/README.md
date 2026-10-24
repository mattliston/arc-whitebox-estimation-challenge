# Conditional quadratic full-tensor information: closed

2026-09-05T19:14Z · experiment · rejected · measured

CP+star+trace contractions passed synthetic dense/rotation certificates below4e-16. All24 generated honest teacher trajectories, old scalar/mechanistic features and old-head outputs were byte-identical. Fixed40-epoch same-capacity probe (20train, held-out37/39/42/48), final epoch: held-out parent residual MSE4.80933814995e-9; zero-feature control 4.81988537970e-9; new full-tensor quadratic features4.82016915271e-9.

Recorded measurement:

    CP+star+trace contractions passed synthetic dense/rotation certificates below4e-16. All24 generated honest teacher trajectories, old scalar/mechanistic features and old-head outputs were byte-identical. New8-mode joint invariants were computed before the current target correction. Three low priority CPU workers, two BLAS threads each, stable5966.50MiB swap. Fixed40-epoch same-capacity probe (20train, held-out37/39/42/48), final epoch: held-out parent residual MSE4.80933814995e-9; zero-feature control 4.81988537970e-9; new full-tensor quadratic features4.82016915271e-9. Joint/control1.00005888; joint/parent1.00225208;0/4 nets improved. MPS fit4.15seconds. This fails the predeclared information gate. These are honest teacher residuals, not final-layer closed-loop or adjusted scores.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/check_conditional_quadratic_tensor_v1.py](code/astra_phase2_reexamination_v1/check_conditional_quadratic_tensor_v1.py)
- [code/astra_phase2_reexamination_v1/conditional_quadratic_tensor_v1.py](code/astra_phase2_reexamination_v1/conditional_quadratic_tensor_v1.py)
- [code/astra_phase2_reexamination_v1/generate_conditional_quadratic_tensor_probe_v1.py](code/astra_phase2_reexamination_v1/generate_conditional_quadratic_tensor_probe_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/run_conditional_quadratic_tensor_probe_rows_v1.py](code/astra_phase2_reexamination_v1/run_conditional_quadratic_tensor_probe_rows_v1.py)

<details>
<summary>Additional programs and dependencies (3)</summary>

- [code/astra_phase2_reexamination_v1/train_conditional_quadratic_tensor_probe_v1.py](code/astra_phase2_reexamination_v1/train_conditional_quadratic_tensor_probe_v1.py)
- [code/astra_phase2_reexamination_v1/train_conditional_skew_energy_probe_v1.py](code/astra_phase2_reexamination_v1/train_conditional_skew_energy_probe_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [CONDITIONAL_QUADRATIC_TENSOR_PROBE_V1_CLOSED.md#L1](CONDITIONAL_QUADRATIC_TENSOR_PROBE_V1_CLOSED.md#L1)
- [CONDITIONAL_QUADRATIC_TENSOR_PROBE_V1_CLOSED.md#L13](CONDITIONAL_QUADRATIC_TENSOR_PROBE_V1_CLOSED.md#L13)

<details>
<summary>All result files and records (5)</summary>

- [CONDITIONAL_QUADRATIC_TENSOR_PROBE_V1_CLOSED.md](CONDITIONAL_QUADRATIC_TENSOR_PROBE_V1_CLOSED.md) · record
- [conditional_quadratic_tensor_certificate_v1.json](conditional_quadratic_tensor_certificate_v1.json) · result
- [conditional_quadratic_tensor_probe_rows_v1.json](conditional_quadratic_tensor_probe_rows_v1.json) · result
- [conditional_quadratic_tensor_probe_verdict_v1.json](conditional_quadratic_tensor_probe_verdict_v1.json) · result
- [CONDITIONAL_QUADRATIC_TENSOR_PROBE_V1.md](CONDITIONAL_QUADRATIC_TENSOR_PROBE_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-conditional-quadratic-full-tensor-information-closed) · [Copy provenance](../../../PROVENANCE.json)

