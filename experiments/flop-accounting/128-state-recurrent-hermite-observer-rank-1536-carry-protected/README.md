# 128-state recurrent Hermite observer on rank-1536 carry: protected error 9.421320969930728e-7

2026-08-23 · experiment · rejected · measured

The exact four-epoch CP1536 H128/P4/C256 recurrent-Hermite fit is finite and improves protected raw mean squared error (MSE) from host 3.7456453025317232e-6 to 9.421320969930728e-7 (74.85%), but misses the <1e-7 gate by 9.421x. A fixed 32-training-row diagnostic is 8.932946971969713e-7 and the held/train ratio is 1.05467, so the miss is fit/optimization rather than protected-fold generalization. Its exact zero-head meter is 209,904,700,250 FLOPs, yielding floor-adjusted 9.421320969930728e-8; the loss flattened across epochs, so this schedule is not extended.

Recorded measurement:

    The exact four-epoch CP1536 H128/P4/C256 recurrent-Hermite fit is finite and improves protected raw MSE from host 3.7456453025317232e-6 to 9.421320969930728e-7 (74.85%), but misses the <1e-7 gate by 9.421x. A fixed 32-training-row diagnostic is 8.932946971969713e-7 and the held/train ratio is 1.05467, so the miss is fit/optimization rather than protected-fold generalization. Its exact zero-head meter is 209,904,700,250 FLOPs, yielding floor-adjusted 9.421320969930728e-8; the loss flattened across epochs, so this schedule is not extended.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/estimator_primary_cp1536_h128.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/estimator_primary_cp1536_h128.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp1536_h128_p4_recurrent_discriminator_v1/run_experiment.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp1536_h128_p4_recurrent_discriminator_v1/run_experiment.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/full_data_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/full_data_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/sealed_role_targets_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/sealed_role_targets_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/target_blind_mini_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/target_blind_mini_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/estimator_control_cp1792_h64.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/estimator_control_cp1792_h64.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/estimator_generalized.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/estimator_generalized.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/observer_model.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/observer_model.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/selection_and_promotion.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/selection_and_promotion.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/checkpoint_converter.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/checkpoint_converter.py)

<details>
<summary>Additional programs and dependencies (9)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/cpu_training_core.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/cpu_training_core.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/feature_store_contract.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/feature_store_contract.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/flopscope_nonzero_worker.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/flopscope_nonzero_worker.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/sealed_bundle_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/sealed_bundle_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/torch_nonzero_worker.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/torch_nonzero_worker.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/trainer_evaluator_scaffold.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/trainer_evaluator_scaffold.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/verify_launch_scaffold.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/verify_launch_scaffold.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/verify_nonzero_parity.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/verify_nonzero_parity.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v6_training_runtime/fused_observer_v6.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v6_training_runtime/fused_observer_v6.py)

</details>

## Evidence

- [LAUNCH_COMMON_DRIVER_METERS_V3.json#L1](LAUNCH_COMMON_DRIVER_METERS_V3.json#L1)
- [STATIC_METER_ESTIMATE_V3.json#L1](STATIC_METER_ESTIMATE_V3.json#L1)
- [INDEPENDENT_LAUNCH_AUDIT_V4.json#L1](INDEPENDENT_LAUNCH_AUDIT_V4.json#L1)
- [PREREGISTRATION.json#L1](PREREGISTRATION.json#L1)
- [RUN_MANIFEST.json#L1](RUN_MANIFEST.json#L1)
- [PREREGISTRATION_SEAL.json#L1](PREREGISTRATION_SEAL.json#L1)
- [RESULTS.json#L45](RESULTS.json#L45)
- [verification.json#L26](verification.json#L26)
- [EXECUTION_RECONCILIATION.json#L1](EXECUTION_RECONCILIATION.json#L1)
- [graph_delta.json#L40](graph_delta.json#L40)

<details>
<summary>All result files and records (10)</summary>

- [LAUNCH_COMMON_DRIVER_METERS_V3.json](LAUNCH_COMMON_DRIVER_METERS_V3.json) · result
- [STATIC_METER_ESTIMATE_V3.json](STATIC_METER_ESTIMATE_V3.json) · result
- [INDEPENDENT_LAUNCH_AUDIT_V4.json](INDEPENDENT_LAUNCH_AUDIT_V4.json) · result
- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [RUN_MANIFEST.json](RUN_MANIFEST.json) · result
- [PREREGISTRATION_SEAL.json](PREREGISTRATION_SEAL.json) · result
- [RESULTS.json](RESULTS.json) · result
- [verification.json](verification.json) · result
- [EXECUTION_RECONCILIATION.json](EXECUTION_RECONCILIATION.json) · result
- [graph_delta.json](graph_delta.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-23-128-state-recurrent-hermite-observer-rank-1536-carry-protected) · [Copy provenance](../../../PROVENANCE.json)

