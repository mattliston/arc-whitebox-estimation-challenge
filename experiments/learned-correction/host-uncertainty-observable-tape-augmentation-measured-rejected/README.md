# Host-uncertainty observable tape augmentation: measured and rejected

2026-08-25 · experiment · rejected · measured

The two-stage aug7 source-information arm is terminal. Stage 1 regenerated the development tape with seven target-free source observables at 688/688 bit-parity against the sealed v3 store. Stage 2, the exact frozen J architecture with only the input widened to 27 raw channels, reached verified out-of-fold (OOF) final mean squared error (MSE) 3.230109930992383e-8 versus J 3.2334958229470984e-8: a 0.1047% improvement with three of four folds improving, 92.01% of gross paired row movement canceling, and the inherited rank-643 breach surviving at 3.3988262727543796e-7.

Recorded measurement:

    The two-stage aug7 source-information arm is terminal. Stage 1 regenerated the development tape with seven target-free source observables at 688/688 bit-parity against the sealed v3 store. Stage 2, the exact frozen J architecture with only the input widened to 27 raw channels, reached verified OOF final MSE 3.230109930992383e-8 versus J 3.2334958229470984e-8: a 0.1047% improvement with three of four folds improving, 92.01% of gross paired row movement canceling, and the inherited rank-643 breach surviving at 3.3988262727543796e-7. Every aug channel's correlation with per-row gain is at most 0.046 in absolute value, so the added information did not couple to the residual. The gate fails every accuracy conjunct. Q lands within 3.1e-6 relative of N's 3.230119894372402e-8 from a materially different mechanism, consistent with a shared development-frame information ceiling near 3.2301e-8. The preregistered frame-mismatch caveat stands: these channels describe the cp1792 development host's approximation gaps while the target is the Q24 host's error, so this null is ambiguous between a true information cap and wrong-frame channels; the native-frame instance resolves that ambiguity.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/full_data_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/full_data_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/target_blind_mini_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/target_blind_mini_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/estimator_control_cp1792_h64.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/estimator_control_cp1792_h64.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/estimator_generalized.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/estimator_generalized.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/estimator_primary_cp1536_h128.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v3/estimator_primary_cp1536_h128.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store/contract.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store/contract.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store/generate_feature_store_once.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store/generate_feature_store_once.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store/host_feature_worker.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store/host_feature_worker.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store/sealed_feature_store_provider.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store/sealed_feature_store_provider.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store/verify_sealed_feature_store.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store/verify_sealed_feature_store.py)

<details>
<summary>Additional programs and dependencies (23)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/contract_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/contract_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/generate_feature_store_once_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/generate_feature_store_once_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/host_feature_worker_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/host_feature_worker_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/host_worker_exec_gate_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/host_worker_exec_gate_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/resource_safety_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/resource_safety_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/sealed_feature_store_provider_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/sealed_feature_store_provider_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/synthetic_group_tree_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/synthetic_group_tree_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/target_blind_cursor_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/target_blind_cursor_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/test_injected_resource_failures_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/test_injected_resource_failures_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/verify_artifacts_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/verify_artifacts_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/verify_static_preflight_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v2/verify_static_preflight_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v3/contract_v3.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v3/contract_v3.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v3/host_feature_worker_v3.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_feature_store_v3/host_feature_worker_v3.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/checkpoint_converter.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/checkpoint_converter.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/flopscope_nonzero_worker.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/learned_residual_observer_design/v4_launch/flopscope_nonzero_worker.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_aug7_source_observable_tape_v1/generate_aug7.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_aug7_source_observable_tape_v1/generate_aug7.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_aug7_source_observable_tape_v1/make_generation_receipt.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_aug7_source_observable_tape_v1/make_generation_receipt.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_aug7_widened_j_cv_v1/analyze_q_vs_j.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_aug7_widened_j_cv_v1/analyze_q_vs_j.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_aug7_widened_j_cv_v1/run_experiment.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_aug7_widened_j_cv_v1/run_experiment.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_aug7_widened_j_cv_v1/verify_results.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_aug7_widened_j_cv_v1/verify_results.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/instrumented_estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/instrumented_estimator.py)

</details>

## Evidence

- [PREREGISTRATION.json#L1](PREREGISTRATION.json#L1)
- [AUDIT_RECEIPT.json#L1](AUDIT_RECEIPT.json#L1)
- [GENERATION_RECEIPT.json#L1](GENERATION_RECEIPT.json#L1)
- [PREREGISTRATION-5cd9a7fb.json#L12](PREREGISTRATION-5cd9a7fb.json#L12)
- [AUDIT_RECEIPT-647bcc17.json#L30](AUDIT_RECEIPT-647bcc17.json#L30)
- [PREREGISTRATION_SEAL.json#L1](PREREGISTRATION_SEAL.json#L1)
- [TRAINING_RELEASE.json#L1](TRAINING_RELEASE.json#L1)
- [CV_RESULTS.json#L1](CV_RESULTS.json#L1)
- [VERIFICATION.json#L1](VERIFICATION.json#L1)
- [Q_VS_J_OOF_DECOMPOSITION_AND_CLOSURE.json#L124](Q_VS_J_OOF_DECOMPOSITION_AND_CLOSURE.json#L124)

<details>
<summary>All result files and records (10)</summary>

- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [AUDIT_RECEIPT.json](AUDIT_RECEIPT.json) · result
- [GENERATION_RECEIPT.json](GENERATION_RECEIPT.json) · result
- [PREREGISTRATION-5cd9a7fb.json](PREREGISTRATION-5cd9a7fb.json) · result
- [AUDIT_RECEIPT-647bcc17.json](AUDIT_RECEIPT-647bcc17.json) · result
- [PREREGISTRATION_SEAL.json](PREREGISTRATION_SEAL.json) · result
- [TRAINING_RELEASE.json](TRAINING_RELEASE.json) · result
- [CV_RESULTS.json](CV_RESULTS.json) · result
- [VERIFICATION.json](VERIFICATION.json) · result
- [Q_VS_J_OOF_DECOMPOSITION_AND_CLOSURE.json](Q_VS_J_OOF_DECOMPOSITION_AND_CLOSURE.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-25-host-uncertainty-observable-tape-augmentation-measured-rejected) · [Copy provenance](../../../PROVENANCE.json)

