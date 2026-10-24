# Complete prefix and endpoint co-adaptation toolchain: validation passed

2026-09-06T18:22Z · experiment · informational · measured

All 78,097 trainable coefficients in the existing prefix messages, encoder, mean heads and gains, plus the actual final Q, were differentiated through the complete dynamic CP/star/trace carrier. Original CPU f64 versus repaired native MPS f32 pooled parameter-gradient relative error was 8.3697402e-6 on rank 35 and 1.0061080e-5 on rank 100. An independent target-free finite test perturbed ALL these groups together along one fixed direction, at relative group-RMS amplitudes +/-0.05 and +/-0.025.

Recorded measurement:

    All 78,097 trainable coefficients in the existing prefix messages, encoder, mean heads and gains, plus the actual final Q, were differentiated through the complete dynamic CP/star/trace carrier. Original CPU f64 versus repaired native MPS f32 pooled parameter-gradient relative error was 8.3697402e-6 on rank 35 and 1.0061080e-5 on rank 100. The worst non-negligible group was 1.9422288e-5; all unchanged gates passed. MPS backward took 3.159 / 2.447 s. This establishes point-VJP agreement, not global smoothness. An independent target-free finite test perturbed ALL these groups together along one fixed direction, at relative group-RMS amplitudes +/-0.05 and +/-0.025. Actual float32 parameter bytes were frozen by the NumPy stage and reused by MPS. The original NumPy zero point was byte-exact. All finite trajectories agreed within 1.495e-6 maximum mean error. CPU/MPS central final-response discrepancies were 1.791% / 3.439% on rank 35 and 1.475% / 2.954% on rank 100, passing the fixed 5% implementation gate. Old finite-difference-versus-local-derivative failures remain failures.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [code/astra_phase2_reexamination_v1/check_joint_prefix_final_finite_v1.py](code/astra_phase2_reexamination_v1/check_joint_prefix_final_finite_v1.py)
- [code/astra_phase2_reexamination_v1/check_joint_prefix_final_gradients_v1.py](code/astra_phase2_reexamination_v1/check_joint_prefix_final_gradients_v1.py)
- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)

<details>
<summary>Additional programs and dependencies (16)</summary>

- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py](code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py)
- [code/astra_phase2_reexamination_v1/joint_prefix_final_numpy_v1.py](code/astra_phase2_reexamination_v1/joint_prefix_final_numpy_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)
- [code/astra_phase2_reexamination_v1/train_joint_prefix_final_v1.py](code/astra_phase2_reexamination_v1/train_joint_prefix_final_v1.py)
- [code/astra_phase2_reexamination_v1/verify_joint_prefix_final_provenance_v1.py](code/astra_phase2_reexamination_v1/verify_joint_prefix_final_provenance_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [JOINT_PREFIX_FINAL_FIDELITY_VERDICT_V1.md#L1](JOINT_PREFIX_FINAL_FIDELITY_VERDICT_V1.md#L1)
- [JOINT_PREFIX_FINAL_FIDELITY_VERDICT_V1.md#L11](JOINT_PREFIX_FINAL_FIDELITY_VERDICT_V1.md#L11)
- [JOINT_PREFIX_FINAL_FIDELITY_VERDICT_V1.md#L20](JOINT_PREFIX_FINAL_FIDELITY_VERDICT_V1.md#L20)

<details>
<summary>All result files and records (4)</summary>

- [JOINT_PREFIX_FINAL_FIDELITY_VERDICT_V1.md](JOINT_PREFIX_FINAL_FIDELITY_VERDICT_V1.md) · record
- [joint_prefix_final_gradients_v1.json](joint_prefix_final_gradients_v1.json) · result
- [joint_prefix_final_finite_numpy_v1.json](joint_prefix_final_finite_numpy_v1.json) · result
- [joint_prefix_final_finite_mps_v1.json](joint_prefix_final_finite_mps_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-complete-prefix-endpoint-co-adaptation-toolchain-validation-passed) · [Copy provenance](../../../PROVENANCE.json)

