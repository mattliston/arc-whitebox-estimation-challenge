# Repaired native Metal implementation: forward and local vector-Jacobian product fidelity established

2026-09-06T14:28Z · experiment · informational · measured

Installed Torch2.7 MPS scalar clamp_min bound reuse was the dominant late forward failure. In same-shaped tensors the first small scalar bound was reused for later1e-10/1e-12/1e-30/0 bounds. The tempting generic learned bidirectional pass was also checked against the wider Phase2 record: public_full688_backward_adjoint_state_cv_v1 already executed and closed.

Recorded measurement:

    Installed Torch2.7 MPS scalar clamp_min bound reuse was the dominant late forward failure. In same-shaped tensors the first small scalar bound was reused for later1e-10/1e-12/1e-30/0 bounds. Three tiny widths and different bound orders reproduced it: first correct, all subsequent different bounds wrong. Explicit tensor bounds were correct. This is an observed local backend defect; no upstream-version claim or public forum post was made. In the model, a1e-12 post-variance floor became1e-10, giving post sigma1e-5 instead of1e-6 for negative-variance neurons. Post-correlation feature columns27/28 differed by up to.556, changing the learned head by~1e-5. Resetting every incoming MPS state to rounded CPUf64 still reproduced this; the moment host itself differed<8e-7. Centered scalar/pair algebra alone did NOT fix the GPU failure. The old failed records remain unmodified. | Net | Max mean difference | Final difference RMS | Memberships | |---|---:|---:|---| |35|1.39839787394e-6|2.38354724806e-7|all30 equal| |100|1.39716386938e-6|2.63501890630e-7|all30 equal| Both pass the unchanged2e-6 maximum gate.1.988seconds for both in the audited replay; process RSS890,011,648B is NOT total GPU memory. Output SHA48736c9b55493164ab012a54ad73fb29efebc314fd6b341eac9771e71cfc96a0. CPUf32 BLAS baseline still fails7.5e-6; do not claim that path was repaired. CPUf64 centered control passes~2e-7. Small scalar/pair value and reverse gradient certificates pass CPUf64/f32 and MPSf32, including alpha0. One fixed random final-layer projection per net, all16x1024 mean-control VJPs, original CPUf64 bank versus repaired MPSf32: relative2.65834453e-6 /5.39764156e-6. Max per-layer relative9.701e-6. All finite. This is strong local point-Jacobian agreement with an already independently certified CPU derivative, not global smoothness. The dynamic finite-step composite remains FAILED: some layer13 perturbations at the fixed2e-4/1e-4 amplitudes change selection membership. A separate zero-point fixed-membership AND order certificate also remains FAILED on rank100large (5.7781% FD-vsAD; half3.6287%). Rank35 both pass. No amplitude, direction, tolerance or old pass flag was changed. The same steps through the ORIGINAL CPUf64 bank explain the latter issue: rank100 CPU FD-vsAD6.0065%/2.8912%. CPU-versus-MPS finite-step discrepancy is only.1703%/.1211% on35 and.2150%/.7173% on100, all below the fixed1% backend-consistency diagnostic. Thus the finite step itself is not a good linearization in that case, even with the original CPU card. No invented GPU derivative fix is indicated by that discrepancy. This does not promote the failed finite-step gates or authorize a blanket training sweep. Peak mixed-worker process RSS7,866,630,144B; largest observed GPU driver allocation6,504,054,784B, not independent additive peak totals. Swap stayed 5910.50MiB. All workers terminated. No target means loaded by these tests. The tempting generic learned bidirectional pass was also checked against the wider Phase2 record: public_full688_backward_adjoint_state_cv_v1 already executed and closed. LANE_direct_diagram_adjoint.md section121.2 records 3.23376600912e-8 versus3.23349582295e-8,330wins/358losses. It must not be launched as if novel merely because its design README sounds unexecuted.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/astra_phase2_reexamination_v1/capture_pair_jet_opportunity_stage_a_v1.py](code/astra_phase2_reexamination_v1/capture_pair_jet_opportunity_stage_a_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [code/astra_phase2_reexamination_v1/certify_pair_jet_coalescence_v1.py](code/astra_phase2_reexamination_v1/certify_pair_jet_coalescence_v1.py)
- [code/astra_phase2_reexamination_v1/check_centered_bank_baseline_v1.py](code/astra_phase2_reexamination_v1/check_centered_bank_baseline_v1.py)
- [code/astra_phase2_reexamination_v1/check_centered_bank_baseline_v3.py](code/astra_phase2_reexamination_v1/check_centered_bank_baseline_v3.py)
- [code/astra_phase2_reexamination_v1/check_centered_mps_bank_gradients_v1.py](code/astra_phase2_reexamination_v1/check_centered_mps_bank_gradients_v1.py)
- [code/astra_phase2_reexamination_v1/check_joint_prefix_final_gradients_v1.py](code/astra_phase2_reexamination_v1/check_joint_prefix_final_gradients_v1.py)
- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)

<details>
<summary>Additional programs and dependencies (21)</summary>

- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
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

- [CENTERED_MPS_NUMERICS_V3_FINAL_VERDICT.md#L1](../whole-estimator-vector-jacobian-product-agrees-finite-dynamic-step/CENTERED_MPS_NUMERICS_V3_FINAL_VERDICT.md#L1)
- [CENTERED_MPS_NUMERICS_V3_FINAL_VERDICT.md#L39](../whole-estimator-vector-jacobian-product-agrees-finite-dynamic-step/CENTERED_MPS_NUMERICS_V3_FINAL_VERDICT.md#L39)

<details>
<summary>All result files and records (1)</summary>

- [CENTERED_MPS_NUMERICS_V3_FINAL_VERDICT.md](../whole-estimator-vector-jacobian-product-agrees-finite-dynamic-step/CENTERED_MPS_NUMERICS_V3_FINAL_VERDICT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-repaired-native-metal-implementation-forward-local-vector-jacobian-product) · [Copy provenance](../../../PROVENANCE.json)

