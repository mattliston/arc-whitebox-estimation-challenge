# Frozen-mean covariance gradient direction, native response at |z|<=1/16: stability 0.151 (gate 0.05), curvature 0.119

2026-09-20 · experiment · rejected · measured

One covariance-only correction direction (a DCD congruence after recentering and masking at layers 1 to 14) was proposed from four MPS forward/backward passes of the older uncompressed carrier model supplied with the actual frozen lazy-rotation P values (the lazy card differs from the older COMP model only in 64 final-head coefficients and one bias), then calibrated natively on net 35 with probes 0, +-1/32 and +-1/64. The final-layer response stability and even-part curvature both fail their gates, so no scalar fit or further network ran; a saved-output diagnostic locates the stability first exceeding 5% at layer index 11.

Recorded measurement:

    pooled gradient norm 8.121537004905289, baseline affine response 39.59182411830987; response stability 0.151045702999259 vs <=0.05; signal RMS 3.9051514087004706e-5 (>=5e-6 passes); curvature 0.11886806250593529 vs <=0.10; stability 7.04/9.56/13.86/15.10% at layers 12..15; each probe call 398186088371 F (+105170674 F over the parent)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/covariance_direction_native_v4/build_probes.py](code/codex_resume_20260919_v1/covariance_direction_native_v4/build_probes.py)
- [code/codex_resume_20260919_v1/covariance_direction_native_v4/run_roster.py](code/codex_resume_20260919_v1/covariance_direction_native_v4/run_roster.py)
- [code/codex_resume_20260919_v1/covariance_direction_native_v4/analyze.py](code/codex_resume_20260919_v1/covariance_direction_native_v4/analyze.py)
- [code/codex_resume_20260919_v1/covariance_direction_native_v4/response_diagnostic.py](code/codex_resume_20260919_v1/covariance_direction_native_v4/response_diagnostic.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/build_zero_v1/estimator.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/build_zero_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/check_meter_lifecycle_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/check_meter_lifecycle_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/compare_ready.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/compare_ready.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/check_small.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/check_small.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/check_small_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/check_small_v2.py)

<details>
<summary>Additional programs and dependencies (66)</summary>

- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/judge_terminal.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/judge_terminal.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/model.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/model.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_model.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_model.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_native_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_wide_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/qualify_wide_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/train.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/train.py)
- [code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/build.py](code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/build_terminal_v1/estimator.py](code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/build_terminal_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/build_zero_v1/estimator.py](code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/build_zero_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/core.py](code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/model.py](code/astra_phase2_reexamination_v1/birth_direction_node_head_v1/model.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [code/astra_phase2_reexamination_v1/current_bd_p_coadaptation_v1/build_terminal.py](code/astra_phase2_reexamination_v1/current_bd_p_coadaptation_v1/build_terminal.py)
- [code/astra_phase2_reexamination_v1/current_bd_p_coadaptation_v1/model.py](code/astra_phase2_reexamination_v1/current_bd_p_coadaptation_v1/model.py)
- [code/astra_phase2_reexamination_v1/current_bd_p_coadaptation_v1/qualify.py](code/astra_phase2_reexamination_v1/current_bd_p_coadaptation_v1/qualify.py)
- [code/astra_phase2_reexamination_v1/current_bd_p_coadaptation_v1/train.py](code/astra_phase2_reexamination_v1/current_bd_p_coadaptation_v1/train.py)
- [code/astra_phase2_reexamination_v1/current_comp_joint_p_covariance_v1/model.py](code/astra_phase2_reexamination_v1/current_comp_joint_p_covariance_v1/model.py)
- [code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/build.py](code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/build.py)
- [code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/model.py](code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/model.py)
- [code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/qualification.py](code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/qualification.py)
- [code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/train.py](code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/train.py)
- [code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/watch_qualification.py](code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/watch_qualification.py)
- [code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/watch_train.py](code/astra_phase2_reexamination_v1/current_comp_joint_p_node_v1/watch_train.py)
- [code/astra_phase2_reexamination_v1/current_comp_receiver_birth_v1/model.py](code/astra_phase2_reexamination_v1/current_comp_receiver_birth_v1/model.py)
- [code/astra_phase2_reexamination_v1/current_comp_receiver_birth_v1/qualification.py](code/astra_phase2_reexamination_v1/current_comp_receiver_birth_v1/qualification.py)
- [code/astra_phase2_reexamination_v1/current_comp_receiver_birth_v1/watch_qualification.py](code/astra_phase2_reexamination_v1/current_comp_receiver_birth_v1/watch_qualification.py)
- [code/astra_phase2_reexamination_v1/current_d50_full_return_capture_v1/common.py](code/astra_phase2_reexamination_v1/current_d50_full_return_capture_v1/common.py)
- [code/astra_phase2_reexamination_v1/current_d50_full_return_capture_v1/native.py](code/astra_phase2_reexamination_v1/current_d50_full_return_capture_v1/native.py)
- [code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/build.py](code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/build.py)
- [code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/build_v1/estimator.py](code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/build_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/build_v1/null_estimator.py](code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/build_v1/null_estimator.py)
- [code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/common.py](code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/common.py)
- [code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/native.py](code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/native.py)
- [code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/native_v2.py](code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/watch.py](code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/watch.py)
- [code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/worker.py](code/astra_phase2_reexamination_v1/current_d50_native_objective_p1_v1/worker.py)
- [code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/common.py](code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/common.py)
- [code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/fit.py](code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/fit.py)
- [code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/fit_v1/estimator.py](code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/fit_v1/estimator.py)
- [code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/native.py](code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/native.py)
- [code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/native_validation.py](code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/native_validation.py)
- [code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/watch_fit.py](code/astra_phase2_reexamination_v1/current_d50_terminal_linear_v2/watch_fit.py)
- [code/astra_phase2_reexamination_v1/current_p1_native_objective_p2_monitor_v2/monitor.py](code/astra_phase2_reexamination_v1/current_p1_native_objective_p2_monitor_v2/monitor.py)
- [code/astra_phase2_reexamination_v1/current_p1_node_composition_v1/build.py](code/astra_phase2_reexamination_v1/current_p1_node_composition_v1/build.py)
- [code/astra_phase2_reexamination_v1/current_p1_node_composition_v1/build_v1/terminal_estimator.py](code/astra_phase2_reexamination_v1/current_p1_node_composition_v1/build_v1/terminal_estimator.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py](code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py)
- [code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)
- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)
- [code/codex_resume_20260919_v1/covariance_direction_native_v4/common.py](code/codex_resume_20260919_v1/covariance_direction_native_v4/common.py)
- [code/codex_resume_20260919_v1/covariance_direction_native_v4/compose.py](code/codex_resume_20260919_v1/covariance_direction_native_v4/compose.py)
- [code/codex_resume_20260919_v1/covariance_direction_native_v4/native.py](code/codex_resume_20260919_v1/covariance_direction_native_v4/native.py)
- [code/codex_resume_20260919_v1/lazy_rotation_objective_v1/candidate.py](code/codex_resume_20260919_v1/lazy_rotation_objective_v1/candidate.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)

</details>

## Evidence

- [RESULT.md#L9](../../../records/research/2026-09-20-covariance-direction-proposals-v1-v2-crashed-sigtrap-mps-allocator/RESULT.md#L9)
- [RESULT.md#L15](../../../records/research/2026-09-20-covariance-direction-proposals-v1-v2-crashed-sigtrap-mps-allocator/RESULT.md#L15)
- [RESULT.md#L18](../../../records/research/2026-09-20-covariance-direction-proposals-v1-v2-crashed-sigtrap-mps-allocator/RESULT.md#L18)
- [RESULT.md#L35](../../../records/research/2026-09-20-covariance-direction-proposals-v1-v2-crashed-sigtrap-mps-allocator/RESULT.md#L35)
- [RESULT.md#L36](../../../records/research/2026-09-20-covariance-direction-proposals-v1-v2-crashed-sigtrap-mps-allocator/RESULT.md#L36)
- [RESULT.md#L43](../../../records/research/2026-09-20-covariance-direction-proposals-v1-v2-crashed-sigtrap-mps-allocator/RESULT.md#L43)
- [RESULT.md#L44](../../../records/research/2026-09-20-covariance-direction-proposals-v1-v2-crashed-sigtrap-mps-allocator/RESULT.md#L44)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.md](../../../records/research/2026-09-20-covariance-direction-proposals-v1-v2-crashed-sigtrap-mps-allocator/RESULT.md) · result
- [RESPONSE_DIAGNOSTIC.md](RESPONSE_DIAGNOSTIC.md) · result
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-frozen-mean-covariance-gradient-direction-native-response-z-1) · [Copy provenance](../../../PROVENANCE.json)

