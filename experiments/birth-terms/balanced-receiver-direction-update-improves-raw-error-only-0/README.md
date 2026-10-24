# Balanced receiver-direction update improves raw error only 0.5570%; original directional pilot remains failed

2026-09-10 · experiment · rejected · measured

A separately qualified balanced direction produced a small actual endpoint gain after the earlier Euclidean information run failed a forward/reverse identity.

Recorded measurement:

    BALANCE_RESULT.md:
    zero-action baseline: mean final served raw MSE1.9864339305920217e-8 to
    1.9753686359110214e-8, ratio0.9944295682274706 (0.5570% improvement),4/4 wins.
    ratio0.992002599265859 was not substituted for the actual endpoint result.
    Per-net actual ratios:0.992652883,0.990884083,0.994595375,0.998748652.
    No adjusted-score, native, held-out, new-bank or sub3/sub2.7 claim follows.
    
    INFORMATION_FAILURE_RESULT.md:
    It missed by1.100134638x. Reverse-dot cancellation ratio16.94437763 is
    ratio0.99790858263551. This is a post-hoc failed-run diagnostic, NOT actual
    raw, native, adjusted or full20 evidence. Its approximately0.209% scale
    
    QUALIFICATION_V3_RESULT.md:
    or raw-error result. All7 wide MPS calls and8 tiny CPU/MPS calls passed;
    1.5469969074929684e-5. Wide reverse/JVP absolute identity errors
    3.721939036810916e-10 /2.37771631639923e-9, within fixed gates.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_comp_receiver_birth_v1/balance.py](code/current_comp_receiver_birth_v1/balance.py)
- [code/current_comp_receiver_birth_v1/diagnose_information_failure.py](code/current_comp_receiver_birth_v1/diagnose_information_failure.py)
- [code/current_comp_receiver_birth_v1/gradient_geometry_diagnostic.py](code/current_comp_receiver_birth_v1/gradient_geometry_diagnostic.py)
- [code/current_comp_receiver_birth_v1/information.py](code/current_comp_receiver_birth_v1/information.py)
- [code/current_comp_receiver_birth_v1/model.py](code/current_comp_receiver_birth_v1/model.py)
- [code/current_comp_receiver_birth_v1/native_geometry_draft.py](code/current_comp_receiver_birth_v1/native_geometry_draft.py)
- [code/current_comp_receiver_birth_v1/probe_ad_scalar.py](code/current_comp_receiver_birth_v1/probe_ad_scalar.py)
- [code/current_comp_receiver_birth_v1/qualification.py](code/current_comp_receiver_birth_v1/qualification.py)
- [code/current_comp_receiver_birth_v1/qualification_v2.py](code/current_comp_receiver_birth_v1/qualification_v2.py)
- [code/current_comp_receiver_birth_v1/qualification_v3.py](code/current_comp_receiver_birth_v1/qualification_v3.py)
- [code/current_comp_receiver_birth_v1/watch_balance.py](code/current_comp_receiver_birth_v1/watch_balance.py)
- [code/current_comp_receiver_birth_v1/watch_information.py](code/current_comp_receiver_birth_v1/watch_information.py)

<details>
<summary>Additional programs and dependencies (62)</summary>

- [code/current_comp_receiver_birth_v1/watch_qualification.py](code/current_comp_receiver_birth_v1/watch_qualification.py)
- [code/current_comp_receiver_birth_v1/watch_qualification_v2.py](code/current_comp_receiver_birth_v1/watch_qualification_v2.py)
- [code/current_comp_receiver_birth_v1/watch_qualification_v3.py](code/current_comp_receiver_birth_v1/watch_qualification_v3.py)
- [code/birth_direction_fit_v1/build_terminal_v1/estimator.py](code/birth_direction_fit_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_fit_v1/build_zero_v1/estimator.py](code/birth_direction_fit_v1/build_zero_v1/estimator.py)
- [code/birth_direction_fit_v1/check_meter_lifecycle_v2.py](code/birth_direction_fit_v1/check_meter_lifecycle_v2.py)
- [code/birth_direction_fit_v1/compare_ready.py](code/birth_direction_fit_v1/compare_ready.py)
- [code/birth_direction_fit_v1/fnp/build.py](code/birth_direction_fit_v1/fnp/build.py)
- [code/birth_direction_fit_v1/fnp/build_v2.py](code/birth_direction_fit_v1/fnp/build_v2.py)
- [code/birth_direction_fit_v1/fnp/check_small.py](code/birth_direction_fit_v1/fnp/check_small.py)
- [code/birth_direction_fit_v1/fnp/check_small_v2.py](code/birth_direction_fit_v1/fnp/check_small_v2.py)
- [code/birth_direction_fit_v1/fnp/core.py](code/birth_direction_fit_v1/fnp/core.py)
- [code/birth_direction_fit_v1/fnp/core_v2.py](code/birth_direction_fit_v1/fnp/core_v2.py)
- [code/birth_direction_fit_v1/fnp/reference_numpy.py](code/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/birth_direction_fit_v1/judge_terminal.py](code/birth_direction_fit_v1/judge_terminal.py)
- [code/birth_direction_fit_v1/model.py](code/birth_direction_fit_v1/model.py)
- [code/birth_direction_fit_v1/native.py](code/birth_direction_fit_v1/native.py)
- [code/birth_direction_fit_v1/native_v2.py](code/birth_direction_fit_v1/native_v2.py)
- [code/birth_direction_fit_v1/qualify_model.py](code/birth_direction_fit_v1/qualify_model.py)
- [code/birth_direction_fit_v1/qualify_native_v2.py](code/birth_direction_fit_v1/qualify_native_v2.py)
- [code/birth_direction_fit_v1/qualify_wide_v2.py](code/birth_direction_fit_v1/qualify_wide_v2.py)
- [code/birth_direction_fit_v1/train.py](code/birth_direction_fit_v1/train.py)
- [code/birth_direction_node_head_v1/build.py](code/birth_direction_node_head_v1/build.py)
- [code/birth_direction_node_head_v1/build_terminal_v1/estimator.py](code/birth_direction_node_head_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_node_head_v1/build_zero_v1/estimator.py](code/birth_direction_node_head_v1/build_zero_v1/estimator.py)
- [code/birth_direction_node_head_v1/core.py](code/birth_direction_node_head_v1/core.py)
- [code/birth_direction_node_head_v1/model.py](code/birth_direction_node_head_v1/model.py)
- [code/centered_bank_autodiff_v1.py](code/centered_bank_autodiff_v1.py)
- [code/centered_bank_autodiff_v3.py](code/centered_bank_autodiff_v3.py)
- [code/centered_torch_numerics_v1.py](code/centered_torch_numerics_v1.py)
- [code/centered_torch_numerics_v3.py](code/centered_torch_numerics_v3.py)
- [code/current_bd_p_coadaptation_v1/build_terminal.py](code/current_bd_p_coadaptation_v1/build_terminal.py)
- [code/current_bd_p_coadaptation_v1/model.py](code/current_bd_p_coadaptation_v1/model.py)
- [code/current_bd_p_coadaptation_v1/qualify.py](code/current_bd_p_coadaptation_v1/qualify.py)
- [code/current_bd_p_coadaptation_v1/train.py](code/current_bd_p_coadaptation_v1/train.py)
- [code/current_d50_full_return_capture_v1/common.py](code/current_d50_full_return_capture_v1/common.py)
- [code/current_d50_full_return_capture_v1/native.py](code/current_d50_full_return_capture_v1/native.py)
- [code/current_d50_native_objective_p1_v1/build.py](code/current_d50_native_objective_p1_v1/build.py)
- [code/current_d50_native_objective_p1_v1/build_v1/estimator.py](code/current_d50_native_objective_p1_v1/build_v1/estimator.py)
- [code/current_d50_native_objective_p1_v1/build_v1/null_estimator.py](code/current_d50_native_objective_p1_v1/build_v1/null_estimator.py)
- [code/current_d50_native_objective_p1_v1/common.py](code/current_d50_native_objective_p1_v1/common.py)
- [code/current_d50_native_objective_p1_v1/native.py](code/current_d50_native_objective_p1_v1/native.py)
- [code/current_d50_native_objective_p1_v1/native_v2.py](code/current_d50_native_objective_p1_v1/native_v2.py)
- [code/current_d50_native_objective_p1_v1/watch.py](code/current_d50_native_objective_p1_v1/watch.py)
- [code/current_d50_native_objective_p1_v1/worker.py](code/current_d50_native_objective_p1_v1/worker.py)
- [code/current_d50_terminal_linear_v2/common.py](code/current_d50_terminal_linear_v2/common.py)
- [code/current_d50_terminal_linear_v2/fit.py](code/current_d50_terminal_linear_v2/fit.py)
- [code/current_d50_terminal_linear_v2/fit_v1/estimator.py](code/current_d50_terminal_linear_v2/fit_v1/estimator.py)
- [code/current_d50_terminal_linear_v2/native.py](code/current_d50_terminal_linear_v2/native.py)
- [code/current_d50_terminal_linear_v2/native_validation.py](code/current_d50_terminal_linear_v2/native_validation.py)
- [code/current_d50_terminal_linear_v2/watch_fit.py](code/current_d50_terminal_linear_v2/watch_fit.py)
- [code/current_p1_native_objective_p2_monitor_v2/monitor.py](code/current_p1_native_objective_p2_monitor_v2/monitor.py)
- [code/current_p1_node_composition_v1/build.py](code/current_p1_node_composition_v1/build.py)
- [code/current_p1_node_composition_v1/build_v1/terminal_estimator.py](code/current_p1_node_composition_v1/build_v1/terminal_estimator.py)
- [code/current_transport_shadow_v1.py](code/current_transport_shadow_v1.py)
- [code/current_transport_shadow_v3.py](code/current_transport_shadow_v3.py)
- [code/full_carrier_autodiff_v1.py](code/full_carrier_autodiff_v1.py)
- [code/full_carrier_bank_autodiff_v1.py](code/full_carrier_bank_autodiff_v1.py)
- [code/gaussian_cdf_quadrature_v1.py](code/gaussian_cdf_quadrature_v1.py)
- [code/joint_prefix_final_data_v1.py](code/joint_prefix_final_data_v1.py)
- [code/tensor_clamp_min_v1.py](code/tensor_clamp_min_v1.py)
- [code/unused_materializations_v1/estimator.py](code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [BALANCE_RESULT.md#L3](BALANCE_RESULT.md#L3)
- [BALANCE_RESULT.md#L11](BALANCE_RESULT.md#L11)
- [BALANCE_RESULT.md#L12](BALANCE_RESULT.md#L12)
- [BALANCE_RESULT.md#L14](BALANCE_RESULT.md#L14)
- [BALANCE_RESULT.md#L15](BALANCE_RESULT.md#L15)
- [INFORMATION_FAILURE_RESULT.md#L3](INFORMATION_FAILURE_RESULT.md#L3)
- [INFORMATION_FAILURE_RESULT.md#L11](INFORMATION_FAILURE_RESULT.md#L11)
- [INFORMATION_FAILURE_RESULT.md#L21](INFORMATION_FAILURE_RESULT.md#L21)
- [INFORMATION_FAILURE_RESULT.md#L22](INFORMATION_FAILURE_RESULT.md#L22)
- [QUALIFICATION_V3_RESULT.md#L3](QUALIFICATION_V3_RESULT.md#L3)
- [QUALIFICATION_V3_RESULT.md#L4](QUALIFICATION_V3_RESULT.md#L4)
- [QUALIFICATION_V3_RESULT.md#L12](QUALIFICATION_V3_RESULT.md#L12)
- [QUALIFICATION_V3_RESULT.md#L13](QUALIFICATION_V3_RESULT.md#L13)

<details>
<summary>All result files and records (4)</summary>

- [BALANCE_RESULT.md](BALANCE_RESULT.md) · record
- [INFORMATION_FAILURE_RESULT.md](INFORMATION_FAILURE_RESULT.md) · record
- [QUALIFICATION_V3_RESULT.md](QUALIFICATION_V3_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-10-balanced-receiver-direction-update-improves-raw-error-only-0) · [Copy provenance](../../../PROVENANCE.json)

