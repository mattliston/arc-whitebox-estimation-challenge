# Partial six-level products improve native40 adjusted score by 1.268286%, with 38/40 wins

2026-09-10 · experiment · adopted · measured

The fixed product composition retained learned physics and passed independent-truth and persistent-client qualification.

Recorded measurement:

    CHECK4_RESULT_20260910.md:
    | Quantity | Current COMP | COMP + partial-D6 | Ratio |
    | Final raw MSE | 1.9933680058595472e-8 | 1.9933019837010814e-8 | .9999668790919329 |
    | Complete mean FLOPs | 475126242316.25 | 468490115996.25 | .9860329198243213 |
    | Mean adjusted | 4.2964338254612944e-9 | 4.2363791826256295e-9 | .9860222116119252 |
    3/4 gate passes. This is a 1.39778% adjusted gain; the .003312% raw change is
    zero added swap. Maximum own primary wall34.562s. This is not NumPy-free
    production qualification; separate60s/.4s gates remain mandatory.
    
    INDEPENDENT14_RESULT_20260910.md:
    bills, mean per-net unclipped noise-corrected raw times max(.1,F/2^41):
    | public8 | 4.77415874629813e-9 | 4.702791678952999e-9 | .9850513836808488 |
    | virgin6 | 3.7345329733435845e-9 | 3.690935258327842e-9 | .9883257919191141 |
    | combined14 | 4.328604843603325e-9 | 4.2691389272565025e-9 | .9862621055754953 |
    Both prospective groups strictly improve. Combined NC raw2.0305056229588224e-8
    Combined observed raw3.8595936464622675e-8 to3.859461048333421e-8, observed
    adjusted8.21423578134458e-9 to8.101497183256336e-9. MC noise subtraction is
    1.8290880235034444e-8 mean raw; NC and observed values are not interchangeable.
    Public8 and combined means are Pareto in raw/F; virgin6 has a tiny raw increase
    1.7609719729578412e-8 to1.761030255761915e-8, outweighed by lower own cost.
    
    NATIVE40_RESULT_20260910.md:
    passes the original40-network native adjusted-objective gate against COMP
    | Exact original40 primary mean | COMP | Partial-D6 composition |
    | Final raw MSE |2.092035131674627e-8|2.0920079482848703e-8|
    | Complete own FLOPs |468313646101.175|462499976825.975|
    | Mean of per-net raw times own multiplier |4.451951498340901e-9|4.395488008616848e-9|
    Adjusted ratio0.9873171372722512 (1.268286% improvement);38/40 strict
    adjusted wins, satisfying the prospectively fixed>=24 gate. Raw ratio
    0.9999870062460495,22/40 raw wins. Complete-F ratio0.9875859494516117.
    (-0.0012994%), not evidence of new raw physics. The benefit comes from
    Both cold binding costs and actual partial-D6 tariffs were included.
    Maximum newly measured call wall time35.258882584s. This is native
    on BOTH public8 and virgin6 noise-corrected adjusted means against
    remains #330390 adjusted4.53886349950395e-9, and the<1e-9 final goal
    
    PRODUCTION_RESULT_20260910.md:
    Official archived bank330390 remains4.53886349950395e-9 adjusted.
    respective adjusted means are4.395488008616848e-9 (existing1e9 references)
    | Call/rank | Own complete F | Wall seconds | Residual seconds | Max absolute native difference, all16 layers |
    | cold35 |464400020054|37.470620667|0.157498096|2.86102294921875e-6|
    |100|465819626962|33.952085542|0.137518361|2.6226043701171875e-6|
    |500|470382991125|34.420548792|0.146824657|2.86102294921875e-6|
    |3016500|445803516158|35.009432792|0.148392022|3.4570693969726562e-6|
    |repeat35|462455049756|36.364029875|0.158089593|2.86102294921875e-6|
    All five pass wall<60s, residual<.4s, completeF<=2^41, finite/nonnegative
    Each passes the unchanged1% relative fidelity gate. The four-net remote
    noise-corrected adjusted mean4.235265081312536e-9 is ONLY this local
    Preparation separately passed15.927392249926925s/60620800B, with no

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/current_comp_partial_d6_v1/build.py](code/current_comp_partial_d6_v1/build.py)
- [code/current_comp_partial_d6_v1/native.py](code/current_comp_partial_d6_v1/native.py)
- [code/current_comp_partial_d6_v1/native40.py](code/current_comp_partial_d6_v1/native40.py)
- [code/current_comp_partial_d6_v1/panel_admit.py](code/current_comp_partial_d6_v1/panel_admit.py)
- [code/current_comp_partial_d6_v1/panel_admit_v2.py](code/current_comp_partial_d6_v1/panel_admit_v2.py)
- [code/current_comp_partial_d6_v1/panel_common.py](code/current_comp_partial_d6_v1/panel_common.py)
- [code/current_comp_partial_d6_v1/panel_common_v2.py](code/current_comp_partial_d6_v1/panel_common_v2.py)
- [code/current_comp_partial_d6_v1/panel_hash_v2.py](code/current_comp_partial_d6_v1/panel_hash_v2.py)
- [code/current_comp_partial_d6_v1/panel_native.py](code/current_comp_partial_d6_v1/panel_native.py)
- [code/current_comp_partial_d6_v1/panel_native_v2.py](code/current_comp_partial_d6_v1/panel_native_v2.py)
- [code/current_comp_partial_d6_v1/panel_score.py](code/current_comp_partial_d6_v1/panel_score.py)
- [code/current_comp_partial_d6_v1/panel_watch.py](code/current_comp_partial_d6_v1/panel_watch.py)

<details>
<summary>Additional programs and dependencies (194)</summary>

- [code/current_comp_partial_d6_v1/panel_watch_score.py](code/current_comp_partial_d6_v1/panel_watch_score.py)
- [code/current_comp_partial_d6_v1/panel_watch_v2.py](code/current_comp_partial_d6_v1/panel_watch_v2.py)
- [code/current_comp_partial_d6_v1/production_check.py](code/current_comp_partial_d6_v1/production_check.py)
- [code/current_comp_partial_d6_v1/production_check_v2.py](code/current_comp_partial_d6_v1/production_check_v2.py)
- [code/current_comp_partial_d6_v1/production_watch.py](code/current_comp_partial_d6_v1/production_watch.py)
- [code/current_comp_partial_d6_v1/production_watch_v2.py](code/current_comp_partial_d6_v1/production_watch_v2.py)
- [code/current_comp_partial_d6_v1/profile_panel_admission_v1.py](code/current_comp_partial_d6_v1/profile_panel_admission_v1.py)
- [code/current_comp_partial_d6_v1/score.py](code/current_comp_partial_d6_v1/score.py)
- [code/current_comp_partial_d6_v1/score40.py](code/current_comp_partial_d6_v1/score40.py)
- [code/current_comp_partial_d6_v1/score_v2.py](code/current_comp_partial_d6_v1/score_v2.py)
- [code/current_comp_partial_d6_v1/watch_build.py](code/current_comp_partial_d6_v1/watch_build.py)
- [code/current_comp_partial_d6_v1/watch_native.py](code/current_comp_partial_d6_v1/watch_native.py)
- [code/current_comp_partial_d6_v1/watch_native40.py](code/current_comp_partial_d6_v1/watch_native40.py)
- [code/current_comp_partial_d6_v1/watch_score.py](code/current_comp_partial_d6_v1/watch_score.py)
- [code/current_comp_partial_d6_v1/watch_score40.py](code/current_comp_partial_d6_v1/watch_score40.py)
- [code/current_comp_partial_d6_v1/watch_score_v2.py](code/current_comp_partial_d6_v1/watch_score_v2.py)
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
- [code/birth_direction_fit_v1/judge_terminal_v2.py](code/birth_direction_fit_v1/judge_terminal_v2.py)
- [code/birth_direction_fit_v1/model.py](code/birth_direction_fit_v1/model.py)
- [code/birth_direction_fit_v1/native.py](code/birth_direction_fit_v1/native.py)
- [code/birth_direction_fit_v1/native_v2.py](code/birth_direction_fit_v1/native_v2.py)
- [code/birth_direction_fit_v1/panel.py](code/birth_direction_fit_v1/panel.py)
- [code/birth_direction_fit_v1/panel_v2.py](code/birth_direction_fit_v1/panel_v2.py)
- [code/birth_direction_fit_v1/qualify_model.py](code/birth_direction_fit_v1/qualify_model.py)
- [code/birth_direction_fit_v1/qualify_native_v2.py](code/birth_direction_fit_v1/qualify_native_v2.py)
- [code/birth_direction_fit_v1/qualify_wide_v2.py](code/birth_direction_fit_v1/qualify_wide_v2.py)
- [code/birth_direction_fit_v1/release_confirmation_v1/check_release.py](code/birth_direction_fit_v1/release_confirmation_v1/check_release.py)
- [code/birth_direction_fit_v1/train.py](code/birth_direction_fit_v1/train.py)
- [code/birth_direction_node_head_v1/build.py](code/birth_direction_node_head_v1/build.py)
- [code/birth_direction_node_head_v1/build_terminal_v1/estimator.py](code/birth_direction_node_head_v1/build_terminal_v1/estimator.py)
- [code/birth_direction_node_head_v1/build_zero_v1/estimator.py](code/birth_direction_node_head_v1/build_zero_v1/estimator.py)
- [code/birth_direction_node_head_v1/core.py](code/birth_direction_node_head_v1/core.py)
- [code/birth_direction_node_head_v1/model.py](code/birth_direction_node_head_v1/model.py)
- [code/build_response_rows_cost_v1.py](code/build_response_rows_cost_v1.py)
- [code/build_response_rows_cost_v2.py](code/build_response_rows_cost_v2.py)
- [code/build_response_rows_current_runtime_v1.py](code/build_response_rows_current_runtime_v1.py)
- [code/candidate_fullpair_runtime_d5_t8192_v1.py](code/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/candidate_response_rows_cost_v1.py](code/candidate_response_rows_cost_v1.py)
- [code/candidate_response_rows_cost_v2.py](code/candidate_response_rows_cost_v2.py)
- [code/candidate_response_rows_current_runtime_v1.py](code/candidate_response_rows_current_runtime_v1.py)
- [code/candidate_response_rows_parent_cost_v1.py](code/candidate_response_rows_parent_cost_v1.py)
- [code/centered_bank_autodiff_v1.py](code/centered_bank_autodiff_v1.py)
- [code/centered_bank_autodiff_v3.py](code/centered_bank_autodiff_v3.py)
- [code/centered_torch_numerics_v1.py](code/centered_torch_numerics_v1.py)
- [code/centered_torch_numerics_v3.py](code/centered_torch_numerics_v3.py)
- [code/check_remote_series_v1.py](code/check_remote_series_v1.py)
- [code/check_response_rows_adjusted_v1.py](code/check_response_rows_adjusted_v1.py)
- [code/check_response_rows_panel_v1.py](code/check_response_rows_panel_v1.py)
- [code/current_bd_p_coadaptation_v1/build_terminal.py](code/current_bd_p_coadaptation_v1/build_terminal.py)
- [code/current_bd_p_coadaptation_v1/model.py](code/current_bd_p_coadaptation_v1/model.py)
- [code/current_bd_p_coadaptation_v1/qualify.py](code/current_bd_p_coadaptation_v1/qualify.py)
- [code/current_bd_p_coadaptation_v1/train.py](code/current_bd_p_coadaptation_v1/train.py)
- [code/current_comp_cstar_information_v1/information.py](code/current_comp_cstar_information_v1/information.py)
- [code/current_comp_cstar_information_v1/model.py](code/current_comp_cstar_information_v1/model.py)
- [code/current_comp_cstar_information_v1/qualification.py](code/current_comp_cstar_information_v1/qualification.py)
- [code/current_comp_cstar_information_v1/watch_qualification.py](code/current_comp_cstar_information_v1/watch_qualification.py)
- [code/current_comp_cstar_proposal_v1/common.py](code/current_comp_cstar_proposal_v1/common.py)
- [code/current_comp_cstar_proposal_v1/model.py](code/current_comp_cstar_proposal_v1/model.py)
- [code/current_comp_cstar_proposal_v1/watch.py](code/current_comp_cstar_proposal_v1/watch.py)
- [code/current_comp_cstar_proposal_v1/worker.py](code/current_comp_cstar_proposal_v1/worker.py)
- [code/current_comp_joint_p_covariance_v1/build.py](code/current_comp_joint_p_covariance_v1/build.py)
- [code/current_comp_joint_p_covariance_v1/fnp.py](code/current_comp_joint_p_covariance_v1/fnp.py)
- [code/current_comp_joint_p_covariance_v1/model.py](code/current_comp_joint_p_covariance_v1/model.py)
- [code/current_comp_joint_p_covariance_v1/qualification.py](code/current_comp_joint_p_covariance_v1/qualification.py)
- [code/current_comp_joint_p_covariance_v1/train.py](code/current_comp_joint_p_covariance_v1/train.py)
- [code/current_comp_joint_p_covariance_v1/watch_build.py](code/current_comp_joint_p_covariance_v1/watch_build.py)
- [code/current_comp_joint_p_covariance_v1/watch_qualification.py](code/current_comp_joint_p_covariance_v1/watch_qualification.py)
- [code/current_comp_joint_p_covariance_v1/watch_train.py](code/current_comp_joint_p_covariance_v1/watch_train.py)
- [code/current_comp_joint_p_node_v1/build.py](code/current_comp_joint_p_node_v1/build.py)
- [code/current_comp_joint_p_node_v1/model.py](code/current_comp_joint_p_node_v1/model.py)
- [code/current_comp_joint_p_node_v1/native.py](code/current_comp_joint_p_node_v1/native.py)
- [code/current_comp_joint_p_node_v1/qualification.py](code/current_comp_joint_p_node_v1/qualification.py)
- [code/current_comp_joint_p_node_v1/score.py](code/current_comp_joint_p_node_v1/score.py)
- [code/current_comp_joint_p_node_v1/train.py](code/current_comp_joint_p_node_v1/train.py)
- [code/current_comp_joint_p_node_v1/watch_native.py](code/current_comp_joint_p_node_v1/watch_native.py)
- [code/current_comp_joint_p_node_v1/watch_qualification.py](code/current_comp_joint_p_node_v1/watch_qualification.py)
- [code/current_comp_joint_p_node_v1/watch_score.py](code/current_comp_joint_p_node_v1/watch_score.py)
- [code/current_comp_joint_p_node_v1/watch_train.py](code/current_comp_joint_p_node_v1/watch_train.py)
- [code/current_comp_partial_d6_v1/build_v1/estimator.py](code/current_comp_partial_d6_v1/build_v1/estimator.py)
- [code/current_comp_partial_d6_v1/build_v1/null_estimator.py](code/current_comp_partial_d6_v1/build_v1/null_estimator.py)
- [code/current_comp_receiver_birth_v1/model.py](code/current_comp_receiver_birth_v1/model.py)
- [code/current_comp_receiver_birth_v1/probe_ad_scalar.py](code/current_comp_receiver_birth_v1/probe_ad_scalar.py)
- [code/current_comp_receiver_birth_v1/qualification.py](code/current_comp_receiver_birth_v1/qualification.py)
- [code/current_comp_receiver_birth_v1/qualification_v2.py](code/current_comp_receiver_birth_v1/qualification_v2.py)
- [code/current_comp_receiver_birth_v1/qualification_v3.py](code/current_comp_receiver_birth_v1/qualification_v3.py)
- [code/current_comp_receiver_birth_v1/watch_qualification.py](code/current_comp_receiver_birth_v1/watch_qualification.py)
- [code/current_comp_receiver_directions_v1/build.py](code/current_comp_receiver_directions_v1/build.py)
- [code/current_comp_receiver_directions_v1/build_v2.py](code/current_comp_receiver_directions_v1/build_v2.py)
- [code/current_comp_receiver_directions_v1/fnp.py](code/current_comp_receiver_directions_v1/fnp.py)
- [code/current_comp_receiver_directions_v1/model.py](code/current_comp_receiver_directions_v1/model.py)
- [code/current_comp_receiver_directions_v1/native.py](code/current_comp_receiver_directions_v1/native.py)
- [code/current_comp_receiver_directions_v1/qualification.py](code/current_comp_receiver_directions_v1/qualification.py)
- [code/current_comp_receiver_directions_v1/train.py](code/current_comp_receiver_directions_v1/train.py)
- [code/current_comp_receiver_directions_v1/watch_native.py](code/current_comp_receiver_directions_v1/watch_native.py)
- [code/current_comp_receiver_directions_v1/watch_qualification.py](code/current_comp_receiver_directions_v1/watch_qualification.py)
- [code/current_comp_receiver_directions_v1/watch_train.py](code/current_comp_receiver_directions_v1/watch_train.py)
- [code/current_d50_d6_cost_census_v1/run.py](code/current_d50_d6_cost_census_v1/run.py)
- [code/current_d50_full_return_capture_v1/common.py](code/current_d50_full_return_capture_v1/common.py)
- [code/current_d50_full_return_capture_v1/native.py](code/current_d50_full_return_capture_v1/native.py)
- [code/current_d50_native_objective_p1_v1/build.py](code/current_d50_native_objective_p1_v1/build.py)
- [code/current_d50_native_objective_p1_v1/build_v1/estimator.py](code/current_d50_native_objective_p1_v1/build_v1/estimator.py)
- [code/current_d50_native_objective_p1_v1/build_v1/null_estimator.py](code/current_d50_native_objective_p1_v1/build_v1/null_estimator.py)
- [code/current_d50_native_objective_p1_v1/common.py](code/current_d50_native_objective_p1_v1/common.py)
- [code/current_d50_native_objective_p1_v1/judge.py](code/current_d50_native_objective_p1_v1/judge.py)
- [code/current_d50_native_objective_p1_v1/judge_v2.py](code/current_d50_native_objective_p1_v1/judge_v2.py)
- [code/current_d50_native_objective_p1_v1/native.py](code/current_d50_native_objective_p1_v1/native.py)
- [code/current_d50_native_objective_p1_v1/native_v2.py](code/current_d50_native_objective_p1_v1/native_v2.py)
- [code/current_d50_native_objective_p1_v1/native_watch.py](code/current_d50_native_objective_p1_v1/native_watch.py)
- [code/current_d50_native_objective_p1_v1/native_watch_v2.py](code/current_d50_native_objective_p1_v1/native_watch_v2.py)
- [code/current_d50_native_objective_p1_v1/panel.py](code/current_d50_native_objective_p1_v1/panel.py)
- [code/current_d50_native_objective_p1_v1/watch.py](code/current_d50_native_objective_p1_v1/watch.py)
- [code/current_d50_native_objective_p1_v1/watch_panel.py](code/current_d50_native_objective_p1_v1/watch_panel.py)
- [code/current_d50_native_objective_p1_v1/worker.py](code/current_d50_native_objective_p1_v1/worker.py)
- [code/current_d50_response10_candidate_v1/build.py](code/current_d50_response10_candidate_v1/build.py)
- [code/current_d50_response10_candidate_v1/build_v1/estimator.py](code/current_d50_response10_candidate_v1/build_v1/estimator.py)
- [code/current_d50_response10_candidate_v1/build_v1/null_estimator.py](code/current_d50_response10_candidate_v1/build_v1/null_estimator.py)
- [code/current_d50_response10_candidate_v1/native.py](code/current_d50_response10_candidate_v1/native.py)
- [code/current_d50_response10_candidate_v1/score.py](code/current_d50_response10_candidate_v1/score.py)
- [code/current_d50_response10_candidate_v1/score_v2.py](code/current_d50_response10_candidate_v1/score_v2.py)
- [code/current_d50_response10_candidate_v1/watch_native.py](code/current_d50_response10_candidate_v1/watch_native.py)
- [code/current_d50_response10_census_v1/capture.py](code/current_d50_response10_census_v1/capture.py)
- [code/current_d50_response10_census_v1/watch.py](code/current_d50_response10_census_v1/watch.py)
- [code/current_d50_serial_d6_kernel_v1/client.py](code/current_d50_serial_d6_kernel_v1/client.py)
- [code/current_d50_serial_d6_kernel_v1/kernel.py](code/current_d50_serial_d6_kernel_v1/kernel.py)
- [code/current_d50_serial_d6_kernel_v1/run.py](code/current_d50_serial_d6_kernel_v1/run.py)
- [code/current_d50_terminal_linear_v2/common.py](code/current_d50_terminal_linear_v2/common.py)
- [code/current_d50_terminal_linear_v2/fit.py](code/current_d50_terminal_linear_v2/fit.py)
- [code/current_d50_terminal_linear_v2/fit_v1/estimator.py](code/current_d50_terminal_linear_v2/fit_v1/estimator.py)
- [code/current_d50_terminal_linear_v2/judge_terminal.py](code/current_d50_terminal_linear_v2/judge_terminal.py)
- [code/current_d50_terminal_linear_v2/native.py](code/current_d50_terminal_linear_v2/native.py)
- [code/current_d50_terminal_linear_v2/native_validation.py](code/current_d50_terminal_linear_v2/native_validation.py)
- [code/current_d50_terminal_linear_v2/panel.py](code/current_d50_terminal_linear_v2/panel.py)
- [code/current_d50_terminal_linear_v2/watch_fit.py](code/current_d50_terminal_linear_v2/watch_fit.py)
- [code/current_d50_terminal_linear_v2/watch_panel.py](code/current_d50_terminal_linear_v2/watch_panel.py)
- [code/current_d50_terminal_linear_v2/watch_validation.py](code/current_d50_terminal_linear_v2/watch_validation.py)
- [code/current_p1_d6_push832_probe_v1/client.py](code/current_p1_d6_push832_probe_v1/client.py)
- [code/current_p1_d6_push832_probe_v1/run.py](code/current_p1_d6_push832_probe_v1/run.py)
- [code/current_p1_native_objective_p2_monitor_v2/monitor.py](code/current_p1_native_objective_p2_monitor_v2/monitor.py)
- [code/current_p1_native_objective_p2_monitor_v2/monitor_tests.py](code/current_p1_native_objective_p2_monitor_v2/monitor_tests.py)
- [code/current_p1_node_composition_v1/admit_v2.py](code/current_p1_node_composition_v1/admit_v2.py)
- [code/current_p1_node_composition_v1/build.py](code/current_p1_node_composition_v1/build.py)
- [code/current_p1_node_composition_v1/build_v1/terminal_estimator.py](code/current_p1_node_composition_v1/build_v1/terminal_estimator.py)
- [code/current_p1_node_composition_v1/endpoint_v2.py](code/current_p1_node_composition_v1/endpoint_v2.py)
- [code/current_p1_node_composition_v1/native.py](code/current_p1_node_composition_v1/native.py)
- [code/current_p1_node_composition_v1/native_v2.py](code/current_p1_node_composition_v1/native_v2.py)
- [code/current_p1_node_composition_v1/panel_admit.py](code/current_p1_node_composition_v1/panel_admit.py)
- [code/current_p1_node_composition_v1/panel_admit_draft_v1.py](code/current_p1_node_composition_v1/panel_admit_draft_v1.py)
- [code/current_p1_node_composition_v1/panel_common.py](code/current_p1_node_composition_v1/panel_common.py)
- [code/current_p1_node_composition_v1/panel_common_draft_v1.py](code/current_p1_node_composition_v1/panel_common_draft_v1.py)
- [code/current_p1_node_composition_v1/panel_native.py](code/current_p1_node_composition_v1/panel_native.py)
- [code/current_p1_node_composition_v1/panel_score.py](code/current_p1_node_composition_v1/panel_score.py)
- [code/current_p1_node_composition_v1/panel_watch.py](code/current_p1_node_composition_v1/panel_watch.py)
- [code/current_p1_node_composition_v1/panel_watch_draft_v1.py](code/current_p1_node_composition_v1/panel_watch_draft_v1.py)
- [code/current_p1_node_composition_v1/score.py](code/current_p1_node_composition_v1/score.py)
- [code/current_p1_node_composition_v1/score_v2.py](code/current_p1_node_composition_v1/score_v2.py)
- [code/current_p1_node_composition_v1/score_v3.py](code/current_p1_node_composition_v1/score_v3.py)
- [code/current_p1_node_composition_v1/watch_native.py](code/current_p1_node_composition_v1/watch_native.py)
- [code/current_p1_node_composition_v1/watch_native_v2.py](code/current_p1_node_composition_v1/watch_native_v2.py)
- [code/current_p1_partial_d6_v1/build.py](code/current_p1_partial_d6_v1/build.py)
- [code/current_p1_partial_d6_v1/build_v2.py](code/current_p1_partial_d6_v1/build_v2.py)
- [code/current_p1_partial_d6_v1/build_v2/estimator.py](code/current_p1_partial_d6_v1/build_v2/estimator.py)
- [code/current_p1_partial_d6_v1/native.py](code/current_p1_partial_d6_v1/native.py)
- [code/current_p1_partial_d6_v1/watch_native.py](code/current_p1_partial_d6_v1/watch_native.py)
- [code/current_transport_shadow_v1.py](code/current_transport_shadow_v1.py)
- [code/current_transport_shadow_v3.py](code/current_transport_shadow_v3.py)
- [code/full_carrier_autodiff_v1.py](code/full_carrier_autodiff_v1.py)
- [code/full_carrier_bank_autodiff_v1.py](code/full_carrier_bank_autodiff_v1.py)
- [code/gaussian_cdf_quadrature_v1.py](code/gaussian_cdf_quadrature_v1.py)
- [code/joint_prefix_final_data_v1.py](code/joint_prefix_final_data_v1.py)
- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
- [code/packed_arena_generic.py](code/packed_arena_generic.py)
- [code/port_joint_p_covariance_working_v1.py](code/port_joint_p_covariance_working_v1.py)
- [code/port_receiver_directions_working_v1.py](code/port_receiver_directions_working_v1.py)
- [code/radial_binary_compact.py](code/radial_binary_compact.py)
- [code/radial_compact_feed_fullpair_v1.py](code/radial_compact_feed_fullpair_v1.py)
- [code/radial_compact_radialfeed_v1.py](code/radial_compact_radialfeed_v1.py)
- [code/radial_cubic.py](code/radial_cubic.py)
- [code/radial_mehler.py](code/radial_mehler.py)
- [code/radial_response_rows_v1.py](code/radial_response_rows_v1.py)
- [code/run_remote_series_v1.py](code/run_remote_series_v1.py)
- [code/tensor_clamp_min_v1.py](code/tensor_clamp_min_v1.py)
- [code/unused_materializations_v1/estimator.py](code/unused_materializations_v1/estimator.py)

</details>

## Evidence

- [CHECK4_RESULT_20260910.md#L3](CHECK4_RESULT_20260910.md#L3)
- [CHECK4_RESULT_20260910.md#L10](CHECK4_RESULT_20260910.md#L10)
- [CHECK4_RESULT_20260910.md#L12](CHECK4_RESULT_20260910.md#L12)
- [CHECK4_RESULT_20260910.md#L13](CHECK4_RESULT_20260910.md#L13)
- [CHECK4_RESULT_20260910.md#L14](CHECK4_RESULT_20260910.md#L14)
- [INDEPENDENT14_RESULT_20260910.md#L3](INDEPENDENT14_RESULT_20260910.md#L3)
- [INDEPENDENT14_RESULT_20260910.md#L9](INDEPENDENT14_RESULT_20260910.md#L9)
- [INDEPENDENT14_RESULT_20260910.md#L13](INDEPENDENT14_RESULT_20260910.md#L13)
- [INDEPENDENT14_RESULT_20260910.md#L14](INDEPENDENT14_RESULT_20260910.md#L14)
- [INDEPENDENT14_RESULT_20260910.md#L15](INDEPENDENT14_RESULT_20260910.md#L15)
- [NATIVE40_RESULT_20260910.md#L3](NATIVE40_RESULT_20260910.md#L3)
- [NATIVE40_RESULT_20260910.md#L8](NATIVE40_RESULT_20260910.md#L8)
- [NATIVE40_RESULT_20260910.md#L11](NATIVE40_RESULT_20260910.md#L11)
- [NATIVE40_RESULT_20260910.md#L13](NATIVE40_RESULT_20260910.md#L13)
- [NATIVE40_RESULT_20260910.md#L14](NATIVE40_RESULT_20260910.md#L14)
- [PRODUCTION_RESULT_20260910.md#L3](PRODUCTION_RESULT_20260910.md#L3)
- [PRODUCTION_RESULT_20260910.md#L6](PRODUCTION_RESULT_20260910.md#L6)
- [PRODUCTION_RESULT_20260910.md#L10](PRODUCTION_RESULT_20260910.md#L10)
- [PRODUCTION_RESULT_20260910.md#L22](PRODUCTION_RESULT_20260910.md#L22)
- [PRODUCTION_RESULT_20260910.md#L24](PRODUCTION_RESULT_20260910.md#L24)

<details>
<summary>All result files and records (4)</summary>

- [CHECK4_RESULT_20260910.md](CHECK4_RESULT_20260910.md) · record
- [INDEPENDENT14_RESULT_20260910.md](INDEPENDENT14_RESULT_20260910.md) · record
- [NATIVE40_RESULT_20260910.md](NATIVE40_RESULT_20260910.md) · record
- [PRODUCTION_RESULT_20260910.md](PRODUCTION_RESULT_20260910.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-10-partial-six-level-products-improve-native40-adjusted-score-1) · [Copy provenance](../../../PROVENANCE.json)

