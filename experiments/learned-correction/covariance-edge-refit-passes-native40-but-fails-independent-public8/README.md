# Covariance-edge refit passes native40 but fails independent public8: adjusted ratio 1.000701542110908

2026-09-21 · experiment · rejected · measured

The fixed terminal fit of 528 existing covariance parameters passed native40, cold14 and confirmation8, then failed the unchanged requirement that both separate-truth groups improve. The local pooled gain did not qualify a release.

Recorded measurement:

    native40 ratio0.996660439050478,29/40wins;cold14 0.997270402736826,10/14;confirmation8 0.995498699789178,5/8;independent public8 1.000701542110908,other6 0.996555250541713,combined14 0.999138351334218.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/run_fit.py](code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/run_fit.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/worker.py](code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/worker.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/verify_fit.py](code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/verify_fit.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/run.py](code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/run.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/build.py](code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/build.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/score_native40.py](code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/score_native40.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/run.py](code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/run.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/score_stage.py](code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/score_stage.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)

<details>
<summary>Additional programs and dependencies (31)</summary>

- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/common.py](code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/common.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/guard.py](code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/guard.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/native.py](code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/native.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/qualify_control.py](code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/qualify_control.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/records.py](code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/records.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/retirement.py](code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/retirement.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/seal_stage.py](code/codex_resume_20260919_v1/current_p_edge_objective_downstream_v2/seal_stage.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/fit_common.py](code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/fit_common.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/guard.py](code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/state_io.py](code/codex_resume_20260919_v1/current_p_edge_objective_fit_v1/state_io.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/common.py](code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/exporter.py](code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/exporter.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/guard.py](code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/native.py](code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/qualify_controls.py](code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/qualify_controls.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/qualify_null.py](code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/qualify_null.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/records.py](code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/records.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/retirement.py](code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/scoring.py](code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/scoring.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/seal_predictions.py](code/codex_resume_20260919_v1/current_p_edge_objective_native_v1/seal_predictions.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/allocator.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/common.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/guard.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/model_edge.py](code/codex_resume_20260919_v1/current_p_edge_objective_qualification_v1/model_edge.py)
- [code/codex_resume_20260919_v1/current_p_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_release_v1/candidate.py)

</details>

## Evidence

- [edge_objective_result_checkpoint123.md#L3](edge_objective_result_checkpoint123.md#L3)

<details>
<summary>All result files and records (3)</summary>

- [edge_objective_result_checkpoint123.md](edge_objective_result_checkpoint123.md) · record
- [edge_objective_closeout_checkpoint123.json](edge_objective_closeout_checkpoint123.json) · record
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-covariance-edge-refit-passes-native40-but-fails-independent-public8) · [Copy provenance](../../../PROVENANCE.json)

