# Adding learned graph-network capacity gives native adjusted ratio 1.0003 with 15/40 wins

2026-09-22T05:16Z · experiment · rejected · measured

Three added arrays (12,352 float32 parameters) were trained for one epoch on 160 networks with every old #331760 value frozen, then validated natively on 40 matched cold parent/trained pairs with sealed predictions and own bills. The adjusted ratio did not improve against either the #331760 parent or the new #331824 bank, so the study closed negative at its original gate.

Recorded measurement:

    vs #331760 parent: adjusted ratio 1.000308587146151, raw ratio 0.999374479164305, 15/40 adjusted wins; vs #331824: adjusted ratio 1.002863174027707, 16/40 wins; gate passed = False

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_fit_v1/fit_common.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_fit_v1/fit_common.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_fit_v1/guard.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_fit_v1/guard.py)

<details>
<summary>Additional programs and dependencies (38)</summary>

- [code/codex_resume_20260919_v1/current_p_hidden_capacity_fit_v1/run_fit.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_fit_v1/run_fit.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_fit_v1/state_io.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_fit_v1/state_io.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_fit_v1/verify_fit.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_fit_v1/verify_fit.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_fit_v1/worker.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_fit_v1/worker.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/common.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/disabled.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/disabled.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/guard.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/native.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/qualify.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/qualify.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/records.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/records.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/retirement.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/retirement.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/run.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/run.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/unit.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/unit.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/zero.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_qualification_v1/zero.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/build.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/build.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/candidate.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/candidate.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/common.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/exporter.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/exporter.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/guard.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/native.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/null_candidate.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/null_candidate.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/qualify_controls.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/qualify_controls.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/qualify_null.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/qualify_null.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/records.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/records.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/retirement.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/run.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/run.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/score_native40.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/score_native40.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/scoring.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/scoring.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/seal_predictions.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_native_v1/seal_predictions.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_training_qualification_v1/allocator.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_training_qualification_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_training_qualification_v1/common.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_training_qualification_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_training_qualification_v1/full.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_training_qualification_v1/full.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_training_qualification_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_training_qualification_v1/model_hidden.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_training_qualification_v1/model_hidden.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_training_qualification_v1/run.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_training_qualification_v1/run.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_training_qualification_v1/state_io.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_training_qualification_v1/state_io.py)
- [code/codex_resume_20260919_v1/current_p_hidden_capacity_training_qualification_v1/unit.py](code/codex_resume_20260919_v1/current_p_hidden_capacity_training_qualification_v1/unit.py)
- [code/codex_resume_20260919_v1/current_p_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_release_v1/candidate.py)

</details>

## Evidence

- [hidden_native_and_submission_checkpoint141.md#L16](hidden_native_and_submission_checkpoint141.md#L16)
- [hidden_native_and_submission_checkpoint141.md#L24](hidden_native_and_submission_checkpoint141.md#L24)

<details>
<summary>All result files and records (3)</summary>

- [hidden_native_and_submission_checkpoint141.md](hidden_native_and_submission_checkpoint141.md) · record
- [PROTOCOL.md](PROTOCOL.md) · record
- [hidden_fit_complete_checkpoint139.md](hidden_fit_complete_checkpoint139.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-22-adding-learned-graph-network-capacity-gives-native-adjusted-ratio) · [Copy provenance](../../../PROVENANCE.json)

