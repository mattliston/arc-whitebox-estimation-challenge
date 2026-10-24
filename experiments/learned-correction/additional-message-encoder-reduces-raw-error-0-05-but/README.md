# Additional message encoder reduces raw error 0.05% but raises work 2.55%

2026-09-22T08:18Z · experiment · rejected · measured

Each of the three 16x64 message maps of the learned correction received 48 fixed complementary directions derived from static bank parameters (combined rank 64 instead of 48) and a new zero-started 128x288 encoder was trained for one epoch with everything else frozen. On 40 matched cold validation pairs the raw error fell slightly but the extra 10.59 GF per network outweighed it, so the trial closed negative.

Recorded measurement:

    native40 raw MSE 1.943204047785454e-8 -> 1.9421752135402906e-8 (ratio 0.999470547497914), 21/40 raw wins; mean cold cost 415.504472505 -> 426.093134265 GF (+10,588,661,760 F per network, +2.54838695%); adjusted 3.669777678749566e-9 -> 3.761261040829282e-9 (ratio 1.024928856755946), 0/40 wins

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
- [code/codex_resume_20260919_v1/current_p_message_complement_design_v1/disabled.py](code/codex_resume_20260919_v1/current_p_message_complement_design_v1/disabled.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_design_v1/sentinel.py](code/codex_resume_20260919_v1/current_p_message_complement_design_v1/sentinel.py)

<details>
<summary>Additional programs and dependencies (39)</summary>

- [code/codex_resume_20260919_v1/current_p_message_complement_design_v1/zero.py](code/codex_resume_20260919_v1/current_p_message_complement_design_v1/zero.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_fit_v1/fit_common.py](code/codex_resume_20260919_v1/current_p_message_complement_fit_v1/fit_common.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_fit_v1/guard.py](code/codex_resume_20260919_v1/current_p_message_complement_fit_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_fit_v1/run_fit.py](code/codex_resume_20260919_v1/current_p_message_complement_fit_v1/run_fit.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_fit_v1/state_io.py](code/codex_resume_20260919_v1/current_p_message_complement_fit_v1/state_io.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_fit_v1/verify_fit.py](code/codex_resume_20260919_v1/current_p_message_complement_fit_v1/verify_fit.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_fit_v1/worker.py](code/codex_resume_20260919_v1/current_p_message_complement_fit_v1/worker.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_qualification_v1/common.py](code/codex_resume_20260919_v1/current_p_message_complement_native_qualification_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_qualification_v1/guard.py](code/codex_resume_20260919_v1/current_p_message_complement_native_qualification_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_qualification_v1/native.py](code/codex_resume_20260919_v1/current_p_message_complement_native_qualification_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_qualification_v1/qualify.py](code/codex_resume_20260919_v1/current_p_message_complement_native_qualification_v1/qualify.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_qualification_v1/records.py](code/codex_resume_20260919_v1/current_p_message_complement_native_qualification_v1/records.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_qualification_v1/retirement.py](code/codex_resume_20260919_v1/current_p_message_complement_native_qualification_v1/retirement.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_qualification_v1/run.py](code/codex_resume_20260919_v1/current_p_message_complement_native_qualification_v1/run.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_v1/build.py](code/codex_resume_20260919_v1/current_p_message_complement_native_v1/build.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_complement_native_v1/candidate.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_v1/common.py](code/codex_resume_20260919_v1/current_p_message_complement_native_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_v1/exporter.py](code/codex_resume_20260919_v1/current_p_message_complement_native_v1/exporter.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_v1/guard.py](code/codex_resume_20260919_v1/current_p_message_complement_native_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_v1/native.py](code/codex_resume_20260919_v1/current_p_message_complement_native_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_v1/null_candidate.py](code/codex_resume_20260919_v1/current_p_message_complement_native_v1/null_candidate.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_v1/qualify_controls.py](code/codex_resume_20260919_v1/current_p_message_complement_native_v1/qualify_controls.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_v1/qualify_null.py](code/codex_resume_20260919_v1/current_p_message_complement_native_v1/qualify_null.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_v1/records.py](code/codex_resume_20260919_v1/current_p_message_complement_native_v1/records.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_v1/retirement.py](code/codex_resume_20260919_v1/current_p_message_complement_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_v1/run.py](code/codex_resume_20260919_v1/current_p_message_complement_native_v1/run.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_v1/score_native40.py](code/codex_resume_20260919_v1/current_p_message_complement_native_v1/score_native40.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_v1/scoring.py](code/codex_resume_20260919_v1/current_p_message_complement_native_v1/scoring.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_native_v1/seal_predictions.py](code/codex_resume_20260919_v1/current_p_message_complement_native_v1/seal_predictions.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/allocator.py](code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/common.py](code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/full.py](code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/full.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/model_bank.py](code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/model_bank.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/model_complement.py](code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/model_complement.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/run.py](code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/run.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/state_io.py](code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/state_io.py)
- [code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/unit.py](code/codex_resume_20260919_v1/current_p_message_complement_training_qualification_v1/unit.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)

</details>

## Evidence

- [NEGATIVE_RESULT.md#L13](NEGATIVE_RESULT.md#L13)
- [message_complement_qualified_checkpoint142.md#L4](message_complement_qualified_checkpoint142.md#L4)

<details>
<summary>All result files and records (3)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · result
- [message_complement_qualified_checkpoint142.md](message_complement_qualified_checkpoint142.md) · record
- [message_complement_handoff_checkpoint146.md](message_complement_handoff_checkpoint146.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-22-additional-message-encoder-reduces-raw-error-0-05-but) · [Copy provenance](../../../PROVENANCE.json)

