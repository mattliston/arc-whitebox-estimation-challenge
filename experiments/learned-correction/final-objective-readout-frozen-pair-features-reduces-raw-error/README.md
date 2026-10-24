# Final-objective readout on frozen pair features reduces raw error 1.03% but worsens adjusted error

2026-09-22T18:21Z · experiment · rejected · measured

A 96-value readout (Theta, 32x3) on a frozen pretrained 140-to-32 physical pair feature map was fitted on 320 graphs / 80 updates against the final-layer objective with every bank parameter frozen; it changes only the off-diagonal incoming K21, the standardized K31 and the physical K22 planes at layers 8-14. Natively it improved pooled raw error on 39/40 networks, but its 9.404 GF per network of added cost made the own-cost adjusted score worse on every network.

Recorded measurement:

    pooled raw error -1.027301% (39/40 raw wins); added cost 9,404,011,746 F per network (2.255-2.379% of the bill); own-cost adjusted ratio 1.012122448736662, 0/40 adjusted wins

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
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_design_v1/disabled.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_design_v1/disabled.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_design_v1/head.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_design_v1/head.py)

<details>
<summary>Additional programs and dependencies (41)</summary>

- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_design_v1/sentinel.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_design_v1/sentinel.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_design_v1/zero.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_design_v1/zero.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_fit_v1/fit_common.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_fit_v1/fit_common.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_fit_v1/guard.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_fit_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_fit_v1/run_fit.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_fit_v1/run_fit.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_fit_v1/state_io.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_fit_v1/state_io.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_fit_v1/verify_fit.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_fit_v1/verify_fit.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_fit_v1/worker.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_fit_v1/worker.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_qualification_v1/common.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_qualification_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_qualification_v1/guard.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_qualification_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_qualification_v1/native.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_qualification_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_qualification_v1/qualify.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_qualification_v1/qualify.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_qualification_v1/records.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_qualification_v1/records.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_qualification_v1/retirement.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_qualification_v1/retirement.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_qualification_v1/run.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_qualification_v1/run.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/build.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/build.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/candidate.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/candidate.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/common.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/exporter.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/exporter.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/guard.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/native.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/null_candidate.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/null_candidate.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/qualify_controls.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/qualify_controls.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/qualify_null.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/qualify_null.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/records.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/records.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/retirement.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/run.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/run.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/score_native40.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/score_native40.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/scoring.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/scoring.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/seal_predictions.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_native_v1/seal_predictions.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/allocator.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/common.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/full.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/full.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/head_translation.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/head_translation.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/model_bank.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/model_bank.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/model_readout.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/model_readout.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/run.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/run.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/state_io.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/state_io.py)
- [code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/unit.py](code/codex_resume_20260919_v1/current_p_incoming_moment_readout_training_qualification_v1/unit.py)
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)

</details>

## Evidence

- [readout_negative_joint_components_checkpoint159.md#L1](readout_negative_joint_components_checkpoint159.md#L1)
- [readout_qualified_checkpoint150.md#L17](readout_qualified_checkpoint150.md#L17)

<details>
<summary>All result files and records (3)</summary>

- [readout_negative_joint_components_checkpoint159.md](readout_negative_joint_components_checkpoint159.md) · record
- [readout_qualified_checkpoint150.md](readout_qualified_checkpoint150.md) · record
- [PROTOCOL.md](PROTOCOL.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-22-final-objective-readout-frozen-pair-features-reduces-raw-error) · [Copy provenance](../../../PROVENANCE.json)

