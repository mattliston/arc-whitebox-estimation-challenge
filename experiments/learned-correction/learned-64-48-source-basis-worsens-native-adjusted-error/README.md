# Learned 64-by-48 source basis worsens native adjusted error: ratio 1.065 and 0/40 wins

2026-09-22T23:21Z · experiment · rejected · measured

The 97-parameter coherent moment carrier (a three-factor fourth-to-third cumulant recurrence feeding a terminal head) gained a learned 64x48 basis map, 3,169 trainable values in all, and was fitted on 320 graphs / 80 updates from identity initialization with every bank value frozen. Its sealed native 40-pair evaluation improved raw error by only 0.14% while its 27.48 GF per network of carrier overhead raised the adjusted score 6.5%, so the line closed negative; the lead's own notes record the same verdict.

Recorded measurement:

    VALID40 raw ratio 0.9985853366822984, own-bill adjusted ratio 1.0646394186900314, adjusted wins 0/40, passed = False (87 calls, 2827.65 s); carrier-plus-basis overhead 27,484,170,972 F warm / 27,484,177,310 F cold; fit-side same-network pass-2/pass-1 ratio 0.9952 on TRAIN160

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
- [code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_message_covariance_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/joint_remainder_carrier_implementation_v1/post_planes_fnp.py](code/codex_resume_20260919_v1/joint_remainder_carrier_implementation_v1/post_planes_fnp.py)

<details>
<summary>Additional programs and dependencies (42)</summary>

- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_bank_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_fit_v1/fit_common.py](code/codex_resume_20260919_v1/learnable_joint_basis_fit_v1/fit_common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_fit_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_fit_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_fit_v1/run_fit.py](code/codex_resume_20260919_v1/learnable_joint_basis_fit_v1/run_fit.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_fit_v1/state_io.py](code/codex_resume_20260919_v1/learnable_joint_basis_fit_v1/state_io.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_fit_v1/verify_fit.py](code/codex_resume_20260919_v1/learnable_joint_basis_fit_v1/verify_fit.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_fit_v1/worker.py](code/codex_resume_20260919_v1/learnable_joint_basis_fit_v1/worker.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/carrier_fnp.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/carrier_fnp.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/disabled.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/disabled.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/sentinel.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/sentinel.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py](code/codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/build.py](code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/build.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/candidate.py](code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/exporter.py](code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/exporter.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/native.py](code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/native.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/null_candidate.py](code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/null_candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/qualify_controls.py](code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/qualify_controls.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/qualify_null.py](code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/qualify_null.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/records.py](code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/records.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/retirement.py](code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/score_native40.py](code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/score_native40.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/scoring.py](code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/scoring.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/seal_predictions.py](code/codex_resume_20260919_v1/learnable_joint_basis_native_v1/seal_predictions.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/allocator.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/allocator.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/carrier_translation.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/carrier_translation.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/common.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/common.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/full.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/full.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/model_bank.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/model_bank.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/model_basis.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/model_basis.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/run.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/state_io.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/state_io.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/unit.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/unit.py)

</details>

## Evidence

- [source_README.md#L3](source_README.md#L3)
- [native40_score.json#L16](native40_score.json#L16)
- [NOTES.md#L120](../../third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L120)

<details>
<summary>All result files and records (6)</summary>

- [source_README.md](source_README.md) · record
- [native40_score.json](native40_score.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record
- [learned_basis_checkpoint161.md](learned_basis_checkpoint161.md) · record
- [carrier_bank_qualified_checkpoint160.md](carrier_bank_qualified_checkpoint160.md) · record
- [NOTES.md](../../third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-22-learned-64-48-source-basis-worsens-native-adjusted-error) · [Copy provenance](../../../PROVENANCE.json)

