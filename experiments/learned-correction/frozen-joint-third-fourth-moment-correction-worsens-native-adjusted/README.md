# Frozen joint third/fourth-moment correction worsens native adjusted error: pooled ratio 1.530216168430

2026-09-21 · experiment · rejected · measured

The full-amplitude frozen moment-source model was inserted into submission #331760 and tested through all downstream interactions.

Recorded measurement:

    Against exact official331760 on the two reused development networks:
    | 35 | 1.517845015 | 1.552368845 |
    | 100 | 1.473698154 | 1.507887431 |
    Pooled raw ratio is1.495640639553; adjusted ratio is1.530216168430.
    The added mean cost is2.2970%.
    fails the0.98 development gate. No native40, confirmation, independent panel,
    The earlier model's proxy0.90 gate remains failed. This new experiment was
    explicitly undertaken despite that scoped failure to test its7–8% next-mean
    exact. The effect includes transfer from the earlier parent to current331760
    variance domains and selectors. Enabled35 repeats exactly. The first8 output
    intended pre-layer8 boundary. Both raw errors then grow through the changed
    their own complete costs were sealed before the1B target arrays were opened.
    All9 calls and6 guarded stages retired. Peak sampled tree RSS was
    6.725452GB, with zero added swap; total guarded
    wall time287.963s. New artifacts before closeout were
    2434553bytes. No data download, new
    Official331760 remains3.507738856975084e-9 adjusted,100/100,zero failures.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/build.py](code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/build.py)
- [code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/candidate.py](code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/candidate.py)
- [code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/closeout.py](code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/closeout.py)
- [code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/common.py](code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/guard.py](code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/head.py](code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/head.py)
- [code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/native.py](code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/null_candidate.py](code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/null_candidate.py)
- [code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/prepare.py](code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/prepare.py)
- [code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/qualify.py](code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/qualify.py)
- [code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/retirement.py](code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/retirement.py)
- [code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/run.py](code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/run.py)

<details>
<summary>Additional programs and dependencies (21)</summary>

- [code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/score.py](code/codex_resume_20260919_v1/current_p_joint_moment_native_v1/score.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/build_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/core_v2.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/fnp/reference_numpy.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native.py)
- [code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py](code/astra_phase2_reexamination_v1/birth_direction_fit_v1/native_v2.py)
- [code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py](code/astra_phase2_reexamination_v1/unused_materializations_v1/estimator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/common.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/guard.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/native.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/native.py)
- [code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/retirement.py](code/codex_resume_20260919_v1/current_p_batch_process_endpoint_v1/retirement.py)
- [code/codex_resume_20260919_v1/current_p_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_release_v1/candidate.py)
- [code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/candidate_zero.py](code/codex_resume_20260919_v1/current_p_signed_branch_qualification_v2/candidate_zero.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/codex_resume_20260919_v1/joint_incoming_moment_head_v1/build.py](code/codex_resume_20260919_v1/joint_incoming_moment_head_v1/build.py)
- [code/codex_resume_20260919_v1/joint_incoming_moment_head_v1/common.py](code/codex_resume_20260919_v1/joint_incoming_moment_head_v1/common.py)
- [code/codex_resume_20260919_v1/joint_incoming_moment_head_v1/head.py](code/codex_resume_20260919_v1/joint_incoming_moment_head_v1/head.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L8](RESULT.md#L8)
- [RESULT.md#L9](RESULT.md#L9)
- [RESULT.md#L11](RESULT.md#L11)
- [REEXAMINATION_INDEX_20260919.md#L379](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L379)
- [REEXAMINATION_INDEX_20260919.md#L382](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L382)

<details>
<summary>All result files and records (5)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [score.json](score.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-frozen-joint-third-fourth-moment-correction-worsens-native-adjusted) · [Copy provenance](../../../PROVENANCE.json)

