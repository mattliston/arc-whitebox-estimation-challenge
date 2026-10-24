# Young negative trace and depth-five covariance products: validation improves adjusted score 1.0748874384119644%

2026-09-30 · experiment · adopted · measured

The combined first-order plane reads, young-negative trace and depth-five symmetric covariance products initially failed the internal0.35-second residual margin. A separately authorized0.4-second extension preserved the original failure and qualified the same source on40 networks and full serving.

Recorded measurement:

    VALID40 adjusted-1.0748874384119644%,CI[-1.432779732929564,-0.7085659005076911]%,34wins;raw+.4591177959888837%,bill-1.5249349966752535%,CPU+.5203286315704103%;original residual projection.357850751>.35 but<.4.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/build.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/build.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/boundary_feasibility_engine_v1/assess.py](code/codex_resume_20260926_v2/boundary_feasibility_engine_v1/assess.py)
- [code/codex_resume_20260926_v2/boundary_feasibility_engine_v3/assess.py](code/codex_resume_20260926_v2/boundary_feasibility_engine_v3/assess.py)
- [code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py](code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py)
- [code/codex_resume_20260926_v2/handle_client_v2.py](code/codex_resume_20260926_v2/handle_client_v2.py)
- [code/codex_resume_20260926_v2/handle_server_v2.py](code/codex_resume_20260926_v2/handle_server_v2.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_gnbatch_young_trace_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_e1_gnbatch_young_trace_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/poll_submission_v1.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/poll_submission_v1.py)

<details>
<summary>Additional programs and dependencies (55)</summary>

- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/prepare_submission.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/prepare_submission.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/probe.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/record_s34_grade_v1.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/record_s34_grade_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/submit_once.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/submit_once.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_hk15_blocked_v2/kernel_blocked.py](code/codex_resume_20260926_v2/hill_climb_hk15_blocked_v2/kernel_blocked.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/build.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/probe.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/probe.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/build.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/build.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/fragment.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/wrapper.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/wrapper.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/boundary_feasibility_assess.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/boundary_feasibility_assess.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/build.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/paired_boundary_assess.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/paired_boundary_assess.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/assess_extension.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/assess_extension.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/build.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/make_assessor.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/make_assessor.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/precision.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/precision.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/symmetry.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/symmetry.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/symmetry_probe.py](code/codex_resume_20260926_v2/hill_climb_precision_symmetry_bundle_v1/symmetry_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/build.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_probe_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_probe_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_young_negative_trace_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_young_negative_trace_v1/fragment.py)
- [code/codex_resume_20260926_v2/native_eval.py](code/codex_resume_20260926_v2/native_eval.py)
- [code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py](code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py)
- [code/codex_resume_20260926_v2/official_limit_research_extension_v1/native8.py](code/codex_resume_20260926_v2/official_limit_research_extension_v1/native8.py)
- [code/codex_resume_20260926_v2/run_boundary_census.py](code/codex_resume_20260926_v2/run_boundary_census.py)
- [code/codex_resume_20260926_v2/run_handle_worker_v3.py](code/codex_resume_20260926_v2/run_handle_worker_v3.py)
- [code/codex_resume_20260926_v2/serving_engine_v1/assess.py](code/codex_resume_20260926_v2/serving_engine_v1/assess.py)
- [code/codex_resume_20260926_v2/serving_engine_v1/common.py](code/codex_resume_20260926_v2/serving_engine_v1/common.py)
- [code/codex_resume_20260926_v2/serving_engine_v1/run.py](code/codex_resume_20260926_v2/serving_engine_v1/run.py)
- [code/codex_resume_20260926_v2/serving_engine_v2/assess.py](code/codex_resume_20260926_v2/serving_engine_v2/assess.py)
- [code/codex_resume_20260926_v2/serving_engine_v2/common.py](code/codex_resume_20260926_v2/serving_engine_v2/common.py)
- [code/codex_resume_20260926_v2/serving_engine_v2/run.py](code/codex_resume_20260926_v2/serving_engine_v2/run.py)
- [code/codex_resume_20260926_v2/setup_only_client_v1.py](code/codex_resume_20260926_v2/setup_only_client_v1.py)
- [code/codex_resume_20260926_v2/setup_stress_driver_v1.py](code/codex_resume_20260926_v2/setup_stress_driver_v1.py)
- [code/codex_resume_20260926_v2/valid40_engine_v1/assess.py](code/codex_resume_20260926_v2/valid40_engine_v1/assess.py)
- [code/codex_resume_20260926_v2/valid40_engine_v1/run.py](code/codex_resume_20260926_v2/valid40_engine_v1/run.py)
- [code/codex_resume_20260926_v2/valid40_engine_v2/assess.py](code/codex_resume_20260926_v2/valid40_engine_v2/assess.py)
- [code/codex_resume_20260926_v2/valid40_engine_v2/build_configs.py](code/codex_resume_20260926_v2/valid40_engine_v2/build_configs.py)
- [code/codex_resume_20260926_v2/valid40_engine_v2/run.py](code/codex_resume_20260926_v2/valid40_engine_v2/run.py)
- [code/codex_resume_20260926_v2/valid40_official_limit_v1/assess.py](code/codex_resume_20260926_v2/valid40_official_limit_v1/assess.py)
- [code/codex_resume_20260926_v2/valid40_official_limit_v1/build_configs.py](code/codex_resume_20260926_v2/valid40_official_limit_v1/build_configs.py)
- [code/codex_resume_20260926_v2/valid40_official_limit_v1/run.py](code/codex_resume_20260926_v2/valid40_official_limit_v1/run.py)
- [code/fable_porter_v1/census_boundary_setup_v2.py](code/fable_porter_v1/census_boundary_setup_v2.py)

</details>

## Evidence

- [handoff_root_archive_20260930T024527Z.md#L53](../depth-three-covariance-diagonal-products-adjusted-gain-0-355945/handoff_root_archive_20260930T024527Z.md#L53)
- [official_bank_S34_v1.json#L77](official_bank_S34_v1.json#L77)

<details>
<summary>All result files and records (7)</summary>

- [handoff_root_archive_20260930T024527Z.md](../depth-three-covariance-diagonal-products-adjusted-gain-0-355945/handoff_root_archive_20260930T024527Z.md) · record
- [RESULT.md](RESULT.md) · record
- [native2_assessment.json](native2_assessment.json) · result
- [broad_assessment.json](broad_assessment.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record
- [official_bank_S34_v1.json](official_bank_S34_v1.json) · record

</details>

## Related work

- shipped in: [Submission #333050: young-column cutoff and depth-5 symmetric covariance products](../../../submissions/333050/README.md)

[Research graph](../../../graph/index.html#node=2026-09-30-young-negative-trace-depth-five-covariance-products-validation-improves) · [Copy provenance](../../../PROVENANCE.json)

