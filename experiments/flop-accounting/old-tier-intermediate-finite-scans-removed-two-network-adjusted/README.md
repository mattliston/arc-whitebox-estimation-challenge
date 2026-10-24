# Old-tier intermediate finite scans removed: two-network adjusted score improves 0.08089221%

2026-09-30 · experiment · adopted · measured

Intermediate old-tier scans were removed while final output finiteness checks remained; the timing of an early nonfinite exception changed.

Recorded measurement:

    TRAIN2 adjusted-.08089221%,2wins,all16 bytes exact.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_s34_oldtier_scan_elision_v1/build.py](code/codex_resume_20260926_v2/hill_climb_s34_oldtier_scan_elision_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_oldtier_scan_elision_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_s34_oldtier_scan_elision_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_oldtier_scan_elision_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_s34_oldtier_scan_elision_v1/fragment.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/boundary_feasibility_engine_v1/assess.py](code/codex_resume_20260926_v2/boundary_feasibility_engine_v1/assess.py)
- [code/codex_resume_20260926_v2/boundary_feasibility_engine_v3/assess.py](code/codex_resume_20260926_v2/boundary_feasibility_engine_v3/assess.py)
- [code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py](code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py)
- [code/codex_resume_20260926_v2/handle_client_v2.py](code/codex_resume_20260926_v2/handle_client_v2.py)
- [code/codex_resume_20260926_v2/handle_server_v2.py](code/codex_resume_20260926_v2/handle_server_v2.py)
- [code/codex_resume_20260926_v2/hill_climb_d5_late_common224_v1/build.py](code/codex_resume_20260926_v2/hill_climb_d5_late_common224_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_d5_late_common224_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_d5_late_common224_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_d5_ynt275_v1/build.py](code/codex_resume_20260926_v2/hill_climb_d5_ynt275_v1/build.py)

<details>
<summary>Additional programs and dependencies (71)</summary>

- [code/codex_resume_20260926_v2/hill_climb_d5_ynt275_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_d5_ynt275_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_d5_ynt275_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_d5_ynt275_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_gnbatch_young_trace_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_e1_gnbatch_young_trace_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/build.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/probe.py)
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
- [code/codex_resume_20260926_v2/hill_climb_s34_gn_pair_reuse_v1/build.py](code/codex_resume_20260926_v2/hill_climb_s34_gn_pair_reuse_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_gn_pair_reuse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_s34_gn_pair_reuse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_gn_pair_reuse_v1/fragment_cached_v2.py](code/codex_resume_20260926_v2/hill_climb_s34_gn_pair_reuse_v1/fragment_cached_v2.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_terminal_full_star_v1/build.py](code/codex_resume_20260926_v2/hill_climb_s34_terminal_full_star_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_terminal_full_star_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_s34_terminal_full_star_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_terminal_full_star_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_s34_terminal_full_star_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_ynt_zonal3_v1/build.py](code/codex_resume_20260926_v2/hill_climb_s34_ynt_zonal3_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_ynt_zonal3_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_s34_ynt_zonal3_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_ynt_zonal3_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_s34_ynt_zonal3_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_s34_ynt_zonal3_v1/probe.py](code/codex_resume_20260926_v2/hill_climb_s34_ynt_zonal3_v1/probe.py)
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
- [code/codex_resume_20260926_v2/s33_two_update_warm_proposal_v1/candidate.py](code/codex_resume_20260926_v2/s33_two_update_warm_proposal_v1/candidate.py)
- [code/codex_resume_20260926_v2/s33_two_update_warm_proposal_v1/native_probe.py](code/codex_resume_20260926_v2/s33_two_update_warm_proposal_v1/native_probe.py)
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

- [handoff_root_archive_20260930T051221Z.md#L10](../learned-covariance-pair-bond-reuse-all-outputs-exact-native/handoff_root_archive_20260930T051221Z.md#L10)

<details>
<summary>All result files and records (1)</summary>

- [handoff_root_archive_20260930T051221Z.md](../learned-covariance-pair-bond-reuse-all-outputs-exact-native/handoff_root_archive_20260930T051221Z.md) · record

</details>

## Related work

- shipped in: [Submission #333099: scatter-based gathers failed one wall and three residual-time limits](../../../submissions/333099/README.md)
- shipped in: [Submission #333111: scatter-based gather with an eager setup prefix of 128 frames](../../../submissions/333111/README.md)

[Research graph](../../../graph/index.html#node=2026-09-30-old-tier-intermediate-finite-scans-removed-two-network-adjusted) · [Copy provenance](../../../PROVENANCE.json)

