# Negative trace-vector depth contrast fails 40-network component confirmation after eight-check gains

2026-09-30 · experiment · rejected · measured

The fixed historical negative depth contrast improved fitting and eight check networks, then failed the40-network win and confidence gates. Its conditional doubled-slope source remained unexecuted.

Recorded measurement:

    FIT4 raw-.061789%,adjusted-.059731%,3/4wins;CHECK8 raw-.384758709%,adjusted-.381011833%,6/8wins,both CIs negative;full40 raw-.1012249724%,adjusted-.2092631587%,21/40wins,adjusted CI[-.4561788035,+.0156629559],maxreg.8522732%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_rrpi_bdepth_minus_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_rrpi_bdepth_minus_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_bdepth_minus_v1/build.py](code/codex_resume_20260926_v2/hill_climb_rrpi_bdepth_minus_v1/build.py)
- [code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py](code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py](code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py)
- [code/codex_resume_20260926_v2/candidate_trace_vector_depth_tilt_minus_v1.py](code/codex_resume_20260926_v2/candidate_trace_vector_depth_tilt_minus_v1.py)
- [code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py](code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/boundary_assess_s34.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/boundary_assess_s34.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)

<details>
<summary>Additional programs and dependencies (33)</summary>

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
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_bdepth_minus_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_rrpi_bdepth_minus_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_bdepth_minus_v1/assess_resources.py](code/codex_resume_20260926_v2/hill_climb_rrpi_bdepth_minus_v1/assess_resources.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_bdepth_minus_v1/prepare_full40.py](code/codex_resume_20260926_v2/hill_climb_rrpi_bdepth_minus_v1/prepare_full40.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_bdepth_minus_v1/valid40_engine_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_rrpi_bdepth_minus_v1/valid40_engine_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_bdepth_minus_v1/valid40_engine_v1/run.py](code/codex_resume_20260926_v2/hill_climb_rrpi_bdepth_minus_v1/valid40_engine_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/prepare_full40.py](code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/prepare_full40.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/valid40_engine_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/valid40_engine_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/valid40_engine_v1/run.py](code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/valid40_engine_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/build.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/endpoint.py](code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/endpoint.py)
- [code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/fit.py](code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/fit.py)
- [code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_probe_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_probe_v1/native_probe.py)
- [code/codex_resume_20260926_v2/native_eval.py](code/codex_resume_20260926_v2/native_eval.py)
- [code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py](code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py)
- [code/codex_resume_20260926_v2/run_boundary_census.py](code/codex_resume_20260926_v2/run_boundary_census.py)
- [code/fable_porter_v1/census_boundary_setup_v2.py](code/fable_porter_v1/census_boundary_setup_v2.py)

</details>

## Evidence

- [HILL_CLIMB_CURRENT.md#L224](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L224)
- [HILL_CLIMB_CURRENT.md#L239](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L239)
- [HILL_CLIMB_CURRENT.md#L235](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L235)

<details>
<summary>All result files and records (6)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [broad_assessment.json](broad_assessment.json) · result
- [build.json](build.json) · result
- [fit4_assessment.json](fit4_assessment.json) · result
- [check8_assessment.json](check8_assessment.json) · result
- [closeout.json](closeout.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-negative-trace-vector-depth-contrast-fails-40-network-component) · [Copy provenance](../../../PROVENANCE.json)

