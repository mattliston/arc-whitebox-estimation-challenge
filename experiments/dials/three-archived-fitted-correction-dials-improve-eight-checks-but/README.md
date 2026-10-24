# Three archived fitted correction dials improve eight checks but regress on 40-network confirmation

2026-09-30 · experiment · rejected · measured

A previously fitted fixed three-dial step was replayed on the current retained source without refitting. It improved fitting and eight check networks, then failed full40 component qualification despite passing resource checks.

Recorded measurement:

    FIT4 raw-.0733710338%,adjusted-.0735071885%,2/4wins;CHECK8 raw-.3842343106527313%,adjusted-.379091170415935%,7/8wins,adjusted CI[-.6602932,-.1468772]%;full40 raw+.1878011658%,adjusted+.0814837429%,17adjusted/16raw wins,maxreg1.6181580%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/candidate.py)
- [code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py](code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py](code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/boundary_assess_s34.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/boundary_assess_s34.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/boundary_feasibility_assess.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/boundary_feasibility_assess.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/build.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/build.py)

<details>
<summary>Additional programs and dependencies (30)</summary>

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
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star8_component40_v1/prepare_full40.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star8_component40_v1/prepare_full40.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star8_component40_v1/valid40_engine_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star8_component40_v1/valid40_engine_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star8_component40_v1/valid40_engine_v1/run.py](code/codex_resume_20260926_v2/hill_climb_rrpi_detached_star8_component40_v1/valid40_engine_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/assess_check8.py](code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/assess_check8.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/assess_fit4.py](code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/assess_fit4.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/assess_resources.py](code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/assess_resources.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/build_candidate.py](code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/build_candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/prepare_full40.py](code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/prepare_full40.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/valid40_engine_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/valid40_engine_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/valid40_engine_v1/run.py](code/codex_resume_20260926_v2/hill_climb_rrpi_native_joint3_recovery_v1/valid40_engine_v1/run.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/build.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star32_q2_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_probe_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_probe_v1/native_probe.py)
- [code/codex_resume_20260926_v2/native_eval.py](code/codex_resume_20260926_v2/native_eval.py)
- [code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py](code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py)
- [code/codex_resume_20260926_v2/run_boundary_census.py](code/codex_resume_20260926_v2/run_boundary_census.py)
- [code/fable_porter_v1/census_boundary_setup_v2.py](code/fable_porter_v1/census_boundary_setup_v2.py)

</details>

## Evidence

- [HILL_CLIMB_CURRENT.md#L247](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L247)
- [HILL_CLIMB_CURRENT.md#L261](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L261)
- [HILL_CLIMB_CURRENT.md#L257](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L257)

<details>
<summary>All result files and records (6)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [broad_assessment.json](broad_assessment.json) · result
- [build.json](build.json) · result
- [fit4_assessment.json](fit4_assessment.json) · result
- [check8_assessment.json](check8_assessment.json) · result
- [closeout.json](closeout.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-three-archived-fitted-correction-dials-improve-eight-checks-but) · [Copy provenance](../../../PROVENANCE.json)

