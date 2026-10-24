# Three redundant dtype casts removed: two-network adjusted increment improves 0.02372371701289122%

2026-09-30 · experiment · adopted · measured

Three proven redundant casts in star factoring, learned-correction return and terminal covariance were removed and qualified further only in the combined scatter source.

Recorded measurement:

    TRAIN2 adjusted-.02372371701289122%,raw0,all16 exact.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_exact_dtype_three_sites_v1/build.py](code/codex_resume_20260926_v2/hill_climb_exact_dtype_three_sites_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_dtype_three_sites_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_exact_dtype_three_sites_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_dtype_three_sites_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_exact_dtype_three_sites_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_dtype_three_sites_v1/screen.py](code/codex_resume_20260926_v2/hill_climb_exact_dtype_three_sites_v1/screen.py)
- [code/codex_resume_20260926_v2/boundary_feasibility_engine_v3/assess.py](code/codex_resume_20260926_v2/boundary_feasibility_engine_v3/assess.py)
- [code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py](code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py)
- [code/codex_resume_20260926_v2/exact_dtype_three_sites_v1/fragment.py](code/codex_resume_20260926_v2/exact_dtype_three_sites_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_bd_delta_uvp_read_cache_v1/screen.py](code/codex_resume_20260926_v2/hill_climb_bd_delta_uvp_read_cache_v1/screen.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_plane_reuse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_exact_plane_reuse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/assess.py)

<details>
<summary>Additional programs and dependencies (19)</summary>

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
- [code/codex_resume_20260926_v2/run_boundary_census.py](code/codex_resume_20260926_v2/run_boundary_census.py)
- [code/fable_porter_v1/census_boundary_setup_v2.py](code/fable_porter_v1/census_boundary_setup_v2.py)

</details>

## Evidence

- [HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z#L105](../../setup-budget/all-setup-keys-fused-eagerly-degraded-setup-drops-bindings/HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z#L105)

<details>
<summary>All result files and records (1)</summary>

- [HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z](../../setup-budget/all-setup-keys-fused-eagerly-degraded-setup-drops-bindings/HANDOFF_CURRENT.md.before_s35_rr_20260930T1308Z) · record

</details>

## Related work

- shipped in: [Submission #333111: scatter-based gather with an eager setup prefix of 128 frames](../../../submissions/333111/README.md)

[Research graph](../../../graph/index.html#node=2026-09-30-three-redundant-dtype-casts-removed-two-network-adjusted-increment) · [Copy provenance](../../../PROVENANCE.json)

