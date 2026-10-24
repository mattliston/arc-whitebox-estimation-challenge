# Batched first-order plane reads: standalone eight-network gain has confidence crossing zero

2026-09-28 · experiment · adopted · measured

Batching six borrowed-arena planes preserved the first-order read output exactly while removing104 calls per network. Boundary qualification passed, the standalone confidence gate failed, and the component later entered the young-negative-trace combination.

Recorded measurement:

    TRAIN8 adjusted-.3081359905425%,7wins,raw-.0269526384%,bill-.285016743%,CI[-.6688088604,+.0947707256];boundary residual.340650751,wall87.121856,setup3.342.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_precision_g3_plane_e1_gnbatch_v2/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_plane_e1_gnbatch_v2/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_plane_e1_gnbatch_v2/build.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_plane_e1_gnbatch_v2/build.py)
- [code/codex_resume_20260926_v2/boundary_feasibility_engine_v2/assess.py](code/codex_resume_20260926_v2/boundary_feasibility_engine_v2/assess.py)
- [code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py](code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_plane_e1_gnbatch_v2/fragment.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_plane_e1_gnbatch_v2/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_plane_e1_gnbatch_v2/native2_parity.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_plane_e1_gnbatch_v2/native2_parity.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_plane_e1_gnbatch_v2/probe.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_plane_e1_gnbatch_v2/probe.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_plane_e1_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_plane_e1_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/boundary_feasibility_assess.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_v1/boundary_feasibility_assess.py)

<details>
<summary>Additional programs and dependencies (19)</summary>

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
- [code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py](code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py)
- [code/codex_resume_20260926_v2/run_boundary_census.py](code/codex_resume_20260926_v2/run_boundary_census.py)
- [code/fable_porter_v1/census_boundary_setup_v2.py](code/fable_porter_v1/census_boundary_setup_v2.py)

</details>

## Evidence

- [handoff_archive_20260928T1402Z.md#L50](handoff_archive_20260928T1402Z.md#L50)

<details>
<summary>All result files and records (6)</summary>

- [handoff_archive_20260928T1402Z.md](handoff_archive_20260928T1402Z.md) · record
- [RESULT.md](RESULT.md) · record
- [native2_assessment.json](native2_assessment.json) · result
- [native8_assessment.json](native8_assessment.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record

</details>

## Related work

- shipped in: [Submission #333050: young-column cutoff and depth-5 symmetric covariance products](../../../submissions/333050/README.md)

[Research graph](../../../graph/index.html#node=2026-09-28-batched-first-order-plane-reads-standalone-eight-network-gain) · [Copy provenance](../../../PROVENANCE.json)

