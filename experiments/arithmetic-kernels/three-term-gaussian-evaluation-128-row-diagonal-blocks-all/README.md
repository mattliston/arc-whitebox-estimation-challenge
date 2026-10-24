# Three-term Gaussian evaluation and 128-row diagonal blocks: all 40 validation scores improve

2026-09-28 · experiment · adopted · measured

The fixed[128,128] diagonal partition reduced client operations while three-term Gaussian evaluation recovered arithmetic cost. The source passed native, early boundary,40-network validation and full serving qualification.

Recorded measurement:

    TRAIN8 adjusted-1.252796500746%,8wins,CI[-1.409043025,-1.103568341]%;VALID40-1.1953067285634167%,40/40wins,CI[-1.3717380078348538,-1.028972795071422]%;raw-.04596230681692681%,bill-1.1488436239046718%;worker20 peak6.958GB;setupstress max3.327253s.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/build.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/build.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/boundary_feasibility_engine_v1/assess.py](code/codex_resume_20260926_v2/boundary_feasibility_engine_v1/assess.py)
- [code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py](code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_hk15_blocked_v2/kernel_blocked.py](code/codex_resume_20260926_v2/hill_climb_hk15_blocked_v2/kernel_blocked.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/probe.py](code/codex_resume_20260926_v2/hill_climb_precision_g3_symdiag128_v1/probe.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/build.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/build.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/candidate.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/fragment.py](code/codex_resume_20260926_v2/hill_climb_precision_g4_symdiag_batched_v2/fragment.py)

<details>
<summary>Additional programs and dependencies (25)</summary>

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
- [code/codex_resume_20260926_v2/native_eval.py](code/codex_resume_20260926_v2/native_eval.py)
- [code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py](code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py)
- [code/codex_resume_20260926_v2/run_boundary_census.py](code/codex_resume_20260926_v2/run_boundary_census.py)
- [code/codex_resume_20260926_v2/valid40_engine_v1/assess.py](code/codex_resume_20260926_v2/valid40_engine_v1/assess.py)
- [code/codex_resume_20260926_v2/valid40_engine_v1/run.py](code/codex_resume_20260926_v2/valid40_engine_v1/run.py)
- [code/fable_porter_v1/census_boundary_setup_v2.py](code/fable_porter_v1/census_boundary_setup_v2.py)

</details>

## Evidence

- [handoff_archive_20260928T1402Z.md#L89](../../series-expansions/batched-first-order-plane-reads-standalone-eight-network-gain/handoff_archive_20260928T1402Z.md#L89)

<details>
<summary>All result files and records (6)</summary>

- [handoff_archive_20260928T1402Z.md](../../series-expansions/batched-first-order-plane-reads-standalone-eight-network-gain/handoff_archive_20260928T1402Z.md) · record
- [native2_assessment.json](native2_assessment.json) · result
- [native8_assessment.json](native8_assessment.json) · result
- [broad_assessment.json](broad_assessment.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record

</details>

## Related work

- shipped in: [Submission #332838: precision changes with symmetric 128-row diagonal covariance blocks](../../../submissions/332838/README.md)

[Research graph](../../../graph/index.html#node=2026-09-28-three-term-gaussian-evaluation-128-row-diagonal-blocks-all) · [Copy provenance](../../../PROVENANCE.json)

