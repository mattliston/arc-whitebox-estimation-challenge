# Nine depth-three setup bindings: requests fall by 996 cold and 870 warm

2026-09-28 · experiment · adopted · measured

Nine fixed-shape depth-three keys were prebound during setup. The original equal-bill assertion failure was preserved and corrected because cold reshape charges moved into setup while warm arithmetic remained unchanged.

Recorded measurement:

    All16 outputs exact on2 networks;requests-996cold/-870warm;cold bill-23,761,920F,warm unchanged;633bindings/609fused;maximum setup stress3.0904s;peak5.959GB,zero swap.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_star32_fused_d3_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star32_fused_d3_v1/candidate.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py](code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py)
- [code/codex_resume_20260926_v2/handle_client_v2.py](code/codex_resume_20260926_v2/handle_client_v2.py)
- [code/codex_resume_20260926_v2/handle_server_v2.py](code/codex_resume_20260926_v2/handle_server_v2.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_fused_d3_v1/assess_probe.py](code/codex_resume_20260926_v2/hill_climb_star32_fused_d3_v1/assess_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_fused_d3_v1/run_probe.py](code/codex_resume_20260926_v2/hill_climb_star32_fused_d3_v1/run_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py)
- [code/codex_resume_20260926_v2/run_boundary_census.py](code/codex_resume_20260926_v2/run_boundary_census.py)
- [code/codex_resume_20260926_v2/run_handle_worker_v3.py](code/codex_resume_20260926_v2/run_handle_worker_v3.py)
- [code/codex_resume_20260926_v2/setup_only_client_v1.py](code/codex_resume_20260926_v2/setup_only_client_v1.py)
- [code/codex_resume_20260926_v2/setup_stress_driver_v1.py](code/codex_resume_20260926_v2/setup_stress_driver_v1.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/fable_porter_v1/census_boundary_setup_v2.py](code/fable_porter_v1/census_boundary_setup_v2.py)

</details>

## Evidence

- [HILL_CLIMB_before_uvp_bcache_v1.md#L60](../../birth-terms/birth-selector-enlarged-32-units-layers-110-expanded-adjusted/HILL_CLIMB_before_uvp_bcache_v1.md#L60)

<details>
<summary>All result files and records (3)</summary>

- [HILL_CLIMB_before_uvp_bcache_v1.md](../../birth-terms/birth-selector-enlarged-32-units-layers-110-expanded-adjusted/HILL_CLIMB_before_uvp_bcache_v1.md) · record
- [probe_assessment.json](probe_assessment.json) · result
- [protocol.json](protocol.json) · result

</details>

## Related work

- shipped in: [Submission #332838: precision changes with symmetric 128-row diagonal covariance blocks](../../../submissions/332838/README.md)

[Research graph](../../../graph/index.html#node=2026-09-28-nine-depth-three-setup-bindings-requests-fall-996-cold) · [Copy provenance](../../../PROVENANCE.json)

