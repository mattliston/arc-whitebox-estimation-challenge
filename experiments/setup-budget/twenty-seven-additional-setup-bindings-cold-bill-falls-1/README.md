# Twenty-seven additional setup bindings: cold bill falls 1,493,091,978 FLOPs

2026-09-28 · experiment · adopted · measured

Twenty-seven keys absent from the earlier training census were appended to setup and qualified with native, worker, boundary and stress tests. The saving was concentrated on the cold first network.

Recorded measurement:

    Cold35 saving1,493,091,978F(-.5399%);warm100 tie;pooled bill-.270478%;setup651/627,max3.30412s;handles49,069;CPUratio1.01142;residual projection.293451s;peak5.844GB,zero swap.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/build.py](code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/build.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py](code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py)
- [code/codex_resume_20260926_v2/handle_client_v2.py](code/codex_resume_20260926_v2/handle_client_v2.py)
- [code/codex_resume_20260926_v2/handle_server_v2.py](code/codex_resume_20260926_v2/handle_server_v2.py)
- [code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/assess_probe.py](code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/assess_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/run_probe.py](code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/run_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_fused_d3_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star32_fused_d3_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_v1/candidate.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [code/codex_resume_20260926_v2/native_eval.py](code/codex_resume_20260926_v2/native_eval.py)
- [code/codex_resume_20260926_v2/run_boundary_census.py](code/codex_resume_20260926_v2/run_boundary_census.py)
- [code/codex_resume_20260926_v2/run_handle_worker_v3.py](code/codex_resume_20260926_v2/run_handle_worker_v3.py)
- [code/codex_resume_20260926_v2/setup_only_client_v1.py](code/codex_resume_20260926_v2/setup_only_client_v1.py)
- [code/codex_resume_20260926_v2/setup_stress_driver_v1.py](code/codex_resume_20260926_v2/setup_stress_driver_v1.py)
- [code/fable_porter_v1/census_boundary_setup_v2.py](code/fable_porter_v1/census_boundary_setup_v2.py)

</details>

## Evidence

- [HILL_CLIMB_before_birth_screens_v1.md#L21](../../third-order-planes/consumer-direction-star-residual-repair-fixed16-gain-has-confidence/HILL_CLIMB_before_birth_screens_v1.md#L21)

<details>
<summary>All result files and records (7)</summary>

- [HILL_CLIMB_before_birth_screens_v1.md](../../third-order-planes/consumer-direction-star-residual-repair-fixed16-gain-has-confidence/HILL_CLIMB_before_birth_screens_v1.md) · record
- [RESULT.md](RESULT.md) · record
- [native2_assessment.json](native2_assessment.json) · result
- [retained_component.json](retained_component.json) · result
- [probe_assessment.json](probe_assessment.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record

</details>

## Related work

- shipped in: [Submission #332838: precision changes with symmetric 128-row diagonal covariance blocks](../../../submissions/332838/README.md)

[Research graph](../../../graph/index.html#node=2026-09-28-twenty-seven-additional-setup-bindings-cold-bill-falls-1) · [Copy provenance](../../../PROVENANCE.json)

