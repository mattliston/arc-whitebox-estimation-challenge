# Base-factor cache and 36 setup bindings combined: eight-network adjusted score improves 0.199586520%

2026-09-28 · experiment · adopted · measured

The exact cache and both setup-key additions were combined and measured as one source. All eight networks retained byte-identical output, while short serving qualification established bounded resources.

Recorded measurement:

    TRAIN8 bill-0.207439895%,adjusted-0.199586520%,raw0,all16 bytes exact8/8;additional6 bill-0.146003599%;setup660/636,max3.26786s;serving peak5.916934GB,49,141handles;boundary requests-2244/-1286;CPUratio0.9922638.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/build.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/build.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py](code/codex_resume_20260926_v2/census_boundary_setup_snapshot_v1.py)
- [code/codex_resume_20260926_v2/handle_client_v2.py](code/codex_resume_20260926_v2/handle_client_v2.py)
- [code/codex_resume_20260926_v2/handle_server_v2.py](code/codex_resume_20260926_v2/handle_server_v2.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_train8_v1/assess.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_train8_v1/assess.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/assess_probe.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/assess_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/d3_delta.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/d3_delta.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/prebind27_delta.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/prebind27_delta.py)
- [code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/run_probe.py](code/codex_resume_20260926_v2/hill_climb_exact_bundle_v1/run_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/assess_probe.py](code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/assess_probe.py)

<details>
<summary>Additional programs and dependencies (19)</summary>

- [code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/build.py](code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/fragment.py)
- [code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/run_probe.py](code/codex_resume_20260926_v2/hill_climb_s32_prebind27_v1/run_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_fused_d3_v1/assess_probe.py](code/codex_resume_20260926_v2/hill_climb_star32_fused_d3_v1/assess_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_fused_d3_v1/assess_probe_v2.py](code/codex_resume_20260926_v2/hill_climb_star32_fused_d3_v1/assess_probe_v2.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_fused_d3_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star32_fused_d3_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_star32_fused_d3_v1/run_probe.py](code/codex_resume_20260926_v2/hill_climb_star32_fused_d3_v1/run_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_star_projection32_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_probe_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_probe_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_v1/build.py](code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_v1/build.py)
- [code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_v1/fragment.py](code/codex_resume_20260926_v2/hill_climb_uvp_base_bcache_v1/fragment.py)
- [code/codex_resume_20260926_v2/native_eval.py](code/codex_resume_20260926_v2/native_eval.py)
- [code/codex_resume_20260926_v2/run_boundary_census.py](code/codex_resume_20260926_v2/run_boundary_census.py)
- [code/codex_resume_20260926_v2/run_handle_worker_v3.py](code/codex_resume_20260926_v2/run_handle_worker_v3.py)
- [code/codex_resume_20260926_v2/setup_only_client_v1.py](code/codex_resume_20260926_v2/setup_only_client_v1.py)
- [code/codex_resume_20260926_v2/setup_stress_driver_v1.py](code/codex_resume_20260926_v2/setup_stress_driver_v1.py)
- [code/fable_porter_v1/census_boundary_setup_v2.py](code/fable_porter_v1/census_boundary_setup_v2.py)

</details>

## Evidence

- [HILL_CLIMB_before_hk_train8_closeout_v1.md#L13](HILL_CLIMB_before_hk_train8_closeout_v1.md#L13)

<details>
<summary>All result files and records (10)</summary>

- [HILL_CLIMB_before_hk_train8_closeout_v1.md](HILL_CLIMB_before_hk_train8_closeout_v1.md) · record
- [RESULT.md](RESULT.md) · record
- [native2_assessment.json](native2_assessment.json) · result
- [retained_bundle.json](retained_bundle.json) · result
- [probe_assessment.json](probe_assessment.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record
- [RESULT-357961b9.md](RESULT-357961b9.md) · record
- [assessment.json](assessment.json) · result
- [protocol.json](protocol.json) · result

</details>

## Related work

- shipped in: [Submission #332838: precision changes with symmetric 128-row diagonal covariance blocks](../../../submissions/332838/README.md)

[Research graph](../../../graph/index.html#node=2026-09-28-base-factor-cache-36-setup-bindings-combined-eight-network) · [Copy provenance](../../../PROVENANCE.json)

