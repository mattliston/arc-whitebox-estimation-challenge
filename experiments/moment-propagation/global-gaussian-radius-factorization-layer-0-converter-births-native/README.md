# Global Gaussian-radius factorization of the layer-0 converter and births: native adjusted +3.17% / +1.36%, v2 worse

2026-10-17T04:10:47Z · experiment · rejected · measured

A coherent layer-0 candidate replaced the input converter by an exact global Gaussian-radius factorization with exact spherical mixed-moment beta and births, radial-subtracted fourth-order term, full post-22 NP diagonal and F4 source; the algebra proofs passed but the native matched pair was adverse. A trace-only v2 that passes the mandatory third-tensor trace exactly was worse. Closed without extension.

Recorded measurement:

    v1 (4e7749ea) vs current-SDK control: raw +3.167409% / +1.359343%, adjusted +3.167623% / +1.360018%, geometric +2.259827% (log SE .883818%), pooled +2.196684%, 0/2 wins; bill pooled +.000437%, requests +28 each. v2 (594acde9): adjusted +3.52318% / +1.44343%, mean +2.48330%, 0/2 wins.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/global_radius_checks_v1/check_analytic_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/global_radius_checks_v1/check_analytic_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/global_radius_checks_v1/check_literal_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/global_radius_checks_v1/check_literal_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/global_radius_checks_v1/price_literal_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/global_radius_checks_v1/price_literal_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/global_radius_grade_v2/grade_fixed2_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/global_radius_grade_v2/grade_fixed2_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/global_radius_v1/build_candidate_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/global_radius_v1/build_candidate_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/global_radius_v1/build_candidate_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/global_radius_v1/build_candidate_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/global_radius_v1/candidate_global_radius_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/global_radius_v1/candidate_global_radius_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/global_radius_v1/candidate_global_radius_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/global_radius_v1/candidate_global_radius_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/global_radius_v1/run_pilot_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/global_radius_v1/run_pilot_v1.py)

<details>
<summary>Additional programs and dependencies (15)</summary>

- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/global_radius_v1/run_pilot_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/global_radius_v1/run_pilot_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/global_radius_v1/trace_identity_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/global_radius_v1/trace_identity_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_radius_birth_v1/grade_fixed2_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_radius_birth_v1/grade_fixed2_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_radius_birth_v1/tiny_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_radius_birth_v1/tiny_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/failure336631_v1/boundary_client_sdk_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/failure336631_v1/boundary_client_sdk_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/failure336631_v1/boundary_run_sdk_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/failure336631_v1/boundary_run_sdk_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/failure336631_v1/candidate_setup_repair_v18c.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/failure336631_v1/candidate_setup_repair_v18c.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_release_v17.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_release_v17.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)
- [code/opus_reexam_20261001_v1/release/common.py](code/opus_reexam_20261001_v1/release/common.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/release/setup_clock.py](code/opus_reexam_20261001_v1/release/setup_clock.py)
- [code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4795](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4795)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4797](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4797)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4799](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4799)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4805](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4805)

<details>
<summary>All result files and records (5)</summary>

- [FIXED2_RESULT_v1.json](FIXED2_RESULT_v1.json) · result
- [CLOSURE_v1.json](CLOSURE_v1.json) · record
- [FIXED2_RESULT_v2.json](FIXED2_RESULT_v2.json) · result
- [PILOT_CONTROLS_v1.json](PILOT_CONTROLS_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-global-gaussian-radius-factorization-layer-0-converter-births-native) · [Copy provenance](../../../PROVENANCE.json)

