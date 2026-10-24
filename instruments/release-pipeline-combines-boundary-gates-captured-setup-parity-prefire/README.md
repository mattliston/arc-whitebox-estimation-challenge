# Release pipeline combines boundary gates, captured-setup parity and prefire signature checks

2026-10-16T01:53:02Z · tool · adopted · measured

The staged release runner was exercised end to end before use and later generalized to arbitrary stack bases and appended fragments. It exported exact archives, verified captured-setup prediction/FLOP parity and required a clean signature check.

Recorded measurement:

    dryrun3m11s;later5m26s;normal+slowranks35/100all16bytes/FLOPsexact

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/captured_setup_replay_generic_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/captured_setup_replay_generic_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/captured_setup_replay_generic_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/captured_setup_replay_generic_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/native_probe_generic_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/native_probe_generic_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/native_probe_generic_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/native_probe_generic_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/release_gov_realfile_finish_v3.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/release_gov_realfile_finish_v3.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/release_gov_realfile_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/release_gov_realfile_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/release_gov_realfile_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/release_gov_realfile_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/s3/assemble_s3_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/s3/assemble_s3_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/s3/assemble_s3_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/s3/assemble_s3_v2.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)

<details>
<summary>Additional programs and dependencies (27)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/fire_once.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/prefire_check_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/fire/prefire_check_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/designation_judge_official120_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/designation_judge_official120_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/early_count_gate_official120_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/early_count_gate_official120_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/release_gate_official120_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/release_gate_official120_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/stage_c_official120_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1/stage_c_official120_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_f384_v1/diag/diag_judge_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_f384_v1/diag/diag_judge_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_lean_v1/candidate_S2t_lean_VB_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_lean_v1/candidate_S2t_lean_VB_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/release_pipeline_v4.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/release_pipeline_v4.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/release_pipeline_v5.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/release_pipeline_v5.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/stage_a_generic_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline/stage_a_generic_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_client.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_client.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_run.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/boundary_run.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/common.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/common.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/designation_judge.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/designation_judge.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/fastgate_pin_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/fastgate_pin_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/project.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/project.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v2.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v2.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v4.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/release_gate_v4.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/rusage.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/rusage.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/server_light.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/server_light.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/setup_clock.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/setup_clock.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_c_boundary.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release/stage_c_boundary.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3666](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3666)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3660](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3660)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3735](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3735)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3752](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3752)

<details>
<summary>All result files and records (8)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [EVIDENCE_FINAL_v1.txt](EVIDENCE_FINAL_v1.txt) · result
- [EVIDENCE_GOV3_v1.txt](EVIDENCE_GOV3_v1.txt) · result
- [EVIDENCE_S2_v1.txt](EVIDENCE_S2_v1.txt) · result
- [EVIDENCE_S2t_v1.txt](EVIDENCE_S2t_v1.txt) · result
- [EVIDENCE_SAFE_v1.txt](EVIDENCE_SAFE_v1.txt) · result
- [fragment_noop_test_v1.py.txt](fragment_noop_test_v1.py.txt) · result
- [EVIDENCE_S3p_base_v1.txt](EVIDENCE_S3p_base_v1.txt) · result

</details>

[Research graph](../../graph/index.html#node=2026-10-16-release-pipeline-combines-boundary-gates-captured-setup-parity-prefire) · [Copy provenance](../../PROVENANCE.json)

