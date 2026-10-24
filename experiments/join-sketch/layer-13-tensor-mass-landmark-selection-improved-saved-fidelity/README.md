# Layer-13 tensor-mass landmark selection improved saved fidelity but lost 0.292914% adjusted natively

2026-10-18T16:51:45Z · experiment · rejected · measured

A single layer-13 selector ranked joined content by tensor mass while leaving earlier layers byte-identical. Saved fidelity gains did not translate into native final accuracy.

Recorded measurement:

    Saved fidelity gain 9.33/6.21%; raw +.506026/+.040135%; bill +.019503/+.020057%; adjusted mean +.292914% → 1.451094598e-9; requests +28.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_tensor_membership_native_v1/grade_fixed2_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_tensor_membership_native_v1/grade_fixed2_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_tensor_membership_native_v1/harness/boundary_client_fusion_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_tensor_membership_native_v1/harness/boundary_client_fusion_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_tensor_membership_native_v1/harness/boundary_run_fusion_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_tensor_membership_native_v1/harness/boundary_run_fusion_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_tensor_membership_native_v1/harness/run_guarded_fusion_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_tensor_membership_native_v1/harness/run_guarded_fusion_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_tensor_membership_native_v1/run_fixed2_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_tensor_membership_native_v1/run_fixed2_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_terminal_spline_v1/build_tensor_membership_l13_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_terminal_spline_v1/build_tensor_membership_l13_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_terminal_spline_v1/candidate_95c_tensor_membership_l13_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_terminal_spline_v1/candidate_95c_tensor_membership_l13_HELD_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_terminal_spline_v1/fragment_tensor_membership_l13_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_terminal_spline_v1/fragment_tensor_membership_l13_v1.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_terminal_spline_v1/probe_tensor_membership_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_terminal_spline_v1/probe_tensor_membership_v1.py)
- [code/opus_reexam_20261001_v1/release/common.py](code/opus_reexam_20261001_v1/release/common.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/release/setup_clock.py](code/opus_reexam_20261001_v1/release/setup_clock.py)
- [code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5725](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5725)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SEARCH_STATE_v21.json](../../../records/research/2026-10-18-constant-bias-additive-variance-oracles-offered-too-little-usable/SEARCH_STATE_v21.json) · record
- [SEARCH_STATE_v20.json](../nystrom-landmark-gram-symmetry-cost-only-0-01315-bill/SEARCH_STATE_v20.json) · result
- [RESULT_v1.json](../../../records/research/2026-10-18-constant-bias-additive-variance-oracles-offered-too-little-usable/RESULT_v1.json) · result
- [RESULT_v1-a6a44da3.json](../../../records/research/2026-10-18-constant-bias-additive-variance-oracles-offered-too-little-usable/RESULT_v1-a6a44da3.json) · result
- [CHOLESKY_QR_CLOSURE_CHECK_v1.md](../../../records/research/2026-10-18-constant-bias-additive-variance-oracles-offered-too-little-usable/CHOLESKY_QR_CLOSURE_CHECK_v1.md) · record
- [RESULT_v1-b3a7d4ff.json](../../../records/research/2026-10-18-constant-bias-additive-variance-oracles-offered-too-little-usable/RESULT_v1-b3a7d4ff.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-layer-13-tensor-mass-landmark-selection-improved-saved-fidelity) · [Copy provenance](../../../PROVENANCE.json)

