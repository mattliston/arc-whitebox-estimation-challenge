# Old-base cross correction increased adjusted error by 0.26769% on two networks

2026-10-18T13:46:35Z · experiment · rejected · measured

A cross correction to the old-base term was tested on the current compact source. Both the measured raw and adjusted mean increased, so it was not extended.

Recorded measurement:

    Raw +.19092%, adjusted +.26769% → 1.450729652e-9.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/re22_cross_old0_v1/run_fixed2_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/re22_cross_old0_v1/run_fixed2_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/re22_cross_old0_v1/analyze_native_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/re22_cross_old0_v1/analyze_native_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/re22_cross_old0_v1/fragment_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/re22_cross_old0_v1/fragment_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/re22_cross_old0_v1/build_and_tiny_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/re22_cross_old0_v1/build_and_tiny_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/re22_cross_old0_v1/candidate_95c_re22cross_old0_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/re22_cross_old0_v1/candidate_95c_re22cross_old0_HELD_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/re22_cross_old0_v1/harness/boundary_client_rtt_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/re22_cross_old0_v1/harness/boundary_client_rtt_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/re22_cross_old0_v1/harness/boundary_run_rtt_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/re22_cross_old0_v1/harness/boundary_run_rtt_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/re22_cross_old0_v1/harness/run_guarded_rtt_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/post336935_training_audit_v1/re22_cross_old0_v1/harness/run_guarded_rtt_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/opus_reexam_20261001_v1/release/common.py](code/opus_reexam_20261001_v1/release/common.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/release/setup_clock.py](code/opus_reexam_20261001_v1/release/setup_clock.py)
- [code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5686](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5686)

<details>
<summary>All result files and records (6)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [RESULT_v1.json](RESULT_v1.json) · result
- [ROOT_REVIEW_v1.json](ROOT_REVIEW_v1.json) · result
- [DECISION_v1.json](DECISION_v1.json) · result
- [ALL_RETIRED_v1.json](ALL_RETIRED_v1.json) · result
- [BUILD_v1.json](BUILD_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-old-base-cross-correction-increased-adjusted-error-0-26769) · [Copy provenance](../../../PROVENANCE.json)

