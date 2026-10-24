# Direct covariance diagonals and first-product diagonal split both worsened native error

2026-10-18T13:56:09Z · experiment · rejected · measured

Cheap direct hidden-covariance diagonal evaluation and a first-product diagonal split were tested in fixed matched native pairs. Both lost raw accuracy and were closed.

Recorded measurement:

    Direct diagonal estimated bill <=.019%; native pairs lose raw accuracy.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/run_fixed2_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/run_fixed2_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/tiny_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/tiny_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/fragment_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/fragment_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/build_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/build_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/candidate_95c_direct_covdiag_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/candidate_95c_direct_covdiag_HELD_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/analyze_fixed2_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/analyze_fixed2_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_first_diag_split_peer_v1/tiny_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_first_diag_split_peer_v1/tiny_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_first_diag_split_peer_v1/analyze_fixed_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_first_diag_split_peer_v1/analyze_fixed_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_first_diag_split_peer_v1/run_control_split_fixed_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_first_diag_split_peer_v1/run_control_split_fixed_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py)

<details>
<summary>Additional programs and dependencies (14)</summary>

- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/harness/boundary_client_fusion_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/harness/boundary_client_fusion_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/harness/boundary_run_fusion_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/harness/boundary_run_fusion_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/harness/run_guarded_fusion_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_diag_direct_v1/harness/run_guarded_fusion_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_first_diag_split_peer_v1/harness/boundary_client_fusion_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_first_diag_split_peer_v1/harness/boundary_client_fusion_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_first_diag_split_peer_v1/harness/boundary_run_fusion_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_first_diag_split_peer_v1/harness/boundary_run_fusion_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_first_diag_split_peer_v1/harness/run_guarded_fusion_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/cov_first_diag_split_peer_v1/harness/run_guarded_fusion_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_lead_v1/cov_first_diag_split_v1/candidate_95c_cov_first_diag_split_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_lead_v1/cov_first_diag_split_v1/candidate_95c_cov_first_diag_split_HELD_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_lead_v1/cov_first_diag_split_v1/fragment_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_lead_v1/cov_first_diag_split_v1/fragment_v1.py)
- [code/opus_reexam_20261001_v1/release/common.py](code/opus_reexam_20261001_v1/release/common.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/release/setup_clock.py](code/opus_reexam_20261001_v1/release/setup_clock.py)
- [code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5688](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5688)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5690](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5690)

<details>
<summary>All result files and records (10)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [RESULT_v1.json](RESULT_v1.json) · result
- [RESULT_v1-3ae11397.json](RESULT_v1-3ae11397.json) · result
- [ALGEBRAIC_TARIFF_v1.json](ALGEBRAIC_TARIFF_v1.json) · result
- [COMPLETE_v1.json](COMPLETE_v1.json) · result
- [CONTROLLER_v1.json](CONTROLLER_v1.json) · result
- [SYMSTRIP_MATCHED_CONTROL_v1.json](SYMSTRIP_MATCHED_CONTROL_v1.json) · result
- [COMPLETE_v1-df532ce3.json](COMPLETE_v1-df532ce3.json) · result
- [CONTROLLER_v1-52ccda14.json](CONTROLLER_v1-52ccda14.json) · result
- [COLD200_CONTROL_v1.json](COLD200_CONTROL_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-direct-covariance-diagonals-first-product-diagonal-split-both-worsened) · [Copy provenance](../../../PROVENANCE.json)

