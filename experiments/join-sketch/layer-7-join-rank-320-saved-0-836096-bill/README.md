# Layer-7 join rank 320 saved 0.836096% bill but added cold requests and 0.608336% raw error

2026-10-18T15:31:39Z · experiment · rejected · measured

Cutting the layer-7 join from rank 384 to 320 traded arithmetic for error on the current estimator. Historical runtime savings did not transfer: the bare cut added cold requests, while a lower-request runtime variant lost adjusted score on two cold checks.

Recorded measurement:

    Rank 35: bill -.836096%, raw +.608336%, adjusted -.232846% → 1.443487610e-9; requests +848, residual instructions +1.922%; runtime variant: bill -.80112%, requests -1583/-1202, adjusted +.45081%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_runtime_margin_v1/price_fusion_literal_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_runtime_margin_v1/price_fusion_literal_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_runtime_margin_v1/runtime_fragment_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_runtime_margin_v1/runtime_fragment_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_runtime_margin_v1/run_fixed2_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_runtime_margin_v1/run_fixed2_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_runtime_margin_v1/candidate_L7rank320_runtime_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_runtime_margin_v1/candidate_L7rank320_runtime_HELD_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_runtime_margin_v1/price_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_runtime_margin_v1/price_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_runtime_margin_v1/grade_fixed2_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_runtime_margin_v1/grade_fixed2_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/bini_apa_v1/price_outer_232_v3.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_compute_v1/bini_apa_v1/price_outer_232_v3.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)

<details>
<summary>Additional programs and dependencies (17)</summary>

- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_rank320_current_v1/candidate_95c_L7rank320_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/l7_rank320_current_v1/candidate_95c_L7rank320_HELD_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_strip5_sym_v1/candidate_95c_fusion_strip5_sym_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_strip5_sym_v1/candidate_95c_fusion_strip5_sym_HELD_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_strip5_sym_v1/fragment_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_strip5_sym_v1/fragment_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_strip5_sym_v1/harness/boundary_client_fusion_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_strip5_sym_v1/harness/boundary_client_fusion_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_strip5_sym_v1/harness/boundary_run_fusion_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_strip5_sym_v1/harness/boundary_run_fusion_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_strip5_sym_v1/harness/run_guarded_fusion_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_strip5_sym_v1/harness/run_guarded_fusion_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_strip5_v1/fragment_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_strip5_v1/fragment_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_runtime_composer_v2/candidate_zero_L7_runtime_HELD_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/post935_compute_v1/l7_runtime_composer_v2/candidate_zero_L7_runtime_HELD_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/price_peel_deunion_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/price_peel_deunion_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1/candidate_v61j.py](code/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1/candidate_v61j.py)
- [code/opus_reexam_20261001_v1/lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py](code/opus_reexam_20261001_v1/lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py)
- [code/opus_reexam_20261001_v1/release/common.py](code/opus_reexam_20261001_v1/release/common.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/release/setup_clock.py](code/opus_reexam_20261001_v1/release/setup_clock.py)
- [code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5704](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5704)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5708](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5708)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [RESULT_v1.json](RESULT_v1.json) · result
- [COMPLETE_v1.json](COMPLETE_v1.json) · result
- [CONTROLLER_v1.json](CONTROLLER_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-layer-7-join-rank-320-saved-0-836096-bill) · [Copy provenance](../../../PROVENANCE.json)

