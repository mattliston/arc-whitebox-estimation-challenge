# Expanded union and layer-5 products: 32-network adjusted -0.674299339%, pending runtime repair

2026-10-18T07:20:48Z · experiment · adopted · measured

Expanded union and layer-5 product coverage, shared products, side batching and bounded end-only retention confirmed an adjusted saving on 32 current-lineage networks. Setup family consolidation and end-only retained-memory limits were necessary to fit the runtime envelope.

Recorded measurement:

    Adjusted -0.674299339% → 1.44171778024e-9; held 25: -0.721282769%; 30/32 and 24/25 wins; raw +0.009155%, bill -0.683352%, SE 0.074918 percentage points; max RSS 7.299 GB.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/build_endtrim095_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/build_endtrim095_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/endtrim095_v1/candidate_union_sidebatch_endtrim095_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/endtrim095_v1/candidate_union_sidebatch_endtrim095_HELD_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/endtrim095_v1/fragment_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/endtrim095_v1/fragment_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/run_shared32_continuation_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/run_shared32_continuation_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/run_shared32_continuation_v4.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/run_shared32_continuation_v4.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/boundary_client_product_census_v4.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/boundary_client_product_census_v4.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/boundary_client_product_census_v5.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/boundary_client_product_census_v5.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/boundary_run_product_census_v4.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/boundary_run_product_census_v4.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/boundary_run_product_census_v5.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/boundary_run_product_census_v5.py)

<details>
<summary>Additional programs and dependencies (12)</summary>

- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/candidate_v61j_union_shared_HELD_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/candidate_v61j_union_shared_HELD_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/final_union_sidebatch_v1/candidate_union_sidebatch_HELD_v1.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/final_union_sidebatch_v1/candidate_union_sidebatch_HELD_v1.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/run_guarded_product_census_v4.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/run_guarded_product_census_v4.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/run_guarded_product_census_v5.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_bill_v1/run_guarded_product_census_v5.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_oct18_lead_v1/run_current_controls25_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_oct18_lead_v1/run_current_controls25_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1/candidate_v61j.py](code/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1/candidate_v61j.py)
- [code/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1/grade_store_v1.py](code/opus_reexam_20261001_v1/lead/runs/opus_fix336687_v1/grade_store_v1.py)
- [code/opus_reexam_20261001_v1/release/common.py](code/opus_reexam_20261001_v1/release/common.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/release/setup_clock.py](code/opus_reexam_20261001_v1/release/setup_clock.py)
- [code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5360](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5360)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5372](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5372)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5379](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5379)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5393](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5393)

<details>
<summary>All result files and records (6)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [E7F1_RELEASE_SEAL_v1.json](E7F1_RELEASE_SEAL_v1.json) · record
- [RESULT32_v2.json](RESULT32_v2.json) · result
- [astra_runtime_v62u_e7f1c3a0_SUMMARY.json](astra_runtime_v62u_e7f1c3a0_SUMMARY.json) · result
- [OWN_PREBIND_77ff48f5_v1.json](OWN_PREBIND_77ff48f5_v1.json) · result
- [ADDENDUM_E7F1_v1.json](ADDENDUM_E7F1_v1.json) · result

</details>

## Related work

- shipped in: [Submission #336910: expanded union and layer-5 product portfolio: failed during setup](../../../submissions/336910/README.md)

[Research graph](../../../graph/index.html#node=2026-10-18-expanded-union-layer-5-products-32-network-adjusted-0) · [Copy provenance](../../../PROVENANCE.json)

