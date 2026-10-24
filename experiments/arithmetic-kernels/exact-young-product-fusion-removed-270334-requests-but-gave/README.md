# Exact young-product fusion removed 270–334 requests but gave no confirmed score improvement

2026-10-18T12:37:58Z · experiment · informational · measured

Exact fusion of young product routes preserved owned-output semantics and reduced requests in two native networks. The small mixed adjusted result was retained only as a possible runtime component.

Recorded measurement:

    First 2: requests -270/-334, bill -.020347%, adjusted -.068795%, 1/2 lower; conditional 1.445861185e-9.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/probe_halfH_metadata_v1.py](code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/probe_halfH_metadata_v1.py)
- [code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/probe_fusion_metadata_v1.py](code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/probe_fusion_metadata_v1.py)
- [code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/build_fusion_candidate_v1.py](code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/build_fusion_candidate_v1.py)
- [code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/build_fusion_candidate_v2.py](code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/build_fusion_candidate_v2.py)
- [code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/analyze_static_price_v1.py](code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/analyze_static_price_v1.py)
- [code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/probe_fusion_metadata_v2.py](code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/probe_fusion_metadata_v2.py)
- [code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py](code/engineer/a62_modules/s49/s56k1tw_xs/candidate.py)
- [code/lead/runs/astra_crunch_compute_v1/bini_apa_v1/price_outer_232_v3.py](code/lead/runs/astra_crunch_compute_v1/bini_apa_v1/price_outer_232_v3.py)
- [code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py](code/lead/runs/astra_nbmerge_s56_v1/shape_tariff_probe_v2.py)
- [code/lead/runs/astra_oct18_accuracy_v1/endtrim095_v1/candidate_union_sidebatch_endtrim095_HELD_v1.py](code/lead/runs/astra_oct18_accuracy_v1/endtrim095_v1/candidate_union_sidebatch_endtrim095_HELD_v1.py)
- [code/lead/runs/astra_oct18_accuracy_v1/post336935_accuracy_v1/fragment_cp_halfH_v1.py](code/lead/runs/astra_oct18_accuracy_v1/post336935_accuracy_v1/fragment_cp_halfH_v1.py)
- [code/lead/runs/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py](code/lead/runs/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py)

<details>
<summary>Additional programs and dependencies (9)</summary>

- [code/lead/runs/astra_oct18_bill_v1/analyze_compact_portfolio_v3.py](code/lead/runs/astra_oct18_bill_v1/analyze_compact_portfolio_v3.py)
- [code/lead/runs/astra_oct18_bill_v1/analyze_union_request_allocation_v1.py](code/lead/runs/astra_oct18_bill_v1/analyze_union_request_allocation_v1.py)
- [code/lead/runs/astra_oct18_bill_v1/candidate_v61j_union_shared_HELD_v2.py](code/lead/runs/astra_oct18_bill_v1/candidate_v61j_union_shared_HELD_v2.py)
- [code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_candidate_v1/candidate_95c_fusion_HELD_v1.py](code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_candidate_v1/candidate_95c_fusion_HELD_v1.py)
- [code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_candidate_v1/fragment_v1.py](code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_candidate_v1/fragment_v1.py)
- [code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_candidate_v2/candidate_95c_fusion_HELD_v2.py](code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_candidate_v2/candidate_95c_fusion_HELD_v2.py)
- [code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_candidate_v2/fragment_v2.py](code/lead/runs/astra_oct18_bill_v1/post935_compute_v1/fusion_candidate_v2/fragment_v2.py)
- [code/lead/runs/astra_oct18_bill_v1/request_backup_cap300_v1/candidate_request_backup_cap300_HELD_v1.py](code/lead/runs/astra_oct18_bill_v1/request_backup_cap300_v1/candidate_request_backup_cap300_HELD_v1.py)
- [code/lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py](code/lead/runs/opus_pubmax_v1/cand_pubmax_B_parity.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5667](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5667)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5676](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5676)

<details>
<summary>All result files and records (9)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [RESULT_v1.json](RESULT_v1.json) · result
- [GUARD_v1.json](GUARD_v1.json) · result
- [PLAN_v1.json](PLAN_v1.json) · result
- [GRADE_STORE_v1.json](GRADE_STORE_v1.json) · result
- [manifest.json](manifest.json) · result
- [end_gc_stress.json](end_gc_stress.json) · result
- [server.json](server.json) · result
- [launcher.json](launcher.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-exact-young-product-fusion-removed-270334-requests-but-gave) · [Copy provenance](../../../PROVENANCE.json)

