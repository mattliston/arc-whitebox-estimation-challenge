# Native evaluation uses 7.9 GB caps and 15% free memory; release gates retain 7.5 GB and 50%

2026-10-19T16:29:42Z · tool · adopted · measured

This lineage peaks at 7.3-7.6 GB RSS on rank 200, so the standard judge's 7.5 GB RSS backstop retired value-only arms regardless of their physics. Copies of the 32-rank paired judge (run_current32_fable_v1, the seven-rank authorized judge without its early return, grading held-out 25 and all 32) were made with a 7.9 GB RSS backstop, then a 7.9 GB footprint cap, then a 15% system-free rule with a matching verify_part assert, plus guard and census-runner copies; a partial grader salvages retired lanes. Release files are still gated on the original 7.5 GB and 50% rules.

Recorded measurement:

    six dial arms retired at rank 200 at max RSS 7.50-7.59 GB; lineage 7.3-7.6 GB (base 7.04, bundle 7.46, dcv002 7.02); platform cap 8 GB; the judge's own min_pressure_free_pct >= 50 assert aborted rc_early15 after 8 parts at min free 46%

## Programs

- [code/astra_oct18_small_stack_qualification_v1/run_current32_fable_v1.py](code/astra_oct18_small_stack_qualification_v1/run_current32_fable_v1.py)
- [code/astra_oct18_small_stack_qualification_v1/run_current7_authorized_v1.py](code/astra_oct18_small_stack_qualification_v1/run_current7_authorized_v1.py)
- [code/astra_oct18_small_stack_qualification_v1/run_current32_rss79_fable_v1.py](code/astra_oct18_small_stack_qualification_v1/run_current32_rss79_fable_v1.py)
- [code/astra_oct18_small_stack_qualification_v1/run_current32_fp79_fable_v1.py](code/astra_oct18_small_stack_qualification_v1/run_current32_fp79_fable_v1.py)
- [code/astra_oct18_small_stack_qualification_v1/run_current32_m15_fable_v1.py](code/astra_oct18_small_stack_qualification_v1/run_current32_m15_fable_v1.py)
- [code/astra_oct18_small_stack_qualification_v1/run_current32_m15b_fable_v1.py](code/astra_oct18_small_stack_qualification_v1/run_current32_m15b_fable_v1.py)
- [code/astra_oct18_small_stack_qualification_v1/grade_partial_fable_v1.py](code/astra_oct18_small_stack_qualification_v1/grade_partial_fable_v1.py)
- [code/astra_crunch_physics_v1/guard_memory15_sci_v1.py](code/astra_crunch_physics_v1/guard_memory15_sci_v1.py)
- [code/astra_crunch_physics_v1/guard_memory50_sci79_v1.py](code/astra_crunch_physics_v1/guard_memory50_sci79_v1.py)
- [code/astra_oct18_bill_v1/run_guarded_product_census_v5_m15.py](code/astra_oct18_bill_v1/run_guarded_product_census_v5_m15.py)
- [code/astra_oct18_bill_v1/run_guarded_product_census_v5_fp79.py](code/astra_oct18_bill_v1/run_guarded_product_census_v5_fp79.py)
- [code/astra_crunch_physics_v1/guard_memory50_v1.py](code/astra_crunch_physics_v1/guard_memory50_v1.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [code/astra_oct18_accuracy_v1/run_shared32_continuation_v1.py](code/astra_oct18_accuracy_v1/run_shared32_continuation_v1.py)
- [code/astra_oct18_bill_v1/run_guarded_product_census_v4.py](code/astra_oct18_bill_v1/run_guarded_product_census_v4.py)
- [code/astra_oct18_bill_v1/run_guarded_product_census_v5.py](code/astra_oct18_bill_v1/run_guarded_product_census_v5.py)
- [code/astra_oct18_lead_v1/run_current_controls25_v2.py](code/astra_oct18_lead_v1/run_current_controls25_v2.py)
- [code/opus_fix336687_v1/candidate_v61j.py](code/opus_fix336687_v1/candidate_v61j.py)
- [code/opus_fix336687_v1/grade_store_v1.py](code/opus_fix336687_v1/grade_store_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5912](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5912)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5915](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5915)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5923](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5923)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5931](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5931)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5977](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5977)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5982](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5982)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-19-native-evaluation-uses-7-9-gb-caps-15-free) · [Copy provenance](../../PROVENANCE.json)

