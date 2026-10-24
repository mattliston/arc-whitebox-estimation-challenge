# Rank-two output projection of the learned correction: fresh-16 adjusted -0.577%, out-of-sample -0.560%

2026-10-14T20:47:12Z · experiment · adopted · measured

GOP2 replaces the full output of the learned correction by an output-optimal rank-2 projection with the full input mean retained, cutting about half a percent of bill at a small accuracy cost. It passed the two-network smoke, a 16-network fresh screen, the owner's shortened 8-network out-of-sample confirmation and the 8-seed seal, and was fired as #336101.

Recorded measurement:

    two-network smoke vs control: adjusted -0.214790% / -0.696525%, paired -0.455949% (bill -0.515440%, MSE +0.059800%); vs GNR2 paired MSE -0.261096%, adjusted -0.162979%; credit 1,359,758,640 FLOPs each; fresh 16: geometric adjusted -0.577068% (log net SE 0.083636%, 15/16 lower); OOS 8 -0.560143%, pooled 24 -0.571426%; current-best fixed 2 bill -0.5302% / -0.5067%; sealed 8 adjusted -0.644465% (SE log 0.141959%)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_gnr2_output_projection_v1/finite_gop2_science_HELD_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/finite_gop2_science_HELD_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/finite_gop2_smoke_HELD_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/finite_gop2_smoke_HELD_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/matchedC_ON_UNVALIDATED_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/matchedC_ON_UNVALIDATED_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_science_judge_HELD_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_science_judge_HELD_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/finite_tch_qualification_HELD_v3.py](code/lead/runs/astra_terminal_central_hole_source_v2/finite_tch_qualification_HELD_v3.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/matched_c_tch_OFF_v2/candidate.py](code/lead/runs/astra_terminal_central_hole_source_v2/matched_c_tch_OFF_v2/candidate.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/matched_c_tch_v2/candidate.py](code/lead/runs/astra_terminal_central_hole_source_v2/matched_c_tch_v2/candidate.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/native_tch_qualification_judge_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/native_tch_qualification_judge_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [code/lead/runs/astra_terminal_central_hole_source_v2/terminal_central_hole_fragment_v2.py](code/lead/runs/astra_terminal_central_hole_source_v2/terminal_central_hole_fragment_v2.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/k4liab_s51k2tw/candidate.py](code/scout_native/k4liab_s51k2tw/candidate.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3003](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3003)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3011](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3011)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3014](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3014)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3016](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3016)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3022](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3022)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3031](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3031)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3053](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3053)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3065](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3065)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3093](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3093)

<details>
<summary>All result files and records (5)</summary>

- [ASTRA_FRESH_RETIREMENT_v1.json](ASTRA_FRESH_RETIREMENT_v1.json) · record
- [ASTRA_OOS24_RETIREMENT_v1.json](ASTRA_OOS24_RETIREMENT_v1.json) · record
- [DECISION_initial8_physics_v2.json](DECISION_initial8_physics_v2.json) · result
- [CENSUS_COMPLETE_HELD_v1.json](CENSUS_COMPLETE_HELD_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Rank-two learned-message compression with a shared second message: two-network adjusted -0.29%](../rank-two-learned-message-compression-shared-second-message-two/README.md)
- shipped in: [Submission #336101: rank-two learned output projection with planner request repair, public score 1.50514e-9](../../../submissions/336101/README.md)

[Research graph](../../../graph/index.html#node=2026-10-14-rank-two-output-projection-learned-correction-fresh-16-adjusted) · [Copy provenance](../../../PROVENANCE.json)

