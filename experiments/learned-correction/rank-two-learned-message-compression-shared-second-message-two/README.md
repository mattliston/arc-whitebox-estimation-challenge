# Rank-two learned-message compression with a shared second message: two-network adjusted -0.29%

2026-10-14T18:39:00Z · experiment · inconclusive · measured

GNR2 compresses the E1, E2 and E3 messages of the learned correction at the current source to rank 2 with a shared second message (M2), keeping the exact odd-message products. The two-network smoke passed its gross gate and was kept as the diagnostic reference for the output-projection variant that replaced it.

Recorded measurement:

    geometric adjusted -0.293447831% (7100136 -0.057902268%, 7100137 -0.528438255%); full bill credit 1,617,680,640 FLOPs per network

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_gnr2_output_projection_v1/build_source_smoke_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/build_source_smoke_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_gnn_nonnegative_rank2_v1/finite_gnr2_smoke_HELD_v1.py](code/lead/runs/astra_gnn_nonnegative_rank2_v1/finite_gnr2_smoke_HELD_v1.py)
- [code/lead/runs/astra_gnn_nonnegative_rank2_v1/fragment_gnr2_UNVALIDATED_v1.py](code/lead/runs/astra_gnn_nonnegative_rank2_v1/fragment_gnr2_UNVALIDATED_v1.py)
- [code/lead/runs/astra_gnn_nonnegative_rank2_v1/matchedC_ON_UNVALIDATED_v1.py](code/lead/runs/astra_gnn_nonnegative_rank2_v1/matchedC_ON_UNVALIDATED_v1.py)
- [code/lead/runs/astra_gnn_nonnegative_rank2_v1/native_gnr2_probe_judge_v1.py](code/lead/runs/astra_gnn_nonnegative_rank2_v1/native_gnr2_probe_judge_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/check_kernel_cost_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/check_kernel_cost_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/finite_gop2_smoke_HELD_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/finite_gop2_smoke_HELD_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/fragment_gop2_UNVALIDATED_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/fragment_gop2_UNVALIDATED_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/kernel_HELD_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/kernel_HELD_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/matchedC_OFF_UNVALIDATED_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/matchedC_OFF_UNVALIDATED_v1.py)
- [code/lead/runs/astra_gnr2_output_projection_v1/matchedC_ON_UNVALIDATED_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/matchedC_ON_UNVALIDATED_v1.py)

<details>
<summary>Additional programs and dependencies (13)</summary>

- [code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py](code/lead/runs/astra_gnr2_output_projection_v1/native_gop2_probe_judge_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/finite_tch_qualification_HELD_v3.py](code/lead/runs/astra_terminal_central_hole_source_v2/finite_tch_qualification_HELD_v3.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/matched_c_tch_OFF_v2/candidate.py](code/lead/runs/astra_terminal_central_hole_source_v2/matched_c_tch_OFF_v2/candidate.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/matched_c_tch_v2/candidate.py](code/lead/runs/astra_terminal_central_hole_source_v2/matched_c_tch_v2/candidate.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/native_tch_qualification_judge_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/native_tch_qualification_judge_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/owned_native_runner_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py](code/lead/runs/astra_terminal_central_hole_source_v2/qualification_lifecycle_helpers_v1.py)
- [code/lead/runs/astra_terminal_central_hole_source_v2/terminal_central_hole_fragment_v2.py](code/lead/runs/astra_terminal_central_hole_source_v2/terminal_central_hole_fragment_v2.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/k4liab_s51k2tw/candidate.py](code/scout_native/k4liab_s51k2tw/candidate.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2972](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2972)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2974](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2974)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2980](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2980)

<details>
<summary>All result files and records (2)</summary>

- [ASTRA_SMOKE_OWNER_RETIREMENT_v1.json](ASTRA_SMOKE_OWNER_RETIREMENT_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Principal-component compression of learned contractions to rank two: bill -0.31%, fresh-16 adjusted -0.245%](../principal-component-compression-learned-contractions-rank-two-bill-0/README.md)
- enables: [Rank-two output projection of the learned correction: fresh-16 adjusted -0.577%, out-of-sample -0.560%](../rank-two-output-projection-learned-correction-fresh-16-adjusted/README.md)

[Research graph](../../../graph/index.html#node=2026-10-14-rank-two-learned-message-compression-shared-second-message-two) · [Copy provenance](../../../PROVENANCE.json)

