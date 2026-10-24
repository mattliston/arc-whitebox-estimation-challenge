# Rank-64 carried covariance, encoder residual and common-content compression audits found no new viable cut

2026-10-18T15:08:30Z · experiment · rejected · measured

Saved-state audits rejected rank-64 low-rank-plus-diagonal covariance fidelity and encoder residual restoration. A proposed common-content reduction duplicated already absorbed changes; energy pruning of a full rank-384 frame provided negligible or inconsistent benefit versus a direct rank-320 cut.

Recorded measurement:

    Current common rank 256 / difference rank 384; full-384 energy pruning to 320: saved error / direct 320 = .99315/1.03320; exact-map ceiling .06146% bill.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct18_bill_v1/post935_compute_v1/cov_lowrank_diag_audit_v1/probe_rank64_v1.py](code/astra_oct18_bill_v1/post935_compute_v1/cov_lowrank_diag_audit_v1/probe_rank64_v1.py)
- [code/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py](code/astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py)
- [code/astra_crunch_runtime_v1/failure336631_v1/server310_openblas_v1/lib/python3.10/site-packages/flopscope/numpy/linalg/_decompositions.py](code/astra_crunch_runtime_v1/failure336631_v1/server310_openblas_v1/lib/python3.10/site-packages/flopscope/numpy/linalg/_decompositions.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5700](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5700)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5702](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5702)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5704](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5704)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5712](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5712)

<details>
<summary>All result files and records (8)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [RANK64_CLOSURE_v1.json](RANK64_CLOSURE_v1.json) · result
- [CLOSURE_v1.json](CLOSURE_v1.json) · result
- [CLOSURE_v1-d3351fde.json](CLOSURE_v1-d3351fde.json) · result
- [RESULT_v1.json](RESULT_v1.json) · result
- [RESULT_v1-1aaca992.json](RESULT_v1-1aaca992.json) · result
- [RANK64_RESULT_v1.json](RANK64_RESULT_v1.json) · result
- [RANK64_GUARD_v1.json](RANK64_GUARD_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-rank-64-carried-covariance-encoder-residual-common-content-compression) · [Copy provenance](../../../PROVENANCE.json)

