# Protected-row reconstruction allocation: price-neutral, fresh-32 MSE +0.087%

2026-10-14T04:59:00Z · experiment · rejected · measured

A fixed change of which rows the reconstruction roster protects at the plane read (same count, three protected events) priced at essentially zero bill and made the fresh networks slightly worse, so it was closed.

Recorded measurement:

    native price MAX log bill 9.73388104558997e-06; fresh 32 (16 pairs): bill_log 0.0009733881045589973%, mse_log 0.0874343905771886%, net_log 0.0884077786817476% (SE 0.06703097715837002%); survivors none

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_reconstruction_row_allocation_v1/fragment_reconstruction_roster_UNVALIDATED_v1.py](code/lead/runs/astra_reconstruction_row_allocation_v1/fragment_reconstruction_roster_UNVALIDATED_v1.py)
- [code/lead/runs/astra_reconstruction_row_allocation_v1/finite_arow_price_v1.py](code/lead/runs/astra_reconstruction_row_allocation_v1/finite_arow_price_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py](code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py)
- [code/lead/runs/astra_reconstruction_row_allocation_v1/native_arow_price_judge_v1.py](code/lead/runs/astra_reconstruction_row_allocation_v1/native_arow_price_judge_v1.py)
- [code/lead/runs/astra_reconstruction_row_allocation_v1/owned_native_runner_v1.py](code/lead/runs/astra_reconstruction_row_allocation_v1/owned_native_runner_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2591](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2591)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2603](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2603)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2621](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2621)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2629](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2629)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2672](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2672)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-protected-row-reconstruction-allocation-price-neutral-fresh-32-mse) · [Copy provenance](../../../PROVENANCE.json)

