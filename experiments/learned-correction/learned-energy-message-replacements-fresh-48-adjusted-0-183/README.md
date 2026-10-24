# Learned energy-message replacements: fresh-48 adjusted -0.183%/-0.340%, native delivery parity fails

2026-10-14T01:38:21Z · experiment · rejected · measured

Two cheaper 'energy' evaluations of the learned correction network (the record's GNN), arms E3 and E23, were priced natively and screened on 48 fresh networks in Torch, where both survived with 16 of 16 lower. The formal native delivery gate (matched-C parity between the native port and the Torch model on two fixed networks) failed for both on both seeds, so the out-of-sample pool could not promote them.

Recorded measurement:

    native price (worst-seed log bill): E3 -0.180897496%, E23 -0.362397788%; fresh 48 net log: E3 -0.183116661% (SE 0.002811934), E23 -0.340277995% (SE 0.025168887), both 16/16 lower net; native 7-call parity: formal delivery FAIL for both arms on both fixed seeds

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_gnn_energy_v1/build_boundary_protocol_v1.py](code/lead/runs/astra_gnn_energy_v1/build_boundary_protocol_v1.py)
- [code/lead/runs/astra_gnn_energy_v1/finite_energy_resume_UNVALIDATED_v2.py](code/lead/runs/astra_gnn_energy_v1/finite_energy_resume_UNVALIDATED_v2.py)
- [code/lead/runs/astra_gnn_energy_v1/finite_energy_science_UNVALIDATED_v2.py](code/lead/runs/astra_gnn_energy_v1/finite_energy_science_UNVALIDATED_v2.py)
- [code/lead/runs/astra_gnn_energy_v1/finite_energy_smoke_UNVALIDATED_v3.py](code/lead/runs/astra_gnn_energy_v1/finite_energy_smoke_UNVALIDATED_v3.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2430](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2430)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2459](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2459)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2489](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2489)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2510](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2510)

<details>
<summary>All result files and records (3)</summary>

- [BOUNDARY_SOURCE_READY_v1.json](BOUNDARY_SOURCE_READY_v1.json) · record
- [ALLOCATION_AUDIT_v1.json](ALLOCATION_AUDIT_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-learned-energy-message-replacements-fresh-48-adjusted-0-183) · [Copy provenance](../../../PROVENANCE.json)

