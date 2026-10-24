# Restoring omitted inherited-core trace had negligible and inconsistent signed-read benefit

2026-10-18T12:37:58Z · experiment · rejected · measured

Saved current states tested restoring the full inherited core trace rather than its attenuated value. The read-error ratios were essentially unchanged, closing a standalone trace correction without native scoring.

Recorded measurement:

    Inherited-trace omission 9–11%; full/attenuated read error 1.000067/1.000641.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct18_lead_v1/probe_core_trace_v1.py](code/astra_oct18_lead_v1/probe_core_trace_v1.py)
- [code/astra_oct18_shape_cost_v1/core_trace_proxy_v1/probe_v1.py](code/astra_oct18_shape_cost_v1/core_trace_proxy_v1/probe_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5667](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5667)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5716](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5716)

<details>
<summary>All result files and records (6)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SEARCH_STATE_v19.json](SEARCH_STATE_v19.json) · record
- [NYSTROM_GRAM_CEILING_v1.json](NYSTROM_GRAM_CEILING_v1.json) · result
- [ACTIVE_REDUCED_OUTPUT_AUDIT_v1.md](ACTIVE_REDUCED_OUTPUT_AUDIT_v1.md) · record
- [RESULT_v1.json](RESULT_v1.json) · result
- [QCORE_REUSE_CORRECTION_RECEIPT_v1.md](QCORE_REUSE_CORRECTION_RECEIPT_v1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-restoring-omitted-inherited-core-trace-had-negligible-inconsistent-signed) · [Copy provenance](../../../PROVENANCE.json)

