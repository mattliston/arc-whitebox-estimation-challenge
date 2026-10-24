# Handle-free round trips are charged as overhead rather than participant residual

2026-10-16T03:25:00Z · finding · rejected · measured

A client-source and boundary regression audit found that coalescing frees would not reduce the binding residual meter. No free-batching patch was built.

Recorded measurement:

    140rows~59.7Kresidualcycles/request;free-tripcoefficient~-11K;corrreq/free-.07;overheadsaving~.1-.2s/network

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3781](../../TEAM_LOG_2026-10-01_to_10-22.md#L3781)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [FREES_RESULT_v1.json](FREES_RESULT_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-handle-free-round-trips-charged-overhead-rather-than-participant) · [Copy provenance](../../../PROVENANCE.json)

