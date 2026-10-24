# Local server rejects the 1168900096-byte scratch arena under its default 100 MiB per-array cap

2026-09-07T02:36Z · incident · informational · measured

The first persistent NumPy-free client/server series for the response-row candidate failed before layer 0 because the standalone flopscope server's per-array guard defaults to 100 MiB (FLOPSCOPE_MAX_ARRAY_BYTES) while the bank allocates a 1,168,900,096-byte scratch arena; the organizers had raised the grader's guard to 4 GiB, so the retry restored FLOPSCOPE_MAX_ARRAY_BYTES=4294967296 for the local test server only, without relaxing the wall, residual or RSS gates.

Recorded measurement:

    Arena 1168900096 bytes vs default 100*1024*1024; grader per-array guard 4 GiB.

## Evidence

- [RESPONSE_ROWS_REMOTE_CONFIGURATION_V1.md#L3](RESPONSE_ROWS_REMOTE_CONFIGURATION_V1.md#L3)
- [RESPONSE_ROWS_REMOTE_CONFIGURATION_V1.md#L15](RESPONSE_ROWS_REMOTE_CONFIGURATION_V1.md#L15)

<details>
<summary>All result files and records (2)</summary>

- [RESPONSE_ROWS_REMOTE_CONFIGURATION_V1.md](RESPONSE_ROWS_REMOTE_CONFIGURATION_V1.md) · record
- [response_rows_current_runtime_openblas_series_v1_resources.json](response_rows_current_runtime_openblas_series_v1_resources.json) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-07-local-server-rejects-1168900096-byte-scratch-arena-under-default) · [Copy provenance](../../../PROVENANCE.json)

