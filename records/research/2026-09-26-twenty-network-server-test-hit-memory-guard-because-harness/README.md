# Twenty-network server test hit the memory guard because harness frees occurred outside an active budget

2026-09-26 · incident · failed · derived

Two resource runs stopped before completion, initially suggesting retained estimator state. Metadata and weak-reference audits instead identified old input-weight buffers whose frees were silently ignored outside an active budget; deleting harness-owned references inside the budget resolved the accumulation.

Recorded measurement:

    Initial version:18 completed networks, group RSS7,500,054,528 bytes,68,002 handles; bounded-cache version:14 completed, handles near58k; old input buffers67,108,864 bytes with16 views; tiny witness both arms charge3072 per operation, deletion after close leaves6 buffers/0 frees, deletion in session0 buffers/6 frees.

## Programs

- [../../../instruments/server-handle-census-session-input-cleanup-validate-long-worker/code/handle_client_v2.py](../../../instruments/server-handle-census-session-input-cleanup-validate-long-worker/code/handle_client_v2.py)

## Evidence

- [RESULTS.md#L544](../../../experiments/third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L544)
- [RESULTS.md#L550](../../../experiments/third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L550)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../../experiments/third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [result.json](result.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-26-twenty-network-server-test-hit-memory-guard-because-harness) · [Copy provenance](../../../PROVENANCE.json)

