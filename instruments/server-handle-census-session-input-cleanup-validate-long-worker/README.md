# Server handle census and in-session input cleanup validate long-worker resource use

2026-09-26 · tool · informational · derived

The handle server instruments ArrayStore.put/free/clear without retaining arrays and enforces the handle limit. The corrected client deletes only harness-owned remote input/output references before an in-session summary flushes frees, preserving estimator state and billing history.

Recorded measurement:

    Corrected20-network test: maximum6,782,844,928 bytes group RSS,58,249 server handles,zero swap; all16-layer predictions on35/100 match native outputs; all14 shared cases retain identical bills.

## Programs

- [code/handle_server_v1.py](code/handle_server_v1.py)
- [code/handle_client_v2.py](code/handle_client_v2.py)
- [code/run_handle_worker_v3.py](code/run_handle_worker_v3.py)

## Evidence

- [RESULTS.md#L560](../../experiments/third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L560)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../experiments/third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [both_tail_worker_qualification_v6.json](both_tail_worker_qualification_v6.json) · result

</details>

[Research graph](../../graph/index.html#node=2026-09-26-server-handle-census-session-input-cleanup-validate-long-worker) · [Copy provenance](../../PROVENANCE.json)

