# Covariance-direction proposals v1/v2 crashed (SIGTRAP in the MPS allocator at a 4.5 GB cap); v4 completed at 8 GB

2026-09-20 · incident · failed · measured

The first proposal process terminated with SIGTRAP after 10.0917 s in a libdispatch same-queue sync trap while the MPS allocator released cached buffers inside an indexing kernel, before any direction was saved; version 2 (cache emptied after each step) logged all 16 forward steps but no completed backward; version 3 failed source admission on a seal self-reference; version 4 completed under an 8 GB device allocation cap with per-layer cache cleanup (largest sampled completed-call driver allocation 4.349 GB).

Recorded measurement:

    v1 SIGTRAP after 10.0917 s, RSS peak 1.028 GB; v2 16 forward steps, no backward; v4 sampled driver allocation 4.349 GB, cap 8 GB

## Evidence

- [RUNTIME_FAILURE.md#L3](RUNTIME_FAILURE.md#L3)
- [RUNTIME_FAILURE.md#L9](RUNTIME_FAILURE.md#L9)
- [RUNTIME_FAILURE.md#L10](RUNTIME_FAILURE.md#L10)
- [RESULT.md#L48](RESULT.md#L48)
- [RESULT.md#L49](RESULT.md#L49)
- [RESULT.md#L50](RESULT.md#L50)
- [RESULT.md#L52](RESULT.md#L52)

<details>
<summary>All result files and records (5)</summary>

- [RUNTIME_FAILURE.md](RUNTIME_FAILURE.md) · record
- [closeout.json](closeout.json) · record
- [closeout-fac55028.json](closeout-fac55028.json) · record
- [closeout-a17e7f56.json](closeout-a17e7f56.json) · record
- [RESULT.md](RESULT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-covariance-direction-proposals-v1-v2-crashed-sigtrap-mps-allocator) · [Copy provenance](../../../PROVENANCE.json)

