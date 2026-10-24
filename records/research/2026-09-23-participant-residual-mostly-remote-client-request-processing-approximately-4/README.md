# Participant residual is mostly remote-client request processing, approximately 4 us per request

2026-09-23T10:58Z · finding · informational · measured

Client profiling attributed most residual time to encoding, remote-array creation and decoding rather than estimator arithmetic. This made request counts a useful safety budget.

Recorded measurement:

    2026-09-23 03:58 [engineer2] WHAT THE 0.4 s RESIDUAL IS MADE OF (fable_engineer2_v1/client_prof.py: one S5e r35 prediction under cProfile in the numpy-free client): the candidate's own Python is ~0.09 s even with profiler inflation (top: _bh_forward/_bh_backward/_exact_view bookkeeping); the rest is the flopscope CLIENT's per-request Python (arg encoding, msgpack walk, RemoteArray creation, response decode, dispatch checks), i.e. residual scales with round trips at ~4 us each (bank 95k round trips -> 0.37-0.46 s locally and 0.385 s worst on the platform; S4/S5e 56-58k -> 0.24-0.31 s). Headroom to the 0.4 s zero-prediction limit: ~30-35k extra round trips per prediction. Any request-adding trade (e.g. last-level copy elimination, +30k) is bounded by this as well as by the wall.

## Programs

- [client_prof.py](client_prof.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L179](../../TEAM_LOG_2026-09-22_to_09-26.md#L179)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-participant-residual-mostly-remote-client-request-processing-approximately-4) · [Copy provenance](../../../PROVENANCE.json)

