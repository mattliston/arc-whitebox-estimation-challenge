# Remote request overhead is dominated by billing and transport, not small NumPy operations

2026-09-23T08:50Z · finding · informational · measured

A request microbenchmark and server profile separated operation time from billing, serialization and transport. Hidden deallocation trips also consumed server time.

Recorded measurement:

    2026-09-23 01:50 [engineer2] SERVER CPU PER REQUEST (fable_engineer2_v1/req_overhead.py, numpy-free client + OpenBLAS server, 5,000 tiny ops each, load 14-15): add on 8 floats (out= held or fresh held) 286-298 us of server CPU per request, getitem view 160 us, a discarded out= alias (op + hidden free trip) 404 us -> a free trip costs ~106 us. cProfile of the server (server_prof.py): the numpy call is ~22 us of it; the rest is flopscope's billing wrapper (op records, dtype-rate resolution, budget lock) plus ZMQ/msgpack. At the measured ~2.4x load inflation this is ~120 us per request on a quiet box, i.e. ~0.29 ms on the platform at its 2.4x factor. So one request costs ~0.51 ms of platform wall (0.22 ms fitted RTT + ~0.29 ms server CPU), not 0.22 ms; requests are ~18-20% of our local server CPU (S4 r35: 47.7k requests + 8.3k free trips). In-process numpy profiles miss this term: judge request-adding trades at ~0.5 ms each.

## Programs

- [req_overhead.py](req_overhead.py)
- [server_prof.py](server_prof.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L142](../../TEAM_LOG_2026-09-22_to_09-26.md#L142)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-remote-request-overhead-dominated-billing-transport-not-small-numpy) · [Copy provenance](../../../PROVENANCE.json)

