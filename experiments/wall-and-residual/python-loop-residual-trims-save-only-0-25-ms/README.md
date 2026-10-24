# Python-loop residual trims save only 0.25 ms; earlier 8-25 ms estimates were misattributed

2026-09-25T07:10Z · experiment · rejected · measured

Stub-operation microbenchmarks isolated estimator bytecode from client-library overhead. The proposed loop and memoization changes were too small to ship, and request reduction remained the useful lever.

Recorded measurement:

    - 2026-09-25 00:10 [engineer2] (b)-TRIMS MEASURED: about -0.25 ms/net, NOT WORTH SHIPPING (lead GO 23:51; fable_engineer2_v1/btrim/). Built S23_btrim_mem.py (6224621b = S23 + prebuilt op-tuple tree loops + _rule_depth memo for push/read + _ipt_fits memo; exactness traced in code, not gated). btrim_microbench.py runs S23 and btrim with identical stub ops, so only our bytecode differs: tree loops save 10 ns/op (0.5/0.7/0.9 us per L4/L5/L6 product, ~0.15 ms/net); rule memo 0.29-0.47 -> 0.09 us/call (~0.05 ms/net); fits memo 0.29 -> 0.09 us/call (~0.05 ms/net).
      MY EARLIER ESTIMATES WERE WRONG (23:51: -15..-25 ms; 00:0x: -8..-13 ms). They subtracted an assumed 4.3 us flopscope cost per request from profiler exclusive times. Our loop bytecode is ~0.06 us/op (the floor bench's python-only 0.03 us agrees), so the tree loops' exclusive time per request IS flopscope's cost in context: 4.52/4.54 us (r35/r100). Recomputed at that rate, the non-request remainder is 65-71 ms/net. It is led by proxy destruction when held results are released (_ipt_product_ab 16-17 ms, _cp_d6_shared ~3 ms, much of _predict_binary's 11-12 ms) and by profiler inflation of small nested functions (_rule_depth profiles 7-9 us/call but measures 0.3-0.5). Our real Python is roughly 10-20 ms/net; no single trim is worth more than ~1 ms.
      CONSEQUENCES: (1) residual room comes only from fewer requests and proxies (E-71). (2) Price the routing pairing on E-71 alone: its -5.0k requests (~-33 ms) fund depth-2 routing (+33 ms), not full routing (+55 ms). (3) Trim #1 is subsumed by the engineer's lvfuse (same three _ipt_* functions, op tuples in entry slot 3, plus fusion), so I dropped it and ceded _ipt_* to the engineer. (4) The #3 memos are exact but worth ~0.1 ms/net; not shipped.
    - 2026-09-25 00:10 lead: PYTHON-TRIM LEVER CLOSED (engineer2 micro-bench, stub ops): the (b) trims save ~0.25 ms/net, not -8..-25 ms. The earlier split subtracted an assumed 4.3 us/request from profiler times, but the profiled loop time is flopscope's own in-context cost (~4.5 us/request); our bytecode is ~0.06 us/op, and real Python is ~10-20 ms/net. Residual can only be bought with FEWER REQUESTS (E-71). E-71 groups 1-4 (~-5k) fund depth-2 routing (+33 ms), not full routing (+55 ms). Engineer2 next: price E-71 group 5 (body request diet), paper only.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer2_v1/btrim`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/S23_btrim_mem.py](code/S23_btrim_mem.py)
- [code/btrim_microbench.py](code/btrim_microbench.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L811](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L811)
- [TEAM_LOG_2026-09-22_to_09-26.md#L812](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L812)
- [TEAM_LOG_2026-09-22_to_09-26.md#L813](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L813)
- [TEAM_LOG_2026-09-22_to_09-26.md#L814](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L814)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-python-loop-residual-trims-save-only-0-25-ms) · [Copy provenance](../../../PROVENANCE.json)

