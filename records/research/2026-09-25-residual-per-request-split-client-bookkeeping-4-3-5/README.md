# Residual-per-request split: client bookkeeping 4.3-5.1 us and estimator Python 1.4-2.4 us

2026-09-25T05:29Z · finding · informational · measured

A boundary microbenchmark and estimator profile explain almost all residual time. Most cost belongs to the client library; remaining estimator reductions are bounded by measured loop and helper overhead.

Recorded measurement:

    2026-09-24 22:29 [engineer2] RESIDUAL FLOOR, part (a) measured (fable_engineer2_v1/resid/resid_floor_bench.py at the client/server boundary, tight single-op loops, GC off): flopscope's OWN residual per request (client code outside its timed dispatch span: span entry/exit bookkeeping + proxy finalizer) = 3.6-5.7 us (median ~5.0): add 5.1, add out= 5.0-5.4, x*y 5.7, x@y 5.3, x.T 4.6, x[1:3] 3.6-4.5, reshape 4.3-4.7, unstack4 5.2-5.7, empty 4.5-5.3; with the result kept alive 3.9 (finalizer ~1.1 us immediate, 0.38 us in bulk). The fnp attribute lookup is 0.04 us (pre-binding ops buys nothing). Local S21 residual (setup client, s1 gate base arms) = 6.6-7.0 us/request, ~= platform 7.3 us -> first estimate: (a) flopscope ~4.3-5.0 us/request (~65-70%), (b) our Python ~2.0-2.7 us/request (~65-90 ms/net; GC passes start inside dispatch spans per E-61 v6, so GC is not in residual). Per-function (b) split: setup-calling probe on S21s1 (resid/residual_probe_v6s.py) queued after the S22 chain. Also re-queued (22:28) the s1zero VALID40 + residual gate behind any fire_chain_S22* (my earlier yield check missed the renamed fire_chain_S22s1_v2.sh; the waiting runners were stopped before they held the lock).
    2026-09-24 23:51 [engineer2] RESIDUAL-PER-REQUEST SPLIT (lead item 2; S21s1 30aea66e, setup-calling client = engineer's residual_probe_v6 + setup(): fable_engineer2_v1/resid/residual_probe_v6s.py, residual6_S21s1_r35r100.json; probe explains 99%): total 6.49 / 6.74 us per request (r35 cold 221 ms / 34.0k; r100 226 ms / 33.6k; platform ~7.3). (a) flopscope's own client code outside its timed span incl. one proxy destruction = 4.3-5.1 us/request (micro-bench at the boundary: span bookkeeping 3.6-5.7 us, destruction 0.38 us in bulk / ~1.1 us immediate; frees ride inside dispatch; GC passes all start inside spans -> 0 ms GC residual). (b) our Python = 1.4-2.4 us/request, best ~2.2-2.4 (75-82 ms/net): tree kernel + dispatch 23-52 ms, per-layer helpers 10-18, _predict_binary loop 8-13. Top 3 (b) trims (estimates, nothing built): tree-kernel level loops with prebuilt per-level op tuples -6..-10 ms; _predict_binary + helpers (owners) -5..-8 ms; dispatch/rule (_rule_depth memo -1.5, _ipt_side hit path -1..-2, wrappers) -4..-6 ms => -15..-25 ms/net. Pairing: full routing +8.6k requests ~ +55 ms; (b) trim + E-71 at -15% requests (~-33 ms) ~ -48..-58 ms -> roughly funds full routing (-0.77% bill); E-71 alone funds depth-2 routing (-0.52%, +33 ms).
    - 2026-09-24 23:51 lead: RESIDUAL SPLIT (engineer2, S21s1, setup client; the probe explains 99% of residual): flopscope client 4.3-5.1 us/request (not ours; only fewer requests or proxies help); OUR Python 1.4-2.4 (best ~2.2-2.4) us/request = 75-82 ms/net. Trimmable: tree-kernel level loops -6..-10 ms, per-layer helpers -5..-8 ms, dispatch/rule -4..-6 ms; total -15..-25 ms. With E-71 (-15% requests = ~-33 ms) that roughly funds the full tree-routing (-0.77% bill, +55 ms); E-71 alone funds depth-2 routing (-0.52%, +33 ms). GO: bit-identical trims #1 + #3 (+ pure-lookup parts of #2) on S23; prodhash identity + residual gate vs S23.

## Programs

- [resid_floor_bench.py](resid_floor_bench.py)
- [residual_probe_v6s.py](residual_probe_v6s.py)
- [../2026-09-25-setup-smoke-incorrectly-rejected-designed-wall-guard-silently-stopped/fire_chain_S22s1_v2.sh](../2026-09-25-setup-smoke-incorrectly-rejected-designed-wall-guard-silently-stopped/fire_chain_S22s1_v2.sh)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L710](../../TEAM_LOG_2026-09-22_to_09-26.md#L710)
- [TEAM_LOG_2026-09-22_to_09-26.md#L772](../../TEAM_LOG_2026-09-22_to_09-26.md#L772)
- [TEAM_LOG_2026-09-22_to_09-26.md#L773](../../TEAM_LOG_2026-09-22_to_09-26.md#L773)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [residual6_S21s1_r35r100.json](residual6_S21s1_r35r100.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-residual-per-request-split-client-bookkeeping-4-3-5) · [Copy provenance](../../../PROVENANCE.json)

