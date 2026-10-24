# Platform timing drift and cold-network residual growth require paired telemetry

2026-09-23T11:10Z · finding · informational · measured

The worst wall excursion came from a slower early part of the run rather than a deterministic cold compute cost. Residual time nevertheless rose systematically and tracked server overhead.

Recorded measurement:

    2026-09-23 04:15 [engineer2] #331992 (S5e) PLATFORM TELEMETRY vs #331961 (S4) vs #331947 (S2), 100 MLPs, mean/median/max s: wall S2 95.7/95.6/105.6, S4 92.5/92.4/101.1, S5e 92.4/93.2/108.0; kernel 85.5/85.3/94.7, 83.7/83.8/91.7, 83.2/84.1/96.1; overhead 9.9/9.8/11.4, 8.6/8.5/10.2, 8.9/8.9/11.6; residual 0.26/0.25/0.30, 0.25/0.25/0.30, 0.27/0.27/0.34; util .1805/.1785/.1741; score 3.339e-9/3.262e-9/3.176e-9. Paired per MLP S5e-S4: wall -0.17 s mean, overhead +0.31 s, residual +18 ms mean (+44 max), flops -2.46%. RISK SITS ON THE COLD FIRST MLP PER WORKER (indices 0-4): S5e wall mean 97.9 max 108.0 s (S4 same MLPs 94.4/97.7), residual mean 0.323 max 0.343 s (S4 0.296/0.302); the other 95: wall max 102.2 (S4 101.1), residual max 0.306 (S4 0.278). Cold MLPs bind every new push5/row-cut frame and start concurrently on all 5 workers. S6 estimate: worst cold MLP ~110-111 s wall, residual ~0.35 s (~50 ms from the 0.4 s zero-prediction cliff).
    - 2026-09-23 04:10 [engineer] S5e WORST MLP (108.0 s, MLP 4) = PLATFORM NOISE, not a cold cost. Telemetry: S5e's run started slow (mean wall vs S4 by MLP block: 0-9 +2.9 s, 10-19 +1.7, 40-99 -0.4..-2.7; request-overhead wall 10.1 s early vs 8.6 s late with identical requests), heavy MLP 39 unchanged (100.9 vs 101.1), per-MLP S5e-S4 sd 4.4 s (S4-S2: 2.0 s). Local QUIET paired cold test (fired S4 vs fired S5e, fresh process per net, box fully quiet): server CPU r35 31.24 vs 31.05 s, r100 29.74 vs 30.03 s (equal); requests +1,756/+2,744 (~+0.9-1.4 s platform), residual +6-8 ms. Extra requests: push5 deeper/packetized products +654, new frames +286, f32 casts +70, row cut +14. Systematic S5e worst ~102.5 s; push5p6 + a +13 s noise excursion would reach ~119 s.
    2026-09-23 04:32 [engineer2] S5e vs S4 PLATFORM, per-MLP (same 100 MLPs, index order): wall S5e-S4 by block 0-4 +3.5 s (sd 6.5), 5-9 +2.3, 10-19 +1.7, 20-39 +0.8, 40-69 -1.4, 70-99 -1.2; the 5 cold diffs -2.8/-4.3/+4.7/+6.5/+13.4 s (t = 1.09): not a cold-MLP cost but an early-run drift (server overhead for identical requests +0.68 s early -> +0.2 s late), agreeing with engineer 04:10. Residual rose SYSTEMATICALLY: +17.6 ms mean on warm MLPs (93/95 up), +27 ms on cold; top-10 residuals = indices 4,3,0,2,8,1,5,19,13,39 (the early ones); p95/p99/max S5e 304/334/343 ms vs S4 278/300/302; across warm MLPs residual correlates +0.98 with server overhead wall (client Python slows with the platform). Rule for S6: platform residual deltas ~1.8x the local paired delta, worst on the first MLPs of a slow start.

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L185](../../TEAM_LOG_2026-09-22_to_09-26.md#L185)
- [TEAM_LOG_2026-09-22_to_09-26.md#L186](../../TEAM_LOG_2026-09-22_to_09-26.md#L186)
- [TEAM_LOG_2026-09-22_to_09-26.md#L191](../../TEAM_LOG_2026-09-22_to_09-26.md#L191)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-platform-timing-drift-cold-network-residual-growth-require-paired) · [Copy provenance](../../../PROVENANCE.json)

