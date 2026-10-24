# Bulk and threshold cache eviction fail residual constraints despite bounded handles

2026-09-23T09:49Z · experiment · rejected · measured

Bulk finalizers and least-recently-used bookkeeping added tens of milliseconds of participant residual time. Per-network eviction of unused push frames passed the twenty-network gate and was selected instead.

Recorded measurement:

    - 2026-09-23 02:49 [engineer] HANDLE CAP SAFETY NET: --lru (evict_fragment_lru.py) = threshold LRU over all frame caches (above 60k cached arrays at a prediction start, drop least-recently-used entries down to 40k; walk gated above 1,200 tracked keys). S5-like with --evict5 --lru = identical to --evict5 in 4 nets (34,467 handles); a 25k/18k test copy triggers at r37: 27,734 handles after 4 nets, predictions bit-identical to the non-evicting build on all 4 nets, re-bind cost +0.12-0.13% on trigger nets (expected ~0.01% average at 60k/40k). Recommended S5 kernel flags: -- --holdelim --bigarena --depthrule depthrule_fragment_push5.py --f32 third --evict5 --lru. File for engineer2's 20-net test: fable_engineer_v1/candStack_l15_push5_f32_ev_lru.py.
    2026-09-23 02:52 porter: S5f (stack_S5f_mem.py, sha c20f7903...) = S5e + --lru: MSE and flops bit-identical to S5e on r35/r100 (one process), validate OK; sha to engineer2 for the 20-net handle test.
    2026-09-23 02:58 [engineer2] HANDLE VALVE TEST (S5 draft + clear-all valve at 20k): predictions bit-exact and handles drop at each clear, but _BH_F (forward frames) keeps 8-12k of the cleared arrays alive, and a clear costs ~+50 ms participant residual per ~20k arrays (RemoteArray finalizers) against the 0.4 s limit (S4 platform residual 0.24-0.30 s). Recommendation to engineer: per-cache ENTRY caps in the existing put helpers (gradual FIFO eviction), sized for ~60k arrays; no clear-all valve. For S5 as built: 87.5k after 20 distinct nets (S4 77.1k passed the platform) -> fire-able at ~20 MLPs per worker with a 12.5% margin.
    2026-09-23 03:15 [engineer2] S5e / S5f 20-NET HANDLE VERDICT (same 20 distinct nets, both runs concurrent so residuals are paired): S5e (--evict5) 34,484 (4) / 60,855 (12) / 74,132 (20), max 74,274 -- below S4's 77,140 from net 12 on, no bill or residual cost -> PASS, recommended for the fire. S5f (--evict5 --lru): LRU fires at net 13 (60,855 -> 36,774), 58,271 after 20, max 60,969, but its bookkeeping adds participant residual on every net once the caches pass ~1,200 keys: +20..+38 ms on nets 6-12 (0.298-0.323 s vs 0.274-0.285 s), +60 ms on the trigger net (0.336 s), ~+20 ms after; and re-binding after the eviction costs +0.2..+0.7% bill per later net (+0.68% on the trigger net). Residual limit 0.4 s (zero-prediction fallback); S4 platform residual 0.24-0.30 s. Live peak both 2,386-2,404 MB. Logs fable_engineer2_v1/h20S5e.out, h20S5f.out.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/evict_fragment_lru.py](code/fable_engineer_v1/evict_fragment_lru.py)
- [code/fable_engineer_v1/depthrule_fragment_push5.py](code/fable_engineer_v1/depthrule_fragment_push5.py)
- [code/fable_engineer_v1/candStack_l15_push5_f32_ev_lru.py](code/fable_engineer_v1/candStack_l15_push5_f32_ev_lru.py)
- [code/fable_porter_v1/stack/stack_S5f_mem.py](code/fable_porter_v1/stack/stack_S5f_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L163](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L163)
- [TEAM_LOG_2026-09-22_to_09-26.md#L164](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L164)
- [TEAM_LOG_2026-09-22_to_09-26.md#L166](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L166)
- [TEAM_LOG_2026-09-22_to_09-26.md#L169](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L169)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-bulk-threshold-cache-eviction-fail-residual-constraints-despite-bounded) · [Copy provenance](../../../PROVENANCE.json)

