# Frame-handle census finds single-use views dominate cache growth

2026-09-23T09:14Z · tool · adopted · measured

A twenty-network census isolated additional cached frames from new push layouts and final-layer row selection. Evicting unused five-level frames reduced handle growth while preserving predictions.

Recorded measurement:

    - 2026-09-23 02:14 lead: S5 FIRE BLOCKED on the server handle store (100k per worker connection). engineer2: the S5 draft's handles grow ~9% faster than S4 (push5 layouts cache ~750 more views per new net): 36,074 vs 33,082 after 4 nets, straight-line cap at ~18-19 distinct nets vs ~20 MLPs per platform worker (S4 ~21). A 20-distinct-net test of S4 / S5 draft / push5p6 is running; the engineer is checking whether push5's layouts can reuse existing frames. No fire before the verdict.
    2026-09-23 02:42 [engineer2] HANDLES OVER 20 DISTINCT NETS in one connection (35,100,37,39,42,48,59,63,66,67,79,93,36,38,40,41,43,44,45,46; run_mem_caches.sh = census + per-cache count of client-held remote arrays): S5 draft 16,836 / 36,074 (4) / 69,831 (12) / 87,495 (20), last 8 nets +2.2k per net; S4 (fired, passed the platform at ~20 MLPs per worker) 15,518 / 33,082 / 61,877 / 77,140, +1.9k per net. S5/S4 = 1.13. Margin to the 100k store cap after 20 nets: S4 23%, S5 12.5%; linear extrapolation hits 100k at ~26 nets (S5) vs ~32 (S4). Who holds them (S5 after 20): _PACKED_FRAME_CACHE 18.5k (254 entries), _BH_O 17.2k (251), _B_FRAME_CACHE 16.7k (421), _READ5_CHILD_CACHE 14.7k (508), _READ5_FRAME_CACHE 9.1k, _A_FRAME_CACHE 7.5k, _BH_X 3.1k; body caches ~250 (constant). The entry caps (8,192 / 4,096) never bind: ~69 arrays per _BH_O entry means the caps allow ~500k handles. push5 adds +5.4k in _BH_O, +2.4k _A_FRAME, +2.2k _B_FRAME by net 20. Live peak flat 2,638-2,679 MB over all 20 (S4 2,951-3,062).
    - 2026-09-23 02:41 [engineer] HANDLE STORE (S5 blocker): my in-process cached-array census (fable_engineer_v1/handle_census.py) reproduces engineer2's server counts (S4-like 33,065 vs 33,082; S5-like l15cut+push5+f32 36,056 vs 36,074 after r35/r100/r37/r39). Excess = single-use frames: push5 +1,166 (five-level frames keyed by per-layer active counts), L15 row cut ~+1,300 (m = |R| changes per net); 69% of all cached frame arrays serve one net (reuse_census.py). Per-net eviction flags for build_eng.py (hook _w5_port_new; predictions unchanged; _predict_binary not redefined): --evict5 (five-level push frames unused by the previous net) S5-like 34,467 handles, util identical except r39 +0.03%; --evict5r (+ frames used only inside the row cut) 33,239 (= S4 +0.5%), util +0.07/+0.14/+0.03% r100/r37/r39; --evictall (all layouts) 26,389 (S4-like 23,873), util +0.12/+0.30% r37/r39. Recommended A (--evict5) for S5; files candStack_l15_push5_f32_{ev,ev5r,evall}.py sent to engineer2's 20-net test.
    - 2026-09-23 02:54 [engineer] HANDLES on the porter's FULL stacks (fable_engineer_v1/handle_census.py; cached arrays = server handles after each net, one process, r35/r100/r37/r39): S4 stack_S4_joinv2 15,501 / 23,590 / 26,268 / 33,065; S5 draft stack_S5_mem 16,818 / 25,595 / 28,405 / 36,056; S5e stack_S5e_mem (--evict5) 16,818 / 25,595 / 27,198 / 34,467 (= S4 +4.2%, S5 draft -1,589; eviction starts at net 3). S5e predictions bit-identical to S5_mem on all 4 nets; util identical except r39 +0.03% (one re-bound frame). engineer2's 20-net test on stack_S5e_mem.py decides; stack_S5f_mem (--evict5 --lru, 60k/40k valve) is the hard-bounded fallback.

## Programs

- [code/fable_engineer2_v1/run_mem_caches.sh](code/fable_engineer2_v1/run_mem_caches.sh)
- [code/fable_engineer_v1/handle_census.py](code/fable_engineer_v1/handle_census.py)
- [code/fable_engineer_v1/reuse_census.py](code/fable_engineer_v1/reuse_census.py)
- [code/fable_engineer_v1/build_eng.py](code/fable_engineer_v1/build_eng.py)
- [code/fable_porter_v1/stack/stack_S5e_mem.py](code/fable_porter_v1/stack/stack_S5e_mem.py)
- [code/fable_engineer_v1/batch_fragment.py](code/fable_engineer_v1/batch_fragment.py)
- [code/fable_engineer_v1/bigarena_fragment.py](code/fable_engineer_v1/bigarena_fragment.py)
- [code/fable_engineer_v1/bindhold_fragment.py](code/fable_engineer_v1/bindhold_fragment.py)
- [code/fable_engineer_v1/build_f32.py](code/fable_engineer_v1/build_f32.py)
- [code/fable_engineer_v1/caps_fragment.py](code/fable_engineer_v1/caps_fragment.py)
- [code/fable_engineer_v1/caps_fragment_v3.py](code/fable_engineer_v1/caps_fragment_v3.py)
- [code/fable_engineer_v1/coldbind_fragment.py](code/fable_engineer_v1/coldbind_fragment.py)

<details>
<summary>Additional programs and dependencies (24)</summary>

- [code/fable_engineer_v1/evict_fragment_all.py](code/fable_engineer_v1/evict_fragment_all.py)
- [code/fable_engineer_v1/evict_fragment_lru.py](code/fable_engineer_v1/evict_fragment_lru.py)
- [code/fable_engineer_v1/evict_fragment_push5.py](code/fable_engineer_v1/evict_fragment_push5.py)
- [code/fable_engineer_v1/evict_fragment_push5r.py](code/fable_engineer_v1/evict_fragment_push5r.py)
- [code/fable_engineer_v1/grouped_fragment_v5.py](code/fable_engineer_v1/grouped_fragment_v5.py)
- [code/fable_engineer_v1/grouped_fragment_v7.py](code/fable_engineer_v1/grouped_fragment_v7.py)
- [code/fable_engineer_v1/grouped_fragment_v7b.py](code/fable_engineer_v1/grouped_fragment_v7b.py)
- [code/fable_engineer_v1/grouped_fragment_v7c.py](code/fable_engineer_v1/grouped_fragment_v7c.py)
- [code/fable_engineer_v1/grouped_fragment_v9.py](code/fable_engineer_v1/grouped_fragment_v9.py)
- [code/fable_engineer_v1/grouped_fragment_v9d.py](code/fable_engineer_v1/grouped_fragment_v9d.py)
- [code/fable_engineer_v1/grouped_read5_fragment_v5.py](code/fable_engineer_v1/grouped_read5_fragment_v5.py)
- [code/fable_engineer_v1/grouped_read5_fragment_v7.py](code/fable_engineer_v1/grouped_read5_fragment_v7.py)
- [code/fable_engineer_v1/grouped_read5_fragment_v7b.py](code/fable_engineer_v1/grouped_read5_fragment_v7b.py)
- [code/fable_engineer_v1/grouped_read5_fragment_v7c.py](code/fable_engineer_v1/grouped_read5_fragment_v7c.py)
- [code/fable_engineer_v1/grouped_read5_fragment_v9.py](code/fable_engineer_v1/grouped_read5_fragment_v9.py)
- [code/fable_engineer_v1/holdelim_fragment.py](code/fable_engineer_v1/holdelim_fragment.py)
- [code/fable_engineer_v1/holdonly_fragment.py](code/fable_engineer_v1/holdonly_fragment.py)
- [code/fable_engineer_v1/lean_fragment_v3.py](code/fable_engineer_v1/lean_fragment_v3.py)
- [code/fable_engineer_v1/push64_fragment.py](code/fable_engineer_v1/push64_fragment.py)
- [code/fable_engineer_v1/select_round_fragment.py](code/fable_engineer_v1/select_round_fragment.py)
- [code/fable_engineer_v1/setupbind_fragment_v3.py](code/fable_engineer_v1/setupbind_fragment_v3.py)
- [code/fable_engineer_v1/sq6_fragment.py](code/fable_engineer_v1/sq6_fragment.py)
- [code/fable_engineer_v1/symout_fragment_v3.py](code/fable_engineer_v1/symout_fragment_v3.py)
- [code/fable_engineer_v1/usedslots_fragment.py](code/fable_engineer_v1/usedslots_fragment.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L153](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L153)
- [TEAM_LOG_2026-09-22_to_09-26.md#L160](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L160)
- [TEAM_LOG_2026-09-22_to_09-26.md#L161](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L161)
- [TEAM_LOG_2026-09-22_to_09-26.md#L165](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L165)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #331992: final third-moment row selection, five-level products, and memory reductions](../../submissions/331992/README.md)

[Research graph](../../graph/index.html#node=2026-09-23-frame-handle-census-finds-single-use-views-dominate-cache) · [Copy provenance](../../PROVENANCE.json)

