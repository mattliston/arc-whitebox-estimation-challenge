# Maximum-shape grouped kernels cut work 2.30-4.43% but raise server compute 27-73%

2026-09-23T04:02Z · experiment · rejected · measured

Grouping kernels and binding maximum-shape views once reduced requests and metered work, including when combined with less frequent old-plane joins. Quiet measurements showed a server-compute increase that made the candidate unsuitable for the wall limit.

Recorded measurement:

    - 2026-09-22 21:02 [engineer] v9 = grouped kernel + once-bound maximum-shape views (free per-shape binding) + held out= aliases (engineer2's free-trip finding): BIT-EXACT r35/r100, util r35 0.17994 (-4.43%), r100 0.18036 (-2.30%); v10 = v9 + push64 (zero-pad 1024-row pushes to /64 -> depth 6): util -4.91%/-3.69%, MSE +0.59%/+0.41% (rounding-level). select() rounding 64/128 is net negative. Boundary round-trip census next (machine at load ~25).
    - 2026-09-22 21:17 [engineer] stacks on engineer2 cand_e2_v7: +v9 util r35 0.17884 (-5.0%) / r100 0.17926 (-2.9%), bit-identical to bank; +v10 0.17795 (-5.5%) / 0.17669 (-4.3%), = bank+push64 predictions (MSE +0.59%/+0.41%). Boundary round-trip census running.
    - 2026-09-22 21:55 [engineer] boundary round trips (census_rt, 3 nets): bank 95.4k/84.1k/79.4k -> v9 67.6k/52.4k/42.6k (-29..-46%) -> e2+v10 60.5k/45.4k/35.1k (-37..-56%). v9 local kernel +5% (squares) to +54% (D6 push k=384); v9d (--dense) +5..+25%, util -3.0%/-2.1% bit-exact. Walls unresolvable at load 15-35: needs one quiet-box timing before fire. RESULT.md updated.
    - 2026-09-22 21:37 [engineer] (timestamp correction: my lines stamped 21:02/21:17/21:55 were written at about 20:55/21:10/21:30 machine time.) candA = bank -> build_e2 -> build_physics -> build_eng.py --grouped --read5 is byte-identical to engineer2's chain/c4_final_phys.py (v9 kernel: grouped requests, held out= aliases, views bound once). A2 = same + --push64 (fable_engineer_v1/candA2_v10_on_c3.py). Side-by-side cold+warm census (requests, free trips, server CPU s) bank / A / A2 running now.
    - 2026-09-22 21:45 lead: CANDIDATE A (candA_v9_e2_k31r.py = engineer v9 + e2 + K31 row .85/.5) VALID40 vs cached bank: adj 0.93727 (-6.27%), 40/40, raw 0.98605, util .17952 vs .18886 (porter). Panel running; memory (engineer2) + quiet-box wall pending. CANDIDATE B = A + join schedule every3 (L7/10/13): torch VALID40 raw pooled 0.9789 (-2.1%), 37/40 (even: 0.9859, 33/40); files fable_fire_v1/candB_A_every3.py, candB_c4_every3.py; porter queues B's VALID40 after A's panel.
    - 2026-09-22 21:52 [engineer] side-by-side census (load 9-13; r35 cold / r100 / r35 repeat): round trips bank 95.4k/84.1k/79.4k, A (v9 stack) 57.5k/41.9k/32.1k, A2 (v10) 57.8k/42.7k/32.5k; server CPU s bank 69/61/63, A 74/76/83, A2 78/86/78; client CPU bank 9.0/7.3/6.8, A 4.8/4.9/4.9; residual bank .52/.42/.39, A .34/.34/.35; local wall bank 81/71/71, A 81/82/90, A2 85/92/83. A trades -38..-47k round trips for +5..+20 s server compute: platform wall roughly a wash, not clearly inside 110 s. Censusing A-dense (--dense) next.
    2026-09-22 22:05 porter: candA gates: parity bit-exact, validate OK, VALID40 adj 0.93727 40/40, panel8 0.9358 / virgin6 0.9300 vs bank -- but WALL NOT PROVEN: paired census at matched load: round trips -37..-56%, server CPU +27..+34%, boundary wall +20..+26% vs bank (bank ~110 s worst); platform est. 110-130 s on the cold net. Details fable_porter_v1/STACKED_BASE.md.
    - 2026-09-22 22:07 lead: QUIET CENSUS (fable_fire_v1/quiet_*_census.json): candA server CPU +27-73% vs bank (66/59/60 s vs 38/44/48 s; r35 cold, r100, r35 warm), round trips -37..-56%, RSS 5.9 vs 5.6 GB; c4 same CPU, RSS 6.8 GB -> A/B (engineer v9 kernels) HELD for wall; engineer finding the compute. TONIGHT'S FIRE CANDIDATE S2 = e2v7 + K31 row + gain b-fix gamma 0.1 (physicist partial strength) + join every3: native r35 util -1.34%, MSE -2.16% (adj -3.5%); VALID40 running (fable_fire_v1/paired_valid40_candS2.json).
    - 2026-09-22 22:32 [engineer] 3rd side-by-side census (load 7-11): server CPU s bank 49.9/47.1/46.5, A 62.3/51.7/76.2, A2 58.4/57.7/64.2; engineer2's memfinal (e2+v9) vs membank 90.9/79.3 vs 68.9/65.2 -> the v9 kernel likely costs +15-30% server compute (3 of 4 pairs), offset by -38..-47k round trips: platform wall effect UNCERTAIN. Fallback built: --holdonly = the bank's own kernel with held out= aliases (bit-exact, util identical to bank, no compute change); census H vs A interleaved running. Server live-array peak (engineer2 run_mem): bank 2.06 GB, v9 alone 2.16 GB, e2+v9 2.73 GB.
    - 2026-09-22 22:37 lead: S2 PANEL: panel8 adj 3.660e-9 vs bank 3.863e-9 (0.9475), virgin6 3.002e-9 vs 3.145e-9 (0.9545) -> PASS. whest validate PASS. S2 waits only on engineer2's handle-growth run (100k array cap). B (A+every3) VALID40 adj 0.90925 (-9.08%), 40/40, below A on 39/40 (porter) -> held for v9 wall.
    2026-09-22 22:37 porter: candB_A_every3 VALID40 vs cached bank: adj 0.90925, raw 0.96413, 40/40 wins, util 0.178116; below candA on 39/40 nets (pooled B/A 0.9701). validate OK (both B files); r35 candB_A == candB_c4 (flops, ops, MSE). Panel running; census held for the quiet window.
    - 2026-09-22 22:45 [engineer] interleaved H/A census pairs (full stack): round trips H 59.4k/48.6k/43.1k vs A 57.5k/41.9k/32.1k vs bank 95.4k/84.1k/79.4k; server CPU s H 70.0/58.7/66.9 & 60.9/46.4/49.2 vs A 100.5/94.9/91.8 & 72.7/70.5/73.9 (+12..+36 s). RECOMMEND H (fable_engineer_v1/candH_holdonly_on_c3.py = e2+physics+hold-only kernel, bit-exact, zero bill/compute change) for the fire; A's -2.3..-4.4% bill costs more platform wall than its extra round trips save. Porter told.
    2026-09-22 22:45 porter: candB_A_every3 PANEL: panel8 0.9120 / virgin6 0.9161 vs bank (A was 0.9358/0.9300). All gates pass except the census (held for the lead's quiet window). Porter idle.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/build_eng.py](code/fable_engineer_v1/build_eng.py)
- [code/fable_engineer2_v1/chain/c4_final_phys.py](code/fable_engineer2_v1/chain/c4_final_phys.py)
- [code/fable_engineer_v1/candA2_v10_on_c3.py](code/fable_engineer_v1/candA2_v10_on_c3.py)
- [code/fable_fire_v1/candA_v9_e2_k31r.py](code/fable_fire_v1/candA_v9_e2_k31r.py)
- [code/fable_fire_v1/candB_A_every3.py](code/fable_fire_v1/candB_A_every3.py)
- [code/fable_fire_v1/candB_c4_every3.py](code/fable_fire_v1/candB_c4_every3.py)
- [code/fable_engineer_v1/candH_holdonly_on_c3.py](code/fable_engineer_v1/candH_holdonly_on_c3.py)
- [code/fable_engineer_v1/batch_fragment.py](code/fable_engineer_v1/batch_fragment.py)
- [code/fable_engineer_v1/bigarena_fragment.py](code/fable_engineer_v1/bigarena_fragment.py)
- [code/fable_engineer_v1/bindhold_fragment.py](code/fable_engineer_v1/bindhold_fragment.py)
- [code/fable_engineer_v1/build_f32.py](code/fable_engineer_v1/build_f32.py)
- [code/fable_engineer_v1/caps_fragment.py](code/fable_engineer_v1/caps_fragment.py)

<details>
<summary>Additional programs and dependencies (26)</summary>

- [code/fable_engineer_v1/caps_fragment_v3.py](code/fable_engineer_v1/caps_fragment_v3.py)
- [code/fable_engineer_v1/coldbind_fragment.py](code/fable_engineer_v1/coldbind_fragment.py)
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

- [TEAM_LOG_2026-09-22_to_09-26.md#L37](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L37)
- [TEAM_LOG_2026-09-22_to_09-26.md#L44](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L44)
- [TEAM_LOG_2026-09-22_to_09-26.md#L50](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L50)
- [TEAM_LOG_2026-09-22_to_09-26.md#L51](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L51)
- [TEAM_LOG_2026-09-22_to_09-26.md#L54](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L54)
- [TEAM_LOG_2026-09-22_to_09-26.md#L55](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L55)
- [TEAM_LOG_2026-09-22_to_09-26.md#L58](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L58)
- [TEAM_LOG_2026-09-22_to_09-26.md#L60](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L60)
- [TEAM_LOG_2026-09-22_to_09-26.md#L66](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L66)
- [TEAM_LOG_2026-09-22_to_09-26.md#L68](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L68)
- [TEAM_LOG_2026-09-22_to_09-26.md#L69](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L69)
- [TEAM_LOG_2026-09-22_to_09-26.md#L70](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L70)
- [TEAM_LOG_2026-09-22_to_09-26.md#L71](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L71)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [STACKED_BASE.md](STACKED_BASE.md) · record
- [paired_valid40_candS2.json](paired_valid40_candS2.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-maximum-shape-grouped-kernels-cut-work-2-30-4) · [Copy provenance](../../../PROVENANCE.json)

