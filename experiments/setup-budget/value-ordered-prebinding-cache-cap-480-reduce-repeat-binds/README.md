# Value-ordered prebinding and cache cap 480 reduce repeat binds while respecting setup limits

2026-09-25T18:35Z · experiment · adopted · measured

After eliminating eviction rebinding, most remaining lazy keys were network-specific. A bounded value-per-time setup list and larger view cap were integrated into the next candidate.

Recorded measurement:

    - 2026-09-25 11:35 [engineer2] LAZY-BIND RESIDUAL PRICED (lead GO; no new run: the engineer's 20-net S27+cap480 session e75v2_lazybind_s27c480.json is the held-out set, and the porter's ipt_key_census_S26w.json nets 300..580 are the selection census).
      Residual lazy binds after cap480 (no rebinds): 0.595% per net over nets 1-10, 0.496% over nets 1-20.
      - 72% (0.358%) comes from keys NEVER used in the 8 selection nets: rare, net-specific (rows = live count, cols = column count) shapes.
      - 28% (0.138%) comes from keys used by exactly 1 of 8 (252 candidate keys not in S27's 372-key _PB_SINGLE).
      Bind bill model: 1.076 x tree numel (2.7% median error).
      (a) Value-ordered list, 3.5 s guard, about +42 keys at about 8 ms/key: capture -0.094% per net (10-net) / -0.061% (20-net).
      (a)+(b) With a 4.0 s guard, about +104 keys: -0.149% / -0.106%.
      All 252 would need about +2 s of setup (over 4.0 s): -0.198% / -0.138%.
      VERDICT: below the 0.2% bar at both horizons. It is bit-identical and free, so it is worth folding into the porter's next key-list re-derivation as a rule change (value per ms, per-side limit tied to the 480 cap) if the setup gate holds, but it is not worth a stand-alone build.
    - 2026-09-25 11:36 lead: (1) k-split does NOT fall back at rank 320 (engineer e76 probe: identical splits at every layer; the rank-320 reads save only -0.19%). The L13 gap is the lazy joins (engineer2), as logged.
      (2) CAP CHOICE = 480 (engineer; capraise480_fragment.py 431d4a5d): horizon-average bill -0.258% vs 320's -0.248%; 0 late rebinds; 55.8k handles at net 20; 0 GB owned. In S28.
      (3) LAZY-BIND RESIDUAL (engineer2, priced from existing data): after cap 480, 0.595%/net over 10 nets; 72% comes from net-specific keys no list can catch. Value-ordered key list + 4.0 s guard = -0.149% (10-net) / -0.106% (20-net). Below the 0.2% stand-alone bar, so it is FOLDED into S28's key-list rule (porter): order by bill saved per ms, per-side limit 480, guard 4.0 s (platform limit 5 s; S27's slowest platform setup 3.16 s). S28's setup gate becomes: at most 4.1 s under concurrent load, all intended keys bound.
    - 2026-09-25 11:40 [engineer] E-75 CAP VERDICT FINAL (e75_cap_verdict.txt): CAP CHOICE 480 (capraise480_fragment.py 431d4a5dc8bac7a2) - both 320/480 PASS every gate on stack_S27_mem.py (20 nets hash-identical; residual cold +0.005/+0.001 s; handles at net 20 51,155 / 55,826 < 70k; cached arrays own 0 GB); bill nets 1-10 -0.201% both, nets 1-20 -0.295 / -0.314%, horizon average -0.248 / -0.258%; 480 has 0 rebinds on nets 13-20. E-76 RANK-320 GAP: the k-split does NOT fall back at rank 320 (identical splits every layer, net 100). The L13 cut's -0.726% sits at _shared_old384_mm L13-15; the older cohort's 11776-column products keep rank 384 and do not shrink (17.6 GF = 5.8% of the net), so the -1.7% prediction (all old-tier x 64/384) overshoots; re-ranking the older cohort at the join is a method question. Small kernel item: some rank-320 products lose Winograd depth (dim/32-indexed tables; ~0.05%). E-74e v5w (ksplit_fragment_v5w.py 57656b02336ec2c7) passes parity: bill -1.147% vs v4w -1.084% (S24+E-73 base, 4 nets), +125 requests, MSE ratio 1.0010; offered to replace v4w in S28.
    2026-09-25 11:42 porter: S28 key-list rule (engineer2, lead 11:5x) implemented: build_pb_keylist_v4.py (candidates >= 1 of 8 nets; value = freq x 1.076 x 7^L x (rows/2^L)(cols/2^L) / 8 ms; per side 480; list capped at 500 = 4.0 s at 8 ms/key; on the S26w census 624 candidates, the 500-prefix holds 98.2% of the value vs 57.8% for S24's old list). fire_chain_S28.sh: v4 rule, _PB_WALL_S = 4.0 override before treuse, setup gates 4.1 s (smoke + concurrent), all intended keys bound in the smoke.
    2026-09-25 11:43 porter: S28 inputs updated: k-split v5w (57656b02; has _KS_OLD_PB_BIND/_KS_ACC_MN, ksfirst-compatible; its 3.5 guard overridden to 4.0), key list --max-total 460 (lead), cap 480, schedule default S27t 384/320/384 (analyst torch held-out -1.368%, pool-80 -1.034%).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/capraise480_fragment.py](code/fable_engineer_v1/capraise480_fragment.py)
- [code/fable_porter_v1/stack/stack_S27_mem.py](code/fable_porter_v1/stack/stack_S27_mem.py)
- [code/fable_engineer_v1/ksplit_fragment_v5w.py](code/fable_engineer_v1/ksplit_fragment_v5w.py)
- [code/fable_porter_v1/build_pb_keylist_v4.py](code/fable_porter_v1/build_pb_keylist_v4.py)
- [code/fable_porter_v1/fire_chain_S28.sh](code/fable_porter_v1/fire_chain_S28.sh)
- [code/fable_fire_v1/fire_candS24/candS24_S23_loopR3_lvf2_lrnys_mem.py](code/fable_fire_v1/fire_candS24/candS24_S23_loopR3_lvf2_lrnys_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1139](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1139)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1140](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1140)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1141](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1141)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1142](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1142)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1143](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1143)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1144](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1144)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1145](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1145)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1146](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1146)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1147](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1147)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1148](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1148)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1149](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1149)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1150](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1150)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1173](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1173)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1178](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1178)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1179](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1179)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [e75v2_lazybind_s27c480.json](e75v2_lazybind_s27c480.json) · result
- [ipt_key_census_S26w.json](ipt_key_census_S26w.json) · result
- [e75_cap_verdict.txt](../increasing-tree-cache-caps-reduces-rebinding-work-while-preserving/e75_cap_verdict.txt) · record

</details>

## Related work

- shipped in: [Submission #332361: middle join rank 320 with refitted learned correction and matched dials](../../../submissions/332361/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-value-ordered-prebinding-cache-cap-480-reduce-repeat-binds) · [Copy provenance](../../../PROVENANCE.json)

