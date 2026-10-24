# Increasing tree-cache caps reduces rebinding work while preserving predictions

2026-09-25T15:43Z · experiment · adopted · measured

Evictions caused a substantial fraction of lazy frame-binding work across worker-length sequences. Larger view caches reduce repeated binding; pricing must account for both ten- and twenty-network worker lifetimes.

Recorded measurement:

    - 2026-09-25 08:43 lead: NEW BILL LEVER (engineer E-75, e75_lazybind_census.py, 8 nets in one session): lazy tree-key binds inside predict (reshape billed at numel, 30-120 Mflop/key; 12-80 net-specific keys/net) cost 0.18-1.16% of each net's bill, mean 0.73%. Eviction rebinds (160/side LRU caps saturate ~net 4) grow: 0 -> 32 keys by net 8, costing 0.44%. A platform worker runs ~20 MLPs, so the steady state is likely higher.
      GO (a): caps 160 -> 320 (bit-identical) on S27's file; gates include a 20-net session steady state, the handle census (100k/worker cap), server peak RSS, and the residual pair; price 480 too.
      (b) Free binds (getitem views of setup-bound super views): paper feasibility with an honest CPU cost.
    - 2026-09-25 08:58 lead: CACHE CAP 320 early (engineer; S26-like candidate, 8 nets in worker order): BIT-IDENTICAL 8/8; bill 0/-.05/-.02/-.01/-.32/-.12/-.37/-.41% (nets 5-8 mean -0.31%, growing); requests -486..-819/net on nets 5-8. Occupancy at 320: 224-270/side after 8 nets, growing 10-20/net (saturates ~net 11-16), so 480 is priced too. Running on S27's file: 20-net bind census (320/480), 20-net handle + peak RSS census, residual pair (heavy steps held until S27's VALID40 starts). Fragments: capraise_fragment.py 1a30b7b3 (320), capraise480_fragment.py 431d4a5d. Router prep sent to the porter and engineer (both rank shapes in cache and prebind).
    - 2026-09-25 09:25 [engineer] E-75 LAZY TREE BINDS: binding a tree view inside predict is billed at numel (tree + its level-1 reshape; 30-120 Mflop per key). e75_lazybind_census(_v2).py, in process, one worker-like session: S27 over 20 nets pays 0.81% of the bill per net in lazy binds (0.19..1.24%), 39% of it eviction REBINDS once the 160-per-side tree caches saturate (~net 4; 30-68 rebinds/net later). CAP RAISE 160 -> 320 (capraise_fragment.py 1a30b7b3fc8c771c, one line, no functions) on stack_S27_mem.py: predictions hash-identical 20/20, bill mean -0.295%, nets 11-20 -0.389%, no net worse; rebinds 584 -> 28; caches full by net 20 (480 being priced). Handle/memory census v2 (owned bytes: views vs copies) and the residual pair (after the porter's S27 residual gate) pending; verdict -> fable_engineer_v1/e75_cap_verdict.txt (porter polls). FREE BINDS via getitem super views: NOT recommended (leave-one-net-out value ~0.1%/net; gapped trees cost +8..+59% on tree level ops).
      Cache cap 320 (engineer): bit-identical, bill -0.295% (nets 11-20 -0.389%), cached entries are views (0 GB owned). The platform runs about 20 nets per worker (5 setup calls per 100 MLPs), so the 20-net census is the right horizon. The verdict, including 480, goes to e75_cap_verdict.txt.
      NOTE for the engineer: this run used 10 workers (10 setups) = about 10 nets per worker, not 20. Price the cap raise on the first 10 nets as well as on 20.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/e75_lazybind_census.py](code/fable_engineer_v1/e75_lazybind_census.py)
- [code/fable_engineer_v1/capraise_fragment.py](code/fable_engineer_v1/capraise_fragment.py)
- [code/fable_engineer_v1/capraise480_fragment.py](code/fable_engineer_v1/capraise480_fragment.py)
- [code/fable_porter_v1/stack/stack_S27_mem.py](code/fable_porter_v1/stack/stack_S27_mem.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1026](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1026)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1027](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1027)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1028](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1028)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1036](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1036)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1052](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1052)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1063](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1063)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1077](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1077)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [e75_cap_verdict.txt](e75_cap_verdict.txt) · record

</details>

## Related work

- shipped in: [Submission #332361: middle join rank 320 with refitted learned correction and matched dials](../../../submissions/332361/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-increasing-tree-cache-caps-reduces-rebinding-work-while-preserving) · [Copy provenance](../../../PROVENANCE.json)

