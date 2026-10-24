# Estimator-body request reductions: bit-exact predictions and 0.58% less metered work

2026-09-23T02:07Z · experiment · adopted · measured

Rewriting the pair map, carried-vector update and learned closure reduced requests and work. Retaining discarded remote-array aliases also removed hidden deallocation round trips.

Recorded measurement:

    2026-09-22 19:07 [engineer2] v1 (pair map + _bd_update + learned closure rewrites) BIT-EXACT vs bank on r35/r100, bill -0.26%; boundary requests r35 cold 58433->55292, r100 warm 50045->46650; server RSS 3.86->4.35 GB. Next: live-loop edits (v2), then read path.
    2026-09-22 20:02 [engineer2] DONE (RESULT.md): body diet cand_e2_v7.py = bank + build_e2.py (9 patches), BIT-EXACT r35/r100, bill -0.58%; boundary r35 cold/r100/r35-repeat requests 58433/50045/45745 -> 53766/45073/40773 (about -4.8k); FOUND hidden 'free' round trips (client frees GC'd handles in an extra ZMQ trip before the next request): bank 36.9k per cold net, 19.7k of them in _packed_product_cached because every out= op returns a discarded alias; holding aliases per call is bit-exact and zero-cost, recipe sent to engineer (~-25k round trips per net expected). Wall/residual unresolvable at load avg ~15.
    2026-09-22 22:08 [engineer2] FINAL CHAIN (fable_engineer2_v1/chain/): bank -> build_e2.py (v8, +AST temp holds) -> build_eng.py --grouped --read5 = c2_final.py, bit-exact r35/r100 all layers; + porter physics 0.85,0.5 = c4_final_phys.py bit-exact vs porter's bank-based physics (r35 MSE 1.6237239631e-08, util .178843). Round trips (requests+hidden frees) r35 cold/r100: bank 95,366/84,143 -> e2 v8 85,121/73,531 -> chain 57,395/41,748. A/B at load~10: e2 v8 beats bank on wall/server CPU/client CPU/residual; engineer kernels add ~15-24% server CPU (flagged). Server live-array peak: bank 2.06 GB, e2 2.63, chain 2.73 GB.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer2_v1/build_e2.py](code/fable_engineer2_v1/build_e2.py)
- [code/fable_engineer2_v1/cand_e2_v7.py](code/fable_engineer2_v1/cand_e2_v7.py)
- [code/fable_engineer_v1/build_eng.py](code/fable_engineer_v1/build_eng.py)
- [code/fable_engineer2_v1/chain/c2_final.py](code/fable_engineer2_v1/chain/c2_final.py)
- [code/fable_engineer2_v1/chain/c4_final_phys.py](code/fable_engineer2_v1/chain/c4_final_phys.py)
- [code/fable_engineer2_v1/hold2.py](code/fable_engineer2_v1/hold2.py)
- [code/fable_engineer2_v1/patches/bd_new.py](code/fable_engineer2_v1/patches/bd_new.py)
- [code/fable_engineer2_v1/patches/edge_new.py](code/fable_engineer2_v1/patches/edge_new.py)
- [code/fable_engineer2_v1/patches/features_new.py](code/fable_engineer2_v1/patches/features_new.py)
- [code/fable_engineer2_v1/patches/final_features_new.py](code/fable_engineer2_v1/patches/final_features_new.py)
- [code/fable_engineer2_v1/patches/fuse_edits.py](code/fable_engineer2_v1/patches/fuse_edits.py)
- [code/fable_engineer2_v1/patches/gnn_new.py](code/fable_engineer2_v1/patches/gnn_new.py)

<details>
<summary>Additional programs and dependencies (36)</summary>

- [code/fable_engineer2_v1/patches/loop_edits.py](code/fable_engineer2_v1/patches/loop_edits.py)
- [code/fable_engineer2_v1/patches/omitted_new.py](code/fable_engineer2_v1/patches/omitted_new.py)
- [code/fable_engineer2_v1/patches/pair_new.py](code/fable_engineer2_v1/patches/pair_new.py)
- [code/fable_engineer2_v1/patches/share_edits.py](code/fable_engineer2_v1/patches/share_edits.py)
- [code/fable_engineer2_v1/patches/vec_edits.py](code/fable_engineer2_v1/patches/vec_edits.py)
- [code/fable_engineer_v1/batch_fragment.py](code/fable_engineer_v1/batch_fragment.py)
- [code/fable_engineer_v1/bigarena_fragment.py](code/fable_engineer_v1/bigarena_fragment.py)
- [code/fable_engineer_v1/bindhold_fragment.py](code/fable_engineer_v1/bindhold_fragment.py)
- [code/fable_engineer_v1/build_f32.py](code/fable_engineer_v1/build_f32.py)
- [code/fable_engineer_v1/caps_fragment.py](code/fable_engineer_v1/caps_fragment.py)
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

- [TEAM_LOG_2026-09-22_to_09-26.md#L12](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L12)
- [TEAM_LOG_2026-09-22_to_09-26.md#L29](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L29)
- [TEAM_LOG_2026-09-22_to_09-26.md#L62](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L62)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #331947: rowwise (3,1) correction, gain correction, and joins every third layer](../../../submissions/331947/README.md)

[Research graph](../../../graph/index.html#node=2026-09-23-estimator-body-request-reductions-bit-exact-predictions-0-58) · [Copy provenance](../../../PROVENANCE.json)

