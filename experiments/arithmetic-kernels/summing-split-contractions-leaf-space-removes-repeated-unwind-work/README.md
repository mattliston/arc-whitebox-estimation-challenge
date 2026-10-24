# Summing split contractions in leaf space removes repeated unwind work

2026-09-25T09:44Z · experiment · adopted · measured

Chunk products accumulate in leaf storage before one unwind. Census-derived setup keys and guarded leaf-buffer binding are necessary to realize the saving without expensive lazy binding.

Recorded measurement:

      GO the S26 build: k-split fragment BEFORE E-73 (fused chunk keys via phase-2 prebind, preallocated buffers, unwind pair) + engineer2's group 5 un-shelved as the request offset.
      SETUP GUARD raised 3.0 -> 3.5 s in S26 only (platform limit 5 s; S24 platform max 2.88 s; +21 keys ~+0.1 s). Gate: local setup under load <= 3.6 s. The analyst's S25 attribution watcher is armed (R320 raw vs treuse/j3 bill, per-net R320 oracle).
    - 2026-09-25 03:05 [engineer] E-74 K-SPLIT BUILT (lead GO via E-72d): ksplit_fragment.py (v1 78e861f6) and ksplit_fragment_v2.py (b133785b) wrap engineer2's altbasis _packed_read5 (no edits; --d6x keeps reads at depth 5, so no overlap). Selection = S24's cost rule (bill + 750k flops/request) with fitted bill models -> 13 read shapes (11-15 splits/net). Setup: phases 1-2 = S24's minus 19 depth-5 keys only split reads used; phase 3 binds 38 chunk keys FUSED + 13 leaf buffers with leftover guard time. v2 sums chunk leaf products in leaf space, ONE unwind: product bench (OpenBLAS, 13 shapes) v1 -> v2: bill -2.54 -> -2.81 GF, requests +1,449 -> +897, backend x1.75 -> x1.68. PARITY v2 (S24+E-73 vs + k-split v2, nets 35/100/37/39): bill -0.72/-1.16/-0.76/-0.94%, final MSE ratio 1.0020/1.0021/0.9970/1.0023 (rounding-level; needs VALID40), layer-15 max diff ~1e-5 of RMS. Candidate candS24_ksplit2_treuse2_mem.py (9ae8ff11) = S24 + k-split v2 + treuse (LAST). Queued on the paired lock: OpenBLAS user-CPU pair, request/setup/handle census (12 nets), then residual pair.
    - 2026-09-25 03:05 lead: K-SPLIT v2 (engineer; chunks summed in leaf space, ONE unwind; ksplit_fragment_v2 b133785b; candS24_ksplit2_treuse2_mem.py 9ae8ff11): 13 read shapes (14 products/net). Bill -0.72..-1.16% (4 nets, ~-0.95%), +0.96k requests (~+6 ms, fits the rule alone), backend ~+6% (+1.5 s), wall ~+3.4 s. Final-MSE ratios 1.0020/1.0021/0.9970/1.0023 (rounding level, but one-sided: VALID40 decides, raw reported separately).
      S26 = S25-or-(S24+j3+E-73) + k-split v2 (before treuse) + group 5 OPTIONAL (before treuse). Extra v2-rule shapes (-0.1%, +0.35k requests) only if group 5 is in. Drop S24's 3 dead batched prebinds (porter) to buy setup time; setup guard 3.5 s.
    2026-09-25 03:13 porter: S26 CHAIN READY (fire_chain_S26.sh <BAR> <ksplit frag> <sha16> <group5 builder|none> <sha16|none> <phase3|none>): S24 + j3cut (build_j3cut on S24) + k-split v2 (b133785b) [+ group 5 built on that exact base] + key list re-derived on the S26 structure by the self-tested d6x rule (census ~6 min in-chain) + _PB_BATCHED () + _PB_WALL_S 3.5 + treuse LAST. Gates: einsum S16/S23/S24, validate, literals; treuse identity (e73_parity, setup, 35/100/37/39); smoke setup <= 3.6 s; VALID40 vs S24 rows (bar -0.6%, raw reported; free attribution vs the S25 j3+treuse rows); residual vs S24 incl. setup walls under concurrent load <= 3.6 s; staging fire_candS26. Open with the engineer: leave k-split phase-3 chunk keys (40 keys, computed after setup by ks_phase3_keys_v2.py) out of the re-derived list so phase 3 binds them fused as designed. Waiting for the k-split v2 gates + the lead's go.
    2026-09-25 03:41 porter: S26 will use k-split v4 (9e63ad1f; engineer: chunk keys come from the porter's census-derived _PB_SINGLE; phase 3 = 22 leaf buffers). ks_shape_check_v2.py (chunk-key coverage by _PB_SINGLE): with S24's old list on S24+j3+v4+treuse, net 35 = 3/46 chunk keys prebound (43 lazy) -> the in-chain re-derived list is what makes v4 pay. Launch on the engineer's v4 all-pass (setup under heavy load + residual pair).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/ksplit_fragment.py](code/fable_engineer_v1/ksplit_fragment.py)
- [code/fable_engineer_v1/ksplit_fragment_v2.py](code/fable_engineer_v1/ksplit_fragment_v2.py)
- [code/fable_engineer_v1/candS24_ksplit2_treuse2_mem.py](code/fable_engineer_v1/candS24_ksplit2_treuse2_mem.py)
- [code/fable_porter_v1/fire_chain_S26.sh](code/fable_porter_v1/fire_chain_S26.sh)
- [code/fable_porter_v1/ks_phase3_keys_v2.py](code/fable_porter_v1/ks_phase3_keys_v2.py)
- [code/fable_porter_v1/ks_shape_check_v2.py](code/fable_porter_v1/ks_shape_check_v2.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L970](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L970)
- [TEAM_LOG_2026-09-22_to_09-26.md#L971](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L971)
- [TEAM_LOG_2026-09-22_to_09-26.md#L976](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L976)
- [TEAM_LOG_2026-09-22_to_09-26.md#L977](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L977)
- [TEAM_LOG_2026-09-22_to_09-26.md#L978](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L978)
- [TEAM_LOG_2026-09-22_to_09-26.md#L984](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L984)
- [TEAM_LOG_2026-09-22_to_09-26.md#L999](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L999)
- [SUBMISSION_LEDGER.md#L885](../../../records/SUBMISSION_LEDGER.md#L885)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [SUBMISSION_LEDGER.md](../../../records/SUBMISSION_LEDGER.md) · record

</details>

## Related work

- shipped in: [Submission #332295: third birth-term row selection, split products, frame reuse, and learned refit](../../../submissions/332295/README.md)

[Research graph](../../../graph/index.html#node=2026-09-25-summing-split-contractions-leaf-space-removes-repeated-unwind-work) · [Copy provenance](../../../PROVENANCE.json)

