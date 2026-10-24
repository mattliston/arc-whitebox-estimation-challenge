# Depth-seven packed and square products combine to reduce arithmetic

2026-09-25T19:54Z · experiment · adopted · measured

Sharing the operand transformation across selected square and packed products adds savings beyond column packing alone. The combined client passes setup, parity, request-census, and memory checks, with a measured runtime cost.

Recorded measurement:

    - 2026-09-25 13:08 [engineer2] OPTION MEASURED, NOT A BUILD REQUEST: d7sq = depth 7 on the 1024^3 products (covariance C.w, newborn common pushes; depth7/d7sq_fragment.py 8b863bbd, composed after d7cp v2). In-process parity (s28d7/cand3sq_T.py c229e1c7): on top of d7cp v2, -1,132 MF on every net (-0.36..-0.39%). That beats the standalone price (18 x -51.7 MF) because the layer-2/3 newborn product now reuses the colpack's depth-7 A tree, which also wins back the 65 MF d7cp lost. d7cp v2 + d7sq vs S28: -0.846/-0.812/-0.817/-0.957% (mean -0.858%), final MSE ratios 1.0029/0.9968/0.9950/0.9989 (rounding level). Wall price of the d7sq step ~+3.7 s on the slowest net (bench rates, WALL v3), rate ~0.10%/s. Ready if the S29 wall ledger has room; its paired CPU would be one more pair (d7cp v2 vs d7cp v2 + d7sq).
    - 2026-09-25 12:54 lead: S29 ADDS d7sq (engineer2; depth7/d7sq_fragment.py 8b863bbd): the 18 per-net 1024^3 products (covariance C.w, newborn pushes) at depth 7, reusing d7cp's depth-7 A tree. With d7cp v2: -0.86% bill vs S28, rounding-level MSE (0.995-1.003 on 4 nets). Wall: about +7 s for both on the slowest net (S27 77.5 s, so about 85 s before the L10 cut's reduction). One paired server-CPU run (S28 vs the built S29) checks the projection against the 90 s rule; d7sq is dropped first if it fails. loopR7s epoch 1 recovers 0.21 pt of the schedule penalty (VALID16, MPS); the epoch 2 CPU check (~13:20) decides whether it goes into S29 (bar 0.2 pt).
    - 2026-09-25 13:10 [engineer2] S29 DEPTH PART CLIENT CENSUS PASS (lead 12:5x: the one gate S29 cannot cover): s28d7/cand3sq_T.py c229e1c7 (= porter's S29 depth composition, byte-identical) with the numpy-free client, nets 35/100: no errors; setup 3.39 s under load with all 465 keys fused (460 list + 5 depth-7) and 489 bound; bills equal parity (-0.846/-0.812%); requests +863 (35 cold) / +531 (100) vs the S28 base census (expected ~+540-600 from depth-7 op counts); server peak RSS 3.60 GB. Gates vs the lead's parity bounds for the bundle: per-layer max|diff|/RMS <= 2.0e-5 (single depth moves 1.0-1.35e-5), final-MSE per net +0.29/-0.32/-0.50/-0.11% (mean -0.16%; max 0.50% = at the bound). Verdict sent to the porter. S29 CPU tally line follows when my S28-vs-S29 paired server-CPU run lands.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer2_v1/depth7`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/d7sq_fragment.py](code/d7sq_fragment.py)
- [code/s28d7/cand3sq_T.py](code/s28d7/cand3sq_T.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1235](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1235)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1236](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1236)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1254](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1254)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-depth-seven-packed-square-products-combine-reduce-arithmetic) · [Copy provenance](../../../PROVENANCE.json)

