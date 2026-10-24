# One-frame pair splitting saves arithmetic but propagated error offsets the gain

2026-09-25T20:15Z · experiment · rejected · measured

Restricting the pair-mean coordinates within one shared frame saves reconstruction and read work but leaves basis work intact. A nested joined representation saves more arithmetic and substantially worsens raw error, so it is rejected.

Recorded measurement:

    - 2026-09-25 13:15 [engineer2] SPLIT u/v OLD TIER, NATIVE BILL, STEP 1 (lead 13:0x; bill prototype, no accuracy claim; fable_engineer2_v1/uvsplit/uvp_fragment_v0.py c247f245 on S28's pre-treuse file + treuse; in process, setup first). Design = NESTED frame: one basis, pair means F_u kept in the frame's leading r_u = 256 coordinates, half differences F_v/2 in all of them, so the basis push and the trailing read stay at rank 384. (Two separate frames would push two bases per layer and read twice at the end: those products run at (256 + 384)/384 = 1.67x, about +0.9% of the bill.) Step 1 splits only the REBUILD (U F) and the READ ((L^2 lam) F^T), for the materialized frame and both lazy-L13 blocks. The joins are unchanged, except that the Nystrom rows now go in energy order, so the QR frame's leading coordinates carry the most energy.
      Bill vs S28: net 35 -1.343%, net 100 -1.613%. Net 100 breakdown: rebuild 33.5 -> 30.0 GF (-10.4% of the function, -1.18% of the bill); read 20.1 -> 18.8 GF (-6.5%, -0.44%); everything else unchanged. The ideal (256+384)/768 = -16.7% is not reached because the Winograd transforms do not shrink: the rebuild transforms U twice (U[:, :256] and U) and pays a pair recombination, and the read's k-heavy products keep their A-side transform, grow the output side, and add ~3 elementwise passes over rows x columns.
      Accuracy glance (no claim): final MSE +0.41% (35), +1.43% (100); per-layer max|diff|/RMS <= 5.6e-5. The u truncation here uses the energy-ordered frame, not u's own principal frame.
      STEP 2 now: split the JOIN (S27 census: joins ~6% of the bill): Q_u from the u columns' own Nystrom range (rank 256), then the v residual's 128 directions, Qn = [Q_u | Q_c]; Fj, core and overlap in u/v form. Step 2 also makes the u frame u-optimal.
    - 2026-09-25 13:20 [engineer2] SPLIT u/v OLD TIER, NATIVE BILL, STEP 2 + VERDICT (fable_engineer2_v1/uvsplit/; bill prototypes on S28's structure, nets 35/100, in process, setup first; the MSE glance is two nets vs the 32M truth, not a claim):
      | prototype | bill 35 / 100 | final MSE 35 / 100 | adjusted, these 2 nets |
      | v0: rebuild + read in u/v form, frame unchanged, u kept in the energy-ordered frame's leading 256 coordinates (uvp_fragment_v0.py c247f245) | -1.343 / -1.613% | +0.41 / +1.43% | -0.94 / -0.20% |
      | v1: v0 + joins in u/v form, nested frame [Q_u (256, from u's own Nystrom) , Q_c (rank - 256 = 128 or 64, v's residual)] (uvp_fragment_v1.py 9c2a93a8) | -2.294 / -2.733% | +5.83 / +6.76% | +3.40 / +3.84% |
      v1's joins save another -0.95..-1.12%, but v then keeps only 64-128 directions of its own, and the MSE jumps ~6%. That matches covhunter's point that v, not u, carries the tail. A nested frame starves v, so the join split only works if v keeps a full frame, which means TWO frames. Two frames push a 640-wide basis every layer and run the trailing read at 640: those products grow 1.67x (~+0.9% of the bill), and v's Nystrom sample goes back to 384 wide. Paper estimate for two frames: about -1% net (not built).
      VERDICT: the native bill of this lever is about -1.3..-1.6% for the accuracy-safe form (v0: rebuild + read only), not the model's -4.6%. The ideal (256 + 384)/768 scaling loses to the Winograd transforms: the rebuild's output transforms don't shrink and U is transformed twice, and the read's k-long products keep their input transform. It only nets anything if its raw cost is under ~1%. v0's two-net glance is +0.4/+1.4%, with a non-optimal u truncation; the u-optimal version (eigen-rotation of the u Gram in frame coordinates at each join, ~0.1-0.3% of the bill) is the one worth a torch or VALID screen, if any.
    - 2026-09-25 13:21 lead: SPLIT u/v NATIVE PROTOTYPES (engineer2, S28 structure, nets 35/100): v0 = rebuild + read split with the frame unchanged (u on the first 256): bill -1.34/-1.61%, MSE glance +0.4/+1.4%. v1 = + nested-frame joins: bill -2.29/-2.73% but MSE +5.8/+6.8% (v gets only 64-128 own directions): DEAD. Two separate frames: 640-wide push/read eat the join saving (about -1% net), not built. The ideal (256+384)/768 scaling fails because Winograd transforms don't shrink.
      NEXT: v0-ROTATED (each join's frame rotated so its leading 256 directions are the pair means' own principal directions; +0.1-0.3% bill). Engineer2 prototypes the native bill; covhunter screens raw on torch VALID40 against the unrotated u-at-256 reference. Bar: raw under +0.5%, giving about -0.7% adjusted or better -> native build for S30.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer2_v1/uvsplit`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/uvp_fragment_v0.py](code/uvp_fragment_v0.py)
- [code/uvp_fragment_v1.py](code/uvp_fragment_v1.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1264](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1264)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1265](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1265)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1266](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1266)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1267](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1267)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1276](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1276)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1277](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1277)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1278](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1278)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1279](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1279)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1280](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1280)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1281](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1281)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1282](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1282)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1283](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L1283)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-one-frame-pair-splitting-saves-arithmetic-but-propagated-error) · [Copy provenance](../../../PROVENANCE.json)

