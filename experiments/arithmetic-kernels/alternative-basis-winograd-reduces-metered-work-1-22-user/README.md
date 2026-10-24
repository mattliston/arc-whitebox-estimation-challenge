# Alternative-basis Winograd reduces metered work 1.22% and user CPU 3.29%

2026-09-24T21:30Z · experiment · adopted · measured

Changing internal operand and output bases reduces transform arithmetic across single, batched and shared-operand tree products. The selected variant avoids the extra request cost of an independently constructed basis.

Recorded measurement:

    2026-09-24 14:30 [engineer2] ALTERNATIVE-BASIS WINOGRAD (Karstadt-Schwartz) PAPER VERDICT on S13's tree kernel (fable_engineer2_v1/altbasis/: ab_census.py = in-process census of every tree product side (depth, dims, P) on S13 r35/r100; ab_price.py). Basis (derived, 12 adds/level): A' = (A11, A12, S2, S3) so the other left factors cost 3 adds (S4 = A12'-A21', A22 = A21'+A22', S1 = A21'+A11'), same for B; output decoded in (C11, C22-C12, C22-C21, C22) with 6 adds; basis changes 2 adds/level per operand. S13 per net: 657 tree sides (196-197 per side at depth 5, 6-13 at 4, 6-9 at 6, 2 at 3); tree transform adds today 25.2 Gflop = 6.9% of the bill (87% of all add/subtract). (a) core 12 adds vs 15: -4.83 Gflop (-1.30% of the bill); (b) basis changes (levels 2..L on the 4 plain chunks per input, inverse on the output, +1 add at the output's level 1): +1.98 Gflop (+0.53%); NET -2.85 Gflop = -0.77% of the bill per net (A -0.82, B -1.14, C -0.89). (c) fusion: the input change at level 1 is FREE inside the tree build (2 copies + 2 basis adds replace 2 of today's 3 copies + ... same billed volume); levels 2..L need one in-place pass per level on the 4 plain chunks (strided) before the level-1 sums; the output's level-1 inverse fuses into the last unwind (8 adds vs 7), levels 2..L need in-place passes on the output. (d) CPU: add+subtract = 45% of S13's local backend (~14 s of ~31 s user per net), tree share 87% -> -11.3% volume = about -1.4 s user/net (-4.4%) if time follows volume (strided basis passes may cost more per element); ops +2,821/net. Platform (project_wall_v3 on S13, worst-MLP kernel 60.5 s): kernel -1.6..-2.5 s, ops +0.85..+1.41 s -> expected worst about -0.9 s central (+0.0 .. -1.7). Not bit-exact (different sums; KS report Winograd-like stability): needs parity + VALID40. Interaction: 12 adds/level make deeper levels relatively cheaper, so S14's depth pricing should be redone if both ship. Recommendation: prototype on _ipt_product (single products) with an in-situ shadow measurement (bill, user CPU, ops, rounding) like read6; then batched/W5/colpack paths.
    2026-09-24 14:39 [engineer2] ALT-BASIS SHADOW PROTOTYPE PASSES the lead's gate (fable_engineer2_v1/altbasis/: ab_fragment.py = new functions only, reusing the engineer's _ipt_side trees; ab_shadow.py in situ on fired S13 2a5dd324, r35 cold / r100 warm, 1 thread; each single tree product run native and alt-basis on the same operands). Found a CHEAPER basis than priced: A's a21 slot holds S1 = a21 + a22, B's b12 slot holds T3 = b22 - b12 (1 add per level each, not 2), output decoded into (C11, C12, D = m5 + m4, C22) with 6 adds, inverse C21 = C22 - D once per level on the C tree's seven level-1 products; op counts equal to the tree kernel's; certified vs float64 A@B at depths 1-6 (ab_certificate.py). In situ, warm r100, 180 single products: bill 226.98 -> 223.82 Gflop = -3.16 Gflop = -0.86% of the net's bill; user CPU 15.36 -> 14.16 s (-7.8% of those products, ~-3.8% of the net); ops about equal; every depth saves (L5 -1.36%, L6 -1.74%, L4 -2.73%); max output difference 1.8e-4 relative (Winograd depth-5 rounding level). Cold r35: -0.51% (first-use view binding). Paper with the 1-add basis: ALL tree products -3.65 Gflop/net (-0.99%), batched -0.64 of it. Next (lead-approved): extend to batched / W5 / colpack as ONE flag --altbasis on S13; parity + VALID40 via the porter.
    2026-09-24 14:45 [engineer2] --altbasis BUILT on S13 (fable_engineer2_v1/altbasis/build_altbasis.py + ab_flag_fragment.py; stack AFTER --ipt): alternative-basis variants of single tree products, the batched P-product (_bb_product), the W5 port and colpack's shared-A (_cp_d6_shared), new functions only, the engineer's _ipt_* trees/regions/caches/depth rule unchanged; IPT dispatchers wrapped behind the literal _AB_ON; view cache LRU-capped (480). Build S13 + --altbasis = altbasis/S13_ab.py 97178798: einsum gate PASS (0 new sites), whest validate PASS. Parity vs fired S13 (OpenBLAS env, 35/100/37/39): bill -0.978/-0.992/-1.002/-1.007% (pooled -0.995% = the paper price -0.99%), final MSE -0.19/-0.10/-0.21/+0.13% (pooled -0.099%, rounding level), adjusted pooled -1.093%, max prediction difference 1.6-2.8e-6 relative. User-CPU pair vs S13 queued on the porter's runner (then project_wall_v3 on #332195). Ready for the porter's VALID40 vs S13 rows.
    2026-09-24 14:52 [engineer2] --ipab INDEPENDENT CHECK (altbasis/ab_vs_ipab.py: S13 in process, 180 single tree products per net, each run native / engineer2-ab / engineer-ipab on the same operands, native kept): steady-state bill (repeat calls, trees bound) mine -0.66/-0.71 Gflop per net vs ipab -0.72/-0.78 (r35/r100): ipab ~9% better (its output inverse bills mn/4 per level vs my 7mn/16 on the C tree); user CPU equal (warm: mine -5.4%, ipab -5.7%); ops: mine -59, ipab +531 per net warm (+1,007 cold) = ipab's per-call output views + own tree binding; max output difference vs native 1.8e-4 / 1.6e-4 (rounding). First-call bill differences are a harness artefact (mine shares native's IPT trees; ipab binds its own). Verdict: keep ipab; my only edge (tree-side C inverse: ~500-1,000 fewer requests/net, ~0.2-0.4 s wall, +0.016% bill) is not worth porting while S15's wall room is ~30 s.
    2026-09-24 14:58 [engineer2] BILL LEFT after IPT + ab (fable_engineer2_v1/altbasis/bill_left_census_v2.py on candS14ab_mem.py, r100 warm 359.4 Gflop/net): tree leaf matmuls 84.0%; tree transforms 7.9% (sub 4.49, add 3.00, mul 0.36); binding 1.7% warm / 2.9% cold (tree views 1.41%); body ~6.1% (matmul 1.79, multiply 1.25 mostly _nonlinear_pair_batch:925-936, tanh 0.52 at _message_edge_source:3799, sum 0.50, billed gathers 0.48 at _deadcut_push:5239 + _predict_binary:3260-3261, QR 0.37 at _shared_old384_push:5080). Next 1%: depth re-priced with ab transform costs (leaf 84%; -12.5% leaf per extra level; wall-limited), --ipt v4 binding -0.6% (built), gathers -0.2..-0.4%, QR -0.2..-0.3%, tanh/pair-series -0.2..-0.5%.
    2026-09-24 15:15 [engineer2] S15 PARITY (e313ab77 vs S14 9003e536, in process, 35/100/37/39): bill -1.208%, MSE +0.06%, adjusted -1.149% pooled (per net -1.233/-1.126/-1.259/-0.943%); max rel prediction diff 2.0-2.2e-6. User-CPU pair e2_S15 vs e2_S14ref queued behind the S15 VALID40 on the lock; project_wall_v3 runs when it lands.
    2026-09-24 15:31 [engineer2] S15 WALL (user-CPU pair e2_S15 vs e2_S14ref, 6 nets, porter tool, 15:20-15:24): user -3.29% (se 0.34%; all 6 nets -1.9..-4.3%), bill -1.22%, op calls +0/net. project_wall_v3 (base S13, S14+ab additive): bill -2.16%, expected worst 70.0 / 69.1 / 68.2 s, room to 100 s 30.0 / 30.9 / 31.8 s.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer2_v1/altbasis/ab_census.py](code/fable_engineer2_v1/altbasis/ab_census.py)
- [code/fable_engineer2_v1/altbasis/ab_price.py](code/fable_engineer2_v1/altbasis/ab_price.py)
- [code/fable_engineer2_v1/altbasis/ab_fragment.py](code/fable_engineer2_v1/altbasis/ab_fragment.py)
- [code/fable_engineer2_v1/altbasis/ab_shadow.py](code/fable_engineer2_v1/altbasis/ab_shadow.py)
- [code/fable_engineer2_v1/altbasis/ab_certificate.py](code/fable_engineer2_v1/altbasis/ab_certificate.py)
- [code/fable_engineer2_v1/altbasis/build_altbasis.py](code/fable_engineer2_v1/altbasis/build_altbasis.py)
- [code/fable_engineer2_v1/altbasis/ab_flag_fragment.py](code/fable_engineer2_v1/altbasis/ab_flag_fragment.py)
- [code/fable_engineer2_v1/altbasis/S13_ab.py](code/fable_engineer2_v1/altbasis/S13_ab.py)
- [code/fable_engineer2_v1/altbasis/ab_vs_ipab.py](code/fable_engineer2_v1/altbasis/ab_vs_ipab.py)
- [code/fable_engineer2_v1/altbasis/bill_left_census_v2.py](code/fable_engineer2_v1/altbasis/bill_left_census_v2.py)
- [code/fable_engineer_v1/candS14ab_mem.py](code/fable_engineer_v1/candS14ab_mem.py)
- [code/fable_engineer_v1/ipab_fragment.py](code/fable_engineer_v1/ipab_fragment.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L410](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L410)
- [TEAM_LOG_2026-09-22_to_09-26.md#L417](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L417)
- [TEAM_LOG_2026-09-22_to_09-26.md#L418](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L418)
- [TEAM_LOG_2026-09-22_to_09-26.md#L424](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L424)
- [TEAM_LOG_2026-09-22_to_09-26.md#L427](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L427)
- [TEAM_LOG_2026-09-22_to_09-26.md#L433](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L433)
- [TEAM_LOG_2026-09-22_to_09-26.md#L444](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L444)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

## Related work

- shipped in: [Submission #332207: alternative-basis Winograd arithmetic](../../../submissions/332207/README.md)

[Research graph](../../../graph/index.html#node=2026-09-24-alternative-basis-winograd-reduces-metered-work-1-22-user) · [Copy provenance](../../../PROVENANCE.json)

