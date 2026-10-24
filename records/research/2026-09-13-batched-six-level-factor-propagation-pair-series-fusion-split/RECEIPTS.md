# fable_wall_repair_v1 — make the c2 card pass the platform's 120 s per-MLP wall at unchanged physics (2026-09-13)

Source of truth: Astra's frozen c2 (SHA c2c2ffad…), copied here as c2_frozen_copy.py (hash verified). Astra's directory is read-only.
Evidence on the platform wall (from Astra's receipts): local numpy-free wall 28.7–29.8 s (329926) passed 100/100; 34.6–43.7 s (330390) passed; 36.8–37.7 s (329884) lost ONE MLP to TIME_EXHAUSTED (score destroyed); 54.6–57.9 s (c2 = 330840) failed every attempted MLP. One timed-out MLP zeroes the whole score, so the margin must be large: target local numpy-free wall <= ~30 s.

Native profile (r35, 1 BLAS thread, numpy backend; the remote client adds ~1.4x): c2 38.4 s: flopscope pointwise ops 25.3 s (48.5k calls: Winograd transform adds), matmul 4.7 s, stack 3.3 s. Big push products through the serial-top D6 kernel 20.6 s (121 calls), 221 five-level square products 11.7 s, shared read 4.7 s, newborn push 7.4 s.

| variant | change | native wall r35 | util r35 | bill vs c2 |
|---|---|---:|---:|---:|
| c2 (frozen) | — | 38.4 s | 0.1834 | — |
| A | big pushes through the D4 packed kernel instead of serial-top D6 | 28.2 s | 0.1910 | +4.1% |
| A2 | A + square-product depth capped at 4 | 20.8 s | 0.1950 | +6.3% |

Expected adjusted (scale Astra's NC-14 3.805e-9 by the bill): A ~3.96e-9, A2 ~4.05e-9; bank is 4.539e-9. A2 chosen for the first bank: its wall is in the 329926 class; A's is in the 329884 class.

## A2 qualification (2026-09-13)
- Output vs the frozen c2 on r35 (1e9 truth): 1.8389e-8 vs 1.8410e-8 (rounding-level; A2 slightly better). Bill 4.288e11 (+6.3%).
- 14-net grader panel (ordinary frame, independent 4M truths, panel_candidate.sh): panel8 adjusted 4.473e-9 (raw 2.297e-8, util 0.1945); virgin6 3.450e-9 (raw 1.781e-8, util 0.1933). Astra's c2 in his NC frame: 4.213e-9 / 3.262e-9. Expected official ~4.0-4.1e-9 (official/local ratio ~0.90 from #329754), bank 4.539e-9.
- Production boundary (numpy-free client, OpenBLAS server, 1 thread, quiet box): walls 36.37 (cold) / 35.68 / 35.84 / 36.11 / 35.63 s on 35/100/500/3016500/35; residual 0.19-0.20 s; repeat byte-identical; under panel load 35.4-41.7 s. The frozen c2 under the same load: 54.4-65.3 s (matches Astra's 54.6-57.9 quiet).
- whest validate: passed. Socket path note: zmq ipc paths must be <104 chars, so the local server socket lives at /tmp/fwr_flops.sock during timing runs (removed after).
- Decision: submit A2 as the wall repair of the c2 card (platform evidence: 34.6-43.7 s passed 100/100; 36.8-37.7 s lost one MLP once; 54-58 s failed all). Risk moderate; a timeout costs no quota.
- Peak combined RSS (server+client) during an A2 boundary run: 5593 MB (limit 8 GB). Walls 35.9-36.6 s.
- Variant A at the boundary (quiet): 43.3-44.2 s, util 0.191 -> upper edge of the passing zone; not submitted.
- Variant A3 (depth 3 rectangles) needs its own arena view layout (the D4 view list fails the frame assertion); not built. Fallback ladder if 330924 times out: A2 + square cap 3 (needs no new layout) then D3 rectangles with a proper layout.
- SUBMITTED: #330924 = variantA2 (copied as cand_c2_wallrepair_A2.py), 2026-09-13 ~15:10 PDT.
- Fallback rung A2b (square depth 3, D4 rectangles): r35 native wall 18.2 s, util 0.2009 (+9.5% bill), final MSE 1.8352e-8 (rounding-level). Expected official ~4.15e-9.

## The real cause and the fix (2026-09-13 afternoon)
- Micro-benchmark: the Winograd transform adds run on views sliced from maximum-shape arena buffers; on the push geometry those strided views are 3.7-5.5x slower than contiguous arrays at every level (identical arithmetic). Those adds are 65% of the card's native time.
- flopscope cost model measured: reshape = 1 flop/element (so exact-shape rebinding is billed); slicing, unstack, zeros, empty = 0.
- Variant C (exact-shape frames via arena reshapes, per-prediction rebinding): byte-identical to A2, native 14.7 s, but +3.6% bill every prediction (frame cache evicted at 128 entries). Variant D (fnp.empty frames, free): bill unchanged but no wall gain (page-fault churn), RSS 7.0 GB. Variant E (= C with a persistent frame cache across predictions, arena is process-level): native 14.1 s; bill 0.2020 on the first prediction (bindings), 0.1972 on the second net (its new shapes), 0.1942 on a repeated net (= A2's bill); peak RSS 5.9 GB; byte-identical to A2.
- E5 (E + five-level squares restored): native 17.9 s, util 0.1993 cold. E6 (E5 + the serial-top D6 push kernel restored with exact-shape top frames): native 23.4 s (c2: 38.4 s), util 0.1908 cold (c2 warm 0.1834), output BYTE-IDENTICAL to the frozen c2 (max abs diff 0.0) = the 3.805e-9 physics and arithmetic exactly.
- Boundary timing of the ladder: time_ladder.sh -> time_ladder.log.

## #330924 (A2) GRADED 0.17235: failure-penalised (many MLPs zeroed) despite a 35.6-36.4 s local boundary wall. E6 boundary 40.4-44.4 s; E5 31.8-35.1 s.
Platform evidence now: 329926 (local ~29 s) passed 100/100; 329884 (37 s, ~39k remote requests) lost 1; A2 (36 s, ~41k calls) lost roughly half; c2 (55-58 s, ~63k) lost all. Consistent with a platform cost dominated by remote CALL COUNT (~2.5-3 ms per call there): safe budget ~30k calls per MLP. Contiguity (numpy time) helps less there than here.

## Request census at the production boundary (2026-09-13 evening; callsite_census.py = numpy-free client + OpenBLAS server, cold r35, one prediction)
Every flopscope op is one synchronous ZMQ round trip (REQ socket, no batching in flopscope_client 0.12). Cold = first prediction of the process (frames bound).
| card | cold requests r35 | of which frames / kernel / body | bill r35 cold | local boundary wall | platform outcome |
|---|---:|---|---:|---:|---|
| 329926 (Astra, passed 100%) | 37,255 | 8.0k / 15.9k / ~13k | 0.2231 | ~29 s | passed |
| 330390 (bank, passed) | 44,135 | 8.8k / 19.0k / ~16k | 0.2146 | 34.6-43.7 s | passed |
| 329884 (Astra) | 39.1k (Astra's count) | — | 0.207 | 37 s | lost 1 MLP |
| A2 = #330924 | 60,262 | 13.5k / 29.3k / 17.2k | 0.1951 | 36 s | ~10 of 57 MLPs failed (0.172) |
| c2 frozen = #330840 | 87,510 | 15.8k / 53.0k+0.9k / 17.2k | 0.1834 | 55-58 s | all failed |
| F (batched D6 push kernel) | 74,425 | 19.8k / 37.0k / 17.2k | 0.1947 | — | — |
| G (F + fused pair series) | 73,075 | 19.8k / 37.0k / 15.9k | 0.1947 | — | — |
| H (G + split frame caches, no dead D6 binding, no no-op slices) | 65,266 | 13.0k / 37.0k / 15.3k | 0.1916 | 41.4 s | — |
H on later nets in the same process: r100 56.6k (frames 5.6k), r500 54.3k, virgin 3016500 54.8k, repeat r35 51.9k (frames 0); walls 38.6-39.5 s.
Platform cost model fitted on the censored outcomes (A2 median ~105 s with 10/57 failures, 329884 ~95 s with 1/57, c2 all): platform wall ~ 1.1-1.4 ms x requests + ~2.0 x local compute (compute = boundary wall - 0.30 ms x requests). H cold: ~125-139 s (fails); 330390: ~113 s; 329926: ~87 s.
Per-shape depth benchmark at the boundary (bench_depths.py -> bench_depths.json, 178 shapes x depths 0/3/4/5 squares, 0/4 reads, 0/4/6 pushes): small products cost 10-20x more WALL under Winograd than as plain matmuls for a 25-35% bill saving on ~1e8-flop products (e.g. 384^3: D0 1.3 ms vs D4 23 ms); the greedy on (bill added)/(platform seconds saved) prices the whole frontier: platform <=120 s +1.2%, <=110 s +2.9%, <=100 s +4.4%, <=95 s +5.6%, <=90 s +7.1% bill (greedy_platform_*s.json).
Variant I = H + a deterministic size rule (naive flops 2mkn thresholds per product kind, fitted to the greedy; never deeper than c2's schedule): rule95 (sq D0<5.33e8<=D3<7.29e8<=D4<1.64e9<=D5; push D0<7.2e8<=D4<8.27e9<=D6; read D0<5.54e8<=D4) r35 native: MSE 1.8374e-8 (c2 1.8410e-8), util 0.2010 cold, native 16.9 s. rule100: 1.8363e-8, 0.1996, 17.4 s.
Zero-bill request cuts still open: Morton (block-major) kernel layout (~-8k requests, needs input blockify copies ~+1% bill -> not free), body fusions (~-3k).
- I95 = SUBMITTED #330931 (cand_c2_wallrepair_I95.py, SHA dd64dfc3…), 2026-09-13 ~21:05 PDT. Quiet-box multinet at the boundary (run_multinet_rss.sh): r35 cold 45,118 requests / 29.5 s / util 0.2010; r100 39.0k / 27.7 s / 0.1970; r500 36.9k / 28.1 s / 0.1964; virgin 3016500 36.9k / 27.2 s / 0.1915; repeat r35 35.4k / 28.0 s / 0.1936. Server peak RSS 5975 MB (8 GB limit). 14-net panel: panel8 raw 2.291e-8 util 0.2014 adj 4.619e-9; virgin6 raw 1.778e-8 util 0.2004 adj 3.569e-9 (A2 for reference: 2.297e-8 / 1.781e-8). rule100 multinet: cold 46.6k, later nets 36-40k, walls 28-30 s.
- Rule ladder built and scored natively on r35 (rule_fits.json -> build_variantI.py): L120 util 0.1977 (+3.2% vs H cold), L110 0.1978, L85 0.2091, L80 0.2132; MSE 1.836-1.838e-8 (c2 1.841e-8).
- Morton (block-major) kernel evaluated on paper and rejected: quadrant extraction becomes 3 unstacks instead of 4 getitems (-3 requests per level per frame, ~-2.5k total), the fused backward cannot share the scratch region with the (2,2) corner axes, and the input/output conversions cost +(mk+kn+mn) per product (~+0.2% bill). Not worth a kernel rewrite.
- #330931 GRADED 2026-09-13 ~22:25 PDT: adjusted 4.188358e-9, raw 2.129570e-8, util 0.1967, "Graded successfully" (no failure penalty). New bank (previous 4.539e-9). Platform calibration point: 45.1k cold requests + 29.5 s local boundary = passes all 57 MLPs.

## Where the score is (2026-09-16, after #330931 banked 4.188e-9; leader 2.45e-9 = raw 1.68e-8 at util 0.146)
- Bill by physics caller (bill_census.py callers, I95 r35 cold, util 0.201): young cube-column push 31% (16.6% + 14.2% tiled for banks wider than 8192 columns), old-tier plane read 15.6%, young plane read 12.3%, old-tier projection/QR path 12.6%, cubic star push 10.3%, covariance push 8.0%, newborn base push 3.2%, exact-view reshapes 3.2% (cold), card body ~5%. 87% is the third-order carrier.
- Bill dials that FAIL: birth gate floor (0.01 -> 0.02: util -7% but MSE +71% because it also drops alive units); symmetric-einsum billing (W^T C W with C declared symmetric bills 0.75x, Gram 0.5x) loses to Winograd (0.55x) on the big products.
- Error profile (oneshot_profile.py, mean teacher-forced to the 1e9 truth each layer; r35/r100): born error per layer card-only 2.5e-10 (L1) -> 3.8e-9 (L10) -> 6.0e-9 (L15); the head removes 9-22% of it; closed-loop 1.84e-8. Accumulation factor ~0.7/layer: final ~= sum_l 0.7^(15-l) born_l (L15 33%, L14 24%, L13 16%, L12 10%). So ~70% of the final error is inherited; knowing m_14 exactly would give a final MSE of 5.2e-9.
- Head ceiling (rows_collect.py, closed-loop rows r35): in-sample ridge on the 30 features + 64 recurrent state explains only 9-13% of the closed-loop error at any layer (current head 4-7%). The accumulated error is not per-unit-predictable; retraining the head is not the lever. The lever is the carrier's plane accuracy per flop (Astra's oracle: true off-diagonal K3 -> next-mean error x0.33; true K3+K4 -> x0.02).
- Tier sweep (sweep_tier.py, r35/r100, native): rank384/age5 (c2) util .201/.197 MSE 1.837/1.806e-8; rank512/age5 util .228/.223 MSE 1.884/1.775e-8 (no raw gain for +13% bill); age4 util .196/.191 MSE 2.07/2.07e-8 (+13% raw); age6 util .205/.202 MSE 1.873/1.775e-8 (wash); rank256 util .175/.172 MSE 2.45/2.45e-8 (+33% raw). The carrier is at a sharp local optimum; its raw is limited by the closure physics, not compression.
- Prefix forcing (prefix_force.py, r35; mean forced to truth through layer L, free after): L-1 1.837e-8, L8 1.764e-8, L11 1.463e-8, L12 1.197e-8, L13 8.88e-9, L14 5.24e-9. The final error is born in layers 12-15 (L15 5.2e-9, L14 3.6e-9, L13 3.1e-9, L12 2.7e-9 marginal), earlier layers 3.7e-9 in total; amplitude decay ~0.9/layer.
- Per-unit oracles at layers 12-15 from the 8M truths (oracle_diag.py, r35): true diag k3 -> final 1.743e-8 (-5%; card k3 diag 6% off), true per-unit k4 -> 1.654e-8 (-10%; card isotropic radial k4 is 24-28% off per unit), both -> 1.530e-8 (-17%).
- Plane oracles at layers 12-15 (oracle_planes.py, MPS 8M truth planes public_g19_attribution_v1/mps_planes_rank0035_new.npz, r35): truth K21 plane -> final 1.703e-8 (-7%); truth K22 plane (incl. per-unit k4) -> 1.634e-8 (-11%); both -> 1.381e-8 (-25%). Card plane errors at L12-15: K21 off-diagonal rel rms 0.60-0.65 (corr 0.80, truth 1.3-1.45x larger), K22 off-diagonal 0.24-0.29.
- Plane error structure (plane_error_structure.py): the K21 error's top singular direction holds 45-66% of its energy (top 8: 81-84%); a scalar rescale only takes the rel error 0.63 -> 0.59; the (a_i+b_j)rho_ij ansatz explains 35-40% of the normalised error.
- Structured corrections FAIL (oracle_struct.py, layers 12-15, r35): card K21 + rank-1 truth component -> final 1.15e-7 (6x WORSE); rank-8 -> 5.5e-8; scalar rescale by the truth slope -> 7.3e-8; learned per-unit k4 (ridge on pre features, leave-one-net-out R2 0.39, k4 error 0.24 -> 0.17 at L12-15) -> 1.92e-8 (+4% worse; at all layers 1.90e-8). The c2 closure is a cancellation optimum: partial or approximate supply fixes break it; only complete consistent supplies help, and even those need joint re-adaptation of the birth/head coefficients (closure-coadaptation-basin-law reconfirmed on c2).
- Row collection (rows_collect.py) for ranks 35, 36, 100, 150, 200 in rows_c2/ (16 layers x 1024 x 30 features + state + host + truth); truthpost exists only for ranks 35..234.
