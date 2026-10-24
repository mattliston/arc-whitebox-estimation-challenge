# COST MAP v1 — the estimator as a cost structure (2026-10-17, [opus-costmap], read-only)

Sources (all numbers cite these): charge census `opus_solo_v1/probe_charges_s2tr_7103256.json` (S2tr 46e9315d, seed 7103256, PB_ALL cold, 258.58 GF, 41,401 charges, every charge with op/shapes/site/layer); product probe `opus_solo_v1/probe_products_s2tr_7103256.json` (132 wrapped top-level products = 68.4% of the bill; the Smirnov kernel and in-function products are below the wrappers); request census `opus_fix336687_v1/profile_v25f/rows.jsonl` row 100 (v25f warm net, 40,265 requests, by_func / by_site); cProfile `profile_v25f/profile_100.pstats`; source `opus_fix336687_v1/candidate_v30.py` (same physics as #336741/#336759: layer loop `_astra_fhm_core` 17327, old tier `_astra_bcc_push_impl` 16248 / `_JOIN_SCHEDULE` 5077 / `_R320_RULE` 7478, Smirnov `_AstraSm40Kernel` 18726 / `_ASTRA_SM40_FAMILIES` 18494 / `_ASTRA_SM40_PLANS` 18493 / `_mainread_lazy_bb_batch` 20464). Oracle values from memories p2-1016-oracle-ladder-s2tr, p2-1011-module-audit-and-star-rank, p2-1016-final-error-origin-budget, p2-1016-nt-join-subspace-closed, p2-1016-roster-alignment-tax-and-dense-remainders, p2-1017-gc-quiet-predict-and-residual-anatomy. "unknown" = no measurement on file.

## 1. Data flow of one layer step (layer l, width n = 1024; `_astra_fhm_core` 17400-17470)

State carried from l-1: `mu` (n,) means; `C` (n,n) f32 post-activation covariance (plus a float64 diagonal path); `b` bias-direction vector; `star` (the star tensor state, rank-truncated, `_SS_RANK`); `radial` per-unit k4 scalar table; the BANK `p` = third-order direction columns (n, K): one (+,-) column pair per unit per cohort, so K grows by ~2n = 2048 per layer until a join; `lam` (K,) column weights; `shared_old` = old-tier state (joined cohorts compressed to rank 384 at joins; `_JOIN_SCHEDULE {7:(0,2), 10:(3,5), 13:(6,8)}` = cohorts 0-2 joined at l=7, 3-5 at 10, 6-8 at 13; `_R320_RULE` 384 at every join; up to 4 recent cohorts stay exact).

Step l (products in formation order; kind, shape class, bill from the census; names from the code):
1. COVARIANCE PUSH `Cpre = w^T C w` (17401): SYM 1024x1024x1024 via `_sym_upper` (4 upper block products, mirrored) — 0.71 GF, **391-430 requests** per layer. `bpre = w^T b` (matvec). Diagonal `vv`, `s = sqrt`, `a = mu/s`, gate `g = Phi(a)`.
2. ROSTERS (`select`, 17411-17415): live roster (gate >= 0.01), response roster (gate_min >= 0.01), 32-rounded; dead-cut roster 64-rounded inside the push (p2-1016-roster-alignment-tax: 12.7 / 29-42 padding rows; load-bearing accuracy, exact rosters +2.4% MSE).
3. BANK PUSH `pp = _deadcut_push(...)` (17418) = `w^T[live] @ p` → `_mm 1024 x K x n_live`: the largest single products: L1 1024x4096x992 4.05 GF ... L5 1024x960x10048 9.12 GF; after the first join the young part is pushed as `_mm m x k x n` with k 5696-8960 (L6 960x928x5696 4.99 GF, L8 896x928x8960 7.26 GF) and the OLD tier as `_shared_old384_mm` m x 384 x 5920 / x 2400 (1.0-1.8 GF each, 2-6 per layer).
4. READ (`_np_left/_cp_read_young_sq`, 17421-17424): the response rows of `pp^2` against the young columns → the CP plane `cp_pre` (n,n) and `_np_forms`; products `_mm 1024 x 992 x 6144`-class (L3 5.70 GF, L4 7.58 GF) — "reads" ≈ 35% of the bill (memory bill-floor); the Smirnov MAIN READ runs the pair-nonlinearity read of the whole bank with a 40/20/10-point quadrature batched over P (`_ASTRA_SM40_PLANS` keys (m, k, n, 384|0): (1024,4096,960,0), (928,8128,896,0), (864,9984,832,0), (768,7168,800,384) ... k up to 11712 = bank width) → `bb_batch` 52.2 GF = 20.2% of the bill, ~2.4K requests.
5. THIRD/FOURTH ORDER TABLES: `K21` (n,n) = cp + star_pre + outer terms (17441-17447), `k4` per unit = radial*vv^2 + K4L hub terms (`_k4l_local`, `_k4l_dk`), `_b6_old/_b6_qb` (old cohorts' energy), `_f4c_step/_f4c_birth` (F4 column shift), `_np_update` (pair law), `_ecut_raw/_xsg_rows` (edge), `_gop2_*` (GNN message products: `_gop2_message_product` 1024x1024 @ 1024x16, `_gop2_gnn_step` 1024x128 @ 128x128), `features` (per-layer feature vector for the learned head/edge), `_astra_hf_grouped`, `_hc_*`, `_re22_*`, `omitted_radial_source`. These are the PHYSICS BODY: 19,031 charges on vectors/matrices for 16.95 GF (6.6% of bill) = 45.9% of charges.
6. NONLINEAR PAIR MAP `_nonlinear_pair_batch(mu, Cpre, K21, k4)` (844): Hermite-series pair map on stacked rows (W-G pair map), 1,872 requests, 0.49 GF — produces the post moments (m, var, post_k3, post_k4) → next `mu`, `C`, bank births (newborn columns), `lam`.
7. STAR push (`_flms_star` / `_bb_star_pre`), `_ott_*` (old-tier tail estimate), `_uvp/_lcr` (u-leg reads), newborn/dead-cut bookkeeping; at join layers: `_astra_bcc_push_impl` + `_rtp_qr_basis` (QR 832x384 / 864x384 = 1.32 GF, 0.5%) + `linalg.solve` + the Nyström projections (`_shared_old384_mm` 928x6144x384, 384x928x6144, 896x384x6144, 800x6144x384 = 8.6 GF at L6; 11.8 GF at L9; 11.3 GF at L12).

Consumers: the pushes feed the reads and tables; the reads feed K21; K21/k4/mu/Cpre feed the pair map; the pair map feeds the next layer's state and the bank births. Error budget (p2-1016-final-error-origin-budget): final MSE 79% inherited from L14 means, 22% L15-local; by origin L13-15 52.5%, L8-12 38%, L0-7 7%; L14-born error sits in ~5-10% kink units.

## 2. Module table (bill = charge census, cold PB_ALL; requests = warm net 100 by_func; accuracy = oracles)

| module / function group | bill GF (%) | requests (% of 40,265) | products / typical shapes | accuracy when removed (+MSE) or made exact (-MSE) |
|---|---|---|---|---|
| Tree-kernel leaves: `_hz_leaf` (old-tier/k-split pushes, k>=2048) | 67.7 (26.2%) | 221 (0.5%) leaves + transforms below | 294 leaf matmuls, ipt layout (7,)*L+(r,c) | — (kernel) |
| `_ipt_leaf_unwind_ab` (generic tree products: reads, pushes k<=1024, cov/star) | 62.5 (24.2%) | 2,557 (6.4%) | 118-121 products, depth 6-7 | — |
| Smirnov main read `bb_batch` + `_astra_sm40_linear_grouped` + `_mainread_lazy_bb_batch` + group_views | 54.5 (21.1%) | 2,441 (6.1%) | 23 bb_batch calls/net, P=40/20/10 batched (m,k,n) with k 4096-11712 | quadrature size vs accuracy: **unknown** (b709 table167 vs 80: -0.02% MSE, -0.09% bill) |
| `_d7_product_kpad` (depth-7 colpack pushes L2-5, 896/1024 dims) | 21.8 (8.4%) | 304 (0.8%) | 216 leaves | — |
| Tree transforms `_ipt_forward_a/b_ab`, `_ks_unwind`, `forward`, `_hzt/_hzd_*`, `_l3o_last`, loads/views (`_rs_load_*`, `_ipt_side`, `_hz_side`, `_rs_basis_views`, `_hzd_load`) | ~22 (8.5%: adds/subs 15%, views ~1%) | **~11,000 (27%)** | alt-basis minimum: 3 core + 1 basis op/level/side, 6 decode/level; 55-104 requests per product | — (byte-identical levers only) |
| k-split `_ks_product`, `_ubx_ks_product`, `_square_product`, `_r7_product` | 7.3 (2.8%) | ~400 | 6-94 products | — |
| Old tier total (`_shared_old384_mm`, all shapes) | 63.3 (24.5%) incl. joins L6 9.0 / L9 11.8 / L12 11.3 | within kernel counts | per-layer m x 384 x 5920/2400 pushes + join projections | exact old tier (no joins) **-1.55% MSE, +18% bill**; rank 320 at L7 +1.72% MSE / -1.08% bill; L10 448: -0.36% / +2.2%; NT (eigen-quality basis) -1.30% MSE / +2.0% bill (rate .64-1.14) |
| Covariance sandwich SYM `_sym_upper` / `_psdm_sym_upper` | 10.6 (4.1%) = 15 x 0.71 | **~6,000 (15%)** = 15 x 391-430 | 4 upper blocks 512^2-class at depth 6-7, mirrored | two-strip: -3,109 req (-7.1%), +0.66% bill, MSE -0.05% (rounding) (bnd_v31t); dense (2 plain products): +21% bill |
| `_astra_fhm_core` body (loop-level vector/matrix algebra) | 1.69 (0.65%) | 2,831 (7.0%) | 178 sites, 15-60 req each; 1,815 vector + 849 matrix charges | — (exact algebra) |
| `_nonlinear_pair_batch` (pair map, Hermite series) | 0.49 (0.19%) | 1,872 (4.6%) | per-layer stacked rows (R, n) | exact pair kernel: KILLED (late |rho| <= .6; truncation is not the late error) |
| `_bd_update`, `_f4c_step/_birth`, `_np_update`, `_npx_left`, `_k4l_*`, `_astra_hf_grouped`, `_gop2_*`, `features`, `_ecut/_xsg`, `_b6_*`, `_hc_*`, `_re22_*`, `omitted_radial_source`, `_ps32`, `select` | ~13 (5%) | **~16,000 (40%)** | mostly (n,) vectors and (n,n)/(n,K) elementwise | whole-module OFF (S47e audit, 16 seeds): F4C +0.40%, kappa_C +1.08%, b6 +1.37%, edge +1.45%, K31L4 +1.66%, K4L +2.7%, head +4-6%, star off +260%: **no module pays to drop** |
| Joins (`_astra_bcc_*`, `_rtp_qr_basis`, `linalg.solve`, `_w3r_product`) | 1.94 (0.8%) + the projections counted in old tier | ~500 | 3 joins | see old tier |
| GC / interpreter | 0 | 0 requests; 15-20 young pauses/net (v34y) vs ~220 | — | byte-identical |

Requests per GF by class (charge census): physics vector/scalar 10,160 charges (24.5%) for 0.40% of bill = **0.10 GF per 1,000 requests**; physics matrix 8,871 (21.4%) for 6.2% = 1.8 GF/1,000; kernel+Smirnov 22,370 (54%) for 94.3% = 10.9 GF/1,000. The residual (requests x ~8 us) is therefore spent 46% on 7% of the bill.

## 3. Product census per layer (wrapped products; `fn mxkxn GF/requests`; from probe_products)

| L | products | GF | req | the layer's big products (k growth) |
|---|---|---|---|---|
| 0 | 5 | 4.7 | 698 | push 1024x2048x1024 1.96; 2 x 1024^3; SYM 0.71 (416 req) |
| 1 | 5 | 8.7 | 797 | push 1024x4096x992 4.05 (152 req); read 1024x1024x2048 1.96; SYM 0.71 (392) |
| 2 | 4 | 6.5 | 710 | read 1024x1024x4096 3.87; SYM; Smirnov 5.8 GF (not wrapped) |
| 3 | 4 | 8.4 | 839 | push 1024x992x6144 5.70 (256 req); d7 4.7 GF; Smirnov 7.0 |
| 4 | 4 | 10.3 | 901 | push 1024x992x8128 7.58 (330); d7 6.2; Smirnov 7.8 |
| 5 | 4 | 11.7 | 952 | push 1024x960x10048 9.12 (384); hz 6.0; Smirnov 8.3 |
| 6 (join 0-2) | 9 | 16.4 | 1,415 | young push 960x928x5696 4.99; OLD projections 928x6144x384 2.24 + 384x928x6144 2.24 + 896x384x6144 2.15 + 800x6144x384 1.95; hz_leaf 15.4 GF |
| 7 | 7 | 12.9 | 1,023 | push 960x928x7360 6.38; OLD 832x384x6144 1.93, 768x6144x384 1.85 |
| 8 | 7 | 13.5 | 1,116 | push 896x928x8960 7.26; OLD 736x6144x384 1.80, 768x384x6144 1.78; PSDM 1024x960x1024 0.70 (430 req) |
| 9 (join 3-5) | 16 | 17.6 | 2,366 | push 896x864x4800 3.52; 7 OLD 1.2-1.96 (5696/5920 widths) + 4 small 384^2 |
| 10 | 9 | 12.6 | 1,381 | push 896x864x6272 4.52; OLD 768x384x5920 1.72, 672x5920x384 1.54, 768x256x5920 1.20, 672x5920x256 1.07 |
| 11 | 9 | 13.4 | 1,400 | push 896x864x7744 5.77; 4 OLD 1.0-1.6 |
| 12 (join 6-8) | 21 | 16.4 | 2,302 | push 896x832x4288 3.05; OLD 384x832x4800 1.57, 832x4800x384 1.57, 640x384x5920 1.44 ... (2400-wide second old tier appears) |
| 13 | 15 | 13.0 | 1,642 | push 832x832x5568 3.77; OLD 640x384x5920 1.44, 544x5920x384 1.30, 2400-wide x4 |
| 14 | 9 | 9.9 | 835 | push 768x800x6784 4.18; OLD 768x384x5920 1.72, 768x256x5920 1.20, 2400-wide x2 |
| none | 4 | 0.8 | 247 | 3 x 512x1024x512 (head/setup-side) |

Bill by layer (all charges, GF): L0 5.0, L1 9.5, L2 13.1, L3 16.2, L4 19.8, L5 20.9, L6 23.2, L7 19.9, L8 21.1, L9 23.5, L10 18.2, L11 19.6, L12 21.3, L13 17.4, L14 11.3. **L0-6 = 108 GF = 42% of the bill for 7% of the final error (origin budget)**; L8-14 = 132 GF = 51% for 90% of the error.

## 4. Re-architecture hypotheses, ranked by (expected score or safety gain) x confidence

| # | hypothesis | bill | requests (residual) | MSE risk | effort | confidence |
|---|---|---|---|---|---|---|
| H1 | **Stacked-state physics body**: keep every per-unit vector of a layer in one (R, n) state array and run the body's algebra as row-broadcast ops / coefficient einsums (`einsum('k,ki->i')` sums of products) instead of ~10,000 separate (n,) ops per net; the E2 diet already stacks the Hermite rows, so the remaining 10,160 vector charges are the target (sites: `_astra_fhm_core` 1,815, `_nonlinear_pair_batch` 1,116, `_ps32` 379, `features` 378, `_bd_update` 366, `_f4c_birth` 320, `omitted_radial_source` 316, `_k4l_dk` 300, `_f4c_step` 294, `select` 276, `_hzd_load` 237, `_np_update` 206, `_xsg_inverse` 204, `_b6_qb/_b6_m2` 392 ...) | ~0 (vector ops are 0.4% of bill; einsum on vectors bills 2 flops/elem) | -3,000 to -6,000 of 40,265 (-7% to -15% residual) | rounding-level if each fused form keeps per-element operation order; f64 sum reorderings must be store-checked | 2-4 days (10K+ lines of fragments; many sites redefine the same functions) | medium: the mechanism is certain (0.10 GF per 1,000 requests), the yield depends on how many chains are fusable |
| H2 | **Smirnov quadrature size per family/layer** (`_ASTRA_SM40_FAMILIES`: L4_all P=40, L5 P=20, L6 P=10): halve P at the near-Gaussian early layers (L2-5 carry 28.8 GF of Smirnov bill = 11% of the total) | -5% to -11% if P halves where the integrand is smooth | -1% to -2% | **unknown** (no oracle on quadrature size; the b709 table167 change was -0.02% MSE); needs a 4-seed native oracle first (3-6 h) | 1 day incl. pricing | low-medium |
| H3 | **Earlier first join** (join cohorts 0-1 at L4-5 instead of 0-2 at L7): the pre-join bank width reaches 10048 at L5 (push 9.1 GF + Smirnov 8.3 GF); an L4 join caps K at ~6000 | -4% to -7% (L4-6 pushes/reads + Smirnov k) | -3% to -6% (fewer k-chunks in the batched reads; unknown exactly) | +0.3% to +1.0% MSE: joins cost ~0.5% each on average (3 joins = 1.55%); "early absorption = content loss" (nt-join memory); the final error is only 7% L0-7-born but a join at L4 propagates through 11 layers | 1 day (JOIN_SCHEDULE is data-driven: `{4:(0,1), 7:(2,3), 10:(4,5), 13:(6,8)}` + prebind keys for the new shapes) | medium on bill, low on MSE (needs the 8-net native check) |
| H4 | **Two-strip covariance sandwich** (SYM 15% of requests for 4% of bill) | +0.66% | -3,109 (-7.1%) | -0.05% (measured, bnd_v31t) | 0 (built: v31t/v35y) | high — the known request/bill frontier (0.21% per 1,000) |
| H5 | **Physics-matrix elementwise fusion** (8,871 (n,n)/(n,K) charges for 6.2% of bill: K21 assembly 17441-17447, cp_pre scatter, `_e2_pp2 * pp`, `_e2_t3 * lam` chains) | ~0 | -1,000 to -2,000 (-2.5% to -5%) | rounding-level | 1-2 days | medium |
| H6 | Old-tier rank / schedule (rank 320 at joins; fewer or more joins) | -1.1% per 64 rank at one join | ~-200 per join | **1:1 or worse** (r7rank320 +1.72% MSE for -1.08% bill; NT subspaces 1:1) | 0 | high that it does NOT pay — closed |
| H7 | Duplicated work across layers | — | — | hunt2: duplicate forward transforms 0.1% of bill; cross-layer identical products: none found (idc1 cut 0.06%) | — | high — closed |
| H8 | Early-layer cheapening of the kernels (depth, roster rounding) | roster rounding is load-bearing (+2.4% MSE for -0.47%); depth frontier -0.42% paper but +6,668 requests (hunt2) | worse | — | — | closed |
| H9 | Lazy-bind coverage (prebind the rare shape keys; 13-98 lazy binds/net ~ +1.6% residual and the early-network variance) | 0 | -500 to -700 (-1.5%) | 0 | 2-4 h, but setup cap risk (platform setup 3.59 s; 5 s cap fails the whole worker) | medium-low |
| H10 | GC: per-predict freeze + coarse young threshold (v34y) | 0 | -3.3% (local) | 0 | done | high |

Where the bill is NOT movable at zero MSE (for the record): kernel leaves 161 GF (62%) run at the Winograd floor (depth 6-7, alt basis, 0.45-0.51 of dense; p2-winograd-kernel-numpy-laws); the Smirnov kernel's 10 bb_batch classes are odd-32 at the next depth (hunt2); roster padding is accuracy; the old tier trades 1:1.

Bottom line: the residual (requests) is dominated by code that bills almost nothing — the physics body's ~19,000 small ops (46% of requests, 7% of bill) and the covariance sandwich (15% of requests, 4% of bill). A stacked-state rewrite of the physics body (H1, plus H5) is the one re-architecture with a plausible -10% to -20% residual at zero bill and rounding-level MSE; the bill side has no zero-MSE lever left above ~0.5% except the untested Smirnov quadrature size (H2) and the earlier first join (H3), both of which need an accuracy oracle before any build.
