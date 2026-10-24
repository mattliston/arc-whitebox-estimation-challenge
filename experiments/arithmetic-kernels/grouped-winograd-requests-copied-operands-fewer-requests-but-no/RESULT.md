# fable_engineer_v1 — Winograd kernel lane: identical predictions at lower cost (2026-09-22/23)

## Big-k split v6 (E-77: kept out of S29) and split u/v old-tier frames, native scope (E-78: paper) (2026-09-25)

**E-77, the big-inner-dim split beyond the bank read.** A census of every product call on S27 net 100 (e77_mm_census.py) plus the v2
bill model suggested -1.3% of the bill. The steady-state bench did not hold that up (e77_sqsplit_bench_s27.log: OpenBLAS, warm keys,
current _mm path vs the v2 leaf-sum split):

| shape (current path) | bill | requests | backend |
|---|---|---|---|
| 896x9984x960 (read6) | -2.14% | 309 -> 179 | x0.92 |
| 576x8576x832 (read6) | -2.82% | 228 -> 179 | x0.98 |
| 1024x4096x1024 (read6) | -1.71% | 154 -> 131 | x1.21 |
| 832x11776x928 (read6) | -2.11% | 258 -> 231 | x1.08 |
| 864x5632x384 (depth-5 read) | -3.60% | 73 -> 135 | x1.72 |
| 672x11776x384 (depth 5) | -2.53% | 147 -> 183 | x1.72 |

- **Why the model overstated it.** engineer2's read6 already runs these shapes at depth 6, and the census bills included first-time
  binds. A warm path census (e77b_path_census.py, S27t net 100) puts read6 at 4 calls, 19.0 GF per net.
- **v6.** ksplit_fragment_v6_final.py (bda9a0ab222ba4a5) = v5w + no 4096 chunk cap + an _mm wrapper that sends products needing
  >= 2 depth-6 chunks whole to the split. Parity vs v5w: bill +0.19 / -0.32 / -0.20 / -0.12% (mean -0.11%; net 35 pays first-time
  binds), requests -75..-552, MSE mean +0.09%. Under the 0.2% bar and mixed: OUT of S29 (lead).
- **Partition with engineer2.** Products that need chunks are mine; whole-fit depth-5 moves (p6) are theirs. p6 hooks below my
  wrappers, so nothing is priced twice.

**E-78, split u/v old-tier frames: native scope, paper only** (S28 a559b073, net 100, 296.5 GF).
- **Where the frame costs sit.** The frame costs 74.5 GF: reconstruct U@F 35.0 GF, inner read 28.3 GF, rank-by-rank join/QR/core
  11.2 GF. The 87 GF at _shared_old384_push are the exact recent cohorts, and the split does not touch them.
- **Price.** At each call's own rank, r_u = 256 saves ~-2.0..-2.5% (best case -2.73%); r_u = 192 saves ~-4.0..-4.5%. The model's
  -4.6% assumed 27.7% of the bill at rank 384. Only 21% scales with the rank, and the L10 cohort is already at 320.
- **The inner must use the per-frame form.** Pair coefficients are +-lambda and F+- = F_u +- F_v/2, so the inner is
  lambda[2uv F_u + (u^2 + v^2/4) F_v], costing (r_u + r_v)/(2r). A naive projection onto both frames costs (r_u + r_v)/r: +5.6%
  at r_u = 256. Cores split per frame, because with equal pair weights the u-v cross terms cancel. The build waits for covhunter's
  torch raw.

## Lazy tree binds and the cache-cap raise (E-75), k-split v5w (E-74e), and the rank-320 gap (E-76) (2026-09-25)

**Lazy binds are billed.** Binding a tree view inside predict is a reshape billed at numel: the tree plus its level-1 reshape, 30-120
Mflop per key. e75_lazybind_census(_v2).py times every bind on the live budget counter. On S27, in one worker-like 20-net session, lazy
binds cost 0.81% of the bill per net (0.19..1.24%). 39% of that is eviction REBINDS: the 160-per-side tree caches fill by about net 4,
then keys that earlier nets used are evicted and bound again, 30-68 per net.

| cache cap on S27 (capraise*_fragment.py appended; one line, no functions) | 320 | **480 (CAP CHOICE)** |
|---|---|---|
| predictions, 20 nets | hash-identical 20/20 | hash-identical 20/20 |
| bill, nets 1-10 / 1-20 / average of the two horizons | -0.201 / -0.295 / -0.248% | -0.201 / -0.314 / **-0.258%** |
| rebinds on nets 13-20 (S27: 37..68 per net) | 1, 0, 2, 2, 6, 3, 3, 11 | all 0 |
| live handles at net 20 (in-process census = server handles; FLAG >= 70k, cap 100k) | 28,770 -> 51,155 | 28,770 -> 55,826 |
| cached arrays owning data / peak RSS | 0 GB / 5.67 -> 5.69 GB | 0 GB / unchanged |
| residual pair, cold (rule <= +0.010 s) | +0.005 s | +0.001 s |

Verdict file: e75_cap_verdict.txt (the porter keys on its CAP CHOICE line); 480 = capraise480_fragment.py 431d4a5dc8bac7a2.
Platform workers run ~10 (S27) or ~20 (S24) nets, and the two caps are identical over the first 10.

**Free binds: not recommended.** The idea is to derive tree views by getitem (billed 0) from setup-bound super views. Row-sliced views
leave gaps between blocks, and a numpy micro-bench prices the gapped trees at +8..+59% on level ops. Leave-one-net-out over 8 nets,
the best 20 super views remove only ~0.11% of the bill, because read widths are net-specific (first-time binds are ~0.6% per net
over ~170 distinct (kind, depth, width)).

**Rank-320 gap (E-76; e76_rank_ks_probe.py, e76b_attrib.py, e76c_callshapes.py; net 100, S27 vs S27s).** The k-split does NOT fall
back at rank 320: identical splits at every layer, reads -0.58 GF. The whole cut, -2.22 GF (-0.726%), sits at the old-tier site
_shared_old384_mm at L13-15. Of that site's 31.5 GF, the 4608-column products of the cohort joined at L13 follow the rank (ratios
0.85, 0.71 when both dims carry it). The 11776-column products of the OLDER cohort keep rank 384 and are unchanged: 17.6 GF, 5.8% of
the net ([896x384]@[384x11776], [576x11776]@[11776x384], ...). So the -1.7% prediction, which scaled every old-tier product by 64/384,
overshoots. Re-ranking the older cohort at the join is a method question. Kernel side: some small rank-320 products lose Winograd depth
(dim/32-indexed depth tables; [576x384]@[384x320] runs as a plain matmul), ~0.05% of the bill.

**k-split v5w (E-74e; ksplit_fragment_v5w.py 57656b02336ec2c7).** It is v4w plus single-chunk depth-6 reads (padded, no split) for
k >= 2048 where the cost rule pays. Parity vs S24+E-73, nets 35/100/37/39: bill -1.147% (v4w -1.084%), requests +1,457 (v4w
+1,332), final-MSE ratio mean 1.0010 (v4w 1.0009). Splits 22-31 per net. Its extra CPU (~6 small depth-6 reads per net) is unmeasured,
estimated ~+1% user CPU.

## Depth-5 read k-split for S26 (E-74 v4): bill -0.93% (4-net parity), +~950 requests/net, user CPU +5.2%, expected worst 73.7 s (2026-09-25)

The S26 fragment is ksplit_fragment_v4.py (9e63ad1f). The wide variant ksplit_fragment_v4w.py (28871d87) is for S26 WITH group 5
only (lead). Both are rounding-level, not bit-exact. S26 order: base (S24 + j3 + R320) + k-split + group 5 if in + treuse LAST.
- **What it does.** It wraps engineer2's altbasis _packed_read5 without editing it; --d6x keeps reads at depth 5 for padding, region
  fit and CPU, not accuracy. A bank read with k >= 4096 is split when S24's cost rule says it pays: bill + 750k flops per request, with
  fitted bill models. m and n are zero-padded to multiples of 64, and k is cut into balanced chunks that fit the depth-6 regions.
  Each chunk's depth-6 leaf product is summed into the output tree through a leaf-sized buffer after the output tree in the C region.
  ONE unwind forms the output (v2 design; v1 unwound every chunk).
- **Prebind (v4).** Bank-read m, k and n vary per net and moved between S23 and S24. v3's S23-derived shape and drop lists covered
  only 4 of net 35's 13 split shapes (porter's ks_shape_check). In v4 the chunk keys come from the porter's census-derived S26 list
  under the standard rule (keys used by >= 2 of the 8 selection nets 300..580), bound in phase 1 and fused in phase 2. Phase 3 binds
  only the 22 leaf buffers, keyed by (M, N), from a 12-net shape census.
- **S26-only setup settings.** _PB_WALL_S = 3.5 s (lead) and _PB_BATCHED = () (porter: the batched prebinds serve no census net).
- **Coverage study** (e74_shape_census_sel8plus4.json: nets 300..580 + 35/100/37/39, 391 reads, 11-19 splits per net). Fixed-size
  chunks do not beat balanced ones: 83% vs 85% of held-out chunk keys covered by a 2-of-8 list. About 15% of chunk keys stay
  net-specific and bind lazily, about 0.16% of the bill per net. The baseline pays the same on its (larger) depth-5 read keys.
- **v2 cost model.** One depth-6 unwind plus its output reshape costs exactly 58.88 flops per M x N element on all 13 shapes. The v2
  bill = v1's fitted model - (nch - 1) x (58.88 MN + mn - leaf size) + mn if the output is padded. That is within 3 Mflop of 23
  measured v2 bills. Requests = 2 per padded operand + 47 per chunk + (nch - 1) + 36 + 2 if padded, exact on 23 shapes. v4w prices
  its rule with this model.

| gate | result |
|---|---|
| product bench, OpenBLAS, 13 shapes (e74_ksprod_bench.py) | v1 -> v2: bill -2.54 -> -2.81 GF, requests +1,449 -> +897, backend x1.75 -> x1.68 |
| parity v2, in process, nets 35/100/37/39 (e74_parity.py) | bill -0.72/-1.16/-0.76/-0.94%; final-MSE ratio 1.0020/1.0021/0.9970/1.0023 (mean +0.09%, rounding-level); layer-15 max diff ~1e-5 of RMS |
| parity v4 with an emulated S26 list (S24's + chunk keys used by >= 2 of 8 nets) (e74_parity_v2.py) | bill -0.89/-1.07/-0.76/-0.99% (mean -0.93%); ops +948/+987/+809/+1,066; MSE as v2 |
| parity v4w, same list rule | bill -1.04/-1.22/-0.89/-1.19% (mean -1.08%); ops +1,222/+1,242/+1,138/+1,725; MSE mean +0.10% |
| user-CPU pair v2, OpenBLAS, concurrent, 6 nets | user CPU +5.15% (se 0.35%), bill -0.96%, op calls +1,065/net |
| wall, project_wall_v3 on S24 #332255 (expected worst 70.7 s) | 73.0 / **73.7** / 74.3 s; room 26.3 s under 100 s |
| request/setup/handle census v2, 12 nets, sequential arms | requests +1,262/net (+827..+1,850; v3-era lists); bill -0.88%; setup 1.683 -> 1.808 s; max handles 30,889 -> 31,607; live-array peak unchanged 2,436 MB |
| setup under heavy load (10 hogs, load ~10, both arms concurrent, setup-only client) | S24+E-73 3.006 s (guard 3.0; 362 bound, 115 fused); v4 **3.507 s <= 3.6** (guard 3.5; 419 bound, 90 fused, 0 of 22 leaf buffers) |
| residual pair v4 (emulated S26 list), concurrent, 3 reps, nets 35 cold/100/37 (gates_e71c.sh; residual_gate_e74d.txt) | cold **+0.005 s** (rule <= +0.010) PASS; r100 +0.007, r37 +0.001 s; requests +1,087/+1,049/+877; bill -0.886/-1.066/-0.755%; setup 1.76-1.89 -> 2.14-2.18 s (441 bound = 419 fused + 22 leaf buffers) |

Outcome: all gates PASS (04:06); the all-pass reached the porter only at 08:30 (my miss: S26 idled ~4 h). S26 = S24 + j3cut + v4 +
treuse (group 5 failed its gates, so v4w, which the lead allowed only with group 5, is shelved). Not built: the level-1 unwind pair and
preallocated chunk/pad buffers from the lead's first GO; v2's single unwind removed the chunk outputs, and the rest was worth ~-0.2k
requests and ~0.015% bill for +67-245 MB of server memory. S26's VALID40 reports raw and adjusted separately (lead).

## K-split measured under OpenBLAS: bill -0.96%, class backend x1.6-1.8, wall +6-7 s; package prices at +9 ms residual vs S25 (E-72d, 2026-09-25)

Lead's GO: bench the chunked depth-6 path to pin E-72c's CPU band, then price the class and the S26 package (k-split + engineer2's
group 5 + top-level fusion + E-73 + j3). e72d_ksplit_bench.py and e72d_ksplit_bench_v2.py run every class, all 24 (25 products per net,
net 100), in process on candS24_treuse2_mem.py after setup. Each class runs at depth 5 as today, and split: m and n padded to
multiples of 64 the way S24 pads (zeros plus copyto, billed), k in balanced chunks that fit S24's depth-6 regions (at most 4096),
one depth-6 tree per chunk, then the chunk outputs added (nch - 1 adds). Three alternating reps each. The transform-reuse key is
cleared before every call. v2 lifts the tree-cache caps and rebinds the class's depth-5 keys fused, as S24's setup leaves them; v1
let the caches evict them.

**The BLAS environment decides the CPU answer.** The .venv numpy is Accelerate. The platform-calibrated instrument is the OpenBLAS
single-thread env: the wall law's 0.78 factor was fitted on it. E-72 and E-72c used Accelerate. Accelerate runs the big depth-5 leaf
matmuls fast and the tiny depth-6 leaves slowly. Under OpenBLAS the depth-5 products are matmul-bound and depth 6 costs far less extra.

| per net, the 24 classes | depth 5 | split, chunk keys fused | split, chunk keys unfused |
|---|---|---|---|
| bill, measured incl. padding | 77.31 GF | 74.34 GF: -2.97 GF = **-0.96%** of S24's 308.9 GF | same |
| requests | 1,800 | 4,435 (+2,635) | 5,519 (+3,719) |
| backend ratio vs depth 5, OpenBLAS | 1 | x1.79 | x1.63 (v1 x1.63 at load ~4) |
| backend, quiet-run base 3.80 s | 3.80 s | ~6.8 s (+3.0 s) | ~6.2 s (+2.4 s) |
| backend, Accelerate (what E-72c extrapolated) | 2.63 s | | 10.07 s (x3.8) |

- **E-72c's CPU band is superseded.** Its 3-5x came from Accelerate rates. Under OpenBLAS the class grows 1.6-1.8x, which is
  +9.5..+11.8% of S24's ~25.4 s of backend per net.
- **Depth 7 re-checked under OpenBLAS** on S24 with E-72's shapes: backend +110% (1024^3), +109%, +151%, +86%, +38%. E-72's verdict
  stands.
- **Accuracy:** the split's error vs float64 is 3.4-5.4e-5, against 2.8-3.9e-5 at depth 5. That is rounding level.
- **Wall.** This uses engineer2's project_wall_v3 law on S24 #332255: expected worst 70.7 s, worst-MLP kernel 63.75 s, r
  0.60/0.78/0.95, 0.30/0.41/0.50 ms per op. Fused chunk keys: +5.3 / **+6.9** / +8.4 s. Unfused: +4.7 / **+6.2** / +7.6 s.
- **Setup.** 50 new chunk keys come in, and 29 prebound depth-5 keys used only by this class drop out: +21 on S24's 359, about +0.1 s
  locally. On the platform (S24 max 2.88 s) the slowest workers reach the 3.0 s guard. There the last chunk keys bind lazily,
  unfused (+22 requests per chunk).
- **Top-level fusion (a) is already in S24.** lvfuse2 fuses levels 1..L of both forward transforms (359 of 362 keys fused in S24's
  setup), so E-71's -0.75k is spent. What remains at the outermost axis is one pair in the level-1 unwind: top-right and bottom-right
  as one op over out's row halves. That is -1 request per tree product, about -0.18k per net. Not built.

| S26 package | bill | requests/net | wall, central |
|---|---|---|---|
| k-split, chunk keys fused, chunk and pad buffers preallocated (-~80) | -0.96% | +2.55k | +6.9 s |
| engineer2 group 5 | -0.016% | -0.97k | -0.4 s |
| level-1 unwind pair (new, not built) | 0 | -0.18k | -0.1 s |
| **vs S25** | **-0.98%** | **+1.40k = ~+9 ms** | **+6.4 s** |
| E-73 + j3 cut (already in S25) | -0.18% | -0.62k | -0.3 s |
| **vs S24** | **-1.16%** | **+0.78k = ~+5 ms** | **+6.1 s -> ~77 s expected worst** |

With unfused chunk keys the package is +1.17k requests worse (+7.6 ms) and 0.7 s lighter on wall. Other request levers: re-partition
the depth-6 regions so 928 x 8128 x 992 and 832 x 8192 x 928 take two chunks (-0.17k), and E-71 group 3, batching same-shape
products (-0.5..-0.7k on S24, not built).

## Depth-5 bank read/push products split along k to depth 6: -1.0% of the bill, 3-5x their backend (E-72c, 2026-09-25; paper, closes on CPU)

SUPERSEDED IN PART by E-72d above: the bill numbers stand, but the CPU band (3-5x) was an Accelerate artifact. Under OpenBLAS it is x1.6-1.8.

Lead's question: what share of S23's bill are the depth-5 bank read/push products with inner dims of 6-8k (928 x 8128 x 992 and the
like)? What would a k-split into ~4k chunks at depth 6, with m and n padded to multiples of 64, net in bill and in CPU?
e72c_ksplit_price.py prices it from the net-100 product census and the E-72 bench logs. Nothing new was run on the kernel.

- **Bill model.** Each depth's transform coefficient is fitted on the measured bills: 18.36 per element of (mk + kn + mn) at depth 5
  (4 products), 30.60 at depth 6 (6 products). All ten measured bills match within 0.1%.
- **Chunks.** Each chunk is the largest multiple of 64, up to 4096, that fits S24's depth-6 regions. The depth-6 A region (90.4M
  floats) caps the chunk at 3072-3776 when m is above 768. The last chunk takes the remainder, rounded up to 64. The chunk outputs are
  added: (chunks - 1) x m'n' adds. Lifting the cap to 8192 changes the bill by 0.01%.

| per net (net 100) | products | depth-5 bill | share of net bill | split to depth 6 | requests | local backend, extrapolated |
|---|---|---|---|---|---|---|
| inner dim >= 4096 | 25 | 77.3 GF | 24.9% | -3.15 GF = **-1.02%** | +3.2k | 0.86 s -> 2.6-4.5 s (+1.7..+3.6 s) |
| lead's 6-8k (k >= 5900) | 19 | 67.9 GF | 21.9% | -2.75 GF = -0.89% | +2.5k | 0.75 s -> 2.3-3.9 s (+1.5..+3.2 s) |

- **Padding.** 16 of the 25 products need padding, and it eats much of their depth-6 saving. 928 x 8128 x 992 saves 2.8% where an
  unpadded depth 6 would save 9.5%. 544 x 6272 x 800 gets 1.2% worse. The pad-in and unpad copies are not in the table. At one
  flop per element they cost another 0.04% of the net bill, so the net is about **-0.98%**.
- **CPU.** This is an extrapolation, not a measurement. The four big depth-5 products run 0.011 s of backend per billed GFLOP. The
  measured depth-6 trees run 0.035-0.060 s per billed GFLOP, and up to 0.11 for narrow shapes, because their leaves are tiny
  (7^6 blocks). So the class's backend grows 3-5x: +1.7..+3.6 s per net, which is +11..+23% of the ~16 s of local backend per net that E-72 measured on S23.
- **Wall.** The backend pair maps about 1:1 onto platform kernel time, plus 0.41 ms per extra request. That projects +8..+16 s on
  S24's worst MLP (72.3 s -> ~80-89 s), under the 100 s rule.
- **Verdict.** The bill gain sits at the -1% line, but its CPU price is 2-4x worse per CPU % than m64 (-0.91% flops for +5.6% CPU on
  the tree kernel). It closes on CPU. It could only reopen as a deliberate spend of ~10-15 s of the ~28 s wall headroom. In that case
  the first step is a chunked depth-6 bench on the four biggest products (e72_pad6_bench.py with a k loop), to pin the 2x-wide CPU
  band before anything is built.

## Transform reuse on S24: bit-identical, bill -0.09%, compute-neutral (E-73 = E-71 group (b), 2026-09-25; recommended)

Build: treuse_fragment_v2.py on the fired S24 (30810da3) = candS24_treuse2_mem.py (32babb65). The tree kernel's forward-A transform
of an operand stays in the arena's A region until the next forward-A. When the very next forward-A has the SAME operand object and
the same (L, m, k), the transform is skipped and the resident tree returned. This covers the two column packets of a push, and
products in the covariance chain that share a left factor.

**Proof** (e73_reuse_audit.py, S24 in process, where flopscope arrays are numpy arrays, nets 35/100/37/39): at every such event (17 /
21 / 22 / 16) the normal recomputation left the A region bit-for-bit unchanged. So skipping it cannot change any value, and it also
covers an operand modified in place. Between the two calls only the previous product's B side and unwind, and pure dispatchers, ran.
**Runtime guard** (conservative): the resident key is cleared by every arena-touching function except the tree views and the audited
dispatchers, by every alias that reaches an older kernel path (frames, the old tree kernel, batched and read paths), and at every
prediction start. v1 also cleared on _D6X_OLD_PPC and refused 11-16 audited reuses per net. That alias holds the ab-era dispatcher
(ab tree kernel, else _AB_OLD_PPC, which still clears), so v2 allows it and reuses exactly the audited events.

| gate | result |
|---|---|
| bit-identity, in process, setup() first, 8 nets (e73_parity.py) | all 16 layers equal (max diff 0.0) on 35/100/37/39/42/48/59/63; reused 15-22 transforms per net |
| bill | -0.073..-0.113% per net, mean -0.09% (-224..-351 MFLOP) |
| user-CPU pair, setup-calling, 6 nets | user -0.02%, backend -0.02% pooled (per net within +-1.7%): neutral |
| paired residual gate vs S24, concurrent, 3 reps | residual -2 / -5 / -7 ms (r35 cold / r100 / r37), PASS; requests -376 / -463 / -483; server CPU 0 / -1% |

The lead's -0.3% estimate assumed costlier transforms. The measured saving is -0.09% of the bill and about -0.45k requests per net.

## One more Winograd level on the big products: -0.55% of the bill at most, +22% server backend (E-72, 2026-09-25; not recommended)

Lead's ask: price depth 7 (or 6 -> 7 wherever dims divide by 128) on the products that dominate the bill, with honest server CPU.
e72_depth7_bench.py runs S23's alt-basis tree kernel (_ipt_product_ab) in-process on random float32 operands at depth 6 and depth 7.
The depth-7 region is sized per class inside S23's arena of 391.6M floats. Steady-state numbers (second call):

| product (m, k, n) | per net | bill at depth 6 -> 7 | ops | backend time | product rel. error |
|---|---|---|---|---|---|
| 1024^3 | 18 | 1.060 -> 1.009 G (-4.9%) | 105 -> 120 | 0.06 -> 0.12 s (+90%) | 2.8e-5 -> 6.1e-5 (x2.2) |
| 512 x 1024 x 1024 | 14 | -2.6% | +15 | +114% | x2.2 |
| 512 x 1024 x 512 | 17 | -0.5% | +15 | +80% | x2.3 |
| 1024 x 1024 x 2048 | 2 | -6.1% (-126M) | +15 | +106% | x2.1 |
| 896 x 384 x 4096 | 6 | -2.5% | +15 | +90% | x2.3 |
| 1024 x 1024 x 4096 (colpack) | 3 | does not fit the arena (474M floats) | | | |

The measurements check engineer2's bill model extended one level to within 3%: bill(7) - bill(6) = -0.112 mkn + 22.0 (mk + kn) +
22.3 mn. Summed over every S23 product class whose dims divide by 128 and whose depth-7 trees fit (63 products per net, r100):
**-1.70 GFLOP = -0.55% of the net's bill**, +3.5 s of local backend per net (about +22%), +945 requests, and about twice the
rounding error on those products. The 1024^3 class alone: -0.30% for +1.1 s. Both backend and matmul time double because depth-7
leaves are 8 x 8 blocks (7^7 of them) and the trees are (7/4)^7 = 50x the operands. The largest depth-5 products (inner dims
6-8k: 928 x 8128 x 992 and the like) cannot move even to depth 6 by padding: their padded depth-6 A tree needs ~224M floats against
the 90M region (e72_pad6_bench.py). They would need inner-dim splitting, which adds accumulation ops and CPU. Verdict: far below the
-2% bar. At this arena size the kernel's depth lever is spent on S23's product mix.

## E-71 group 1 built and gated: bit-identical, -3.0k requests, but +7.9% server CPU (E-71b, 2026-09-25; NOT recommended)

Build: lvfuse_fragment_v2.py on S23 9247a212 = candS23_lvfuse2_mem.py (89762ed3). Each tree level runs its component-pair ops as one
strided op (A 3 -> 2, B 3 -> 2, unwind 6 -> 4 per level). The fused op tuples live in the tree cache entry's slot 3. They are built
in setup by a TWO-PHASE prebind: phase 1 is S23's prebind exactly, phase 2 fuses the bound keys, most frequent first, with the
guard's leftover time. So prebind coverage never drops below S23's. (v1 7f031040 fused while binding; v3 47fe9755 leaves the unwind
unfused.)

| gate | result |
|---|---|
| bit-identity, in-process, setup() first, r35/100/37/39 (e71_parity.py) | all 16 layers array-equal (max diff 0.0), flops identical, 359 keys fused (v1 and v2) |
| request census, 12 nets, setup client (run_mem_setup_pair_e71.sh, v1) | 31,532 -> 28,534 per net = -3.0k (-9.5%), per net -2.7..-3.3k; util identical; setup 1.28 -> 1.83 s; handles 28.5k -> 30.9k |
| paired residual, concurrent pairs, 3 reps, both arms fully prebound (gates_e71c.sh, guard 10 s in both measurement builds) | residual -6 / -6 / -15 ms (r35 cold / r100 / r37), cold rule PASS; server CPU +13 / +11 / +8% |
| setup-calling user-CPU pair, 6 nets (run_usercpu_pair_e71.sh) | user CPU +7.9% pooled, backend +18.6%: adds 22.3 -> 37.5 s for 16% fewer calls, subtracts 14.8 -> 18.9 s |

Why: a plain-numpy micro-benchmark on a depth-6 level array shows each fused pair op takes 1.5-3x the time of the two
single-component ops it replaces, whenever the component axis is not the array's outermost axis. At the outermost axis it takes the
same time. Numpy iterates and buffers interleaved strided views much worse than single-index views. So in-product fusion is free only
at the top level (about -0.75k requests per net). With the team's wall law (platform kernel % = 0.78 x local user % + 0.41 ms/op) the
build costs about +1.3-1.5 s of platform wall per MLP for about -9 ms of residual. The residual gain is also half the 6.5 us/request
estimate: these tiny level requests are cheap on the client.

Under concurrent load (the first paired gate, load 10-15) setup slowed ~3x. Both arms hit the 3.0 s guard: S23 bound 131 of 362
keys, v2 bound 142 and fused none. The two-phase design held: never below S23's coverage.

Compute-safe remainder of the priced list: top-level-only fusion about -0.75k; group 2, transform reuse, about -0.6k with bill down to
-0.3% (needs an arena-overlap proof); group 3, batching same-shape products on a batch axis, about -0.7k. That is about -2.1k (-6.6%)
in my lane, short of the 15% target; engineer2's body diet adds about -1..-1.5k.

## Request diet on S23: census and priced groups (E-71, 2026-09-25; paper + census, nothing built)

Lead: residual is the binding budget, and requests are the lever we control (~6-7 ms of residual per 1k requests). Target at least
15% fewer requests per net, bill neutral or better. Census at the production boundary (numpy-free client + flopscope server, setup
then nets, S23 = porter stack_S23_mem.py 9247a212): request_census_e71.py decodes every request's op and attributes it to its call
site; product_census_e71.py logs every tree-kernel side call (layer, physics site, depth, dims, operand ids, requests inside).

| S23 per net | requests | free trips | flops |
|---|---|---|---|
| r35 (first net after setup) | 32,153 | 2,694 | 310.7 G |
| r100 (second net) | 31,741 | 2,655 | 310.4 G |

By op (r100): add 9,397 (29.6%), subtract 7,918 (24.9%), multiply 3,413 (10.8%), getitem 2,049, unstack 1,886, true_divide 860,
copyto 832, reshape 717, matmul 569, astype 428, transpose 403, sum 292. The tree kernel runs 188 products per net, all single: S23's
s1 = 0 left the covariance product C32 @ w without its batch partner. They make 546 side calls (170 A, 188 B, 188 unwinds; 18 A
transforms are already reused) and 19.4k requests (61%): about 30 per A side, 30 per B side, 46 per unwind. Depths: 100 products at
6 levels, 75 at 5, 11 at 4, 2 at 3. The physics body makes the other ~12.6k (39%): _predict_binary 4.0k, _bd_update 1.0k,
_gn_dcov 0.7k, features 0.5k, derivatives 0.3k, gnn_step 0.2k and a long tail.

Priced groups (r100, requests per net; a getitem view bills 0 flops, a reshape bills its element count, a paired op bills its
elements = the two separate ops, all measured in flopscope):

| # | group | requests saved | bill | status |
|---|---|---|---|---|
| 1 | level-op fusion in the tree kernel: each level's ops on component pairs run as one strided op. Forward A (X3, X1) = (X2, X6) - X5; forward B (Y6, Y4) = (Y5, Y3) - (Y2, Y5); unwind (Z5, Z0) = Z0 + (Z5, Z2) and (Z2, Z3) = Z5 + (Z3, Z1). Ops per level 3 -> 2, 3 -> 2, 6 -> 4 | -3.6k gross; -3.1k with the pair views bound for the setup-prebound keys and for other keys at their 3rd use (binding them for every key: -2.4k) | 0 | bit-identical: same per-element arithmetic. Flopscope handles the negative-step and overlapping pair views correctly (checked) |
| 2 | reuse the forward-A transform of a left operand repeated within a layer (23 operands, 30 redundant transforms) | -0.7k | -0.3% if the reusing products run back to back (TA still in the arena); ~0 with a copy (a copy of TA bills ~30M flops, the transform ~40M) | bit-identical; needs per-site reordering |
| 3 | batch same-shape products within a layer (15 groups, 19 extra products; many are the column packets of one wide product sharing A) | -0.6k after group 1 (-0.8k before) | 0 | needs a shared-A x batched-B kernel variant |
| 4 | level-1 load: the two copies per load as one strided copy; unwind level 1: (tr, br) = Z5 + (Z3, Z1) | -0.4k and -0.2k | 0 | bit-identical; layout-dependent |
| 5 | physics body (engineer2's lane): per-unit (1024-vector) elementwise chains, select() at 7 requests x 46 calls | est. -1..-1.5k (15-20% of the body) | 0 | engineer2 |
| - | per-call operand and output views (reshape + unstacks) | 0 | - | dead: operands are fresh arrays (140 distinct left operands in 170 calls, 186 in 188 on the right) |
| - | alt-basis basis passes (L per side per call, ~2.4k) | 0 | - | dead: sequential passes along different axes |

Total in my lane (1-4): about -5.0k per warm net (-15.6%) at zero or negative bill; with engineer2's body diet about -6..-6.5k
(-19..-20%). Group 1 alone is -9.7%. At ~6-7 ms per 1k requests that is about -20 ms of residual per net for group 1 and about
-33 ms for groups 1-4. Group 1's pair views cost ~22 view requests per tree key. If they were bound in setup for all 359 prebound
keys (~7.9k requests, ~0.6-0.8 s) setup would pass its 3.0 s guard. Binding them at a key's first predict use instead costs each
worker's first MLP about +7.9k requests once.

## MPS eval mode for research grids + FIFO MPS queue (E-70, 2026-09-24)

Lead: accuracy grids on the torch cards (analyst k34_spec, covhunter k3+k4 pricing) are the bottleneck on the largest lever; give them
MPS float32 evals with the E-68 fixes, a one-job FIFO queue, and a parity check on their own rows.

**Drop-in (mps_card.py).** For any card a script builds itself (a covstep.HookCard: s6bcard, s6bcard_joint2, dial modules, any hooks):
build it and install everything under mc.on_mps() with mc.to_mps(payload), then mc.bind(cd, w) (the net's float64 CPU mean-path
islands + pair-form k3), then mc.run(cd) -> (16, 1024) float64 on CPU. Research code needs no other change: the float32 mode now also
turns x.numpy() on MPS tensors into a CPU copy, keeps seeded CPU-generator draws on CPU in their requested dtype (the same numbers as the
CPU run), maps x.double() and new_tensor(float64 numpy) to float32, and sets up its own import paths. bind() builds its float64 weights
under the island bypass (called inside on_mps() it had produced MPS float32 weights: run 2 crash, kept).
**Queue (mps_lock.py).** FIFO tickets in /tmp/fable_mps_queue; the oldest ticket whose pid is alive (same start time) holds MPS; dead
tickets are ignored, never deleted by others. `mps_lock.py run <label> -- <cmd>`, `with mps_lock.hold(label)`, `mps_lock.py status`.
refit_mps_v3.py queues itself. Launch order: machine_gate.sh (memory), then the queue.

**Parity on their own rows** (CPU float64 in the same process reproduces every logged row exactly):

| script, card, net | rows | MPS vs logged, per row | effect vs base: MPS vs logged | forward s, CPU 3 thr / MPS |
|---|---|---|---|---|
| covhunter k34_supply_price (joint2 + round-1 dials + lrstar + gn_linear + ps, loopR1), r35 | 10 | -0.031..+0.011% | <= 0.034 pt (effects -12.6..+1.1%) | 6.0-7.6 / 1.6-4.4 |
| same, r100 | 10 | -0.024..+0.023% | <= 0.029 pt (effects -10.3..+3.9%) | 5.8-7.7 / 1.6-3.5 |
| analyst k34_spec (s6bcard + B2 ep1, their run() exec'd unmodified), r150 | 7 | -0.040..+0.030% | fraction left <= 0.0004 | 6.6-7.4 / 1.6-3.3 |
| covhunter's trusted r150 rows (his pick: base, ctrl_s.5, L12-14 eps1 s.5, L2-14 eps.5 s.5; 8M truth) | 4 | -0.011..+0.016% | <= 0.017 pt (effects -19.4..+1.5%) | 5.3-5.8 / 1.4-3.3 |
| covhunter's double check (s6bcard + B2 ep1, no hooks; bit-exact in his and the analyst's code), r150 | 2 | +0.0014%, +0.0026% | - | - |

Covered: truth k3/k4 injections (eps, band, all-units), learned-correction scaling, noise (seeded draws identical), top-K units by
adjoint / random / fake scores, layer bands. The analyst's adjoint mode on MPS: gradients at L12-14 within 2.3-2.9e-3 of their CPU file,
top-128 selections identical (128/128 at each layer). Memory: the per-unit truths are 1 MB per net, second-order truths 134 MB per net,
vs the MPS pool of 6-8 GB on a 48 GB box. Scripts: e70_parity_k34.py, e70_parity_analyst.py (their files are never written: the
analyst's adjoint output went to e70_analyst_parity/).

## Torch mirror of S23's Nystrom old-tier join, certified against native (E-69, 2026-09-24)

Lead: S23 deploys s1 = 0 plus the porter's Nystrom join (PT-42 build_join_nystrom.py: at the L7/10/13 joins the test matrix
Omega = W[live, :384] becomes the 384 live rows of largest Gram diagonal); the torch card must mirror it before the next refit.
join_nystrom_torch.py: install(card, torch) swaps the card's push for s2card.make_push with that join block (same math as the porter's
torch arm C in join_sketch_test.py: d = (J*J) w [+ rowsum((Qp core) * Qp)], rows = sort(stable argsort(-d)[:384]),
Y = J (w o J[rows]^T) [+ Qp core Qp[rows]^T]; QR, factor and core updates unchanged). It asserts the plain push (fine_ckpt=False).
Certificate (e69_native_preds.py in the .venv with flopscope, the porter's in-process recipe; e69_join_cert.py): native S23 (9247a212)
and S23pre (the same build without the join) vs the torch loop card configured as S23 (s6bcard_loop3 = E-67 + s1-zero star; round-1
dials, equal to the porter's s21_dials_used_round1.json; loopR1 params), with and without the drop-in, CPU float64 and MPS float32:

| net | join effect on final MSE: native / torch CPU / torch MPS | torch vs native final MSE (with join) | L15 prediction gap to native, with / without join |
|---|---|---|---|
| 35 | +0.103% / +0.093% / +0.096% | CPU -0.047%, MPS -0.039% | 6.6e-6 / 6.8e-6 |
| 100 | +1.958% / +1.941% / +1.975% | CPU +0.647%, MPS +0.610% | 6.4e-6 / 6.7e-6 |

The mirror reproduces the native join's effect within 0.03 pt and adds no torch-native gap (the r100 +0.65% gap is the card's own,
present without the join: +0.664%). Native flops: S23 -1.8% / -1.9% vs S23pre. MPS vs CPU (with join): L15 2.7-2.9e-7.

## Float32 refit card on MPS: 3.5-3.8x faster steps, certified (E-68, 2026-09-24)

Lead's task: the loop's clock is the refit epoch. Can the loop card train in float32 on MPS? Stop rules: (a) CPU float32 vs float64,
final MSE <= 0.05% and median gradient diff <= ~1e-2; (b) MPS step <= 1/3 of the CPU drop-in's; (c) 20 updates from one start, same
batches: train-loss trajectories match and VALID16 within +-0.1%. Card for all tests: the physicist's s6bcard_loop with covhunter's
live round-1 dials (s6bcard_jfitR1, joint2) + the E-67 drop-in (fine_ckpt=False + pair_series loop form), params j05 ep2.

**How the card runs in float32 without editing it.** A torch function mode (e68_f32_cpu.py; on MPS mps_card.MpsF32Mode together with
torch.device('mps')) rewrites every float64 dtype request to float32 and casts float64 results to float32; on MPS, CPU tensors in a
mixed-device call move to MPS. Five traps, each hit once and fixed:
- Tensor.backward is intercepted by the mode and runs with the mode popped, so the checkpoint recompute came back float64: call
  torch.autograd.backward(loss) inside the mode (it is not intercepted), with the float64 loss formed outside.
- torch._C.DisableTorchFunction did not hold inside the checkpoint recompute on the MPS autograd worker thread: islands use a bypass
  counter that the mode checks.
- Casting float64 INPUTS also rewrote property reads (x.dtype) and broke checkpoint's metadata check on float64 island tensors: only
  CPU->MPS moves cast; torch.Size must stay a Size.
- torch._dynamo must be imported before entering torch.device('mps') (checkpoint imports it lazily); MPS tensors cannot .to('cpu',
  float64) in one call (move, then cast).
- Caching island weights by MPS storage address served the previous net's weights when the allocator reused the address (net 37 final
  MSE off by 10^7 relative): each card now binds its own float64 weights from its numpy array.

**(a) Where float32 loses accuracy.** Pure float32 (CPU): final MSE +0.025% (r37) / -0.041% (r39), L15 predictions 1.35e-6 / 1.52e-6
relative, median gradient diff 5.4e-3 / 2.1e-2 (r39 FAILS the gradient gate). The gradient error is ~10^4 x the L15 prediction error:
every gradient is seeded by the residual (prediction - truth, ~1.2e-4 RMS), and a 1.5e-6 relative prediction error (~5e-7 absolute)
is ~0.4% of it, amplified by cancellation across units (E-67's pair series shows the same ratio: 1e-7 predictions, 1e-3 gradients).
Island sweep (each part in float64 inside the float32 card, e68_islands2.py, r39): the step's two mean-path products w64.T @ m and
w64.T @ b cut the L15 error 1.52e-6 -> 4.8e-7; the bank k3 sums (sum_j lam_j p_j^3 over up to 24,832 columns, in +/- pairs) complete
the fix for gradients; the pair map, _bd_update, push, read, features, cov_pm and all hooks each move it <= 30%.
Float32 + the two fixes vs float64:

| net | final MSE | L15 predictions | gradients median / max |
|---|---|---|---|
| 37 (CPU f32, f64 k3 island) | +0.018% | 4.4e-7 | 1.4e-3 / 5.8e-3 |
| 39 (CPU f32, f64 k3 island) | -0.027% | 5.0e-7 | 2.6e-3 / 4.9e-3 |
| 37 (CPU f32, pair-form k3) | +0.027% | 4.2e-7 | 2.3e-3 / 3.3e-3 |
| 39 (CPU f32, pair-form k3) | -0.034% | 5.1e-7 | 3.3e-3 / 5.6e-3 |
| 37 (MPS f32, mps_card) | +0.025% | 4.5e-7 | 1.3e-3 / 5.0e-3 |
| 39 (MPS f32, mps_card) | -0.056% | 5.2e-7 | 5.0e-3 / 1.1e-2 |

The k3 pair form: sum_pairs c (p+ - p-)(p+^2 + p+ p- + p-^2), exact in real arithmetic (each layer appends positives then negatives
with lam = (c, -c); checked exact on all 15 sites); float32 error 0.8-1.2e-7 vs 1.4-2.9e-7 naive (e68_k3pair_test.py). A float64 k3
island on MPS cost 1.4-1.7 s per step (100 MB per call to CPU); the pair form runs on MPS. Selection logs: no unit-set flips between
float32 and float64 (only orders of tied ids). The islands are ~0.5 GFLOP per forward vs ~70 GFLOP for the per-layer covariance
products alone (kill rule was 20%).

**(b) MPS speed** (e68_mps_cert.py; alternating steps, CPU arm = float64 drop-in at 5 threads, machine load 7-9):

| net | CPU step s | MPS step s | MPS / CPU |
|---|---|---|---|
| 37 | 20.7-24.4 | 5.3-7.3 | 0.266 |
| 39 | 22.2-22.5 | 5.4-7.0 | 0.287 |

MPS floor ~5 s (no islands); the mean-path islands cost ~0.2 s; erfc is the only CPU fallback. MPS driver pool peak 6.1-8.1 GB
(allocated 1.6 GB).

**(c) Training certificate** (refit_mps.py = the physicist's refit loop with a device switch; e68c_mps_u20.log, e68c_cpu_u20.log,
e68c_eval.log). 20 updates from j05 ep2 on the jfitR1 loop card, groups full+P, lr 1e-4 / 5e-5, warmups 3 / 5, clip 1, loss x 1e8/4,
TRAIN160 order from rng(0), MPS float32 (mps_card) vs CPU float64 (both with the E-67 drop-in):
- Per-update batch loss, MPS vs CPU: max |diff| 0.021%, mean -0.002%, no drift (first 5 updates <= 0.015%, last 5 <= 0.021%); per net
  <= 0.08%; gradient norm <= 2.5% (typically < 0.6%).
- VALID16 on the CPU float64 card: start 1.667714e-8; after 20 MPS updates 1.651960e-8 (-0.945%); after 20 CPU updates 1.652000e-8
  (-0.943%). MPS-trained vs CPU-trained -0.0024% pooled, per net <= 0.049%; both improve 13/16 nets. PASS (bar +-0.1%).
- The MPS card's own VALID16 at the start params vs the CPU float64 card: +0.005% pooled, per net <= 0.081% (median 0.021%).
- Wall for 20 updates: MPS 466 s, CPU 1454 s (0.32; CPU at 5 threads, load 9-10). The physicist's live round-1 run (CPU, card without
  E-67) took 1442 s for the same 20 updates, and its batch losses match the MPS run within 0.034% per update.
Projected loop refit epoch (40 updates + VALID16 before/after): ~16-17 min on MPS vs ~50 min on CPU.

**Adoption.** refit_mps_v2.py = the physicist's refit_s6b.py (valid_before, per-epoch VALID and params, SPIKE line) with --device
mps|cpu on the loop card + E-67 drop-in; LOOP_DIALMOD required. One MPS job at a time; launch behind the machine gate.
refit_mps_v3.py = v2 + --card (default s6bcard_loop3 = the physicist's loopR2 card: E-67 drop-in + the s1-zero star,
lrstar_s1zero_hook.py). Certificate on s6bcard_loop3 (e68_mps_cert_v2.py, jfitR1 dials, MPS float32 vs CPU float64): r39 final MSE
-0.040%, L15 4.6e-7, gradients 2.6e-3 / 5.1e-3, step 7.0 vs 21.9 s (0.32); r42 (the E-67 knife-edge net) +0.031%, L15 3.9e-7 (no flip),
gradients 1.0e-3 / 3.4e-3, step 7.2 vs 26.0 s (0.28).

**E-67 addendum (VALID16 scale, e67_ps_valid16.py).** At j05 ep2 on the jfitR1 loop card, the pair series moves VALID16 from
1.666309e-8 (bit-equal to the physicist's own loopR1 valid_before) to 1.667714e-8 = +0.084% pooled. 15 nets move <= 0.04%; net 42 moves
+1.18%: a knife-edge flip, not rounding. Same-card comparisons are unaffected; valid_before numbers across the switch shift by that amount.

## Refit step on the loop card: no nested checkpoints + loop-form pair series = -18% (E-67, 2026-09-24)

Re-profile of one refit step on the physicist's loop card (fable_physicist_v1/refit6b/s6bcard_loop.py: covhunter's joint2 dial module +
the S18 low-rank star hook + gn_linear_hook; fine_ckpt as the harness passes it; groups full+P; final-MSE loss), net 37, 3 threads
(loop_step_profile.py, loop_fn_bwd_probe.py, loop_mem_probe*.py).

| part of the 24.4 s step | seconds | how measured |
|---|---|---|
| forward (per-layer checkpoints: nothing saved) | 5.7 | step timer |
| per-layer checkpoint recompute in the backward | ~5.6 | with-grad forward of every layer (probe) |
| nested fine-checkpoint recompute (pair map, _bd_update, star, push, read) | ~1.5-2.5 | fine_ckpt on vs off |
| gradient arithmetic | ~11 | remainder; flat: 22k elementwise muls, none above ~5% |

Backward by function (isolated replays, forward / backward seconds per net): pair map 1.29 / 4.45, push 0.37 / 1.04, _bd_update
0.41 / 0.86, read 0.28 / 0.94, message edge 0.46 / 0.60, features 0.14 / 0.46, cov_pm hook 0.18 / 0.42, star 0.19 / 0.25, gnn 0.07 / 0.24.
The pair map is the bottleneck: it runs three times per step (forward, layer recompute, fine recompute) plus a backward of ~3.5x its
forward, about 8.3 s = 34% of the step. Its float32 (3,3,n,n) broadcast products make autograd's backward heavy.

**Nested fine checkpoints buy nothing.** Peak RSS of a fresh process (load + 2 steps): 8.38 GB with them, 8.39 GB without (the peak is
set by one layer's working set either way). Gradients are bit-identical (23 tensors, max abs diff 0, loss equal) on both certificate
cards. They cost 7-13% of the step.

**Dropping the per-layer checkpoint does not fit.** Saved activations per layer are 0.35-2.1 GB (23.8 GB for all 16 layers without fine
checkpoints, 11.8 GB with them), and each GB held saves only ~0.2-0.4 s of recompute. Not safe on the shared 48 GB machine.

**The drop-in (arm C): fine_ckpt=False + pair_series_torch.install (E-64b).**

| card, net, threads | C / A step | B / A step (fine off only) | predictions C vs A | final MSE C vs A | gradients C vs A |
|---|---|---|---|---|---|
| s6bcard_jfitON2 (joint2), r37, 3 | 0.820 (4 pairs: .814/.825/.869/.773) | 0.93 median (8 pairs) | 1.04e-7 (L15 2.3e-7) | -0.012% | median 1.05e-3, max 8.9e-3 |
| s6bcard_jfitR1 (live loop module), r39, 5 | 0.817 (3 rounds: .764/.851/.844) | 0.870 | 9.8e-8 (L15 2.5e-7) | -0.003% | median 1.28e-3 |

Gradient floor for comparison (loop_grad_detail.py, R1 card, r39): the same arm A at 3 instead of 5 threads differs by 1.1e-4..8e-4 per
tensor; C differs by 7e-4..2.9e-3 (the gn_linear_hook level). One outlier: e.rho_scale (a scalar whose gradient, 1.4e-11, is ~1000x below
the other scalars) moves 20% (thread floor 0.3%); its Adam step sign is unchanged. Peak RSS C 8.59 GB (+0.2 GB).
rho_scale check (lead's condition 3): e.rho_scale IS trained (it is in the 'full' group, lr 1e-4; torch Adam eps 1e-8). The harness
multiplies the loss by 1e8/batch and clips the total norm to 1, so Adam sees its gradient far above eps and moves it about lr per step on
either setup; Adam is scale-invariant and the sign agrees, so the 20% change barely alters that step. What makes it immaterial is the loss:
the final MSE's slope in rho_scale is 1.4e-11 per net (MSE 1.4e-8), so even its whole-epoch movement (<= 40 x 1e-4) is worth <= ~4e-6
relative MSE. Condition met; drop-in handed to the physicist and covhunter for their round-2 runs (lead GO 2026-09-24 ~21:00).

Install (in the loop card's card(); pair_series_torch.py is in fable_engineer_v1, already on the card's sys.path):
    cd = DM.card(torch, w, p, a, c, e, fine_ckpt=False, **kw)    # was fine_ckpt=fine_ckpt; the star re-wrap then goes away
    gl.install(cd, torch)
    import pair_series_torch as ps; ps.install(cd, torch)

Not built: a hand-written backward for the pair series (autograd's is ~3.5x forward; a fused analytic one might save ~5% more).

## Bank-read plane rank truncation killed (E-66, 2026-09-24)

The lead's long shot: if the star-plane win meant MSE-relevant plane content is low-rank, the young + old bank READ (about 30% of the
bill) could be computed by a randomized range finder. read_lowrank_s18.py: S18-physics torch card (s6bcard_j05 + S17 literals + the
low-rank star hook), no refit, the porter's 8 VALID nets; the read plane (response rows x live columns, before the pair map) replaced
at every layer by its exact float64 top-r SVD.

| rank r | pooled final MSE change | per net | nets better | plane Frobenius lost (mean) |
|---|---|---|---|---|
| 256 | +140% | +104..+192% | 0/8 | 13.6% |
| 128 | +459% | +367..+554% | 0/8 | 25.7% |
| 64 | +1,005% | +836..+1,314% | 0/8 | 37.4% |

The read plane's spectrum is flat and its content is MSE-critical (the star term is ~3% of the kink-block norm; the bank read ~80%).
Kill bar was r256 <= +1%; no native pricing.

## Computed-but-unused audit of S18 (E-65): native is lean; torch refits run the pair map twice (2026-09-24)

**Method.** liveness_probe_s18.py: on the S18-physics torch card (s6bcard_j05 + S17 literals + the low-rank star hook), net 37, no
checkpointing, a zero-valued probe (requires_grad) is added to every large intermediate; after backward of a random functional of
pred[15] (the scored layer), an exactly-zero probe gradient marks an element nothing downstream reads. Native call sites were then
read against the result.

| output | unused on net 37 | native status | recoverable bill |
|---|---|---|---|
| old-frame push U (1024 x 384) | dead-unit rows, 9-16% | _shared_old384_push pushes all rows (~0.6% of the bill) | about -0.06..-0.1% |
| _bd_update stored / increment | dead rows + unselected columns, 6-18% | elementwise (~0.4% of the bill) | about -0.05% |
| L15 young push | rows outside the row cut, 28% | native _l15_rowcut_k3 already pushes only kept rows | 0 |
| read planes | ~0.1% | response rows x live columns only | 0 |
| covariance push Cpre, star s1, young push pp (L1-14) | fully read (pp's dead rows feed per-unit k3 via the cp diagonal) | | 0 |

No native build is worth a VALID40. **Torch-side find:** gn3_knobs' 'cov_pm' hook (GN v3 knobs iso, k21n, k31) recomputes the WHOLE
pair map at layers 2-14 with the modified K21 / K31 remainder and keeps only its C (13 of the 27 pair-map calls per net). Natively the
same knobs are the exact linear correction _gn_dcov (the series is linear in T21 and T31 and the knobs leave diag(K21) unchanged).

**gn_linear_hook.py (torch drop-in, the native way).** gl.install(card, torch[, th]) replaces the 'cov_pm' hook with C + dcov at
layers 2-14 (row scale read from the card's _ROWF; refuses a k22 knob, which is not linear). gn_linear_cert.py on the S18-physics card:
predictions 1.0e-7 / 9.5e-8 relative (all layers) and 2.3e-7 (L15) on nets 37 / 39, final MSE +0.034% / -0.002%; gradients of the 23
trained tensors 1.2e-3 / 8.6e-4 median, 2.4e-3 / 6.7e-3 max relative (float32 order); alternating refit-style steps (fine_ckpt,
full+P, 3 threads) 30.1 -> 28.0, 35.6 -> 28.7, 36.5 -> 29.0 s: -16% overall, -20% after the first round, mostly in backward.
On covhunter's S19-physics dial card (s6bcard_jfitON2 + lr32; k4 band-scaled through pair_in and dialed c0/c1 through _ROWF), the
drop-in matches the card's own GN hook to 1.1e-7 relative (L15 2.5e-7), final MSE -0.009 / -0.012% on nets 37 / 39
(gn_linear_cert_jfit.py). HANDED to covhunter and the physicist on the lead's GO for all next refits, with the S18 star hook
(covhunter's VALID40-scale check of that hook: +0.18% pooled vs the no-hook j05 rows, native S18 +0.23%).

## Covariance diag+low-rank killed (E-63) and the refit step profiled (E-64), 2026-09-24

**Covariance diag + low-rank (lead's test; cov_lowrank_s18.py).** S18-physics torch card (s6bcard_j05 + S17 literals + the low-rank
star hook), no refit, the porter's 8 VALID nets (37, 39, 42, 48, 59, 63, 66, 67). The covariance leaving every layer 0-14 is replaced by
its rank-r eigen-truncation with the diagonal reset to the exact variances.

| rank r | pooled final MSE change | nets better |
|---|---|---|
| 512 | +10,284% (about 100x) | 0/8 |
| 256 | +46,670% | 0/8 |
| 128 | +183,656% | 0/8 |

The base arm reproduces the certified hook value (net 37: 1.70482e-8). Even half the spectrum carries structure the 15-layer
propagation cannot lose. The lead's second arm, truncation at layers 8-14 only (cov_lowrank_s18_late.py): r512 +393% (about 5x), r256
+3,282%, 0/8 better at both. The bill side was negative anyway: the diagonal part's push W^T D W is still a full weighted Gram.

**Refit step profile (refit_step_profile.py, refit_fn_timing.py).** Mirrors refit6b_aux.py on the S18-physics card, fine_ckpt, groups
full+P, net 37, 3 threads at load 12. Forward without grad 8.7 s, with grad 9.1 s, backward 31.8 s (including checkpoint recompute):
about 41 s per net. Self CPU (38.6 s): elementwise mul 36% (27,439 calls), sum/add/copy/fill/div about 25% more, matmul 6%. By card
function (forward): _nonlinear_pair_batch including _pair_series_fused 41% (no grad) / 47% (with grad), on float64 [3,3,1024,1024]
and [3,1024,1024] tensors: memory-bandwidth-bound. Push, read and star together ~13%. Fusing the pair map (torch.compile on that env
function, float64 kept) is the speed lever (~-25..-30% per step if backward follows forward's split). torch.compile does NOT work on
this machine (anaconda torch 2.7.0, macOS: a minimal float64 elementwise function hung in inductor's compile workers, and died silently
with TORCHINDUCTOR_COMPILE_THREADS=1).

**Loop-form pair series (pair_series_torch.py, optional drop-in).** Replaces the torch card's _pair_series_fused (the zero.py broadcast
form that materializes (6,3,n,n) and eight (3,3,n,n) float32 temporaries) by the same Hermite series accumulated term by term.
pair_series_cert.py on net 37 (S18-physics card): predictions differ by 1.1e-7 relative (all layers), final MSE by +0.011%;
gradients of the 23 trained tensors differ by 8.4e-4 median / 1.5e-3 max relative. A control with both arms original gives exactly 0,
so this is the float32 summation order amplified by cancellation in the loss gradient, not noise. Alternating timing
(pair_series_timing.py; refit-style step, fine_ckpt, full+P): after warm-up 31.2 -> 30.1 s and 33.1 -> 28.3 s, i.e. about -9% +- 5%
per step (load fell 12.5 -> 9 during the run). A modest option for the refit owners, not adopted by me.

## Low-rank star plane for S18: power iterations are essential (E-62, 2026-09-24; S18 #332226 GRADED 2.6705e-9 = NEW BANK)

The star plane A (the K21-plane residual leaving each layer, 1024x1024 float32) feeds only the next layer's star term and the L15
star diagonal. The porter's torch test (s6bcard, B2 ep1, 8 VALID nets, no refit) found exact rank-32 SVD truncation at +0.37% raw
(CI -0.72..+1.56%). Native design (lrstar_fragment_k*.py, lrstar2_*.py): A ~ Q B by a randomized range finder (Omega = a fixed n x k
Gaussian from a seeded generator, drawn once per worker; optional power steps with re-orthonormalization; optional exact rank-r
truncation by a small SVD of B); push = (w^T Q) B and s2 = ((w^T o w^T) B^T)(w^T Q)^T are thin products; s1 = (push o w^T) w stays a
full product and shares ONE batched depth-6 pair with the covariance's C w (same right operand), so 2 of the 4 batched 1024^3
products per layer disappear. flopscope prices (k = 48): a thin 1024x1024xk product 1.0e8 flops, QR 9e6, SVD of B 1.6e7; a dense
1024^3 product 2.15e9.

| build (on S16r3b) | final MSE vs S16: 37 / 39 / 35 / 100 (mean) | bill: 37 cold / 39 second | adjusted |
|---|---|---|---|
| candS16r3b_lr48_mem.py (2037e374): rank-48 sketch, no power steps | +8.54 / +7.44 / +10.92 / +4.51% (+7.85%) | about -6.0% | about +1.4% |
| candS16r3b_k48_q2_r32_mem.py (90e2c18b): 2 power steps, rank 32 | +1.96 / +0.23 / -0.91 / +1.39% (+0.67%) | -4.66 / -4.52% | about -4.0% |
| candS16r3b_k48_q2_rfull_mem.py (b35d9319): 2 power steps, rank 48 kept | +0.75 / +0.26 / -2.02 / +1.13% (+0.03%) | -4.21 / -4.08% | about -4.1% |
| candS16r3b_k48_q1_r32_mem.py (6f74da86): 1 power step, rank 32 | +1.53 / +0.41 / -0.35 / +2.03% (+0.90%) | -5.58 / -5.41% | about -4.6% |

Torch exact rank-32 truncation on the same nets: 37 +2.58%, 39 +0.53%. The native star plane is NOT low-rank: 98% of its energy needs
rank 131-720 (layer 1 ~720, deep layers ~150), and the rank-32 SVD tail is 25-91% of its Frobenius norm. A plain sketch keeps about
that much energy but in the wrong directions; with 1-2 power steps its error matches the exact rank-32 tail to the third decimal on
every layer (lrstar_check.py diagnostics), and the MSE follows the torch test. Each power step costs ~0.9% of the bill. Per-net noise is
about +-1%, so 4 nets do not rank the three power-step variants; the torch VALID40 at r = 32/64 (porter) sets r, then native VALID40.
Paired user CPU (porter's run_usercpu_pair.sh, e_lrq1r32 vs e_S16r3bref) is queued: the in-process walls above include diagnostic SVDs.

**On the j05 card (the S17 physics).** Porter's stack_S17_j05_mem.py (ae52f49a) vs the same stack + lrstar2_k48_q1_r32.py
(candS17j05_k48_q1_r32_mem.py, b848aeea), in process, same run positions:

| net | S17 j05 final MSE | + low-rank star (rank 32, 1 power step) | change | bill |
|---|---|---|---|---|
| 37 (cold) | 1.6813e-8 | 1.7056e-8 | +1.45% | 357.92 -> 337.94 G (-5.58%) |
| 39 | 1.4326e-8 | 1.4336e-8 | +0.08% | 372.64 -> 352.56 G (-5.39%) |
| 35 | 1.5642e-8 | 1.5377e-8 | -1.70% | 354.63 -> 334.67 G (-5.63%) |
| 100 | 1.5569e-8 | 1.5568e-8 | -0.01% | 350.27 -> 330.31 G (-5.70%) |

Mean MSE change -0.05%, so about -5.6% adjusted on these nets (S16 card: +0.90% MSE). The S17 stack uses exactly the same tree cache
keys as S16 on 35 and 100 (ipt_key_census_S17j05_2.json), so the prebind key list carries over.

**S18 = porter's stack_S18_S17p_lrq1r32_mem.py (ab8c8ba8) = S17p (S16 + view trim + j05 + prebind v2b, 9a4ccc1a) + lrstar2_k48_q1_r32.py.**
Engineer gates, all PASS: whest validate; the porter's einsum gate (0 new repeated-operand sites); paired user CPU vs S17p on
35,100,37,39,42,48 (usercpu_pair_S18.log): user -6.35% (se 0.32%, every net -5.4..-7.6%), bill -5.60%, op calls -1,797 per net, no
page faults; project_wall_v3 on S16's page (#332210, 72.6 s): expected worst 69.6 / 68.7 / 67.8 s. Client/server with setup() on
35/100 is byte-identical (max abs diff 0.0) to the in-process S17-j05 + lrstar run (chain_S18gate.log): the seeded draw, QR, small SVD
and None-indexing behave identically through the numpy-free client. **VALID40 vs S17p (porter, 18:50): adjusted -5.352% pooled (95% CI
-5.598 .. -5.097%), 40/40 better (per net -3.56 .. -6.97%, median -5.31%); util -5.569%; raw +0.232% pooled (bar +-1.0%): PASS.**
Residual gate vs S17p PASS (cold -0.025 s). **FIRED 19:01 as #332226; GRADED 2.6705e-9 = NEW BANK (-5.24% vs S17 #332224 2.8182e-9;
expected 2.667e-9): raw 1.7577e-8 (+0.40%), util .15199 (-5.62%), 50/50 public nets better. Platform telemetry (setup_telemetry.py vs
S17): setup 5/5 SUCCESS at 1.93-2.05 s; warm-MLP bill 352.40 -> 332.35 G; residual warm mean .240 -> .234 s, max .267 -> .247 s;
roundtrip max 73.7 -> 68.3 s.**

**Warm-started sketch: dead (E-62b).** Could the power step (~0.9% of the bill) be replaced by starting each layer's sketch from the
previous plane's subspace? lrstar_warmstart_probe.py (net 37, S17-j05 + lrstar, 14 planes, rank 32 from a k = 48 sketch): mean relative
error with a random sketch q=0 .434, q=1 .372, exact SVD .366; warm starts at q=0 from the previous plane's U or V, raw or pushed by the
consuming w or w^T, topped up to 48 random columns: .429-.434, no better than random. Consecutive planes' leading subspaces are
unrelated, so the power step stays.

**Sketch size 40 instead of 48: dead.** candS18_k40_mem.py (38ff7090 = S17p + lrstar2_k40_q1_r32.py) vs S18, in process, nets
37/39/35/100: bill -0.36% on every net, final MSE +0.22 / +0.78 / +0.02 / +0.61% (mean +0.41%, all worse): adjusted about even. The
16-column oversampling pays for itself; S18's k = 48 stays.

**Torch hook for refits on the S18 physics (E-62c).** lrstar_torch_hook.py replaces the torch card's _flms_star and _flms_star_diagonal
with the native S18 factorization (same Omega draw: numpy default_rng(20260924), 1024 x 48 float32; 1 power step; rank 32). The basis
is computed on a detached copy of the plane and the plane enters as Q_r (Q_r^T A), so the forward value equals the native projection
and gradients avoid QR/SVD backward. lrstar_torch_cert.py on the j05 torch card (S17 literals): hook vs base +0.80% (net 37) / +0.11%
(39), the porter's exact rank-32 +1.34 / +0.48%, native S18 +1.45 / +0.08%; base reproduces the porter's harness exactly; one backward
on net 37 gives a finite loss and 27 finite gradients (max 2.0e-4).

## Residual trim on S16: the view cut is real, the GC cut is not; tree prebinding in setup (E-61, 2026-09-24)

**Builds** (all = fired S16 #332210, a4a1e0f7, plus one appended fragment; predictions byte-identical to S16 in process and client/server, bill identical):

| build | file (sha) | fragment | content |
|---|---|---|---|
| S16r | candS16r_mem.py (c87a3dda) | resid_fragment.py | view trim + GC threshold 20000 + gc.freeze after each predict; kept its level-1 intermediates: max handles +454, failed the no-rise gate |
| S16r2 | candS16r2_mem.py (0a33c080) | resid_fragment_v2.py | view trim + GC threshold, no freeze, no kept intermediates |
| S16r3b | candS16r3b_mem.py (ca974768) | resid_fragment_v3b.py | view trim only (recommended residual layer); candS16r3_mem.py (a3b1d50a) is the same code with a stale header, unused |
| S16r3bp | candS16r3bp_mem.py (2e62d7e2) | + prebind_fragment.py | S16r3b + setup() prebinding of 358 tree cache entries; binds least frequent first (bug, see below) |
| S16r3bp2b | candS16r3bp2b_mem.py (579056e9) | + prebind_fragment_v2b.py | most frequent first, LRU re-order, 3.0 s guard (recommended prebind build; candS16r3bp2_mem.py bf4ee0e3 = same code, stale header, unused) |

**The GC part buys no residual.** gc_garbage_probe.py (DEBUG_SAVEALL sample, client/server) finds that the client's cyclic garbage is
100% flopscope's own per-request serializability check: `_describe_unserializable` defines a recursive nested `walk` on every call,
leaving one function-tuple-cell cycle per request (~31k per net). No RemoteArray is ever in it, so a higher threshold cannot delay a
server-handle free. residual_probe_v6.py labels each GC pass by where it starts: on S16 (181-290 passes per net) and S16r2 (5-8), on
nets 35/100/37, every pass starts inside a flopscope dispatch span (flopscope's timed-dispatch frame on the stack), so GC time is
billed as dispatch. The threshold trims 20-25 ms of wall per net and raises in-process peak RSS by 2-3 GB (in-process mode only);
gc.freeze would pin the cycles pending at each freeze. Both are dropped in v3.

**The view trim (natural-order level-1 views, basis views in the tree entry, intermediates not kept):**

| per net, S16 -> trim | r35 (cold) | r100 | r37 | r39 |
|---|---|---|---|---|
| requests | 39,841 -> 38,386 | 37,004 -> 35,439 | 35,120 -> 33,470 | 36,696 -> 35,111 |
| free-only round trips (dispatch, not residual) | 2,202 -> 3,096 | 2,121 -> 2,562 | 2,123 -> 2,238 | 2,111 -> 2,466 |

Max handles over the 4 nets: S16 25,687 / 25,821 / 25,820; S16r2 23,344 / 23,424; S16r3b 23,418 (-9.5%). Live-array peak 2,435.2 MB and
final live 1,894.5 MB identical in every run. Server peak RSS: S16 6,230 / 6,087 / 6,231 MB, S16r2 6,155 / 6,234 MB (ABBA pairs
+68 / +4 MB, inside S16's own 143 MB spread). Local residual change vs S16 in four same-window comparisons: -11.3, -7.2, -1.1,
-11.4 ms (mean -7.8 ms; the request cut predicts ~-8 ms at ~5 us per request). Per-net noise is +-10 ms at load ~8 and far more at
load 10-11 (the S16r3b run_mem at 17:17 ran at 46-54 s wall per net: its residuals are unusable, its handles are not).
The free-only trips rise because the dropped intermediates die mid-binding: each is a full round trip inside the dispatch span.

**Why the residual is ~5 us per request and only fewer requests cut it:** flopscope's timed-dispatch wrapper spends ~2-3 us per
request outside its own span (context-manager entry, the caller-provenance frame check, exit), and each request's result proxy is
retired later by a weakref finalizer (~1 us) in the participant's code (`_BH_HOLD.clear()` shows as ~100 us per product).

**Request fusion is dead.** Fusing two tree level ops into one broadcast op saves one request each, but numpy's ufunc overlap check
(solve_may_share_memory at max_work=1) gives up (TooHard) on the fused B pattern ([Y6,Y4] = [Y5,Y3] - [Y2,Y5]) at levels 1-4, which
forces a temporary copy of the output: more server memory traffic. The C-side pairs overlap outright. The A pattern ([X3,X1] = [X2,X6]
- X5) is proven disjoint but saves only ~1.1k requests per net and costs ~1.0k binding requests on a cold net. Every op of the
current kernel is proven disjoint, so today's kernel pays no hidden overlap copies.

**Tree prebinding in setup (S16r3bp).** A worker's first MLP binds ~219 tree cache entries at ~22 view requests each: the cold
premium in the residual probe is ~19 ms locally (binding 23.4 ms cold, 12.8 ms on the second net, 4.7 ms by the third). The keys
(kind, depth, rows, cols) are MLP-specific but recur: ipt_key_census.py (in process, fired S16) over 8 nets outside VALID40 (ranks
300-580) finds 609 distinct keys; recurrence counts 1..8 of 8 hold 251/103/54/48/51/31/21/50 keys.

| prebound set | keys (max per side) | coverage of held-out nets 35 / 100 / 37 / 39 |
|---|---|---|
| used in 8 of 8 | 50 | 21% / 22% / 22% / 22% |
| used in >= 4 of 8 | 201 (70) | 61% / 58% / 58% / 54% |
| used in >= 2 of 8 (built) | 358 (121) | 76% / 68% / 70% / 62% |

The prebound entries fit under the 160-entry caps with room for a cold net's ~23 new entries per side; least-frequent keys are
bound first so the common ones are evicted last. flopscope_server's ConnectionStore documents that handles minted at import or in
setup() survive across MLPs (budget_open/close reset only the flop counter), and the first predict still checks the arena with one
free view request and falls back to lazy binding if it is gone. Measured below (S16r3bp2b).

| prebind result (client/server with setup(); predictions byte-identical to S16 on r35/r100 in every run) | r35 cold | r100 second |
|---|---|---|
| bill, all 358 bound (S16r3bp cs_probe, S16r3bp2b cs_probe) | 359.50 -> 353.50 G (-1.67%) | 354.21 -> 352.83 G (-0.39%) |
| setup wall for all 358 | 1.73 s (load 10), 2.97 s (load 12) | |
| S16r3bp at load 14: guard hit after 175 (the LEAST frequent) | bill -0.47%, requests -1,054 | requests -296 |

Binding reshapes are billed at numel (E-38), so every prebound hit moves flops out of the predict: the platform average is about
-0.25..-0.35% of the bill at 10-20 MLPs per worker. v1 bound the least frequent keys first (so the most frequent would be evicted
last), which made its 2.0 s guard drop the most useful keys under load; v2b binds the most frequent first and then re-orders each
LRU with touch() (no requests). The platform's setup timer covers the module import too (whestbench runner: start_wall before
load_estimator_from_path); the import takes ~0.15 s in the numpy-free client, so setup stays under ~3.3 s of the 5 s limit. Max
handles over 4 nets: 25,301 (175 bound) vs 23,448 without. 12-net gate (run_mem_setup.sh, S16r3bp2b vs S16r3b, nets 35,100,37,39,42,48,
59,63,66,67,71,73): max handles 28,061 vs 28,031, live peak and final live identical; requests / bill by position: cold -3,767 / -1.67%,
second -1,033 / -0.39%, third to fifth -195..-360 / -0.05..-0.11%, zero from the sixth net (the caches then hold the MLPs' own shapes).
Platform average: about -0.12..-0.23% of the bill at 10-20 MLPs per worker, plus the cold MLP's -3.8k requests.
**Platform-verified on S17 #332224** (fallback path forced once in process, prebind_fallback_test.py: after setup the arena was
replaced by an object whose indexing raises; the first predict fell back to lazy binding with byte-identical predictions) (graded 2.818e-9; S16 + j05 + this view trim + this prebind; setup_telemetry.py on eval_332224 vs
eval_332210): setup() SUCCESS on all 5 workers at 1.93-2.04 s wall with 12.0 GFlop each booked to setup_executions; the cold MLPs'
bill premium over the warm mean fell from +2.6% to +0.9%; residual cold .227-.241 s (S16 .253-.275), warm mean .240 (S16 .262), max
.267 s (S16 .313). Local residual
timings at load 10-14 are useless (S16 at 46-68 s per net); requests and flops are deterministic.

**B5 kernel items (lead's strategy plan) are falsified as a 3-6% lever.** The <4x4x4:48> rational scheme (Dumas, Pernet,
Sedoglavic, Tichavsky, arXiv 2609.12027, 10 Sep 2026: canonical 212 additions + 4 scalings, leading constant 7.75; square
alternative basis 172 + 4, constant 6.5) against two levels of our 12-add Strassen (49 products, 132 block additions): the saved
product pays for the +44 block additions at the top two 4x4 levels only, about -2.7% on depth-6 1024^3 products and at most about
-2.2% of the bill. But every block addition of the tree kernel is one request: ~176 per 4x4 level against ~30 today, +146 per
product per level, +32k..+64k requests per net, +160..+320 ms of residual against the 0.4 s cap. Dead. Transform sharing is
bounded at -0.3..-0.5% by engineer2's dataflow census; depth is at 0.498-0.523 (depth 6) and 0.562-0.565 (depth 5) of dense, the
leaf floor plus additions, so in-kernel padding is ~0; remaining depth -0.6% for +4.4 s (engineer2).

**Files.** gc_garbage_probe.py, residual_probe_v6.py (+ run_*.sh), ipt_key_census.py (+ ipt_key_census_S16sel8test4.json; its
'misses' column is wrong for single products: the test used the kind in the key), census_setup.py / run_mem_setup.sh (engineer2's
census and runner plus a setup() call), cs_probe_setup.py / run_cs_probe_setup.sh, chain_S16r2.sh, chain_S16r2_ba.sh,
chain_S16r3b.sh, chain_S16r3bp.sh and their logs.

## Residual budget: ranked client-side costs on S15 (E-60, 2026-09-24; measured, nothing built)

**Result.** The platform's residual is the participant's own Python: wall minus dispatch, where dispatch covers every flopscope
op's full client round trip. On S15 (#332207) it is 266 ms on the cold net r35 and 251 ms on the warm net r100. A new probe
explains 263 of the 266 ms cold. The largest bill-free item is Python's cyclic garbage collector. Tuning it saved 25-36 ms per net
in measurement. With per-call view trimming in the tree kernel, the achievable cushion in my lane is about -39 ms cold and -50 ms
warm.

**Probe.** residual_probe_v3.py / v4.py run at the client/server boundary (numpy-free client and OpenBLAS server):
- Every function the candidate defines is wrapped. Its exclusive residual is its wall, minus the dispatch accrued inside it, minus
  its wrapped callees, minus the wrapper's calibrated cost (474 ns per call).
- Requests (Connection.send_recv) are counted per innermost function.
- gc.callbacks time each collection, and flopscope._handles.enqueue_free times the handle finalizers.
- v4 can apply a GC policy from the harness, leaving the candidate untouched.

**Ranked, S15, per net (cold r35 / warm r100):**

| rank | source | exclusive ms | requests | bill-free cut | est. saving |
|---|---|---|---|---|---|
| 1 | cyclic GC (265/209 young, 24/19 middle, 2 full passes) | 42.5 / 48.4 | 0 | gc.set_threshold(20000, 10, 10) at import + gc.freeze() after each prediction | measured -25 / -36 ms |
| 2 | tree/alt-basis kernel per-call client work (unwind 40.0, B 28.6, A 20.8, batched 21.4, product wrapper 14.5, bit views 8.6, colpack 4.1, _mm 3.2) | 141 | ~22k | per-call views in the input's natural axis order: no transpose, no unused sub-quadrant split, no move-axis | about -1.8k requests, -2.3k proxies: about -12 ms |
| 3 | physics loop and helpers (_predict_binary 33.5, _bd_update 6.2, _gn_dcov 5.5, features 4.3, message source 3.2, others) | about 60 | about 7.3k | request diets in physics code (owners) | not in my lane |
| 4 | tree binding (_ipt_side 12.4 / 6.9, alt-basis views 9.7 / 6.2) | 22 / 13 | 4.2k / 2.1k | one cache entry per tree side instead of two lookups | about -1..-2 ms |
| 5 | handle finalizer bodies (35.5k per net) | 9-10 | 0 | shrink with item 2 | included above |

The GC measurement (harness policy, nets 35/100/37):

| net | default GC | threshold 20000 + freeze |
|---|---|---|
| r35 cold | 42.5 ms | 17.2 ms |
| r100 | 48.4 ms | 12.9 ms |
| r37 | 37.6 ms | 12.7 ms |

Awaiting the lead's GO. Item 1 sets the client process's GC policy process-wide. Item 2 edits engineer2's alt-basis functions and
my tree-side views.

## Alternative-basis tree kernel (E-58) and batched products at six levels (E-59), 2026-09-24

**E-58 --ipab: built, superseded by engineer2's --altbasis (the lead assigned the lever to engineer2; I missed the assignment).**
ipab_fragment.py writes the tree kernel's recursion in an alternative basis (after Karstadt & Schwartz):
- A's a21 slot holds a11 - a21, B's b12 slot holds b12 - b11, and C is produced with c12' = c12 - c22, at every level.
- Each level then encodes A and B with 3 subtractions each and decodes C with 6 adds (12 instead of 15).
- The basis changes are one operation per level on the original-size operands: A and B after the load, and C's inverse on the
  output's bit-split view.
- Products sit at chunks m1 0, m7 1, m2 2, m4 3, m5 4, m6 5, m3 6. The data chunks are A [0:4], B [0:7:2] and C [0:4], all
  slices, with no in-place encode step.
- It covers single products, colpack's pair, the batched products and the W5 port. Certified in float64 by ipab_cert.py; its
  float32 rms error equals Winograd's.
- On S13 + tree read6 (candS13r6xab_mem.py 79353b8f), in process: bill -1.040% (r35 cold) and -1.055% (r100 warm); ops +224 / +102
  per net (the output-side inverse needs view requests); predictions max rel change 1.95e-6 / 1.73e-6.
- engineer2's version (S13_ab 97178798) applies C's inverse on the seven level-1 products, so its op counts match the tree kernel's.
  Its 4-net parity is bill -0.995% and MSE -0.10%. It is the one in VALID40.

**E-59 lead for the depth re-pricing (engineer2's): batched 1024^3 products at six levels.** _mm_batch pins the covariance/star
products (P=2, about 31 calls per net) at depth 5. On the tree kernel (bb6_bench, one warm call):

| depth | bill | ops | user CPU | max rel err vs float64 |
|---|---|---|---|---|
| 5 | 2.340 G | 133 | 279 ms | 3.2e-5 |
| 6 | 2.167 G | 148 | 355 ms | 6.7e-5 |

Per net that is about -5.3 G (-1.45% of the bill) for about +2.4 s of local user CPU.

**Built (lead GO, 15:10): --bb6 on S15** = fable_porter_v1/stack/stack_S14_ab_mem.py (e313ab77, S14 + engineer2's --altbasis) +
bb6_fragment.py -> candS16bb6_mem.py (a4a1e0f7).
- _mm_batch sends 1024^3 batches to _bb_product at depth 6. engineer2's alt-basis batched variant is depth-generic, so these products
  also get the 12-add transforms.
- The depth-6 A region grows from 30.1M to 90.4M floats, enough for P <= 3 trees. B and C keep 120.5M, so 331.3M floats in all,
  below the 344.2M that depth-5 regions already touch.
- Single six-level products never have an A tree above 30.1M, so only their placement moves.

In process vs S15 (OpenBLAS env, 1 thread; user CPU unpaired at load 16):

| net | bill | ops | user CPU | predictions, max / rms rel change |
|---|---|---|---|---|
| r35 cold | -1.482% | +423 | 34.8 -> 37.4 s | 2.0e-6 / 1.5e-6 |
| r100 warm | -1.525% | +420 | 32.9 -> 35.9 s | 2.1e-6 / 1.4e-6 |

Gates (chain_S16bb6.log):
- Einsum gate PASS (gate_S16bb6_einsum.txt); validate PASS.
- Client/server predictions byte-identical to in-process on r35/r100.
- engineer2 verified bit-exact against their independent batched depth-6 wrapper on 4 nets.
- Porter-tool paired user-CPU census vs S15 (6 nets): flops -1.511%, user CPU +7.84% (204.8 vs 189.9 s, about +2.5 s per net). That
  is about +3.7 s of expected worst at engineer2's 0.47 s per 1%.
- residual_probe (client/server, S15 then S16, sequential under load 16): only _bb_product changes, +6.1 ms cold and -0.1 ms warm.
  That is the one-time binding of the fixed-shape depth-6 batched trees. The porter's paired residual gate decides.

## Depth on the tree kernel: S14 candidate = S13 + tree six-level reads + m64 (E-54..E-57, 2026-09-24)

**Result.** S14 = S13 (#332195) + exact tree six-level reads + deadcut rows rounded to 64 + m64 = candS14_mem.py (9003e536). Its
bill is 0.94-0.96% below S13 on r35/r100, and its predictions change at rounding level (max relative 2.9-3.6e-6). The m64 part
needs a VALID40. The tree reads are bit-exact and give back most of m64's CPU. Each piece was measured on the porter's paired
6-net census:

| piece (vs its base) | flops | user CPU | predictions |
|---|---|---|---|
| E-55 tree six-level reads (vs S13; candS13r6x_mem.py a7e18ab7) | -0.089% | -4.90% | identical |
| E-54 m64 + deadcut rows rounded to 64 (vs rows rounded to 64 alone; candS13m64 vs candS13r64) | -0.914% | +5.64% (+1.75 s/net) | rounding-level |
| rows rounded to 64 alone (vs S13, 2 nets in process) | +0.05 / -0.02% | | rounding-level |

On the old kernels m64 was -0.71% for +7.97% (+3.2 s/net). On the tree kernel it saves more and costs less, about 0.5% of bill
per local CPU second. S13's measured room below 100 s is 22-37 s (engineer2).

**E-55: tree six-level reads.** engineer2's read6 is a standard depth-6 Winograd with the old formulas. A full six-level tree of
A and B would need 512M floats against the 392M-float arena, so the tree version runs in TWO passes over the root level. The root
is written from the inputs and read only by the final unwind, so its chunk order is free:
- Pass 1 holds m5, m3, m6, m4 (the s1->s2->s4 and t1->t2->t4 chains stay inside it).
- Pass 2 holds m1, m2, m7.
- Each pass runs levels 2..6 and its leaf on 4/7 or 3/7 trees, and writes its root range of one full C tree. The unwind 6..2 and
  the level-1 unwind then run once.
It is byte-identical to S13 on r35/r100 and has 1.03-1.04k fewer ops per net (ipt_read6_fragment.py).

**Tried and dropped:**
- **--ipt v4 row-capacity trees** (bind each tree once at row capacity, slice the leaf's first dim per shape): bit-exact, flops
  -0.358%, but user CPU +18.4%. Each op falls apart into one short run per leaf block, and the deepest level has 16,807-117,649
  of them. The capacity-bound tree reads (ipt_read6_fragment_v2.py) fail the same way: with v4, flops -0.52%, user +14.3%.
- **All divisible reads at six levels** (read6all_fragment.py): on the tree kernel six levels always pays in bill per read, but
  over 2 nets the extra reads added -0.06..+0.06% (binding of new read shapes eats it).
- **Six-level squares** (sq6ipt_fragment.py): no product changes. The square tables cap these shapes at five, and the biggest
  squares (the covariance halves 512x1024x1024 and 512x1024x512) would get 8-row leaves for about 0.15% of bill.
- **Non-Winograd bill** (11% of S13): 4.6 points of it are the tree reads themselves. The GNN and message matmuls multiply by a
  few channels, so no Winograd lever is left there.

Files: candS14_mem.py, candS13r6x_mem.py, candS13m64_mem.py / candS13r64_mem.py (v3 base), ipt_read6_fragment.py (exact) and
_v2.py (capacity, dropped), ipt_fragment_v4.py + build_ipt_v4.py (dropped), read6all_fragment.py, sq6ipt_fragment.py,
pair_S13m64_usercpu.log, pair_S13r6x_usercpu.log, pair_ipt4_usercpu.log, pair_S13b_usercpu.log, gate_S14_einsum.txt,
gate_S14_whest_validate.log.

## --ipt: an in-place tree Winograd kernel, bit-identical; S13 FIRED #332195 (E-53, 2026-09-24)

**Result.** S13 = S12 (c20d67c7) + --ipt v3 = candS12ipt3_mem.py (2a5dd324). Its predictions are byte-identical to S12, in process
and at the client/server boundary. Every Winograd product now runs on ONE tree array per operand instead of one array per level,
except engineer2's six-level reads. The porter's paired 6-net census gives flops -1.75% and user CPU -35%. Server CPU, wall and
requests all fall, and so do server live memory and handle counts. Every gate the lead listed is green. S13 is recommended in
place of S12: the same predictions at util x0.9825, so about 2.963e-9 against S12's about 3.016e-9. This is the cheaper-transform
lever: the old kernels spent most of their CPU moving data, not adding.

**S13 gates** (gate_S12ipt3_checks.txt):

| gate | S12 | S13 |
|---|---|---|
| predictions r35/r100, in process and client/server | reference | byte-identical |
| einsum site gate, whest validate | | PASS, PASS |
| paired user-CPU census, 6 nets (porter's tool): flops, user CPU, backend | | -1.750%, -35.24%, -35.70% |
| server live-array peak, max handles (engineer2's run_mem.sh, r35+r100) | 2,555 MB, 25,655 | 2,424 MB, 16,202 |
| server CPU r35 / r100 | 45.3 / 44.5 s | 32.2 / 31.0 s |
| server wall r35 / r100 | 50.4 / 49.2 s | 36.8 / 34.9 s |
| requests r35 / r100 | 48,469 / 43,061 | 39,309 / 37,135 |
| round trips r35 / r100 | 50,793 / 45,259 | 41,540 / 39,256 |
| residual r35 / r100 | 0.30 / 0.28 s | 0.27 / 0.23 s |

Per net in the paired census, flops fell 2.49% on the cold r35 and 1.30-1.78% on the five warm nets.

**How the layout works.**
- Each operand is stored as a tree of shape (7,)*L + leaf. Every block sits in a room of 7 chunks, and its four quadrants are four
  of those chunks: A's quadrant (i,j) at chunk 4i+2j, B's at 2i+j, C's at i+2j.
- The next level's three plain slots (A: a11, a12, a22; B: b11, b21, b22) are therefore already in place. Below level 1 there are
  no level copies.
- The new slots go into the free chunks, one of them in place over the quadrant it replaces. The leaf is ONE batched matmul over all
  7^L chunk pairs, and the unwind runs in place.
- The deepest level is OUTERMOST in memory, so an op at level j is 4^(L-j) contiguous runs of 7^(j-1) leaf blocks. A first
  version with the root outermost was 30-65% SLOWER than the old kernel, because each deepest-level op touched one leaf block in
  seven across a 420 MB tree.
- The formulas, operands and operand order match the old kernels, so the arithmetic is identical. Only the memory layout and the
  order of independent ops change.

**Why it is faster** (op_cpu_level.py on S12, r35, per net): transform adds and subtracts take 21.5 s, level copies 3.4 s and leaf
matmuls 9.9 s of 46.6 s thread CPU. Most transform subtracts ran at 0.72-0.97 ns per element on strided quadrant views, against
about 0.45 ns for contiguous ops.

**Packet bench** (ipt_bench_v2.py, S11 kernel vs IPT on the same inputs, OpenBLAS env, 1 thread, 6 interleaved pairs):

| six-level push packet | bit-identical | bill | ops | user CPU |
|---|---|---|---|---|
| 1024x896x4096 | yes | -1.55% | 119 -> 103 | 374 -> 181 ms |
| 1024x384x4096 | yes | -1.33% | 119 -> 103 | 257 -> 134 ms |
| 1024x896x1920 | yes | -1.90% | 119 -> 103 | 233 -> 114 ms |

**Card builds** (run_card.py, one process per candidate, r35 cold then r100 warm, OpenBLAS env, 1 thread, S12 and the build
alternated, 2 rounds):

| build | r35 cold: bill, ops, user CPU | r100 warm: bill, ops, user CPU |
|---|---|---|
| S12 (c20d67c7) | 382.33 G, 38,979, 36.1-36.4 s | 374.09 G, 37,437, 35.6-36.0 s |
| v1: products through _packed_product_cached, colpack's pair, read5 (candS12ipt_mem.py 869c6eb5) | -2.18%, 33,345, 25.5 s | -0.71%, 32,908, 25.0-25.1 s |
| v2: v1 + the batched covariance/star products + the W5 port (candS12ipt2_mem.py 2aa61974) | -2.47%, 32,785, 23.1-23.2 s | -0.91%, 32,459, 22.8 s |
| v3 = S13: v2 with tree caps 160 and chunk views by one unstack per level (candS12ipt3_mem.py 2a5dd324) | -2.49%, 23.3 s | -1.75%, 23.0 s |

- v2 RAISED requests at the boundary (r100: 43,061 -> 45,864). A net uses about 70 tree shapes per side and about 40 of them are
  new on each warm net. v2's caps of 48/48/32 thrashed, and every binding created its 7 chunk views one getitem at a time (about 50
  requests per tree). v3 keeps 160 trees per side, creates chunk views with one unstack per level, and holds binding temporaries.
  Warm bill -0.91% -> -1.75%; boundary requests now fall 14-19%.

- Predictions are bit-identical to S12 on both nets in every round.
- The cold net saves more because a tree binds once per shape and costs less than the old per-level frames.
- The per-call cost is three reshapes of the inputs and output into bit-split views (billed at numel), paid out of the copies saved.
- engineer2's six-level reads keep their own kernel: their trees would need 512M floats against the 392M-float arena.

**Boundary checks** (cs_probe.py): single products of every family give the same checksums as S12. Server RSS after the probe is
1,517 vs 1,793 MB, and server wall per product is 35-50% lower (six-level push packet 0.43 -> 0.22 s). The 3.8 GB S12 server peak
in the porter's 12:10 smoke came from a busier window; in the same window S13 is lower on every memory number.

**What it frees.** If the platform kernel follows local user CPU at 0.57-0.94 (engineer2's transform calibration) or 1.2-1.5 (the
general ratio), -13 s of local user CPU per net is about -7 to -20 s of platform kernel. Another ~5k fewer requests save about
-1 to -2.5 s. That room can buy deeper Strassen, which is now cheaper per level too. The m64 pair on the OLD kernels (porter's tool,
6 nets) was flops -0.71% for user CPU +7.97%. It should be re-measured on --ipt before anything is decided.

Build: `python3 fable_engineer_v1/build_ipt_v3.py <S12.py> <out.py>` (a post-stage, applied last). Files: ipt_fragment_v3.py
(ipt_fragment_v1.py + ipt_fragment_v2_extra.py for the earlier builds), build_ipt.py, build_ipt_v2.py, build_ipt_v3.py, ipt_bench.py
(v1 layout), ipt_bench_v2.py, op_cpu_level.py (+ op_cpu_level_S12.json), run_card.py, pair_ipt.sh / pair_ipt2b.sh (+ logs and
pair_ipt_runs/), cs_probe.py + run_cs_probe.sh (+ cs_probe.log, cs_runs/), check_ipt3.sh, smoke_ipt2.sh, mem_ipt3.sh,
pair_ipt3_usercpu.log, gate_S12ipt3_*.

## Copy-free level L-1, B side (E-52, 2026-09-24): bit-identical, superseded by --ipt

- The B side of level L-1 normally copies its three plain quadrants into the level array. The prototype skips that copy and runs
  level L's B transforms and the three affected leaf products in four groups.
- The first version was not bit-identical. In the six-level layout, the level-6 A array shares arena offsets with the level-4 B
  array, which the copy-free leaf still reads.
- Moving level-6 A into the free region [236642560, 266760704) fixed it (copyfree_bench_v2.py).
- Per six-level packet: bill -0.65% to -0.69%, ops +20, user CPU neutral (paired medians -9 to +13 ms, noise +-20 ms).
- --ipt removes every level copy below level 1 with FEWER ops, so this variant is not worth building.

## --m64: six-level pushes for any row count divisible by 64 (E-51, 2026-09-24): built and measured, not recommended

**Result.** build_m64.py (a post-stage, applied last) lets pushes with 64 <= m < 1024 and m % 64 == 0 take the six-level path. These
are deadcut's kept rows with the porter's --round=64. It is bit-exact at m = 1024: the null build on colpack alone matches colpack in
predictions and bill on r35 and r100. Under deadcut it cuts the bill another ~0.6%, but it costs 3.4-4.6 s of local user CPU per
net and gives back most of deadcut's CPU saving. That is about 0.1% of bill per platform-second, worse than read6q (~0.15%/s) and far
worse than column keying (~free). Recommendation: keep deadcut on the current kernels and spend its freed wall elsewhere.

**Changes:**
- The depth rule offers four, five and six levels (padded six when k or n is not a multiple of 64) to those m, priced exactly as at
  m = 1024.
- _mm's padded-six dispatch accepts m % 64 == 0.
- _packed_product_4 runs m % 64 rows through the six-level layout, sharing colpack's left transform when present.

**Row bench** (m64_bench.py, one warm _mm call, OpenBLAS env; the porter's shapes):

| product | m | current: depth, bill per row vs m = 1024, user CPU | --m64: bill vs current, user CPU vs current |
|---|---|---|---|
| young push k 864, n 6144 | 960 / 896 / 832 | depth 5, +5.1 / +5.3 / +5.5%, 455-469 ms | depth 6, -4.5 / -4.3 / -4.1%, +49 / +75 / +68% |
| old rebuild k 384, n 12288 | 960 / 896 / 832 | depth 5, +6.0 / +6.2 / +6.4%, 590-637 ms | depth 6, -5.3 / -5.2 / -5.0%, +63 / +71 / +93% |
| U projection k 864, n 384 | any | square path, +0.2-0.6% per row | unchanged |

The six-level transforms dominate the CPU. The five-level path that deadcut's pushes fall back to is where deadcut's -10% user CPU
comes from, not from the rows it drops.

**Whole net** (in process, .venv; bill vs S11):

| build | r35 cold | r100 warm |
|---|---|---|
| colpack | -0.293% | -0.345% |
| + deadcut -3.5, rows rounded to 32 (porter's) | -0.534% | -0.888% |
| + deadcut -3.5, rows rounded to 64 | -0.482% | -0.953% |
| + deadcut -3.5, round 64 + --m64 | -1.059% | -1.552% |
| + deadcut -3.0, rows rounded to 32 | -1.291% | -1.531% |
| + deadcut -3.0, rows rounded to 64 | -1.270% | -1.394% |
| + deadcut -3.0, round 64 + --m64 | -1.858% | -2.037% |

- At -3.5, --m64's user CPU (A-B-A-B, in process) was +4.6 s on r35 and +3.4 s on r100 (+7-10%).
- Predictions under --m64 move at rounding level (max relative 1.5-1.9e-6).
- The analyst puts -3.0 at raw +0.97% vs +0.09% at -3.5. Its extra ~0.7% of bill (current kernels) therefore roughly cancels in the
  adjusted score, with or without --m64.
- The porter's paired 6-net user-CPU run of --m64 vs deadcut r64 is queued behind the VALID40 lock and will give the official number.

Files: build_m64.py, candS11cp_dc35r64_mem.py (V0), candS11cp_dc35r64_m64_mem.py (V1), candS11cp_m64null_mem.py (null),
candS11cp_dc30r32/r64(_m64)_mem.py, m64_bench.py (+ m64_bench_V0/V1.jsonl), card_census_v3.py (census with keyword pass-through).

## Copy-free levels and less frame binding, priced in wall terms (E-50, 2026-09-24; measurements, nothing built)

Question (lead): if R320 frees about 6 s of expected-worst wall, how much of the binding and copy bill can be bought for 3-5 s?
The measurements show the two good options cost about 0 s of wall, not 3-5 s. The expensive variants are not worth buying.

**Inputs measured on S11:**
- **Where the local user CPU goes** (op_cpu_profile.py, per net, OpenBLAS env, 1 thread): transform adds and subtracts 24-26 s,
  leaf matmuls 11 s, level copies (stack, copyto) 3.6-3.9 s. Binding reshapes cost no measurable CPU (they are views). S11 issues
  37-38k billed ops per net.
- **Bills:** binding reshapes 6.34 U on cold r35 (3.53%) and 3.15 U on warm r100 (1.78%); the warm mean over 5 nets is 1.12%.
  Level copies are 3.3 U on both nets.
- **Warm binding by frame class** (warm_frame_census.py, nets 35 then 100, 37, 39, 42, 48 in one process): read frames lead
  (read5 child 0.25%, read A 0.24%, read B 0.18% of the bill), then six- and five-level push B and output frames.
- **Strided-frame CPU** (stride_bench.py and a rows-only rerun, user CPU at S11's real level shapes):
  - Slicing rows only (blocks stay contiguous) costs nothing measurable: adds 0.91-1.01x, copies 0.94-1.08x.
  - Slicing both dims costs +24% on adds overall (+37-56% on the read's A side) and +17% on read leaf matmuls.
  - E-44's +1.5-5 s estimate for column keying was measured while the box was paging (see E-46); it does not hold.

**Priced options** (platform wall = local user CPU x 1.2-1.5, plus 0.5 ms per request):

| option | bill saved: r35 cold / r100 warm (warm mean) | requests per net | local user CPU per net | platform wall | verdict |
|---|---|---|---|---|---|
| E6a column keying: bind each frame at the layout's row capacity, key it on its column dims, slice rows (views stay block-contiguous); read A/B/output, read5 child and push/square B frames | -1.15% / -0.90% (-0.66%) = -2.07 / -1.59 U | -479 / -358 (-243) | about 0 (+-0.5 to 1 s) | about -0.2 s, +-1.2 s | BUY: exact (same values), cheapest; needs a prototype to confirm |
| E6b full capacity keying (slice both dims) | -3.13% / -1.78% (-1.12%) | about -1.2k / -0.6k | about +6-7 s | +7 to +10 s | no |
| E6c row keying (slice columns) | -1.86% / -1.34% (-0.95%) | about -0.7k | +24-56% on the keyed frames' ops | several s | no |
| E5a copy-free level L-1 in every product (the plain quadrants read in place, one level above today's last-level elimination) | -0.85% | +45 per product, about +12.6k | -1.8 s | +3.6 to +4.1 s | no (0.2% per s) |
| E5b the same in the ~60 largest products only (young pushes and reads, rebuild packets) | about -0.45% | about +2.7k | about -1.6 s | about -0.5 to -1.0 s | maybe: paper estimate; a grouped last-level kernel rewrite (about a day) |
| E5c copy-free at every level | | ops grow 4-5x per level | | | infeasible |

Why copies cost requests: each level stores its 7 sub-blocks in one array so the next level runs as one batched op. Leaving the 3
plain quadrants in place splits the next level into 4-5 groups (the transforms, and the leaf products per slot group). A big
product saves about as much copy time (0.52 ns per copied element) as its extra requests cost. A small product does not.

**Answer.**
- Column keying (E6a) buys 0.66% warm and 1.15% cold of exact bill for about 0 s of wall. It also removes 240-480 requests per net.
- Selective copy elimination (E5b) might add about 0.45% at about 0 s.
- Together that is about 1.1-1.6% of exact bill without spending R320's room. Every other variant costs well over 3-5 s.
- Next step: prototype E6a in the read frame builders first, then the push and square B frames. Check it with paired per-call user
  CPU and run_mem.sh ABBA on r35 and r100.

Files: op_cpu_profile.py and op_cpu_S11.json, warm_frame_census_S11.json, stride_bench.py.

## --colpack: the exact per-column package, bit-identical (E-49, 2026-09-24)

**Result.** candS11cp_mem.py (sha c6ca3ebf) = S11 (acf67b9d) + --colpack. Its predictions are bit-identical to S11 on r35 and
r100. The bill falls 0.293% on the cold net and 0.345% on the warm net, and every gate passes.

**How to build.** `python3 fable_engineer_v1/build_colpack.py <post-diet S11 file> <out.py>`, run AFTER
fable_engineer2_v1/build_e2mem.py. The flag's loop edits sit on lines the diet rewrites, so it cannot run inside build_stack.sh.
It refuses to overwrite. Every edit is count-asserted on the input's live definitions. It reads colpack_fragment_e1.py from its
own directory.

**What it does:**
- **E1: the second 4096-column packet skips the left-operand transform.** A six-level push wider than one packet now transforms
  its left operand once. The six-level layout reuses every A-level region of the arena during the first packet's unwind. So the
  A side's levels 5 and 6 (level 5 whole; level 6's used slots 2 and 4-6) are bound in a side buffer: 34.4M floats, allocated once
  per process. The second packet skips the A transform and reads the same values. This covers the young pushes and the old-tier
  rebuild products.
- **E3: the read reuses the squares.** pp*pp is formed before the read, and the read and inner take their squared response rows
  from it (the same products).
- **E4: the gated young columns stay compact.** They are kept only on the rows the next push reads, formed from the read's
  already-gathered rows. This removes the gating of all 1024 rows and the row gathers of p at the next push, the joins and L15.

**Measured on S11 r35 cold / r100 warm:**

| check | result |
|---|---|
| predictions (.venv, 2 threads) | bit-identical on both nets |
| bill | -0.527 U (-0.293%) / -0.609 U (-0.345%) |
| saving by piece (r35 / r100) | E1 -0.33 U (+0.08 U of extra side-buffer binding on the cold net only); E4 -0.22 U; E3 -0.06 U |
| einsum gate (einsum_site_diff.py vs S11) | PASS: 0 new repeated-operand sites; the only change is the gating einsum's operands |
| whest validate | PASS |
| requests / round trips (run_mem.sh, deterministic) | -508 / -453 cold; -689 / -634 warm; free trips +55 |
| server CPU, ABBA means (load 11-14, noisy) | -10.0 s cold, -4.7 s warm: no increase |
| residual, ABBA means | -0.10 s cold, -0.07 s warm |
| server live-array peak | +150 MB (the 138 MB side buffer plus the kept rows) |
| process peak RSS (in-process, both nets) | 5,757 vs 5,729 MB (+28 MB). The server RSS peaks (5.6 GB vs 3.7-4.5 GB) vary 0.8 GB between identical S11 runs, so they are noise. |
| in-process user CPU | -1.7 s cold, 0.0 s warm |

Files: build_colpack.py, colpack_fragment_e1.py, candS11cp_mem.py, card_census_v2_S11cp.json (parity), gate_S11cp_einsum.txt,
gate_S11cp_whest_validate.log, chain_colpack.sh and colpack_[1-4]_*.log (the ABBA server check).

## Per-column cost audit at fixed physics (E-48, 2026-09-24; paper plus one cheap measurement run, nothing built)

Question (lead): births carry 70% of the bill, so is there an algebraic reordering that lowers the per-column price at fixed
physics (bit-identical, or at most 1e-6 relative change on predictions)? Current prices on r35: old 3.36e-4 U per column per
layer, young 7.0e-4 at L8-14, join 6.6e-4 once per joined column.

**Answer.** At fixed physics, S11's per-column work is already at its minimum product structure, and the old-tier read is
already done inside the frame. The only fixed-physics savings are shared transforms and small trims:
- 0.50 U per net (0.28% of the bill) with no memory cost;
- 0.94 U more (0.52%) that needs about 1.1 GB of transforms cached across layers;
- 0.36 U (0.20%) at rounding level.
Every larger saving drops rows from the column work. That changes the third cumulant (k3) of the dropped units, so it is a
physics change: 2.3-4.5% of the bill, depending on which rows go.

**Why rows cannot be dropped at fixed physics.** Each layer sums every column's cube over all 1024 rows to get each unit's k3
(diagonal = sum over columns of lam * pp^3). The pair map uses k3 for every unit, dead units included, with no mask. The mean
takes k3 * e / 6 with e = -a*pdf/v. For a dead unit at a = -3 with skewness 0.3, that term is 1.7 times its Gaussian mean. Dead
units' means feed the next layer through W. The omitted-radial source also reads the columns on all alive rows. So every
column is needed on every row. The card's own comment says the same: "Full1024 rows are mandatory for the original diagonal
and omitted-radial consumers."

**Minimum work per column per layer, against S11** (r35 shapes; "plain" = 2 flops per multiply-add, no Winograd):

| column | work every column must do | plain | S11 bills | S11 / plain |
|---|---|---|---|---|
| old (L11: resp 640) | rebuild on all 1024 rows (1024 x 384) + its read inside the frame (resp x 384) | 5.95e-4 U | 3.35e-4 U | 0.56 |
| young (L11: prev 864, resp 640, live 864) | push on all 1024 rows (1024 x prev) + read (resp x live) | 13.4e-4 U | 7.15e-4 U | 0.53 |
| join, once (L10: prev 832) | range finder 2 x prev x 384, new factors prev x 384, joined core 384 x 384 | 10.3e-4 U | 6.3e-4 U | 0.61 |

The ratio below 1 is the Winograd saving. The depth rule already sits at its bill optimum (E-46). So S11's per-column
products are the minimum products; what remains is duplicated transform work inside them.

**Candidates** (r35, per net; r100 gives the same within 0.2 U):

| # | candidate | physics | does S11 do it? | saved on r35 | risk |
|---|---|---|---|---|---|
| 1 | Old-tier plane read inside the frame on the response rows (the lead's example) | exact | The read: YES. S11 forms inner = (U[resp] F)^2 lam F^T at rank 384, then inner @ U[live]^T; it never forms resp x n_old x n. The response-row rebuild: NO, because the k3 diagonal needs all rows. | 0 at fixed physics. At L14 S11 costs 10.0e9 multiply-adds, not 16e9 (rebuild 6.4e9 + inner 3.4e9 + plane 0.2e9). The response-row form is 7.0e9, but the other 480 rows still need rebuilding for the diagonal (3.0e9). | none: nothing to build |
| 1p | Same, dropping the old-tier part of k3 on non-response rows | physics | no | 6.5 U (3.6%) | changes k3 of dead and saturated units and the omitted-radial source |
| 2 | Push young columns only on the rows the next read and push need (the alive rows) | physics | no | 0 at fixed physics; 8.0 U (4.5%) with the old rebuild also on alive rows | dead units lose the CP part of k3 (see above) |
| 2b | Drop only rows with a < -3.5 from the column work (the L15 row-cut rule, one-sided, at L1-14) | physics | only at L15 | 4.2 U (2.3%) | smallest physics change of the row cuts: 121-161 rows at L12-14 |
| 3 | Young push reusing covariance-push products | exact | nothing to reuse | 0 | The covariance push forms C*W and W^T(C*W). W^T*C = (C*W)^T would come free, but young columns and newborn bases are not columns of C (base = alive * (g * Cpre + change)). Only W^T's transform is shared, and only where prev = 1024 (L1-4): about 0.02 U per layer. |
| 4 | One W[prev]^T transform for every packet of the young push | bit-identical | partly: the port shares one leaf between the newborn and frame pushes; the six-level young push recomputes it per 4096-column packet | 0.33 U (0.18%), about 0.03 U per layer | the push kernels compute both operands' transforms together, so it needs a kernel refactor |
| 5 | Cache the transformed F (rebuild) and (F*lam)^T (inner) between joins; F is constant from one join to the next | bit-identical | no | 0.94 U (0.52%), 0.11-0.30 U on non-join layers | +1.1 GB live memory. In E-45, +0.7-1.5 GB cost 5-13 s of server CPU. |
| 6 | Trims: reuse pp^2 for the reads' squares, skip gating of old columns (never read), skip the roster concatenation | bit-identical | no | 0.17 U (0.10%) | none |
| 7 | Joined core and core update as upper-block products (the symout trick) | rounding-level | no (symout did it for the covariance) | 0.36 U (0.20%) | Frame rounding flows into every later layer. The card moves MSE by up to 0.3% per net between BLAS environments, so a 1e-6 prediction bound is unlikely to hold. |

Fixed-physics total: 0.50 U (0.28%) bit-identical (4 + 6); 1.44 U (0.80%) with the memory cache (5); 1.80 U (1.0%) with the
rounding-level item (7).

**What this means.**
- The per-column price is at its floor at fixed physics, within about 1% of the bill. The price levers left are physics
  changes; the column count (the physicist's test 1) is the other lever.
- The cheapest physics test is 2b. Apply the L15 row cut (|a| <= 3.5 keeps the CP part of k3) to the dead side at L1-14, as a
  no-refit VALID16 on the torch card. It is worth 2.3% of the bill if the MSE cost stays under about 2%.
- 1p and 2 are larger (3.6% and 4.5%) but change more units.
- Of the fixed-physics items, only 4 + 6 (0.28%) are safe to build. 5 needs a memory budget decision.

**Addendum (op-level split; card_census_v2.py records the op name of every billed op, S11 r35 and r100).** The column work on
the warm net r100 is 128.8 U per net. It splits by op class as follows:

| op class | U per net | share of column work |
|---|---|---|
| leaf products (matmul) | 109.4 | 85% |
| Winograd transform and unwind adds | 12.3 | 9.6% |
| frame binding (reshape, billed per element; 6.2 U on the cold net) | 3.2 | 2.4% |
| copies (Winograd level copies: stack 2.2, copyto 0.5; concatenations 0.2) | 2.8 | 2.2% |
| other (QR 0.6, billed row gathers, finiteness checks) | 1.1 | 0.9% |

So 85% of the per-column price is leaf products at the bill-optimal depths. Every exact option works on the other 15%:

| # | exact option | exact? | saved per net | cost and risk |
|---|---|---|---|---|
| E1 | one W[prev]^T transform per young push, shared by its 4096-column packets (= item 4) | bit-identical | 0.12-0.33 U (0.07-0.18%) | The six-level layout reuses the transformed-W^T regions during packet 1's unwind. Sharing therefore needs either a billed copy of the transformed block (as the port does; the net saving falls to 0.12 U) or a layout that reserves it (0.33 U, about 160 MB). |
| E3 | reuse pp^2 for the reads' squares; skip the roster concatenation (item 6 without the gating) | bit-identical | 0.12 U (0.07%) | none |
| E4 | keep the gated columns only on the rows the next push reads, formed from the read's already-gathered rows. This removes the gating of all 1024 rows and the billed gathers of p (next push, joins, L15). | bit-identical | 0.22 U (0.12%) | touches every consumer of p |
| E2 | cache the transformed F and F*lam from each join to the next. L10's F also serves the lazy prefix at L13-14, so it is reused 4 times. (= item 5, recounted) | bit-identical | 1.15 U (0.64%) | about 1.1 GB extra live memory at L13-14 |
| E5 | copy-free Winograd levels: read the plain quadrants in place at every level, as the last-level copy elimination already does | bit-identical | at most 2.7 U (1.5%) | 2-4 times more transform ops per level (requests), and a kernel rewrite |
| E6 | less frame binding (E-44's column keying: bind on column counts only, slice rows) | bit-identical | at most 3.2 U warm (1.8%); E-44 put the ceiling at 0.83% | E-44 measured +1.5-5 s of CPU (strided views) |

Totals:
- The cheap exact package (E1 + E3 + E4) saves 0.46-0.67 U (0.26-0.37%).
- With the memory cache (E2) it is 1.8 U (1.0%).
- E5 and E6 could add up to 3.3%, but they cost requests or CPU. S11's worst platform round trip is already 99.3 s.
- Even with every exact option, the per-column price falls by at most about 4%. Without new memory or CPU costs, it falls by 0.26-0.37%.

Files: card_audit_e48.py (pricing from the E-47 census), card_zcount.py and card_zcount_S11.json (per-layer row classes:
dead, response, saturated, a < -3 / -3.5 / -4, a > 3.5; S11 bit-identical, util 0.175513 / 0.172278).

## Native bill and column census of S11 (E-47, 2026-09-24)

Question (lead): the price base for the lean-card candidates. The physicist's fold_bill.py is calibrated on S2; S11 adds
push5p6, symout v3, lean v3, coldbind3 and the B2 refit. Deliver per layer L0-15 on r35 cold and r100 warm (.venv): (a) columns
per tier, (b) billed flops by the physicist's function groups, (c) the card sums as shares of the bill, (d) the per-column
marginal bill for young and old columns.

**Method.** card_census.py ran the fired S11 file (fable_fire_v1/fire_candS11/candS11_S10L_B2e1_mem.py, sha acf67b9d) in the
.venv (flopscope 0.10.0, 2 BLAS threads, nice 10), r35 then r100 in one process. Every billed op goes to (layer, function,
source line). The layer is the loop variable of _predict_binary. The function is the innermost function of the file that is
not a kernel helper (the analyst's bill_by_layer_v2 rule, with the S5-S11 kernels added). Port products are labelled by their
role. Column counts are read at every push and read. The attributed bill equals the total exactly on both nets. The r35 bill
is 179.7 U (util 0.17551); the r100 bill is 176.4 U (util 0.17228). **1 U = 2·1024³ flops (one dense 1024³ product).**

Groups:
- covariance push: the covariance lines of _flms_cov_star (upper blocks, casts), half of its batched stage-1 call (C@w and
  w^T@A are one call of two equal products), the layer-0 Gram, and the covariance diagonal at L15.
- star push: the rest of _flms_cov_star and the star diagonal at L15.
- young push: _push_bank_shared_newborn (inherited young columns and the ± newborn assembly), without its port product.
- old-tier functions: _slr_* and _shared_old384_* except the read (joins, QR, rebuild, inner), without the port product.
- atoms/plane read: _e2_shared_old384_read_young (the young plane product and the old plane product).
- port body: the W-left port products, split by role below.
- other: pair map, learned parts, BD update, GNN, edge, card algebra.

Columns: young exact = roster minus frame columns (after that layer's join). Old frame = columns held in the rank-384 frame.
The star is one dense 1024x1024 matrix from layer 1 on; it has no columns. Births = units selected at the previous layer (2
columns each). Active rows = the push's k; at L15 the push runs on the row-cut rows.

**(a)+(b) r35 cold, U per layer:**

| L | young cols | old cols | star | births | active rows | covariance push | young push | star push | old-tier functions | atoms/plane read | port body | other | layer total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 1024 | 0.45 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.18 | 0.64 |
| 1 | 2048 | 0 | 1024 | 1024 | 1024 | 1.05 | 0.01 | 1.70 | 0.00 | 1.16 | 0.60 | 0.50 | 5.01 |
| 2 | 4096 | 0 | 1024 | 1024 | 1024 | 0.98 | 1.16 | 1.66 | 0.00 | 2.29 | 0.56 | 0.48 | 7.13 |
| 3 | 6144 | 0 | 1024 | 1024 | 1024 | 0.98 | 2.24 | 1.66 | 0.00 | 3.33 | 0.56 | 0.50 | 9.27 |
| 4 | 8128 | 0 | 1024 | 992 | 1024 | 0.98 | 3.03 | 1.66 | 0.00 | 4.01 | 0.58 | 0.51 | 10.76 |
| 5 | 9984 | 0 | 1024 | 928 | 992 | 0.98 | 4.21 | 1.66 | 0.00 | 4.47 | 0.54 | 0.52 | 12.38 |
| 6 | 11712 | 0 | 1024 | 864 | 960 | 0.98 | 4.83 | 1.66 | 0.00 | 4.98 | 0.49 | 0.54 | 13.47 |
| 7 | 7232 | 6144 | 1024 | 832 | 960 | 0.98 | 2.66 | 1.66 | 6.94 | 2.87 | 0.65 | 0.53 | 16.29 |
| 8 | 8768 | 6144 | 1024 | 768 | 896 | 0.98 | 3.40 | 1.66 | 2.20 | 3.31 | 0.59 | 0.76 | 12.91 |
| 9 | 10240 | 6144 | 1024 | 736 | 896 | 0.98 | 3.83 | 1.66 | 2.12 | 3.30 | 0.56 | 0.77 | 13.21 |
| 10 | 6016 | 11712 | 1024 | 672 | 832 | 0.98 | 2.03 | 1.66 | 8.80 | 2.06 | 0.49 | 0.77 | 16.79 |
| 11 | 7360 | 11712 | 1024 | 672 | 864 | 0.98 | 2.71 | 1.66 | 3.95 | 2.38 | 0.52 | 0.78 | 12.97 |
| 12 | 8640 | 11712 | 1024 | 640 | 864 | 0.98 | 3.35 | 1.66 | 3.87 | 2.44 | 0.49 | 0.78 | 13.58 |
| 13 | 5184 | 16384 | 1024 | 608 | 832 | 0.98 | 1.80 | 1.66 | 8.97 | 1.56 | 0.46 | 0.79 | 16.21 |
| 14 | 6400 | 16384 | 1024 | 608 | 832 | 0.98 | 2.15 | 1.66 | 5.37 | 1.61 | 0.44 | 0.79 | 13.00 |
| 15 | 7488 | 16384 | 1024 | 544 | 768 (rows 736) | 0.56 | 2.15 | 0.43 | 2.79 | 0.00 | 0.00 | 0.18 | 6.10 |

**(a)+(b) r100 warm, U per layer:**

| L | young cols | old cols | star | births | active rows | covariance push | young push | star push | old-tier functions | atoms/plane read | port body | other | layer total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 1024 | 0.43 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.18 | 0.61 |
| 1 | 2048 | 0 | 1024 | 1024 | 1024 | 0.98 | 0.01 | 1.66 | 0.00 | 1.09 | 0.56 | 0.44 | 4.74 |
| 2 | 4096 | 0 | 1024 | 1024 | 1024 | 0.98 | 1.02 | 1.66 | 0.00 | 2.18 | 0.56 | 0.48 | 6.88 |
| 3 | 6144 | 0 | 1024 | 1024 | 1024 | 0.98 | 2.01 | 1.66 | 0.00 | 3.16 | 0.56 | 0.50 | 8.87 |
| 4 | 8128 | 0 | 1024 | 992 | 1024 | 0.98 | 3.03 | 1.66 | 0.00 | 3.80 | 0.55 | 0.51 | 10.52 |
| 5 | 9984 | 0 | 1024 | 928 | 992 | 0.98 | 3.99 | 1.66 | 0.00 | 4.53 | 0.50 | 0.52 | 12.18 |
| 6 | 11776 | 0 | 1024 | 896 | 960 | 0.98 | 4.61 | 1.66 | 0.00 | 4.86 | 0.49 | 0.54 | 13.13 |
| 7 | 7296 | 6144 | 1024 | 832 | 928 | 0.98 | 2.70 | 1.66 | 6.58 | 2.99 | 0.61 | 0.54 | 16.05 |
| 8 | 8832 | 6144 | 1024 | 768 | 928 | 0.98 | 3.54 | 1.66 | 2.16 | 3.09 | 0.58 | 0.76 | 12.77 |
| 9 | 10240 | 6144 | 1024 | 704 | 864 | 0.98 | 3.83 | 1.66 | 2.06 | 3.26 | 0.52 | 0.77 | 13.09 |
| 10 | 5952 | 11776 | 1024 | 672 | 864 | 0.98 | 2.04 | 1.66 | 8.89 | 2.02 | 0.49 | 0.77 | 16.84 |
| 11 | 7296 | 11776 | 1024 | 672 | 864 | 0.98 | 2.68 | 1.66 | 3.89 | 2.34 | 0.49 | 0.78 | 12.82 |
| 12 | 8576 | 11776 | 1024 | 640 | 864 | 0.98 | 3.22 | 1.66 | 3.81 | 2.30 | 0.48 | 0.78 | 13.23 |
| 13 | 5120 | 16384 | 1024 | 576 | 832 | 0.98 | 1.60 | 1.66 | 8.81 | 1.43 | 0.44 | 0.78 | 15.70 |
| 14 | 6272 | 16384 | 1024 | 576 | 800 | 0.98 | 2.13 | 1.66 | 5.30 | 1.64 | 0.44 | 0.79 | 12.93 |
| 15 | 7360 | 16384 | 1024 | 544 | 800 (rows 736) | 0.56 | 2.18 | 0.40 | 2.73 | 0.00 | 0.00 | 0.18 | 6.05 |

**(c) Card sums and shares:**

| net | covariance push | young push | star push | old-tier functions | atoms/plane read | port body | other | total |
|---|---|---|---|---|---|---|---|---|
| r35 cold, U | 14.8 | 39.6 | 23.7 | 45.0 | 39.8 | 7.5 | 9.4 | 179.7 |
| r35 cold, share | 8.2% | 22.0% | 13.2% | 25.0% | 22.1% | 4.2% | 5.2% | 100% |
| r100 warm, U | 14.7 | 38.6 | 23.6 | 44.2 | 38.7 | 7.3 | 9.3 | 176.4 |
| r100 warm, share | 8.3% | 21.9% | 13.4% | 25.1% | 21.9% | 4.1% | 5.3% | 100% |

- r35 port body by role: newborn_common 6.0 U (3.3%), shared_basis 1.6 U (0.9%).
- r35 largest 'other' functions: _nonlinear_pair_batch 2.36 U, _predict_binary 1.58 U, _message_edge_source 1.50 U, gnn_step 1.30 U, _e2_messages 0.75 U, _bd_update 0.65 U, omitted_radial_source 0.36 U, features 0.29 U.
- Against the S2 calibration in fold_bill.py, the shares barely moved. Young push with its newborn port product is 25.3% (S2
  25.6%). The read is 22.1% (21.4%). The old tier with its frame port product is 25.9% (25.4%). The star is 13.2% (12.3%) and
  the covariance 8.2% (8.5%).

**(d) Per-column prices, 1e-4 U per column per layer (r35 per layer; the last row is r100):**

| L | young cols (newborn) | old cols | resp x live rows | inherited push | newborn push | young read | young total | old rebuild | old inner | old total | join once per joined col | fixed: cov + star + old fixed + other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 (0) | 0 | - | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.45 + 0.00 + 0.00 + 0.18 = 0.63 |
| 1 | 2048 (2048) | 0 | 1024 x 1024 | 0.00 | 2.93 | 5.67 | 8.62 | 0.00 | 0.00 | 0.00 | 0.00 | 1.05 + 1.70 + 0.00 + 0.50 = 3.25 |
| 2 | 4096 (2048) | 0 | 1024 x 1024 | 5.63 | 2.75 | 5.58 | 9.79 | 0.00 | 0.00 | 0.00 | 0.00 | 0.98 + 1.66 + 0.00 + 0.48 = 3.12 |
| 3 | 6144 (2048) | 0 | 992 x 1024 | 5.44 | 2.75 | 5.42 | 9.98 | 0.00 | 0.00 | 0.00 | 0.00 | 0.98 + 1.66 + 0.00 + 0.50 = 3.13 |
| 4 | 8128 (1984) | 0 | 928 x 992 | 4.91 | 2.91 | 4.93 | 9.36 | 0.00 | 0.00 | 0.00 | 0.00 | 0.98 + 1.66 + 0.00 + 0.51 = 3.15 |
| 5 | 9984 (1856) | 0 | 864 x 960 | 5.17 | 2.93 | 4.47 | 9.23 | 0.00 | 0.00 | 0.00 | 0.00 | 0.98 + 1.66 + 0.00 + 0.52 = 3.16 |
| 6 | 11712 (1728) | 0 | 832 x 960 | 4.83 | 2.85 | 4.25 | 8.80 | 0.00 | 0.00 | 0.00 | 0.00 | 0.98 + 1.66 + 0.00 + 0.54 = 3.17 |
| 7 | 7232 (1664) | 6144 | 768 x 896 | 4.76 | 2.62 | 3.75 | 8.03 | 2.05 | 1.69 | 3.74 | 7.14 (6144) | 0.98 + 1.66 + 0.63 + 0.53 = 3.81 |
| 8 | 8768 (1536) | 6144 | 736 x 896 | 4.69 | 2.45 | 3.61 | 7.91 | 1.94 | 1.62 | 3.56 | 0.00 | 0.98 + 1.66 + 0.39 + 0.76 = 3.78 |
| 9 | 10240 (1472) | 6144 | 672 x 832 | 4.36 | 2.45 | 3.09 | 7.18 | 1.94 | 1.49 | 3.43 | 0.00 | 0.98 + 1.66 + 0.35 + 0.77 = 3.75 |
| 10 | 6016 (1344) | 11712 | 672 x 864 | 4.33 | 2.29 | 3.20 | 7.08 | 2.06 | 1.47 | 3.53 | 6.33 (5568) | 0.98 + 1.66 + 0.93 + 0.77 = 4.34 |
| 11 | 7360 (1344) | 11712 | 640 x 864 | 4.49 | 2.28 | 3.05 | 7.15 | 1.94 | 1.42 | 3.35 | 0.00 | 0.98 + 1.66 + 0.36 + 0.78 = 3.78 |
| 12 | 8640 (1280) | 11712 | 608 x 832 | 4.54 | 2.37 | 2.69 | 6.92 | 1.94 | 1.35 | 3.29 | 0.00 | 0.98 + 1.66 + 0.33 + 0.78 = 3.75 |
| 13 | 5184 (1216) | 16384 | 608 x 832 | 4.51 | 2.29 | 2.81 | 6.81 | 1.95 | 1.28 | 3.23 | 6.21 (4672) | 0.98 + 1.66 + 1.06 + 0.79 = 4.48 |
| 14 | 6400 (1216) | 16384 | 544 x 768 | 4.14 | 2.15 | 2.36 | 6.13 | 1.95 | 1.21 | 3.16 | 0.00 | 0.98 + 1.66 + 0.48 + 0.79 = 3.91 |
| 15 | 7488 (1088) | 16384 | - | 3.07 | 1.63 | 0.00 | 2.87 | 1.56 | 0.00 | 1.56 | 0.00 | 0.56 + 0.43 + 0.23 + 0.18 = 1.40 |
| L1-14 | 101952 (22784) | 86336 | | 4.71 | 2.62 | 3.80 | 8.05 | 1.97 | 1.39 | 3.36 | 6.60 | 3.61 per layer |
| L1-14 (r100) | 101760 (22656) | 86528 | | 4.58 | 2.53 | 3.71 | 7.84 | 1.96 | 1.32 | 3.28 | 6.52 | 3.59 per layer |

- A young column costs about 8e-4 U per layer on average. That is 9-10e-4 at L1-6, where the read covers about 1000 x 1000
  rows, and 6-8e-4 at L7-14.
- An old column costs 3.2-3.7e-4 U per layer (rebuild 2.0e-4 over all 1024 rows, inner 1.4e-4). A join adds 6.2-7.1e-4 once per
  joined column. The factor rotation at a join costs 0.29e-4 U per existing old column.
- At L15 (row cut, no read) a young column costs 2.9e-4 and an old column 1.56e-4.
- The physicist's S2 prices (young 6.8e-4, old 3.3e-4 per column per layer) hold at L8-14 within about 5%: S11 has young
  7.0e-4 and old 3.3e-4 there. At L1-6 a young column costs 8.6-10.0e-4, 27-47% more than 6.8e-4.
- The fixed cost per layer is about 3.6 U: covariance 0.98, star 1.66, frame port product plus old plane product plus QR 0.3-1.1,
  other 0.5-0.8.

**Bill carried by one birth over its whole life** (card_census_birthprice.py; linear prices, fixed parts excluded), r35:

| birth layer | births | joined at | young phase | join | old phase | L15 | lifetime per birth | cohort total |
|---|---|---|---|---|---|---|---|---|
| 0 | 1024 | 7 | 11.86e-3 | 1.43e-3 | 5.46e-3 | 0.31e-3 | 19.06e-3 | 19.52 |
| 1 | 1024 | 7 | 9.56e-3 | 1.43e-3 | 5.46e-3 | 0.31e-3 | 16.76e-3 | 17.16 |
| 2 | 1024 | 7 | 7.35e-3 | 1.43e-3 | 5.46e-3 | 0.31e-3 | 14.55e-3 | 14.90 |
| 3 | 992 | 10 | 10.17e-3 | 1.27e-3 | 3.31e-3 | 0.31e-3 | 15.07e-3 | 14.94 |
| 4 | 928 | 10 | 8.15e-3 | 1.27e-3 | 3.31e-3 | 0.31e-3 | 13.05e-3 | 12.11 |
| 5 | 864 | 10 | 6.28e-3 | 1.27e-3 | 3.31e-3 | 0.31e-3 | 11.17e-3 | 9.65 |
| 6 | 832 | 13 | 8.89e-3 | 1.24e-3 | 1.28e-3 | 0.31e-3 | 11.73e-3 | 9.76 |
| 7 | 768 | 13 | 7.17e-3 | 1.24e-3 | 1.28e-3 | 0.31e-3 | 10.00e-3 | 7.68 |
| 8 | 736 | 13 | 5.58e-3 | 1.24e-3 | 1.28e-3 | 0.31e-3 | 8.41e-3 | 6.19 |
| 9 | 672 | - | 6.83e-3 | 0.00e-3 | 0.00e-3 | 0.62e-3 | 7.44e-3 | 5.00 |
| 10 | 672 | - | 5.29e-3 | 0.00e-3 | 0.00e-3 | 0.62e-3 | 5.90e-3 | 3.97 |
| 11 | 640 | - | 3.78e-3 | 0.00e-3 | 0.00e-3 | 0.62e-3 | 4.40e-3 | 2.81 |
| 12 | 608 | - | 2.32e-3 | 0.00e-3 | 0.00e-3 | 0.62e-3 | 2.94e-3 | 1.79 |
| 13 | 608 | - | 0.90e-3 | 0.00e-3 | 0.00e-3 | 0.62e-3 | 1.52e-3 | 0.92 |
| 14 | 544 | - | 0.00e-3 | 0.00e-3 | 0.00e-3 | 0.33e-3 | 0.33e-3 | 0.18 |

- r35 all births carry 126.6 U of the card's 179.7 U (70.4%): the part of the bill that scales with the number of births.
- r100: all births carry 123.6 U of the card's 176.4 U (70.1%): the part of the bill that scales with the number of births; its per-birth prices are within 5% of r35.
- Dropping a birth born at layer c saves its lifetime price. For test 1, the bill ratio is 1 minus the sum over dropped births
  of these prices, divided by the card total.

Files: card_census.py (instrument), card_census_S11.json (+ _pred_r35/_r100.npy), card_census_table.py (a-c),
card_census_table_v2.py (d; it prints the v1 tables first), card_census_birthprice.py (lifetime prices).

## Depth decisions re-priced in situ: compute, not paging (E-46, 2026-09-23)

Question (lead): re-price the push depth choices (four, five or six levels, and the padded six of push5p6) on in-process backend
time, give a Pareto table of bill against time over a few MU values, and find the largest bill saving at time <= the S6b rule's.

**Method.** depth_shadow_v5.py runs whole networks on the fired S7 file and executes every push and read at every allowed depth
inside the same call, in random order; only the S7-rule output goes on, so the network is exactly S7. Per execution it records bill,
backend time, user and system CPU, page faults and counted ops. Two concurrent runs, warm nets 35, 100, 37, 39, 42, 48 (a cold
warm-up net first in each run). depth_pareto_v3.py builds per-class tables, paired differences, a refit of the old seconds model
and the Pareto. Control: depth_shadow_v6.py --mlock=1 wires the kernel arena after the first net (single run, nets 35 and 39).

**Finding 1: backend time on this machine was dominated by paging.** In the concurrent runs each warm net took 1.9-2.9M page
faults and 21-39 s of system CPU. Backend time per net ranged 113-178 s while user CPU stayed at 54.6-56.1 s. The excess of
backend over user CPU tracks the fault count per call (correlation 0.93-0.98, about 65 ms per 1,000 faults). Six-level pushes
took the most faults (median 15k per call, up to 200k), because their arena region sits idle between uses and the memory
compressor pages it out (the machine compressed and decompressed several GB per second, other jobs oversubscribe the 48 GB).
With the arena wired, six-level backend time equals user CPU (18.7 vs 18.6 s per net; faults 16k per net instead of 893k), and
user CPU is unchanged across all regimes (within 2%). So a backend-priced rule measured under paging would chase this machine's
memory pressure. Concurrent pairs double the footprint and can cause the paging themselves.

**Finding 2: the compute pricing of the old rule was right.** push5/push5p6 were priced on isolated in-process numpy time (E-28
bench), not on server CPU. In situ, per-class user CPU differences agree with that model within about 20%. A refit of the same
model form on in-situ user CPU predicts per-class depth differences to 0.1 s and cross-validates between runs. The padded-six part
of S7 costs +2.79 s user CPU per net (+5.3%) for -0.457% bill; the platform showed S7 vs S6b kernel +5.0% (4.15 s).

**Pareto, per net against the S6b rule on the same kernels** (6 warm nets; user CPU is paging-free; the paging-regime backend is
shown for reference; platform kernel ~ 1.5x local user seconds):

| policy | bill | user CPU | backend (paging) | pushes at 4/5/6 |
|---|---|---|---|---|
| S7 rule (push5p6) | -0.457% | +2.79 s (+5.3%) | +9.4 s | 0/8/29 |
| MU 0.4 per-class optimum | -0.430% | +2.06 s | +6.7 s | 0/9.3/27.7 |
| S6b rule (push5) | 0 | 0 | 0 | 0/15/22 |
| best at user CPU <= S6b (MU 0.5) | -0.151% | -0.23 s | | |
| same, deployable refit rule (MU 0.62) | -0.081% | -0.03 s | | |
| MU 0.6 | +0.218% | -2.75 s (-5.2%) | -17.5 s | 0/26.7/10.3 |
| MU 0.8 | +0.396% | -3.69 s | -18.7 s | 0.2/28.3/8.5 |
| all at five levels at most | +1.154% | -5.88 s (-11.2%) | -30.9 s | 0/37/0 |
| MU 1.2 | +1.214% | -6.80 s (-13.0%) | -29.9 s | 1.8/32/3.2 |

The wired control run gives the same frontier on backend time (S7 vs S6b: -0.341% bill, backend +2.72 s = +3.8%, user +2.47 s;
best at backend <= S6b: -0.134%).

Slices of push5p6 added to the S6b rule (padded six only when k >= kmin): kmin 992: -0.143% bill for +0.60 s user (0.91 Gflop per
user-second); kmin 928: -0.294% for +1.44 s (0.78); kmin 864: -0.416% for +2.48 s (0.65); all: -0.457% for +2.79 s (0.63).
Reads whose five-level bill is not lower than the four-level one (4.5 per net, n = 384): four levels save 0.22 s user per net at
-0.004% bill (free but tiny).

**Answer.** At compute <= the S6b rule, the best depth policy saves 0.08-0.15% of the bill: the S6b rule is on the compute
frontier, so a new depth rule is not worth building on its own. The depth dial trades bill for compute at 0.5-0.9 Gflop per
user-CPU second (about 0.35-0.6 per platform-kernel second); symout v3 buys about -1% util for about 1-1.3 s of platform wall
(+2.5k requests, compute about neutral), several times better, so wall budget should go to symout first. If S10' has wall left after symout, the best-rate depth spend is the
padded six on k >= 928 (or >= 992). If wall must be bought back, thin-k six-level pushes (k = 384) go to five levels at 0.52 Gflop
per second (+0.495% bill, -3.64 s user).

Files: depth_shadow_v5.py / v6.py (instrument), depth_shadow_s5A.json, s5B.json (6 warm nets), s6L.json (wired control),
depth_pareto_v3.py, depth_pareto_s5_user.txt, depth_pareto_s5_backend.txt, depth_pareto_s6L_backend.txt,
depth_model_s5_user.json (refit coefficients). Earlier drafts of the instrument (depth_shadow.py, v2-v4, backend_profile.py) were
superseded before producing results (v3 and v4 runs were stopped after their cold nets).

## --lean: fewer round trips at identical predictions (E-45, 2026-09-23)

Priced from S8's request budget (47.4k requests + 2.6k free trips per cold net; residual is about 5-7 us per round trip on the
platform). Item 1 held each layer's temporaries to the layer boundary; item 2 took the kernel quadrants and corners with fnp.split.

**v2 (items 1+2 as priced) FAILED; clean negative result.** It was bit-exact and cut round trips by about 1.8k per net. But on S8 the
live-array peak rose +1,469 MB (deterministic), server CPU +8.8/+13.0 s and residual +164/+389 ms (paired ABBA, cold/warm). On
S8' (no hold2, so no scope registration) it cost +701 MB and +5 s. Extending lifetimes to batch frees costs more than the trips save.
hold2's function-scoped holds already did the cheap part: 7.9k free trips without hold2 vs 2.6k with it.

**v3 (lean_fragment_v3.py, the current --lean)** changes no lifetime beyond a function or kernel call:
- fnp.split quadrants and corners, 3 requests per operand instead of 4, with the same views;
- function-scoped holds in select(), the old-tier finite guard and _sym_upper;
- read5 packet slices held for the kernel call.
File: candS7L3_mem.py (23731b49) = the fired S7 recipe + --lean.

| paired ABBA vs fired S7 (run_mem.sh) | cold r35 | warm r100 |
|---|---|---|
| requests | -717 | -717 |
| free trips | -251 | -221 |
| round trips | -968 | -938 |
| freed handles (split's two half-views per operand) | +1,434 | +1,434 |
| live-array peak / max handles | unchanged (2,405.3 MB) / +8 | |

Predictions, util and cached arrays are bit-identical to S7 on r35, r100, r37 and r39, and the einsum/outer audit is identical. CPU,
residual and wall are unresolved at load 15-18: the runs flipped between two regimes (77-79 vs 89-91 s server CPU). The expected platform
effect is -0.2..-0.45 s of wall per net. Residual is about -1..-3 ms: fewer trips at 5-7 us each, partly cancelled by 1,434 more handle
finalizers at about 2.5 us each. fnp.split changes the request count only; the kernel arithmetic is unchanged.


## --symout v3: covariance only, no einsum (E-43 v3, 2026-09-23)

**Why v3.** v2's old-tier joined-core einsum('ia,a,ja->ij', Fj, weight, Fj) raised flopscope SymmetryError on r100. That was the
porter's S8 build in the .venv. flopscope infers the output as symmetric from the repeated operand and validates the float32 result
with allclose(R, R.T, atol=1e-6, rtol=1e-5), and the BLAS rounding asymmetry can exceed that tolerance. The lead's rule: no
inferred-symmetric f32 einsum ships.

**What v3 keeps.** symout_fragment_v3.py keeps only the covariance upper blocks: two plain _mm products with G21 mirrored. The
builder's call-site edits for the two einsum items are removed.

**Files.** candS8so3_mem.py (5116cee9) is the fired S7 recipe (#332026) + --symout. candS8so3sb_mem.py (43a1adab) adds --setupbind.
The diff against the fired S7 file is purely additive: 39 lines (_sym_upper, _flms_cov_star), with no existing line changed.

**Symmetry audit.** An AST pass over every einsum, outer, as_symmetric and symmetrize call finds the same 90 sites in both files.
The only einsums with a repeated operand (einsum('i,j,ij->ij', m2, m2, cov) and einsum('ij,ij,j->ij', bp, bp, _e2_bb)) are already
in S7 and cannot be inferred symmetric. Every outer(x, x) is in S7 and is exactly symmetric.

| in-process (OpenBLAS env) vs fired S7 | r35 cold | r100 | r37 | r39 |
|---|---|---|---|---|
| util | -1.021% | -0.966% | -0.981% | -0.927% |
| final MSE vs 8M truth | +0.108% | -0.104% | +0.116% | -0.021% |

Pooled final MSE is +0.028%, the largest prediction change is 1.1e-5, and handles fall by 183. In the .venv, r35 then r100 run
clean. On the server path (engineer2's census, cold r35) requests rise by 2,525 (+5.6%). Because v3 has no einsum, it loses v2's
read-path saving, so this is more than v2's +1.6k. The residual rose +7 ms on paired means at load 19-26 (repeats differ by about
90 ms); at the quiet 4-5 us per request the expected rise is +10-13 ms. With --setupbind on top it stays bit-identical, with a
further -0.42% on the cold net and the warm net unchanged.


## Requests, residual and warm binding: priced, nothing built (E-44, 2026-09-23)

**Symout's extra requests.** engineer2's census on the cold r35 puts v2 at +1,642 requests. The covariance's two standalone kernel
calls add 2,450, the P=2 batch gives back 420, and the old-core einsum saves about 300 read-path requests. Batching three equal
512-wide covariance blocks (a v3) would recover only about 400, because an extra product inside a batch still costs about 30
requests against 87 standalone. It would also bill 0.435 instead of 0.428 units, so it was not built.

**Residual.** client_profile.py (cProfile of the numpy-free client) shows the residual spread thinly over flopscope's per-request
machinery, about 14 us per request under load 17. At that load cb3's own residual reads 0.548 s, against about 0.18 s in engineer2's
quieter runs. The quiet delta for v2 should therefore be about +7-8 ms, not the +23 ms seen under load.

**Warm binding.** warm_frame_census.py on the symout v2 build, nets 2-6 of one process:

| warm binding by class | share of the warm bill |
|---|---|
| total, mean of 5 warm nets | 1.29% (0.62-1.62% per net) |
| read5 child frames / READ A / READ B / READ O | 0.27% / 0.26% / 0.20% / 0.01% |
| push output + B frames (PUSH5, PUSH6) | 0.46% |
| squares | 0.07% |
| ceiling for column keying: frames whose (builder, layout, level, column count) was bound earlier in the process | 0.83% |

Column keying means binding at full row capacity and slicing the rows. It would realistically save 0.5-0.7%, but every such frame
becomes row-strided. Under load, sliced elementwise ops ran +30-100% slower and leaf matmuls +4-15%, and the B side is about 40% of
push transform time. The estimated cost is +1.5 to +5 s of CPU per net and 4-6 h of work, so it fails the >= 0.8% for <= +1 s bar.
Contiguous frames for varying shapes need one billed reshape per shape. The only ways around that are strided views (CPU), padding to
fewer shapes (the extra arithmetic bills more than it saves), or pre-binding many shapes (the 100k handle cap).


## --symout v2 (E-43 v2, 2026-09-23): the covariance as 2 products

symout_fragment_v2.py computes the covariance's second product as the top half-rows [G11 G12] (512x1024x1024) plus G22
(512x1024x512), with G21 mirrored. That bills 0.428 units against 0.435 for v1's three blocks, and runs 38 ms per layer faster than
v1. File: candS6b_cb3so2_mem.py (b05b80dd), which supersedes v1.

| in-process (OpenBLAS env) vs cb3 | r35 cold | r100 | r37 | r39 |
|---|---|---|---|---|
| util | -1.210% | -1.139% | -1.124% | -1.086% |
| final MSE vs 8M truth | -0.098% | -0.138% | +0.117% | -0.129% |

Pooled final MSE is -0.056%. Compute is neutral (product_profile, a concurrent pair on the cold r35: wall 72.21 vs 72.24 s,
process CPU 63.84 vs 63.82 s). The covariance's two products cost +1.78 s and the smaller P=2 batch saves 1.79 s. On the server
path (engineer2's census, cold r35): requests +1.6k (+3.7%), residual +23 ms, server peak RSS -428 MB. Under load 17 the local
server CPU read +2.8 to +4.7 s in paired runs, tracking the round trips rather than the compute. The expected platform cost is
about +0.8 s per net (1.6k requests x 0.5 ms); engineer2's quiet gate decides.


## Symmetric-output products: --symout (E-43, 2026-09-23; lead-approved, rounding-level)

`--symout` = symout_fragment.py + two count-asserted call-site edits in the builder (needs --batch):
- **Covariance push w^T (C w):** the second product is computed as its two 512-wide diagonal blocks and one off-diagonal block through
  _mm, with the other block mirrored (_sym_upper). The two star products stay batched, now at P=2.
- **Layer-0 Gram w^T w:** its diagonal blocks are symmetric einsums (_sym_gram). _resident_gram512 itself is untouched, because
  its other caller, _cov_compact, passes a weighted pair.
- **Old-tier joined core Fj diag(weight) Fj^T:** it is one einsum('ia,a,ja->ij'). Its SymmetricTensor output passes the finite
  check; _mm and dense adds treat it exactly like a plain array.

| in-process (OpenBLAS env) vs cb3 | r35 cold | r100 | r37 | r39 |
|---|---|---|---|---|
| util | -1.159% | -1.082% | -1.072% | -1.037% |
| final MSE vs 8M truth | -0.092% | -0.267% | +0.109% | -0.034% |
| handles kept | -654 | -844 | -844 | -1126 |

Pooled final MSE over the 4 nets is -0.067%, and the largest prediction change is 1.1e-5, so the change is rounding level. The
expected server CPU is +0.6 s per net: the covariance adds 1.4 s, the old core saves 0.63 s and layer 0 saves 0.14 s. File:
candS6b_cb3so_mem.py (cfe454f0). With --setupbind (v3 skips the now-unused P=3 batch frames), candS6b_cb3sosb_mem.py (957637b6)
is bit-identical to it, with a further -0.461% on the cold r35. Against cb3 the combined stack is about -1.6% cold and -1.1% warm.


## Setup-time frame binding: --setupbind (E-42, 2026-09-23)

Estimator.setup runs once per worker before any predict and outside every predict's BudgetContext. The platform books it separately
as setup_executions (eval pages: marius 3.6 Gflop, reds 15 Gflop, ours 0). frame_census.py and cold_premium_census.py (r35, r100
and r37, each cold in a fresh process) show where a cold net's extra bill goes:

| cold net's extra bill (vs the same net warm) | share of the cold bill |
|---|---|
| all of it | 3.5-3.7% |
| frame reshapes / read5 child reshapes / pair-map arena | 3.00% / 0.54% / 0.03% |
| frames whose shapes are fixed by the width and kernel tiles (25) + the 1024-unit pair-map arena | 0.53% |
| the other 240-250 frames per cold net (shaped by the MLP's active counts; 86 of them recur in all 3 nets, 1.0%) | about 3.0% |

`--setupbind` (setupbind_fragment_v2.py) binds the 26 fixed-shape items in setup, through the kernels' own builders, largest
first, inside a 2.0 s wall. Nothing reads an MLP. If setup has no flopscope session (the local client harness raises
NoBudgetContextError), it opens its own BudgetContext. On any failure it restores every cache and global, so predict then starts
exactly as without the flag.

| check | result |
|---|---|
| predictions, in-process r35/r100/r37 and server-path r35 | bit-identical (server path: 0 of 16384 entries differ) |
| cold r35 bill | -0.53% (in-process and server path alike) |
| warm r100, r37 bill and handles | unchanged |
| setup on the local server path | 0.47 s (server CPU 0.22 s) + 0.12 s from client start through import |
| setup arrays in later predicts | used (the cold bill drops by exactly their binding) |
| S7-lite flags (caps2, bindhold, hold2) | bit-exact, cold -0.535%, warm unchanged |

Worth -0.03% (5 cold MLPs per 100) to -0.05% (10 cold) of the average bill. Files: candS6b_cb3sb2_mem.py (87ece267) and
candS7lite_pad07_cb3sb_mem.py (f48a0f9f).


## Symmetric-output billing: what S6b could save (E-41, 2026-09-23; measurement only)

flopscope bills a symmetric-output einsum (Gram, weighted Gram) at half the dense cost, and ops on its SymmetricTensor output at
about half. sym_census.py classes every product of S6b + --coldbind3 by its output; the shares are the same on the cold r35 and
the warm r100.

| product class (per net) | share of bill | symmetric? | best alternative, units of 2mkn (server path) | bill saved | server CPU |
|---|---|---|---|---|---|
| covariance second product w^T (C w), 14 | 4.3% | output only, not a Gram | upper blocks, 1 level: 0.551 -> 0.435 | -0.91% | +1.4 s |
| old-tier joined core Fj diag(w) Fj^T, 3 | 0.78% | weighted Gram | einsum('ia,a,ja->ij'): 0.59-0.62 -> 0.503 | -0.13% | -0.63 s |
| layer-0 W^T W, 1 | 0.24% | Gram (3 of 4 blocks already) | einsum diagonal blocks: 0.435 -> 0.396 | -0.02% | -0.14 s |
| old-tier core sandwich, 2 | 0.06% | output only | - | ~0 | - |
| all other products: star, C w, w^T A (square 20.5%), pushes and reads (rectangular 68.2%) | 88.7% | no | - | 0 | - |
| elementwise ops on symmetric matrices | 0.22% | - | SymmetricTensor typing | <= 0.1% | - |

- Deeper block recursion does not help: the kernel runs the 256-wide off-diagonal blocks dense, so 2-3 levels bill 0.46-0.52.
- The Cholesky route (C = L L^T, cov = (L^T w)^T (L^T w)) costs 1.11-1.22 units for both covariance products against 1.10 now.
  C is also singular whenever units die.
- On the production path (numpy-free client + OpenBLAS flopscope_server), the bills equal in-process. The symmetric einsum is
  BLAS-backed: 30 ms against 17 ms for a plain GEMM and 130-240 ms for our Winograd paths.
- Net: the two win-win items (weighted Gram einsum, layer-0 einsum diagonals) cut about 0.15% of the bill and 0.77 s of server
  CPU per net. The covariance upper blocks cut 0.91% for +1.4 s, about 2.5 Gflop of bill per CPU-second (pad07: about 0.7).
  None is bit-exact (rounding level; the einsum path is the more accurate one).


## S7-lite depth fragments: padded D6 at a stricter rate (E-30b, 2026-09-23)

The first S7-lite fragments (depthrule_fragment_push5p6_mu06.py / _mu07.py) were wrong. In the v3 rule one MU prices every push
decision, so raising MU to thin out the padded six-level pushes also dropped 18 aligned pushes per net from depth 6 (S6b's choice)
to depth 5. As built they raise the bill against S6b. make_depthrule_fragment_v3b.py fixes this: every push keeps the push5
rule's choice (MU 0.4), and a 1024-row push whose k or n is not a multiple of 64 takes the padded depth-6 product only where that
beats the push5 choice at MU_PAD. Over all 227 product shapes of the 4-net profile, the only changes against push5 are depth 5 to
padded depth 6 (7 shape classes at MU_PAD 0.7, 13 at 0.6).

| per net vs S6b (4-net profile, model) | padded moves | bill | model compute |
|---|---|---|---|
| depthrule_fragment_push5p6pad_mu07.py | 3.0 | -0.277% | +1.35 s |
| depthrule_fragment_push5p6pad_mu06.py | 4.8 | -0.384% | +2.00 s |
| push5p6 at MU 0.4 (S7; +4.0% server CPU measured) | 7.8 | -0.518% | +3.03 s |
| old _mu07 fragment (wrong) | 3.0 padded + 18 demoted | +0.297% | -3.00 s |
| old _mu06 fragment (wrong) | 4.8 padded + 18 demoted | +0.190% | -2.35 s |

End to end: candS7lite_pad07_cb3_mem.py (15d55e77) = the S6b recipe + the MU_PAD 0.7 fragment + `--caps --caps2 --batch --bindhold
--coldbind3` + mem,yright,arena,hold2. In one process against S6b + --coldbind3: util -0.084% on the cold r35 and -0.206% on r100
(the model said -0.10% / -0.27%; the padded shapes bind new frames), final MSE -0.043% / -0.110% (rounding level), handles flat.
Server CPU is engineer2's gate (model +1.35 s per net, about +1.8%).


## Scratchless unwind: --coldbind3 (E-40, 2026-09-23)

`--coldbind3` = `--coldbind2` + one builder edit. The Winograd unwind at each level computes tl=p0+p1, s0=p0+p5, s1=s0+p6,
s2=s0+p4, tr=s2+p2, bl=s1-p3, br=s1+p4, and it kept s0, s1, s2 in a bound scratch region (about 44% of every output frame's
binding). Product p1 is dead once tl is written, and p0 and p5 are dead once s0 is. So the output frames (_o_frame, _bh_o_frame,
_bb_o_frame) now hand the unchanged unwind code (p1, p0, p5) as its scratch and no longer bind the scratch region. The ops, their
order, their shapes and their contiguity are unchanged. The builder edit is count-asserted, three lines. In every layout, each
unwind destination is disjoint from the products it reads, for both the tail leaf and the layout leaf (checked on the runtime
tables).

| in-process, one process | r35 cold | r100 | r37 | r39 |
|---|---|---|---|---|
| predictions vs S6b + --coldbind | bit-identical | bit-identical | bit-identical | bit-identical |
| util vs --coldbind2 | -0.359% | -0.114% | -0.069% | -0.153% |
| util vs --coldbind | -0.587% | -0.183% | -0.098% | -0.213% |
| handles kept (cumulative), --coldbind -> --coldbind3 | 17001 -> 15450 | 25678 -> 23414 | 27253 -> 24893 | 34426 -> 31514 |

Binding falls to 3.54% of the cold r35 bill (4.11% with --coldbind, 5.33% in S6b), 1.63% of the warm r100 (1.81%, 2.05%) and
0.25% of a repeat (0.33%, 0.33%). Against S6b, the three steps together cut about 1.86% of the cold bill and 0.18-0.47% of the
next three nets. Over 100 MLPs with 5-10 cold starts, that is about -0.44% (5 cold) to -0.51% (10 cold) of the average bill, if nets 2-20 in a
worker behave like nets 2-4. File: candS6b_cb3_mem.py (23d4f339) = the S6b recipe with `--coldbind3` + mem,yright,arena. The
server CPU should not move, because the ops are identical; engineer2's paired gate decides.


## Used-slot last level: --coldbind2 (E-39, 2026-09-23)

`--coldbind2` = `--coldbind` + usedslots_fragment.py. With last-level copy elimination on both sides, the kernel writes and reads
only slots 2, 4, 5 and 6 of the A side's last Winograd level and slots 3-6 of the B side's, but the frames reshaped all seven slots.
Where every product's leaf lands in the free arena tail (checked per layout from the specs: the level-L unwind scratch bounds the
leaf), both sides always eliminate their copies, so the last level is now bound as the used slots only (4/7 of it on each side).
The read layout (read5 packets split the whole last level) and the A side of the square-5 layout (the W5 port copies the whole A
leaf) keep full frames. Cache keys, evict5 and caps are unchanged. Every layout in S6b passes the tail check.

| vs S6b + --coldbind (in-process, one process) | r35 cold | r100 | r37 | r39 |
|---|---|---|---|---|
| predictions | bit-identical | bit-identical | bit-identical | bit-identical |
| util | -0.229% | -0.069% | -0.029% | -0.060% |
| handles kept by the caches (cumulative) | -495 | -758 | -782 | -938 |

Binding falls from 4.11% to 3.89% of the cold r35 bill, from 1.81% to 1.74% of the warm r100 and from 0.33% to 0.30% of a
repeat (bind_census_S6bcb2.json). Expected average over 100 MLPs: about -0.06% (5 cold) to -0.07% (10 cold), on top of
--coldbind's -0.25% to -0.31%. File: candS6b_cb2_mem.py (ecf9b9c1) = the S6b recipe with `--coldbind2` + mem,yright,arena. The
S7-lite flag set (`--caps --caps2 --bindhold`, push5p6 MU 0.7) also builds with `--coldbind2`.

Remaining cold binding (3.89%): the read path 1.64 (READ A 0.67, read5 children 0.54, READ B 0.44), output frames of the pushes,
squares and batched products about 0.75 plus the tail leaves 0.33, and the first four levels of the push/square A and B frames.


## Cold-MLP binding cost: --coldbind (E-38, 2026-09-23)

Each platform worker (re)start makes one MLP bind every frame from scratch. flopscope bills reshape at numel even for views. On
#332005 the platform made 10 setups. bind_census.py (wrapping every binding op; S6b; nets back to back like one worker) measured
binding at 5.33% of the cold r35 bill, 2.05% of the next net and 0.33% of a repeat.

| cold r35 binding, by class | % of bill |
|---|---|
| read path: read5 child frames / READ A / READ B / READ O | 0.88 / 0.66 / 0.43 / 0.06 |
| layout-region binding at process start (_cp3_bind_arena_views) | about 0.9 |
| six-level push frames (B / O) | 0.52 / 0.36 |
| relocated leaves (holdelim tail) | 0.33 |
| five-level push frames (B / O / A) | 0.33 / 0.22 / 0.09 |
| depth-5 squares | about 0.2 |

`--coldbind` removes two wastes. (a) The layout-region views were never read by live code (AST-verified: only their identity and
_VIEW_SPECS are used), so the tuple now holds markers. (b) Read5 child frames are bound once per read shape, with the packet axis
leading, and 2-slot packets use the free [:2] slice.

| vs S6b (in-process) | r35 cold | r100 | r37 | r39 |
|---|---|---|---|---|
| predictions | bit-identical | bit-identical | bit-identical | bit-identical |
| util | -1.28% | -0.25% | -0.08% | -0.26% |

Binding falls from 5.33% to 4.11% on the cold r35 and from 2.05% to 1.81% on the warm r100. Cold round trips fall by 983. The
expected average is -0.25% with 5 cold MLPs per 100 and -0.31% with 10. The file is candS6b_coldbind_mem.py (f9e67942), with
engineer2's gate. Possible next step: bind only the used slots of the last level when copy elimination applies (about -0.2% cold,
-0.05% warm).


## Status 2026-09-23 ~05:50 (lead)

S7 is HELD: push5p6's +4.0% server CPU (about +3.4 s platform) and +6 ms residual are too much for -0.5% adjusted, given the +13 s
platform excursion already on record. S7c2 (caps sized for 20-net workers) stays parked with S7 until the fired S6b (#332005)
telemetry shows the real margins. If the margins are wide, the candidate is S7c2 + engineer2's hold2 (about -1 s wall). Fired from
this lane tonight: HE/HEB (S3/S4), push5 + f32 third + evict5 (S5e #331992), and --batch (S6b #332005). No new builds tonight.


## S7 candidate (E-37, 2026-09-23 05:05): S6b + --bindhold + push5p6 + --caps

`candS7_mem.py` (5bf2a07e) is built through the porter's chain: `--gn=gn_knobs/v3.json --l15cut --join=v2 --refit=.../params_epoch2.pt`
and the kernel flags `--holdelim --bigarena --depthrule depthrule_fragment_push5p6.py --f32 third --evict5 --caps --batch --bindhold`,
then build_e2mem.py mem,yright,arena. The S6b part reproduces the porter's stack_S6b_mem (a514c21b) byte-for-byte.

| vs S6b (in-process) | r35 cold | r100 | r37 | r39 |
|---|---|---|---|---|
| util | -0.21% | -0.38% | -0.83% | -0.52% |
| final MSE (rounding) | +0.04% | -0.09% | -0.02% | +0.01% |
| adjusted | -0.17% | -0.46% | -0.85% | -0.52% |

Census, r35 cold: round trips fall from 55,353 to 52,508 (free trips 8,535 to 5,710). It is queued for engineer2's gate after S6b.

Gate (engineer2, paired 20 nets vs the fired S6b #332005): adjusted -0.50%; server CPU +3.99% (push5p6); handles 63,042 vs 74,701;
live peak equal; residual +5.8 ms mean, creeping from +4 ms (nets 2-8) to +9..+13 ms (nets 13-19). The creep is eviction cost:
once the v1 caps fill, each net frees about as many arrays as it binds, at about 2.5 us per finalizer. The fix is `--caps2`
(caps_fragment_v3.py, caps at about 1.1x the 20-net entry counts). S7c2 = candS7c2_mem.py (791ea953). In-process over 20 nets its
arrays reach 75,846 at net 20, which tracks S6b uncapped. Only _BH_F reaches its cap (net 16), and its evictions free nothing
because the underlying _A/_B frames stay cached. The store stays bounded near 80k beyond about net 22.


## Request cut: batched independent products (E-35, 2026-09-23 04:45), for S6

`--batch` (batch_fragment.py; layouts from make_batch_layout.py) runs each layer's five FLMS 1024^3 depth-5 products as two
P-batched kernel calls:
- stage 1: C32@w and w^T@A32;
- stage 2: w^T@(C32@w), pw@w and (w^T*w^T)@pushed^T.
Every level array of the batched layout carries the batch axis right before its two trailing dims, so the bank frame builders batch
the products with no change. Level 1 reads and writes each product's own arrays. Levels 2-5, the leaf products and the unwinds
5-2 run once for all P. The leaf is placed after all other regions, so the last-level copy elimination always applies. The P=2
layout spans 142.6M floats and the P=3 layout 213.9M, both inside the existing arena.

| vs fired S5e (in-process) | r35 cold | r100 | r37 | r39 |
|---|---|---|---|---|
| predictions | bit-identical | bit-identical | bit-identical | bit-identical |
| util | +0.38% (one-time batched frame binding) | 0 | 0 | 0 |
| numpy calls (requests) | 37,172 -> 34,204 | 34,990 -> 31,882 | | |

That is about -1.5 s of platform wall and -12 ms of residual per MLP at 0.5 ms and 4 us per request. The file is
candS5e_batchv1_mem.py (cf49f47f), with engineer2's paired gate.

Gate verdict (engineer2, inside the porter's S6b, paired over 20 nets vs S5e): predictions bit-exact vs S6 without --batch; server CPU
-2.12% (about twice the request overhead alone, so the batched levels also run more efficiently); residual -12.6 ms (cold net 217 vs
230 ms); round trips -3.2k (-7%) per warm net and -2.6k on the cold net; handles +569 constant; live peak -25 to -40 MB; cold bill
+0.34% one-time.

Cold-MLP round trips (engineer2: the first MLP per worker makes about 12k extra requests and 2.9k extra free trips, all frame
binding, and sets both the worst wall and the worst residual). `--bindhold` (bindhold_fragment.py) holds the binding intermediates
(arena slices and discarded level arrays from _exact_view, _read5_view and _cp3_bind_arena_views) in _BH_HOLD until the kernel
call ends, so they free together.

| census r35 cold | requests | free trips | round trips | util |
|---|---|---|---|---|
| S5e | 49,440 | 8,508 | 57,948 | .180464 |
| + --bindhold | 49,440 | 5,811 | 55,251 | .180464 |
| + --batch | 46,816 | 8,586 | 55,402 | .18115 |
| + --batch --bindhold | 46,816 | 5,755 | 52,571 (-9.3%) | .18115 |

Residual was unchanged in mirrored pairs (0.2156 / 0.2146 vs 0.2140 / 0.2144 s). The combined file candS5e_batch_bindhold_mem.py
(844ae0eb) is with engineer2's gate.


## S5e graded 3.176e-9; its 108 s worst MLP is platform noise (E-34, 2026-09-23 04:15)

S5e #331992 graded 3.1763e-9 (-2.64% vs S4). Its worst MLP was 108.0 s (MLP 4, a cold first-of-worker net); S4's was 101.1 s.

- Telemetry (eval_331961.json vs eval_331992.json): the S5e run started slow. By MLP index block, 0-9 was +2.9 s vs S4, 10-19
  +1.7 s, and 40-99 -0.4 to -2.7 s. Its request-overhead wall drifted from 10.1 s to 8.6 s with identical requests. The heavy MLP 39
  is unchanged (100.9 vs 101.1 s). Per-MLP S5e minus S4 has sd 4.4 s (S4 minus S2: 2.0 s).
- Local quiet paired cold test (fresh client and server per net): server CPU r35 31.24 (S4) vs 31.05 s (S5e); r100 29.74 vs
  30.03 s. Requests +1,756 / +2,744: about +0.9-1.4 s of platform wall at 0.5 ms each. Residual +6-8 ms.
- Extra requests, by in-process numpy calls on r35: push5's deeper and packetized products +654, new exact frames +286, f32
  casts +70, the row cut +14.

Conclusion: the systematic worst is about 102.5 s. The observed +13 s single-MLP excursion is the real margin risk: with push5p6
(+3.7 s) the heavy net could reach about 119 s.


## S5 fired (S5e, #331992) and the S6 candidate (E-33, 2026-09-23 03:21)

The lead fired S5e as #331992: S4 + L15 row cut + push5 + f32 third + `--evict5` + engineer2's memory diet (stack_S5e_mem.py,
c34bdd79). engineer2's 20-net server run: S5e ends at 74,132 handles, below S4's 77,140, with no residual or bill cost. `--lru` is
retired: its per-prediction walk added 20-60 ms of participant residual per net against the 0.4 s limit.

S6 candidate `candS6_p6caps_mem.py` (sha 0fd6043c). It is the fired S5e recipe with `--depthrule depthrule_fragment_push5p6.py` and
`--caps`, then build_e2mem.py. The same recipe with push5 and without --caps rebuilds S5e byte-for-byte.

| vs S5e (in-process, one process: r35 cold, r100, r37, r39) | r35 | r100 | r37 | r39 |
|---|---|---|---|---|
| util | -0.21% | -0.38% | -0.83% | -0.52% |
| final MSE (rounding) | +0.24% | +0.08% | -0.04% | -0.02% |
| adjusted | +0.04% | -0.30% | -0.87% | -0.54% |

It is with engineer2 for one gate: 20-net handles, residual per net, and CPU vs S5e (the padded D6 compute is still unmeasured).

Gate result (engineer2, paired 20 nets vs S5e at the same load):
- --caps passes. Handles are flat at about 63k from net 16 (62,684 after 20; S5e 74,132), peak memory is identical, and residual is
  +10 ms on average (+4 to +21 ms).
- push5p6 cuts util 0.43% on average for +4.40% server CPU (about +1.9 s per net at load 8).
- Dial if headroom is tight (model, per net): padded-D6 MU 0.6 gives -0.40% bill for +2.1 model-s (about +3.1% CPU); MU 0.7 gives
  -0.28% / +1.35; MU 0.8 gives -0.16% / +0.7.


## S6 handle valve: gradual LRU caps on the frame caches (E-32, 2026-09-23 03:30)

`--caps` (caps_fragment.py) turns every frame cache into a least-recently-used dict with an entry cap. A hit refreshes its entry and
a put evicts the oldest entry, one at a time (30-75 arrays), so client finalizers spread over the net. Parents refresh their
children on a hit, and each child cap is at least the sum of its parents' caps, so a frame still referenced by a live entry is
never evicted. Caps: _BH_F, _BH_O, _PACKED_FRAME and _READ5_FRAME 150; _BH_X 500; _A/_B 320; bank _O 160; _READ5_CHILD 320.
This replaces `--lru`, whose one-shot drop of about 20k arrays would add about 50 ms of residual (engineer2).

| 20 distinct nets, one process, S5-like stack with --evict5 | arrays after net 20 | bill vs uncapped | predictions |
|---|---|---|---|
| no caps | 73,152 (still growing) | reference | reference |
| --caps | 61,928 (flat from net 13 at 56.7-62.0k) | mean +0.039% (0 through net 9; +0.01% to +0.28% after) | bit-identical, 20/20 |
| --caps at quarter scale (6 nets) | about 18k | +0.4% to +1.3% per net | bit-identical |


## Server handle store: per-net eviction of single-use frames (E-31, 2026-09-23 02:40-03:00)

The flopscope server caps its array store at 100k handles per worker connection, about 20 networks. Every cached exact-shape frame
view is one handle. handle_census.py counts the arrays reachable from the module caches after each net, and it reproduces
engineer2's server numbers: S4-like 33,065 vs 33,082; S5-like 36,056 vs 36,074. reuse_census.py shows that 69% of cached frame
arrays serve one net only, because their shapes carry per-layer active counts or the L15 row count |R|.

| stack (4 distinct nets: r35, r100, r37, r39) | handles after 4 nets | util change vs no eviction (r35/r100/r37/r39) |
|---|---|---|
| S4-like kernel (S3-HEB) | 33,065 | reference |
| S5-like: --l15cut + push5 + --f32 third | 36,056 | reference |
| + --evict5 (five-level push frames the previous net did not use) | 34,467 | 0 / 0 / 0 / +0.03% |
| + --evict5r (also frames used only inside the row cut) | 33,239 | 0 / +0.07 / +0.14 / +0.03% |
| + --evictall (every layout) | 26,389 | 0 / 0 / +0.12 / +0.30% |
| + --evict5 --lru (60k/40k; never triggers in 4 nets) | 34,467 | 0 / 0 / 0 / +0.03% |
| + --evict5 --lru at 25k/18k (test copy) | 27,734 | 0 / 0 / +0.12 / +0.13%; predictions bit-identical |

All flags hook `_w5_port_new` at the start of `_predict_binary` and never redefine `_predict_binary`, so builders that patch the
last def keep working. Frames are views of the arena, so predictions never change: the LRU test copy is bit-identical to the
non-evicting build on all 4 nets. Recommended S5 kernel flags:
`-- --holdelim --bigarena --depthrule depthrule_fragment_push5.py --f32 third --evict5 --lru`.

On the porter's full stacks (with gn knobs and engineer2's diet), cached arrays after r35 / r100 / r37 / r39:

| stack | r35 | r100 | r37 | r39 |
|---|---|---|---|---|
| S4 (stack_S4_joinv2) | 15,501 | 23,590 | 26,268 | 33,065 |
| S5 draft (stack_S5_mem) | 16,818 | 25,595 | 28,405 | 36,056 |
| S5e (stack_S5e_mem, --evict5): the lead's S5 | 16,818 | 25,595 | 27,198 | 34,467 |

S5e predictions are bit-identical to S5_mem on all 4 nets; util is identical except r39 +0.03%. engineer2's 20-net server run:
S4 77,140 and S5 draft 87,495 handles after 20 nets. The lead picked --evict5 for S5. The --lru valve (stack_S5f_mem) is queued
for S6. engineer2's view-binding idea (about 1.4% of bill per net in billed reshapes) is v9 again: bit-exact but compute-walled.


## S6 candidate: padded six-level pushes on top of push5 (E-30, 2026-09-23 02:15)

`candS3_HEB_push5p6.py` (sha 3e166670) = push5 with `--depthrule depthrule_fragment_push5p6.py` (make_depthrule_fragment_v3.py, MU 0.4).
Six levels now also cover 1024-row pushes whose k or n is not a multiple of 64. k and n are zero-padded to multiples of 64 and the
batched D6 push returns the leading n columns (`_push6_padded`). At MU 0.4 the model sends all 23 such shape classes to padded D6.
Their rates are 0.44-0.90 Gflop per model-second.

| vs push5 | r35 cold | r100 | r37 | r39 |
|---|---|---|---|---|
| util | -0.20% | -0.41% | -0.80% | -0.55% |
| final MSE (rounding) | +0.33% | -0.39% | +0.12% | +0.06% |

Requests are about equal (47,683 vs 47,703 on r35): the padded shapes reuse the D6 4096-packet frames. Compute is still open. The
model says about +3 s loaded, likely +1-2 s quiet. Quiet window 5 overlapped two other flopscope lanes and drifted 9 s
(push5 41.61 / 32.17 s, push5p6 41.15 / 39.08 s). The file is with engineer2's concurrent-pair gate.


## float64 share of the bill and the float32 patch (E-29, 2026-09-23 01:57)

flopscope bills float64 at twice the float32 rate. The census wraps flopscope's own charge function and resolves the billing dtype
of every call (dtype_census.py -> dtype_census_S3HE.json). S3-HE bills 2.0% of its flops at the float64 rate on r35 and on r100,
so converting all of it would save at most 1.0% of the bill.

| float64 by array class (r35) | % of bill | where |
|---|---|---|
| n x columns third-order planes | 0.92 | the diagonal sum(pp^3 lam) (0.48), omitted_radial_source's two matvecs (0.38), terminal k3 |
| n x n covariance state | 0.55 | C, K21, cp updates in _predict_binary, features, the pair map, at 0.036% or less per site |
| batched n x n | 0.21 | _bd_update's (4,n,n) direction stack, the pair series |

Patch "third" (build_f32.py; `build_eng.py --f32 third`). The planes pp and lam are already float32 (kernel outputs), and the card
cast them to float64 only to sum them. The patch keeps the sums in float32 and widens only the reduced vectors.

| vs S3-HE | third | third + bd |
|---|---|---|
| util r35 / r100 | -0.52% / -0.52% | -0.61% / -0.62% |
| float64 share (r35) | 1.10% (from 2.01%) | 0.89% |
| final MSE r35 / r100 / r37 / r39 | -0.21 / -0.08 / +0.12 / +0.16% (mean -0.003%) | +0.24 / -0.25 / +0.11 / +0.27% (mean +0.09%) |
| quiet server CPU r35 cold (HE 35.80, 31.31 s) | 33.61, 31.05 s: neutral (only the second pair is clean, -0.26 s; engineer2's run overlapped the first) | 31.35, 31.24 s: neutral |
| server peak RSS | about -0.15 GB (two 187 MB float64 plane copies gone) | about -0.1 GB |

Any rounding-level change moves the per-net final MSE by about 0.2% sd; push5 read -0.10/-0.05/+0.06/+0.35%. So a bound of +0.05% on
single nets cannot be certified; VALID40's mean decides. The patch composes with the physicist's L15 row cut: it patches the row
cut's own float64 k3_rows line when that is present. Recommendation: "third" into S5, "bd" optional. The remaining 0.89% is the
covariance state and was left alone.

engineer2 reviewed "third" (09-23 04:05): it is correct, omitted_radial_source is the only reader of the cache, and it matches his v7
and v8 bodies and the diet. His note: lam carries plus/minus pairs, so the sums cancel. Measured in situ (f32_precision_check.py,
r35, float32 vs float64 on the same inputs): the diagonal cancels up to 65x with relative error at most 6e-7; the omitted-radial
contraction cancels up to 81x with relative error at most 1.2e-6. That is about 4 orders of magnitude below the kernel's own float32
rounding (about 1e-5).

## Depth trade menu relative to S5 (E-28, depth_menu.py on depth_bench_v3merged.json; per net, mean of r35/r100/r37/r39)

Seconds are model seconds (in-process numpy time on a loaded box).

| buy bill | products/net | bill | compute | Gflop per s |
|---|---|---|---|---|
| five-level pushes not /64 -> padded D6 | 7.8 | -0.52% | +3.0 s | 0.68 |
| 1024^3 squares D5 -> D6 | 72 | -1.41% | +9.25 s | 0.60 |
| other squares D5 -> D6 | 11 | -0.02% | +0.8 s | 0.11 |

| free compute | products/net | bill | compute | Gflop per s freed |
|---|---|---|---|---|
| k=384 pushes D6 -> D5 | 18 | +0.57% | -4.35 s | 0.52 |
| thin reads (n=384) D5 -> D4 | 20 | +0.61% | -2.4 s | 1.00 |
| k=384 pushes D6 -> D4 | 18 | +1.64% | -6.4 s | 1.02 |
| k>=768 /64 pushes D6 -> D5 | 7.2 | +0.68% | -2.1 s | 1.28 |
| other squares D5 -> D4 | 28 | +0.20% | -0.6 s | 1.29 |
| 384-row pushes D5 -> D4 | 4 | +0.14% | -0.4 s | 1.39 |
| k>=768 /64 pushes D6 -> D4 | 7.2 | +1.68% | -3.1 s | 2.14 |
| 1024^3 squares D5 -> D4 | 72 | +2.25% | -3.45 s | 2.58 |
| wide reads D5 -> D4 | 17 | +2.06% | -2.6 s | 3.13 |

Quiet calibration (paired census, r35 cold): k=384 D6 -> D5 freed only 0.7-1.35 s against 4.35 model seconds; 1024^3 D5 -> D6 cost
+5.9 s against 9.25; the push5 fill cost +0.5 s against 1.0. The loaded bench over-weights the memory-bound deep push transforms.

---

## S5 candidate: five-level pushes under a compute budget (E-28, 2026-09-23 01:30)

**Deliverable:** `fable_engineer_v1/candS3_HEB_push5.py` (sha256 36573984…) = S3-HEB + `build_eng.py --depthrule depthrule_fragment_push5.py`.
Functionally identical to the measured `candS3_HEB_dr2a.py`; only a provenance comment differs.
It lowers the bill by about 1.9% on warm nets at the same local compute as S3-HEB.

| vs S3-HEB (same process order: r35 cold, then warm nets) | r35 | r100 | r37 | r39 |
|---|---|---|---|---|
| util (native, in-process) | .18251 vs .18341, -0.49% | .17779 vs .18133, -1.95% | .17542 vs .17877, -1.87% | .18457 vs .18823, -1.94% |
| final-layer MSE (rounding only) | -0.10% | -0.05% | +0.06% | +0.35% |
| adjusted = MSE x util | -0.59% | -2.00% | -1.81% | -1.60% |

Paired quiet census, r35 cold, server CPU in seconds (quiet_pairs_v2.sh pauses the heavy non-flopscope jobs and leaves ZeroMQ lanes running; three windows):

| window (order) | S3-HEB | push5 (dr2a) | dr2b | dr2c |
|---|---|---|---|---|
| 1 (HEB, 2a, 2b, 2c) x2 | 36.16, 38.69 | 43.79, 39.25 | 43.75, 39.89 | 45.95, 43.07 |
| 2 (2b, HEB, 2a, 2a, HEB, 2b) | 31.69, 34.53 | 32.15, 32.79 | 30.92, 31.32 | |
| 3 (HEB, 2a, 2c, 2c, 2a, HEB) | 34.09, 31.32 | 32.06, 32.01 | | 38.02, 37.88 |

push5 minus HEB, paired: -2.03, -1.74, +0.46, +0.56, +0.69, +7.63 s. The median is +0.5 s, inside the +2.5 s budget. The +7.6 s pair is
the first round of window 1, where dr2b (which the in-process profile shows is faster than HEB) also read +7.6 s, so it is a
window artifact. Requests +1,394 (+3.0%, about +0.3 s at 0.22 ms each on the platform). Server peak RSS 6.00-6.07 GB vs 5.66-5.99 GB.

**What changed.** A push whose k or n is not a multiple of 64, and every 384-row push, could only take four Winograd levels. It now
takes five, through a new five-level push layout. `make_push_layout.py` generalizes the frozen W-F generator and reproduces its D6
layout exactly at L=6. Pushes wider than 4096 run as a 4096 packet plus the rest, as the six-level pushes do. Six-level pushes,
reads, squares and W5 products are unchanged. The rule is `bill + MU x seconds`, with MU = 0.4 Gflop of bill per compute-second and
a fitted time model; at this MU it is equivalent to "five levels wherever six are impossible".

**Why the lead's first idea (pull D6 off the k=384 pushes) does not win the bill.** Measured as dr2b: it saves about 1 s of local
compute, but it gives back 0.5% of the bill (+0.02% on r35, -1.45% on r100). The only things that time could buy cost about the same
per second. Six-level 1024^3 squares (dr2c) buy -1.1% more bill for +4.4 to +5.3 s, which is over the budget.

### The compute model (in-process numpy seconds, 1 thread; depth_bench.py, leaf_rate_bench.py)

- Transforms dominate. Their time follows the elements written, (m k + k n + m n) x S(L), with S(L) = sum_{j<=L} (7/4)^j = 19.6,
  36.0 and 64.7 for L = 4, 5, 6. subtract(out=) writes 1.6-2.9 G elements per second (19-35 GB/s).
- Leaf GEMMs run at 34-67 Gflop/s as one contiguous batched call. The inner dimension 6 of a k=384 six-level leaf gives 34 Gflop/s;
  inner dimensions 12-32 give 57-61. In the kernel the leaf stage works on strided quadrant views and runs about 4x slower.
- So each extra level adds about 80% transform traffic for a 12.5% cut in leaf flops. The output side, m n S(L), does not shrink
  with k, so thin products gain the least from depth.
- A 6-feature non-negative fit per (kind, depth) reproduces 35 push shapes to 4-6% rms. The depth rule uses it.

Marginal exchange rates, Gflop of bill saved per extra in-process second on a loaded box (depth_bench_v2/v3):

| step | pushes k=384 | pushes k>=768 | 1024^3 squares | reads |
|---|---|---|---|---|
| D4 -> D5 | 1.2-3.5 | 2.5-10.7 | 2.5 | 1.0-5.9 |
| D5 -> D6 | 0.2-0.74 | 0.76-1.9 | 0.56 | no layout |

### The frame bill decides depth changes on new shapes

Exact-shape frames are bound by billed reshapes, one flop per element, each time a new (layout, depth, m, k, n) appears. An
unpacketized five-level push binds about 0.5 Gflop per new shape, 15% of the product. The D6 pushes share their 4096-packet frames
across all wide pushes with the same k. The first variants (dr1a/b/c, unpacketized) lost 0.6-0.8% of the bill to binding on the
cold net (dr1a r35 util +0.62%). The 4096 packets (dr2) removed it.

| variant | rule | util r35 / r100 vs HEB | MSE r35 / r100 | local compute vs HEB |
|---|---|---|---|---|
| dr1a | unpacketized D5, MU 0.6 | +0.62% / -0.72% | -0.14% / -0.13% | in-process only |
| dr2a = push5 | packetized, MU 0.4 | -0.49% / -1.95% | -0.10% / -0.05% | median +0.5 s |
| dr2b | packetized, MU 0.6 (k=384 D6 -> D5) | +0.02% / -1.45% | -0.15% / -0.14% | about -1 s |
| dr2c | push5 + D6 for 1024^3 squares | -1.62% / -3.24% | +0.12% / -0.41% | +4.4 to +5.3 s |

About 2 s of the budget is unused. The frontier past push5 is flat, at about 0.5 Gflop of bill per extra in-process second (roughly
0.2% of the bill per quiet second). The candidates are padded D6 on the five-level pushes (push64 style) and a subset of the 1024^3
squares at six levels. The squares have no clean shape split. Neither is built.

Files: make_push_layout.py (layouts), depthrule_fragment_v0p.py (layout + hook), make_depthrule_fragment_v2.py (rule generator),
depth_bench.py / depth_bench_v3merged.json (timings), depth_plan_v2.py (planner), product_profile.py, pred_mse.py, quiet_pairs_v2.sh.
candS3_HEB_dr0_bench.py is a mis-build (holdelim appended twice) and was never used.

## Call-site TIME profile of S3-HEB (E-27, 00:30): product_profile.py -> product_profile_S3HEB.json

Load was 14-15, so seconds run about 1.5x a quiet run. numpy time on r35: 73.8 s = kernel 53.3 + body 20.5 (r100 warm: 38.6 + 13.1).
By class: six-level pushes 30% of the time for 26.6% of the bill (4.8 Gflop per second), five-level reads 14.7 s / 29.4%,
D5 squares 10.2 s / 23.8%, four-level pushes 2.8 s / 8.3%. Body hot spots (engineer2): _predict_binary:3240-3244 (float64
diagonal, 3.9 s) and _nonlinear_pair_batch:906-919 (3.75 s). The kernel time grouped by the body line that requests the product:
_push_bank_shared_newborn:3059 11.7 s, _slr_reconstruct (k=384) 12.2 s, _shared_old384_read 8.7 s.

---

## Server compute by TIME (S3-HE, r35 cold; time_profile.py in-process with the server's numpy 2.4.6 + OpenBLAS, 1 thread)

| part | seconds (load 12-15, ~1.5x a quiet box) | notes |
|---|---|---|
| kernel, Winograd add/subtract | 26-28 | forward subtracts 11.1, backward adds 7.4, read-packet subtracts 2.4 |
| kernel, leaf matrix products | 10.4-10.7 | ~30 GFLOP/s on average; D6 pushes with k=384 have leaves of inner dimension 6 |
| kernel, copies (stack/copyto) | 6-7 | HEB: -1.4 s, but its leaf products +1.1 s (4 matmuls on strided views) |
| card body (engineer2's lane) | 10-13 | _nonlinear_pair_batch multiply 2.0, _predict_binary multiply 1.3, _nonlinear_pair_batch sum 0.6 |

The transforms write ~4.4e10 elements per network (~0.5 TB of traffic) at 20-35 GB/s, i.e. near single-thread memory bandwidth.
A bit-exact time cut must move fewer elements; the only traffic left to remove is the level-(L-1) copies (~2.4 s local), which
costs ~+30k requests (~+6.6 s platform at 0.22 ms each). So -5 s bit-exact is out of reach inside the kernel.

## Under the compute-bound platform model (lead, from the platform's own telemetry)

Platform kernel time is about 2.4x our local quiet single-thread server CPU; overhead is only ~0.22 ms per request. So request
diets buy little wall, and any bill cut may add at most ~+2.5 s of local server CPU vs S3. Re-ranked kernel options:

| option (build_eng.py flags on S2) | util vs S2 (r35 / r100) | parity | local compute vs S3-HE | status |
|---|---|---|---|---|
| S3-HE `--holdelim` | -0.88% / -0.93% | bit-exact | reference (1.7 s below S2) | validated base |
| **S3-HEB `--holdelim --bigarena`** (candS3_HEB.py) | -1.26% / -1.18% | bit-exact | lower: D6 pushes 5-21% faster in-process (their last-level copies go too) | engineer2's memory/timing gates (+0.40 GB live arena) |
| S3-HEB + `--push64` (candS3_HEB_p64.py) | -1.97% / -3.38% (adj -1.75% / -3.31%) | rounding-level, MSE +0.22% / +0.08% | push64 +1.5..+4 s vs HEB's savings | queued behind HEB |
| rule re-priced for cheap requests (130k vs 750k flops per call) | -0.16% (r35 census replay) | rounding-level | more depth on mid-size products | not worth it |
| sq6 (depth-6 squares) | -2.05% / -2.23% | rounding-level | +6..+11 s | out |
| v9 view binding | -4.4% cold / -2.3% | bit-exact | +15..+28 s | out |
| level-(L-1) copy elimination | ~-0.7% (bill model) | bit-exact | uncertain (more, smaller ops) | parked |

---


## LANE DONE (after midnight): the kernel layer is `--holdelim`; depth on the freed wall is marginal

Three paired quiet windows (every other heavy python job paused; candidates alternating; numpy-free client + OpenBLAS server,
1 thread; nets r35 cold / r100). Even paused, the machine varied 5-10 s in server CPU between runs (the bank's own r35 read
36.4, 37.7 and 47.7 s), so deltas below ~5 s are not resolved. Wall models (lead's rule: <= the bank under both):
porter P(c) = c x requests + 2 x (wall - 0.3 ms x requests), c = 1.1-1.4 ms; engineer2 E = 1.25 ms x requests + 2 x server CPU.

| build on the fired S2 (build_eng.py flags) | util vs S2 (r35 / r100) | MSE vs S2 | adjusted vs S2 | platform wall vs bank, porter model | verdict |
|---|---|---|---|---|---|
| S3 `--holdelim` (candS3_S2_HE.py) | -0.88% / -0.93% | identical | -0.88% / -0.93% | r35 -3.5..-23 s, r100 ~-20 s | clears; bit-exact |
| S3 + `--sq6` (candS3_sq6.py) | -2.05% / -2.23% | +0.07% / +0.04% | -1.98% / -2.19% | r35 -9..+9 s | marginal |
| S3 + `--push64` (candS3_p64.py) | -1.51% / -2.92% | +0.22% / +0.08% | -1.29% / -2.84% | r35 +1.4 / +4.9 s, r100 -5.5 / -8.1 s | fails r35 |
| S3 + `--sq6 --push64` (candS3_sq6_p64.py) | -2.68% / -4.22% | +0.29% / -0.01% | -2.39% / -4.23% | r100 +0.8..+3.9 s (requests-only models) | fails r100 |

- sq6 costs +6 to +11 s of server CPU per network against S3 (one outlier +24 s): 72 squares x ~+110 ms, because 117,649
  16x16x16 leaf GEMMs and a sixth level of transforms replace 16,807 32x32x32 ones. Its flops-per-platform-second (~264M) is
  below the bank depth rule's own price (~600M per platform second), which is why the rule never chose it.
- Keep the depth builds for later: if S3's graded platform wall shows real headroom, S3 + push64 is the cheapest depth trade.
- RSS of every variant 4.4-6.2 GB (within S2 + 0.5 GB).

---


## S3 (night): hold + copy elimination on the fired S2, plus depth-6 squares paid for by the freed wall

Paired QUIET boundary census (fable_fire_v1/quiet_census.sh: every other heavy job paused; numpy-free client + OpenBLAS server,
1 thread; nets r35 cold / r100). Native util and MSE from one process (r35 then r100). Base = the fired S2
(fable_fire_v1/candS2_e2v7_k31r_bfix01_every3.py: e2 v7 + K31 row + b-fix 0.1 + every3 join, bank kernel).

| build (build_eng.py flags on S2) | round trips | server CPU s | local wall s | util vs S2 | MSE vs S2 | adjusted vs S2 | parity vs S2 |
|---|---|---|---|---|---|---|---|
| bank (reference) | 95,366 / 84,143 | 36.4 / 34.0 | 41.9 / 38.6 | — | — | — | — |
| S3 = `--holdelim` (candS3_S2_HE.py) | 53,797 / 45,983 | 31.9 / 29.3 | 35.4 / 32.3 | -0.88% / -0.93% | identical | -0.88% / -0.93% | bit-identical, 16 layers |
| S3 + `--sq6` (candS3_sq6.py) | 55,575 / 47,620 | 38.0 / 35.5 | 41.5 / 38.7 | -2.05% / -2.23% | +0.07% / +0.04% | -1.98% / -2.19% | rounding-level (2.9e-6 / 2.3e-6 rel RMS) |
| S3 + `--sq6 --push64` (candS3_sq6_p64.py) | 56,150 / 49,205 | 39.3 / 38.7 | 43.2 / 42.3 | -2.68% / -4.22% | +0.29% / -0.01% | -2.39% / -4.23% | rounding-level (2.7e-6 / 2.5e-6) |

- Platform wall estimate vs the bank (0.7-1.2 ms per round trip + 2x server CPU): S3 -36..-59 s, S3+sq6 -23..-45 s,
  S3+sq6+push64 -15..-41 s. Server peak RSS: bank 6.31 GB, S3 6.00, S3+sq6 5.82, S3+sq6+push64 6.20 (S2 6.06 in its quiet run).
- sq6: square products (all dims <= 1024) may take a sixth Winograd level when all dims divide 64 and the depth rule's own value
  says so (in practice the 1024^3 products, 72 per network: -5.5% bill each, +110 ms local each). The hold-elim kernel runs it on
  the SQUARE6 arena layout. push64: 1024-row pushes whose k or n is not a multiple of 64 are zero-padded to depth 6 when it wins.
- v9's view-binding cut stays parked: the frame bill is 1.1% of the bill per later network (12-network replay), 69% of it in
  the read path whose inner dimension k takes ~64 distinct values; every sliced-view form costs 1.1-3.7x on the transform ops.
- Recommendation: next fire S3 + sq6 (about -2% adjusted, wall well under the bank's); S3 + sq6 + push64 if VALID40 confirms.

---


## FINAL (late evening): fire with the hold + copy-elimination layer on the bank's own kernel (HEB)

| kernel layer on the e2 + physics base (build_eng.py flags) | util vs bank-kernel base (native, r35 / r100) | round trips (boundary, r35 cold / r100 / r35 repeat) | server CPU vs H | parity |
|---|---|---|---|---|
| bank kernel (base c3_e2_phys; bank alone for round trips) | — | 95,366 / 84,143 / 79,395 (bank alone) | — | — |
| H `--holdonly` | 0 | 59,403 / 48,632 / 43,066 | — | bit-exact |
| **HEB `--holdelim --bigarena`** (candHEB_holdelim_big_on_c3.py) | **-1.23% / -1.18%** (bank alone) | 60,998 / 49,687 / 43,670 | same (45.6-61.8 s vs 51.5-60.2 s, load 7-11) | **bit-exact** |
| HE `--holdelim` (candHE_holdelim_on_c3.py) | -0.86% / -0.91% (bank alone) | as HEB | same | bit-exact |
| A `--grouped --read5` (v9, candA_v9_on_c3.py) | -4.43% / -2.30% (bank alone) | 57,535 / 41,889 / 32,123 | +12 to +36 s per net | bit-exact |

- HEB = the bank's own kernel structure and frames, (1) every discarded out= alias and per-call view held and released once per
  call (engineer2's hidden-free-trip finding: bank 36.9k free trips per cold net), (2) no copies at the last Winograd level: the
  leaf products read the plain quadrants (A: a11, a12, a22; B: b11, b21, b22) from the previous level (4-5 leaf matmuls instead
  of 1; same operand pairs and order). Where the layout reuses that level's region for the leaf, the leaf moves to the free
  arena tail; --bigarena grows the arena by 396 MB of address space so the D6 pushes qualify (up to ~0.48 GB resident).
- Per-product local time equals the bank's (1024^3 D5 0.062 -> 0.061 s, D6 push 0.233 -> 0.233 s, read D5 0.278 -> 0.275 s);
  boundary server CPU equals H's within load noise. Server peak RSS (macOS, noisy): HEB 7.12 / 4.99 GB, H 5.49 GB.
- A (v9) bills much less but its grouped ops and maximum-shape views cost 12-36 s of server CPU per network, which outweighs its
  extra 2-11k round trips; not recommended unless a quiet-box timing proves wall headroom.
- HEB stack parity (native): bit-identical to its e2+physics base on r35/r100, all 16 layers; util .18487 / .18132 vs base
  .18719 / .18350. Against the bank: util -1.81% / -1.78%, MSE -1.47% / -1.16% (physics), adjusted -3.26% / -2.91% on r35/r100.
- View binding stays exact. Keying views by the recurring row count and slicing columns would cut the frame bill from 1.11% to
  0.12% per later network (12-network key replay), but numpy runs the transforms 2.4-3.7x slower on column-sliced views and
  1.1-1.6x slower on row-sliced views (bank-layout microbenchmark), so the exact frames' bill is the price of speed.

---


## Latest (late evening): use the hold-only kernel; v9's bill cut costs too much server compute

Interleaved boundary censuses of the full stack (bank -> build_e2 -> build_physics -> kernel layer; numpy-free client + OpenBLAS
server, 1 thread; census_cpu.py = engineer2's tool with server CPU seconds; nets r35 cold / r100 / r35 repeat):

| kernel layer (build_eng.py flag) | round trips | server CPU s, pair 1 (load 11-14) | server CPU s, pair 2 (load ~10) | native util r35 / r100 | parity |
|---|---|---|---|---|---|
| bank kernel (reference: bank alone) | 95,366 / 84,143 / 79,395 | — | — | .18828 / .18460 | — |
| H `--holdonly` (candH_holdonly_on_c3.py) | 59,403 / 48,632 / 43,066 | 70.0 / 58.7 / 66.9 | 60.9 / 46.4 / 49.2 | = e2+physics base (.18719 / .18350) | bit-exact vs its base |
| A `--grouped --read5` = v9 (candA_v9_on_c3.py) | 57,535 / 41,889 / 32,123 | 100.5 / 94.9 / 91.8 | 72.7 / 70.5 / 73.9 | .17884 / .18050 (porter) | bit-exact vs its base |

- H is the bank's own kernel with every discarded `out=` alias and per-call quadrant view held and released once per call
  (engineer2's hidden-free finding). Bit-identical on the bank (r35/r100, all 16 layers), util identical, no compute change.
- A saves only 2k-11k more round trips than H but costs 12-36 s more server CPU per network (+19% to +62%). At the platform's
  ~2x on local compute that is a net wall LOSS against H. Other censuses under changing load: A vs bank server CPU -6% to +64%;
  engineer2's run_mem: e2+v9 90.9/79.3 s vs bank 68.9/65.2 s. Server live-array peak (engineer2): bank 2.06 GB, v9 alone 2.16 GB.
- Recommendation: fire candidate on H. A's bill cut (-2.3% warm, -4.4% cold network) is only worth it if a quiet-box timing shows
  the platform wall absorbs its extra compute; near the bank's 110 s worst I expect it cannot. The v10 (--push64) and --dense
  variants inherit A's compute profile (--dense stack hit two client timeouts at the boundary under load; not diagnosed).

---


## Update (evening): free view binding reaches -4.4% util on the first network, bit-exact

The lead asked for (1) free view binding and (2) depth-enabling rounding. Both measured natively (one process, r35 then r100, the
lead's protocol; local flopscope 0.10, 2 BLAS threads). Bank: util 0.18828 / 0.18460, MSE 1.648017257e-8 / 1.680305197e-8.

| variant (builder flags) | util r35 / r100 | util vs bank | MSE r35 / r100 | adjusted vs bank | parity vs bank |
|---|---|---|---|---|---|
| v7c `--grouped --read5 --v7c` (grouped kernel, relocated leaf, held aliases) | 0.18742 / 0.18347 | -0.46% / -0.61% | identical | -0.46% / -0.61% | bit-identical, 16 layers |
| **v9 `--grouped --read5`** (v7c + views bound once per process) | **0.17994 / 0.18036** | **-4.43% / -2.30%** | identical | -4.43% / -2.30% | **bit-identical, 16 layers** |
| v10 `--grouped --read5 --push64` (v9 + depth-6 padding of pushes) | 0.17904 / 0.17778 | -4.91% / -3.69% | 1.6578e-8 / 1.6872e-8 (+0.59% / +0.41%) | -4.35% / -3.30% | rounding-level: final-layer rel RMS 5.7e-6 |
| v9d `--grouped --read5 --dense` (v9, faster numpy: single corner writes, exact narrow/half-empty views) | 0.18257 / 0.18073 | -3.03% / -2.10% | identical | -3.03% / -2.10% | bit-identical, 16 layers |
| engineer2 cand_e2_v7 + v9 (cand_stack_e2_v9.py) | 0.17884 / 0.17926 | -5.02% / -2.89% | identical | -5.02% / -2.89% | bit-identical, 16 layers |
| engineer2 cand_e2_v7 + v10 (cand_stack_e2_v10.py) | 0.17795 / 0.17669 | -5.49% / -4.28% | as v10 | -4.93% / -3.89% | = bank + push64, bit-identical |
| bank + push64 only | 0.18714 / 0.18171 | -0.61% / -1.57% | as v10 | -0.02% / -1.16% | as v10 |
| bank + select() rounds to 64 | 0.19027 / 0.18365 | +1.06% / -0.51% | 1.6066e-8 / 1.7731e-8 | -1.48% / +4.98% | predictions change (more units) |
| bank + select() rounds to 128 | 0.19502 / 0.19100 | +3.58% / +3.47% | 1.7436e-8 / 2.1636e-8 | +9.6% / +33.2% | predictions change |

Production boundary (numpy-free client + OpenBLAS server, 1 thread; census_rt.py = engineer2's tool; three nets in one process:
r35 cold, r100, r35 repeat). A round trip is a request or a hidden `free` trip (the client frees discarded handles in a separate
trip before the next request; engineer2's finding). The bank row is engineer2's measurement with the same tool.

| variant | requests | hidden free trips | round trips | vs bank | util (0.12 server) | server peak RSS |
|---|---|---|---|---|---|---|
| bank | 58,433 / 50,045 / 45,745 | 36,933 / 34,098 / 33,650 | 95,366 / 84,143 / 79,395 | — | .18838 / .18469 / .17977 | 3.46 GB |
| v9 | 55,398 / 42,703 / 34,698 | 12,242 / 9,657 / 7,896 | 67,640 / 52,360 / 42,594 | -29% / -38% / -46% | .18003 / .18045 / .17861 | 3.32 GB |
| e2 + v10 (cand_stack_e2_v10.py) | 51,021 / 38,511 / 30,050 | 9,468 / 6,840 / 5,030 | 60,489 / 45,351 / 35,080 | -37% / -46% / -56% | .17795 / .17669 / .17655 | 4.67 GB |

Walls and residuals were not resolvable: load average 15-35 all evening (v9: 171 / 120 / 99 s, residual 0.88 / 0.60 / 0.39 s;
e2+v10: 101 / 87 / 99 s, 0.47 / 0.32 / 0.38 s; the bank alone took 88-148 s under the same load).

Local kernel time per product (in-process, 1 thread, median of 5, each run paired with the bank in the same process; loaded
machine, +-15% noise):

| product (m, k, n, depth) | v7c vs bank | v9 vs bank | v9d vs bank |
|---|---|---|---|
| square 1024^3, D5 (72 per net) | +16% | +5% | +5% |
| push 1024 x 384 x 4096, D6 (25 per net) | +36% | +54% | +25% |
| push 1024 x 960 x 4096, D6 | — | — | +13% |
| read 928 x 8128 x 992, D5 | +5% | +14% | +37% (noise) |

Where the time goes (D6 push): grouped writes into strided PAIRS of corners cost 1.5-2x the two separate writes in numpy (v9d
writes corners singly), and a view sliced from a half-empty maximum base spreads the data over twice the memory (v9d keeps exact
rows then). The platform's wall model (about 1 ms per round trip + 2x local compute) makes v9's 28k-37k fewer round trips worth
roughly 28-37 s per network against an estimated 5-7 s of extra local compute (10-14 s at 2x), but that is a model, not a
measurement.

### (1) Free view binding — what and why
- The bank binds a view set for every new product shape by reshaping arena regions (billed 1 flop per element). Over 12 networks in
  one process (frame_keys.py + frame_schemes.py, model = measured bill to 3 digits) that costs 4.1% of the bill on the first network
  and about 1.1% on every later network (0.4-1.8%), because every network brings new alive/birth counts, hence new shapes.
- v9 reshapes each arena region ONCE per process into its layout's maximum blocked shape; every exact view is a slice of that base
  (billed 0). The bank's own maximum-shape binding at the first prediction (0.56% of a network) is replaced by a free registration,
  since the grouped kernels read only (arena, specs). What stays billed per new shape: the three top-buffer views (m*k + k*n + m*n).
  Frame share of the bill: bank 4.01% / 1.76% -> v9 0.79% / 0.12%.
- Keying views on the column count only (rows sliced) was priced first and saves only 0.46 points (frames 0.65% steady state).
- Cost: views sliced from maximum-shape bases have strided rows; a numpy microbenchmark of the forward transforms gave +7% to +42%
  (noisy, loaded machine). Memory: views only, no new server memory beyond the 100 MB top buffers.

### (2) Depth-enabling rounding
- select() rounding to 64/128 includes extra units: the bill of every product grows and the predictions change; on r35 only three
  1024-row pushes were depth-capped by the /64 rule. Net negative on adjusted.
- push64 instead zero-pads k, n of those pushes to multiples of 64 inside _mm when the depth rule's own value (bill + 750000 per
  request, + padding copies) says the padded depth-6 product wins. Physics unchanged; the product changes at rounding level
  (5.7e-6 final-layer relative RMS), which moves the final MSE by +0.4-0.6% on these two networks (the gate selections amplify it).
- Deeper squares (depth 6 for the 72 1024^3 products): -5.5% bill per product but +157 ms local each (117,649 16x16x16 GEMMs),
  about +11 s local per network. Rejected.

### Deliverables
- `build_eng.py <base.py> <out.py> --grouped --read5 [--push64]` (appends fragments; checks every overridden function exists;
  refuses to overwrite). Fragments: `grouped_fragment_v9.py`, `grouped_read5_fragment_v9.py`, `push64_fragment.py`.
  Overrides: _packed_product_cached, _packed_read5, _w5_port_product_body, _cp3_bind_arena_views (+ _mm with --push64).
- Built: cand_v9.py (sha 13a621e5), cand_v10.py (743c6e24), cand_v9d.py (63257b5a); stacked on engineer2's cand_e2_v7.py:
  cand_stack_e2_v9.py (b4fae364), cand_stack_e2_v10.py (100f9b18). Builder flag --dense selects v9d.

### Recommendation
For the next fire, stack e2 + v9 (bit-exact: util -5.0% r35 / -2.9% r100) or e2 + v10 (+ push64, rounding-level: util -5.5% / -4.3%,
MSE +0.59% / +0.41%) after ONE quiet-box boundary timing against the bank (5 nets, walls, residual, output sha); the round-trip
cut (-37% to -56%) should dominate the extra local compute, but that is unmeasured. If the quiet-box wall is worse, use --dense
(v9d: -3.0% / -2.1%, faster numpy). The r35 numbers include the first network's one-time binding; over many networks the frame
part of the saving settles near 1 point (bank ~1.1% of the bill per later network -> v9 ~0.1%), so the board-average util cut of
e2 + v10 should be ~3.5-4%, not 5.5%. Deeper squares and select() rounding are dead.


---

## First pass (afternoon), kept for the record


Lane: cut the bank card's (#331824 = cand_bank.py, sha d79463f3…) cost with identical predictions. Verdict: **no fire-ready variant; the
>= 4% util bar is not reachable by kernel engineering at identical predictions.** The best variant (v5) is bit-exact and cuts warm
requests by 22%, but its bill is flat (+0.29% cold r35, +0.02% r100) and its cold-network request count is 9% higher.

## Table (util: native flopscope, one process r35 then r100 = the lead's protocol; requests: numpy-free client + OpenBLAS server)

| variant | util r35 / r100 | vs bank | requests cold r35 / r100 / warm r35 | boundary wall r35 cold / worst (s) | residual worst (s) | parity |
|---|---|---|---|---|---|---|
| bank (cand_bank.py) | 0.18828 / 0.18460 | — | 58,433 / 50,045 / 45,745 | run 1: 51.3 / 64.6; run 2: 56.5 / 148.5 (load spike) | run 1: 0.358; run 2: 0.674 (spike) | reference |
| v2 grouped main kernel (cand_grouped_v2.py) | 0.18911 / 0.18517 (boundary) | +0.44% / +0.31% | 57,176 / 43,938 / 36,715 | 72.0 / 72.0 | 0.388 (cold r35) | output sha identical on 5 nets |
| v5 = v2 + grouped read5 + last-level copy elimination (cand_grouped_v5.py) | 0.18883 / 0.18464 | +0.29% / +0.02% | 63,557 / 47,288 / 35,459 | 63.6 / 65.3 | 0.346 | output sha identical on 5 nets; final MSE identical (1.648017257e-8 / 1.680305197e-8) |

Walls are from a machine shared with the lead's jobs (backend time 39-55 s for the SAME bank bytes across runs); treat them as
noisy (the bank's own worst wall ranged 64.6 s to 148.5 s between two runs). Bank run 1 and v2 ran back to back; v5 and bank run 2
ran back to back later under heavier load (load average ~12). No wall or residual claim is made for any variant.

## What was built (all bit-exact; files in this directory)
- `grouped_fragment_v5.py` (+ `grouped_read5_fragment_v5.py`), appended to the bank source by `build_grouped.py`. Same Winograd
  recursion, same operand pairs and operation order per element, so every product is bit-identical (13/13 production shapes via
  `test_kernel.py`; full predictions identical locally and at the production boundary).
- Request structure per Winograd level: 12 requests (4 A, 4 B, 4 backward) instead of 18, by applying each ufunc to a strided PAIR of
  quadrants/slots (every arena region viewed once as a fully blocked array; slot order permuted 3,0,1,2,4,5,6 so each side's copies are
  one stack; backward s1/s2 in the u-scratch so no op reads and writes overlapping memory). Per product: D5 square 104 -> 67
  requests, D6 push 122 -> 77, D5 read 142 -> 97. The top operands are copied once into fixed blocked buffers (bill +m*k+k*n+m*n).
- Last-level copy elimination: the leaf products read the level-(L-1) quadrants directly (4 leaf matmuls instead of 1) and skip 3/7 of
  the last level's slot writes. Worth 1.27% of the bill in principle; only ~0.8% is realisable because the push/square arena
  layouts reuse the level-(L-1) B region for the leaf products (static overlap check per shape; that side keeps its copies).

## Why the bill does not move (measured)
- Transform einsum (the lead's idea 1) is not viable: flopscope bills einsum as (2K-1) x outputs, so a 7x4 transform costs 12.25 m k
  per level versus 1.75 m k for the add/subtract/copy chain (7x). Transforms are 10.6% of the bill.
- Bill anatomy of the bank (r35 cold, product_census.py): Winograd kernels 91.2% (leaf matmuls 76.6%, transforms 10.6%, view-binding
  reshapes 4.1%), plain contractions 2.6%, rest body.
- View binding ("frames") is a recurring per-network cost, not a one-off: every network brings new product shapes, and each new shape
  reshapes its arena regions (billed 1 per element). frame_bill.py: bank 4.01% of the bill on the first network, 1.76% on the second.
  The blocked layout adds the top-buffer views: v5 4.72% / 2.22%. Without frames v5 is -0.45% on both networks; with them, flat.
- Deeper recursion: at most 2.0-2.35% of the bill even ignoring requests (divisibility-limited), and it is wall-limited by tiny leaf
  GEMMs (a D6 1024^3 square = 117,649 16x16x16 BLAS calls; estimated +8 s local for the 72 squares alone). Not pursued.
- Grouped ops that write into strided quadrant PAIRS are 1.5-2x slower in numpy than two separate ops (bench in scratchpad; the
  forward pairs into contiguous slots are as fast). Per 1024^3 product: +5.6 ms local for 15 fewer requests (~ -7 ms platform at
  2x local + 1.25 ms/request); on the D6 pushes the local cost exceeds the request saving.

## Recommendation
Not fire-ready: v5 is bit-exact and saves ~10k requests per warm network, but the bill is flat and the cold network has +5.1k requests
(new blocked frames), which is the network most at risk of the 120 s wall. The single engineering lever left with size is the frame
bill (1.8% of the bill on a later network, 4.0% on the first): key the arena views on the trailing (column) dimension only and slice
rows (free), so the few recurring column counts are bound once per process; expected ~-1.5% on later networks at unchanged
predictions, cost = somewhat slower strided rows. Even with that and the copy elimination, kernel engineering tops out near -2% util;
the 4% (3e-9) step needs physics or a cheaper carrier, not the kernel.
