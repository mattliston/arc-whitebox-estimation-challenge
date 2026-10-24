# Stacked base and fire candidate A: gate record (porter, 2026-09-22)

**Verdict.** Candidate A passes parity, whest validate, the paired VALID40 and the 14-net panel. It does NOT clear the wall gate:
at matched load its server compute is 27-34% above the bank's. The bank already peaks at about 110 s of the 120 s platform limit, and the platform
estimate for A's first (cold) net is roughly 110-130 s. Do not fire A until a quiet-box or platform-calibrated timing settles the wall.

## Candidate A
`fable_fire_v1/candA_v9_e2_k31r.py`, sha256 50666329fb1789340fc9083622e72d5da90861031d24cd096caf32345f999c2f (verified).
Composition: engineer `cand_stack_e2_v9.py` (bank -> engineer v9 kernels -> engineer2 build_e2) -> `build_physics.py` k31row 0.85,0.5@1-14, extra plane off.
The physicist's TRAIN pick c0 0.80 (expected +0.2-0.3 points) was not used; the lead kept 0.85.

## Gates
| gate | result | pass? |
|---|---|---|
| 1. parity (`gate_candA_bill_r{35,100}.log`) | r35 MSE 1.6237239631400155e-08, r100 1.6609057416338460e-08, bit-identical to the K31-row-on-bank values; util r35 0.178843 (bank 0.188284, -5.01% bill), r100 0.180502 | yes |
| 2. whest validate (`gate_candA_whest_validate.log`) | success, exit 0 (contract only: a 2x4 network, where the estimator returns its zero fallback) | yes |
| 3. paired VALID40 vs cached bank rows (`paired_valid40_candA.json`) | adj_ratio 0.93727 (-6.27%), raw 0.98605, adj wins 40/40 (raw 38/40), util 0.179520 vs 0.188858 (-4.94% bill); per-net adjusted -8.78% .. -4.57% (median -5.93%); MSE bit-equal to the bank-based K31-row run on all 40 | yes (bar: <= 0.97, >= 24/40) |
| 4. 14-net panel vs `fable_refit_v1/panel_bank331824.log` (`panel_candA.log`, `gate_candA_panel.txt`) | panel8 adjusted 3.615e-9 vs 3.863e-9 (ratio 0.9358); virgin6 2.925e-9 vs 3.145e-9 (0.9300); raw 2.010e-8 vs 2.042e-8 and 1.631e-8 vs 1.666e-8 | yes (also the 2% margin of gate_and_fire.sh) |
| 5. boundary census, cold+warm (`census_candA_paired2.log` vs `census_bank_paired2.log`) | see below | NO (not proven inside the safe zone) |

### Census (numpy-free client + OpenBLAS server, 1 thread; r35 cold, r100, r35 warm in one client; back-to-back runs at load about 6)
| | bank | candA |
|---|---|---|
| requests | 58,433 / 50,045 / 45,745 | 50,815 / 37,815 / 29,810 |
| hidden free trips | 36,933 / 34,098 / 33,650 | 9,501 / 6,839 / 5,078 |
| round trips | 95,366 / 84,143 / 79,395 | 60,316 / 44,654 / 34,888 (-37% / -47% / -56%) |
| server CPU | 50.2 / 47.1 / 47.1 s | 64.2 / 59.9 / 63.1 s (+28% / +27% / +34%) |
| local boundary wall | 57.5 / 53.6 / 53.1 s | 69.7 / 64.2 / 66.9 s (+21% / +20% / +26%) |
| residual (cap 0.4 s) | 0.334 / 0.302 / 0.274 s | 0.294 / 0.247 / 0.225 s |
| server peak RSS | 5.96 GB | 5.87 GB |

Platform estimate. The September fit (memory: platform-wall-is-request-count-plus-compute) is
`platform ≈ 1.1-1.4 ms x requests + 2 x (boundary wall - 0.3 ms x requests)`. Absolute local walls are load-inflated tonight, so only
paired differences are used. On r35 cold, A lands about +20 s above the bank if only requests count, and about +2 s if each hidden free trip costs
like a request. With the bank at about 110 s worst, A's worst case is roughly 110-130 s. The extra time is server compute, presumably
from the deeper v9 schedules. A's client CPU is lower than the bank's.
Notes: an earlier A census at load 10 gave server CPU 80.6/77.2/79.0 s (`census_candA.log`). The first bank attempt timed out while four
servers competed (`census_bank_paired_attempt1_TIMEOUT.log`); the retry is the one used.

Options to settle the wall:
- a quiet-box paired timing;
- cap the v9 schedule depth on the push/read products (the Winograd depth trades bill for wall; D6 -> D4 was -27% wall for +4% bill in September);
- engineer2's chain without the engineer kernels (e2 v8 + physics: about -0.6% bill, round trips -11%, compute near the bank's).

## Earlier stacked base on engineer2 v5 (no engineer kernels): `cand_stack_e2v5_k31r.py`, sha fbee9331...
- Paired VALID40 vs cached bank: adj 0.98034 (-1.97%), raw 0.98605, adj wins 40/40; util 0.187767 (-0.578% bill); MSE bit-equal to the bank-based
  K31-row run on r35, r100 and all 40 VALID nets. Its panel and census were not run: the lead moved the fire base to engineer v9 (candidate A).

## Tools (fable_porter_v1/)
- `run_fire_gates.sh <cand> <tag>`: VALID40 under /tmp/fable_paired.lock, then the panel step, then the census.
- `run_census_porter.sh`, `census_boundary.py`: copies of engineer2's census writing here.
- `paired_eval_cached.py` + `bank_valid40_cache.json`; `t31_bill.py` (exact flops and ops).

---

## Update 22:45: A == engineer2's c4, and candidate B (every3 join schedule)
- **A and c4.** candA and `fable_engineer2_v1/chain/c4_final_phys.py` (sha d8985714...) have equal flops: r35 cold 393,280,994,057; r100 warm (second net
  in a process) 394,205,094,801, from the candA census. A cold r100 in its own process is 396,927,095,295 (frame binding). MSE is bit-exact, so A's VALID40
  and panel certify c4. `whest validate` on c4: success, exit 0.
- **Census.** A's census above was run before the lead's hold message, at load about 6 and paired with the bank. The lead now runs the quiet-window
  censuses (fable_fire_v1/quiet_*); porter runs no more censuses unless asked.

### Candidate B = `fable_fire_v1/candB_A_every3.py` (sha 9c96a91b..., = candA + the lead's every3 join schedule: old cohorts join at L7/10/13)
| gate | result | pass? |
|---|---|---|
| paired VALID40 vs cached bank (`paired_valid40_candB_A_every3.json`) | adj 0.90925 (-9.08%), raw 0.96413, wins 40/40 adj and 40/40 raw; util 0.178116 vs 0.188858 (-5.69% bill) | yes (bar: adj <= 0.95, >= 28/40) |
| vs candA per net (`compare_to_parent_run.py`) | adjusted below A on 39/40; pooled B/A adj 0.9701, raw 0.97777; per-net adj change vs A median -2.90%, range -6.04% .. +0.30% | yes (bar: most nets) |
| whest validate (candB_A_every3 and candB_c4_every3) | both success, exit 0 | yes |
| r35 parity candB_A_every3 vs candB_c4_every3 (sha bc6b4b3b...) | identical: flops 390,379,668,443 (util 0.177524), ops 30,941, MSE 1.6380296553665177e-08 | yes (VALID40 certifies both) |
| 14-net panel (`panel_candB_A_every3.log`, `gate_candB_A_every3_panel.txt`) | panel8 3.523e-9 vs bank 3.863e-9 (0.9120; A 0.9358); virgin6 2.881e-9 vs 3.145e-9 (0.9161; A 0.9300); raw 1.974e-8 / 1.619e-8; raw per net vs bank slightly worse only on r35 (+1.1%) and 3016502 (+1.3%), both covered by the -5.6% bill | yes (also the 2% margin) |
| census | held for the lead's quiet window | |

---

## Update 22:55: one-command S3 chain (for the engineer's fixed v9')
The lead fired S2 (`fable_fire_v1/candS2_e2v7_k31r_bfix01_every3.py`: e2 v7 + K31 row + b fix 0.1 + every3; gated by the lead).
Next is S3 = e2 + v9' + K31 row + b fix 0.1 + every3.

`bash fable_porter_v1/build_S3.sh <out.py> [build_eng flags]` runs these steps:
1. bank (sha pinned);
2. `fable_engineer2_v1/build_e2.py`, final default patches, output pinned to engineer2's announced v8 sha 1e33bace...;
3. `build_physics.py` 0.85,0.5@1-14;
4. `build_bfix_v2.py` gamma 0.1;
5. `fable_engineer_v1/build_eng.py <flags>`, skipped without flags;
6. `fable_join_v1/build_join_schedule.py` every3.

Intermediates and a sha list are kept in `<out>_build/`.
- **New step `build_bfix_v2.py`** finds the live-loop b update by AST, so it works on the bank, e2 v7 and e2 v8 (whose hold2 stage rewrote the b line
  into `_e2_H(...)` form), and checks the order of the dependent statements. On v8 it writes the same operations with holds.
  On the bank and on v7 + physics its output equals my PT-1 build and the lead's S2 intermediate respectively, apart from the appended comment line.
- **Fragment safety:** no engineer fragment redefines `_predict_binary`, `_shared_old384_push`, `_slr_*` or `_nonlinear_pair_batch`, so the kernel layer
  cannot silently undo the b fix or the join schedule.

Native checks (t31_bill.py, exact):
| build | r35 flops | r35 ops | r35 MSE | r100 flops | r100 MSE |
|---|---|---|---|---|---|
| S2 (lead, e2 v7) | 408,474,919,414 | 34,681 | 1.6123680337995120e-08 | | 1.576382292246693e-08 (lead) |
| `candS3_noeng.py` (no kernel layer: S2's math on e2 v8) | 408,474,919,414 | 34,681 | identical | 412,920,343,447 | 1.5763822922466930e-08 (identical) |
| `candS3_H.py` (`--holdonly`: bank kernel + held aliases, engineer's pick) | 408,474,919,414 | 34,681 | identical | | |
| `candS3_v9.py` (`--grouped --read5` = current v9, stand-in for v9') | 390,379,729,973 (-4.43%) | 31,016 | identical | 395,215,048,206 (-4.29%) | identical |

`candS3_H.py` has S2's predictions and bill, and per the engineer's census it should need fewer round trips (bank 95.4k -> about 59k). It is a
wall-safer form of tonight's fire math and was not censused by me. `candS3_v9.py` carries the v9 server-CPU issue until v9' lands.

---

## Update 23:20: S3 = fired S2 + engineer HEB kernel layer (held aliases + last-level copy removal + bigger arena)
The engineer's `fable_fire_v1/candS3_S2_HEB.py` has not arrived yet. The gates were pre-run on the porter-built equivalent
`candS3_HEB_porter.py` = `build_S3.sh candS3_HEB_porter.py --holdelim --bigarena` (S2's math on e2 v8, then `build_eng.py --holdelim --bigarena`).
| gate | result | pass? |
|---|---|---|
| parity with S2 (`s3check_candS3_HEB_porter_r{35,100}.log`) | r35 MSE 1.6123680337995120e-08 and r100 1.5763822922466930e-08, bit-identical to S2; flops r35 403,324,339,195 (-1.261% vs S2), r100 cold 408,160,042,789 (-1.153%) | yes |
| paired VALID40 vs cached bank (`paired_valid40_candS3_HEB_porter.json`) | adj 0.93406, raw 0.95822 (= S2), wins 40/40, util 0.184110; per-net MSE bit-identical to S2's VALID40 rows on 40/40; util vs S2 0.98781; adjusted below S2 on 40/40 | yes (expected 0.9456 x 0.988 = 0.9342) |
| 14-net panel (`panel_candS3_HEB_porter.log`) | panel8 3.615e-9 (0.9358 of bank; 0.9877 of S2's 3.660e-9); virgin6 2.966e-9 (0.9431; 0.9880 of S2's 3.002e-9); raw per net identical to S2's panel | yes (also the 2% margin) |
| whest validate | pending, to be run on the engineer's file | |
When the engineer's file arrives: r35/r100 flops and MSE must equal this build's for these gates to transfer (as A/c4); then whest validate.

---

## S3 (the stack BASE, not a fire) = `fable_engineer_v1/candS3_S2_HE.py` (sha ac03c6bc...) = fired S2 + `build_eng.py --holdelim` (no --bigarena)
The lead's memory decision: skip --bigarena (a 7.12 GB RSS reading against the 8 GB limit). candS3_S2_HEB will not be built; my HEB gates
above stand in for S3's VALID40 and panel, because the predictions are bit-identical.
| check | result |
|---|---|
| whest validate (`gate_candS3_S2_HE_whest_validate.log`) | success, exit 0 |
| r35 cold (`s3he_seq_35_100.log`, bill_sequence.py) | flops 404,856,028,333 (= engineer's census; -0.886% vs S2's 408,474,919,414), util 0.184107; MSE 1.612368033799512e-08 = S2 to 16 digits |
| r100 warm (second net in the same process) | flops 399,769,306,753 (= engineer's census), util 0.181794; MSE 1.576382292246693e-08 = S2 |
| r100 cold (own process, `s3he_bill_r100_cold.log`) | flops 409,181,874,775 (-0.905% vs S2-math cold 412,920,343,447); MSE = S2 |
| VALID40 / panel | per-net MSE = S2's (bit-identical predictions; verified on all 40 nets by the HEB run). Estimated adjusted about 0.9456 x 0.991 = 0.937 (not measured: util scaled by the r35/r100 ratio) |

---

## Update 00:25: covhunter's GN knobs (native layer), the stack builder, and a join-builder issue
### GN knob layer: `build_gn_knobs.py <base> <knobs.json> <out>` (`gn_knobs/v3.json` = covhunter's valid40_gn2_v3 theta)
Knobs iso -0.094, k21n -0.028, k31 -0.212 (L2-14, covariance-only); edge +0.164 (L8-14); rc -0.08 (L1-14). Values and placement were read from
`fable_covhunter_v1/gn2_knobs.py` and `covstep.py`; covhunter was asked to confirm.
- covariance-only knobs: covhunter's certified closed form added right after the live pair-map call, before the omitted-radial source
  (his cov_pm hook point). The n^2 products are float32, the card's series precision. e2's cached denominator is keyed on the identity of C, so it is recomputed from the new C.
- edge: a wrapper around the last `_message_edge_source`. rc: `C = C - rc (C0 - C)` around the live recentering statements.
- Edits are AST-located; they apply to the bank and to e2 v7/v8 with or without physics. Disabled knobs run the original statements.
| check | result |
|---|---|
| knobs off, on the bank-based K31-row file and on S3 (r35) | bit-exact: flops, ops and MSE identical to each base |
| certificate vs the definition (`gn_certificate.py`, r35, L2/5/9/12/14): second pair-map call with K21+dK21 and T31 x 0.788 | relative gap 0.8-1.5e-4 (float32 series rounding, covhunter's figure), correlation 0.99999999, diagonal untouched |
| r35 | K31-row+GN MSE 1.5820e-8 vs 1.6237e-8 (-2.57%); S3+GN 1.5715e-8 vs 1.6124e-8 (-2.53%); +0.18-0.19% flops, about +100 requests per layer |
| **VALID40 parity vs torch** (`gn_parity_valid40.txt`) | knobs/row pooled MSE: native 0.98342 vs torch 0.98337; wins 33/40 vs 32/40; per-net correlation 0.973, native minus torch +0.010% (sd 0.389%); worst nets agree (r135 +2.20% vs +2.26%) |
| K31-row + GN vs cached bank (`paired_valid40_gn_v3_bankphys.json`) | adj 0.97144, raw 0.96970, 38/40; util +0.174% vs the K31-row file (adjusted knobs/row 0.98512) |

### Stack builder: `build_stack.sh <out.py> [--e2=v7|v8] [--gn=<knobs.json>] [--refit=<params.pt>] [-- <build_eng flags>]`
Chain: bank -> build_e2 (v7 default, the fired lineage; pinned shas) -> K31 row -> b fix 0.1 -> [GN] -> [refit literals] -> [kernel layer] -> every3.
`check_layer.sh <with> <without> <ranks>` compares native MSE and flops of two builds that differ by one layer.
- `-- --holdelim` reproduces S3 exactly: r35 flops 404,856,028,333 and MSE equal to the engineer's candS3_S2_HE.py.
- `--gn=gn_knobs/v3.json -- --holdelim` reproduces my S3+GN built directly on the engineer's S3 (flops 405,642,599,623; MSE 1.5715236354257291e-08).
- `--refit=fable_refit_v1/params_orig.pt` gives the same program as without it (literal round trip). The analyst's refit_s2 .pt files are p/a/c/e dicts, so they plug in directly.
- The sq6 kernel layer: `-- --holdelim --sq6 [--push64]`.

### Join-builder issue (reported to the lead)
`fable_join_v1/build_join_schedule.py` as revised at 23:11 puts `len(state['joins']) != len(_JOIN_SCHEDULE)` in the else branch. With every3 that
branch runs at every non-join layer after the first join, so candidates raise 'Shared-old384 terminal join history' at L8. The crashing builds are
kept as `stack/*_BROKENJOIN.py`. Both porter chains use `build_join_schedule_v1_fired_S2.py`, the code of the graded S2, until the builder is fixed.
Suggested fix: `if (layer == 15 and len(state['joins']) != len(_JOIN_SCHEDULE)) or not old_columns:`.

### S3 + GN (`stack/stack_S3_gn.py` = build_stack.sh --gn=gn_knobs/v3.json -- --holdelim; v1 join builder): gates
| gate | result |
|---|---|
| paired VALID40 vs cached bank (`paired_valid40_stack_S3_gn.json`) | adj 0.92453, raw 0.94344, 40/40 (adj and raw); util 0.185069 |
| vs the fired S2 (per net, fable_fire_v1/paired_valid40_candS2.json) | pooled adj 0.97772 (-2.23%), raw 0.98457; adjusted below S2 on 37/40; median -2.09%, range -5.19% .. +1.42% |
| 14-net panel (`panel_stack_S3_gn.log`) | panel8 3.572e-9 (0.9247 of bank; 0.976 of S2's 3.660e-9); virgin6 2.925e-9 (0.9300; 0.974 of S2's 3.002e-9); 2% margin passes |
| whest validate (`stack/gate_stack_S3_gn_whest_validate.log`) | success |
Just under the lead's 3% bar over S2.

### sq6 kernel layer on S3+GN (`check_layer.sh stack/stack_S3_gn_sq6.py stack/stack_S3_gn.py 35,100`)
r35: MSE x1.00053, flops -1.173%, ops 37,827 vs 36,278. r100: MSE x0.99793, flops -1.178%, ops 36,826 vs 35,323. So the MSE change is at rounding
level and the bill is 1.2% lower. Projected S3+GN+sq6 is about 0.966 of S2 (-3.4%). The VALID40 waits for the lead's go.

---

## Update 01:15: S4 fire file, knob-port reconciliation, and new build_stack.sh options
### S4 (the lead fires it) = `fable_fire_v1/candS4_S3HEB_gn.py`, sha256 082f2b501047d4470dc83ee6517c18b87c9fbce84801655709e84b31afba34d1
Byte-identical copy of `stack/stack_S3HEB_gn.py` = `build_stack.sh --gn=gn_knobs/v3.json -- --holdelim --bigarena` (join v1_fired_S2).
- Parity vs the gated `stack_S3_gn.py` (HE): r35 MSE bit-identical (1.5715236354257291e-08), flops -0.378%; r100 bit-identical, flops -0.249%.
- whest validate passes on the fire file, stack_S3HEB_gn.py and stack_S3_gn.py.
- Its gates were measured on stack_S3_gn.py, whose MSE is identical: VALID40 adj 0.92453 (40/40), panel8 0.9247, virgin6 0.9300 vs bank.
### Knob-port reconciliation (the lead asked; `gn_recon_*_S2_r35.log`)
On the same fired S2 with the v3 constants, r35 MSE: porter port 1.5715236e-8 (-2.533%); covhunter's port 1.5743404e-8 (-2.358%); the porter port
with the correction in float64 (`diag_gn_v3_S2_f64delta.py`, diagnostic only) 1.5735093e-8 (-2.410%). So the correction's precision alone moves r35
by 0.13 points; the rest comes from operation order in the edge and recentering scaling. The placements are identical, and the recentering at L0 is
exactly zero (rc at L0-14 gives a bit-identical r35). The card amplifies rounding-level differences into about 0.1-0.2% per net; this averages out
(VALID40 parity vs torch 0.005% pooled). The lead retired covhunter's native script for fires.
### build_stack.sh options (the old version is kept as build_stack_v1_S4.sh; with no new options the output is byte-identical)
`--k31row=c0,c1` (default 0.85,0.5), `--bfix=gamma` (default 0.1), `--l15cut[=zc]` (physicist's `build_l15_rowcut.py`, zc 3.5), `--join=v1|v2` (default v1).
- Regression: the default rebuild and explicit defaults are byte-identical to the S4 fire file.
- `--join=v2` on S4: r35 flops and MSE identical to S4. The terminal check is `layer >= max(_JOIN_SCHEDULE)`.
- `--l15cut` on S4 (`stack/stack_S4_l15cut.py`; `stack/parity_l15cut_on_S4.txt`, nets 35 cold then 100 warm in one process): r35 MSE -0.035%,
  util -0.59%, adjusted -0.62%; r100 MSE +0.226%, util -0.84%, adjusted -0.61%. The physicist's native S2+cut matched his torch
  (+0.009% vs +0.011%; +0.434% vs +0.442%).

---

## Update 02:10: S4 fired as #331961 (graded 3.262e-9 per the lead's board map); S5 base = S4 + L15 cut
- S4 VALID40 (`paired_valid40_stack_S4.json`, stack_S3HEB_gn.py = the fired file): adj 0.92147, raw 0.94344, 40/40, util 0.184461; per-net MSE
  bit-identical to S3+GN on 40/40, util x0.99671 (HEB).
- S5 = `build_stack.sh --gn=gn_knobs/v3.json --l15cut -- --holdelim --bigarena` (`stack/stack_S4_l15cut.py`, join v1; sha 8ac08ae34d5154cc...)
  VALID40 vs cached bank: adj 0.91543, raw 0.94337, 40/40. **Vs S4 exactly per net:** adj 0.99344 (-0.66%), raw 0.99993, adjusted below S4 on 37/40,
  raw below on 24/40, per-net median -0.67% (range -1.36% .. +0.80%), util x0.99353. Passes the lead's S5 bar (-0.5%, >= 24/40).
- Join-v2 twin `stack/stack_S5_l15cut_joinv2.py` (sha e85368329e11d2e5...): r35/r100 MSE, flops and ops identical to
  the v1 file; whest validate passes on both.
- The analyst's plane-feed layer was closed on S2 (+61% bill for -2.4% raw); there is nothing to port. covhunter keeps GN v3 (no gn3).

---

## Update 02:40: S5 = S4 + L15 cut + push5 kernel + f32 third + memory diet (fire file `stack/stack_S5_mem.py`)
Built with the lead's commands (from P2):
  `build_stack.sh fable_porter_v1/stack/stack_S5.py --gn=fable_porter_v1/gn_knobs/v3.json --l15cut --join=v2 -- --holdelim --bigarena --depthrule fable_engineer_v1/depthrule_fragment_push5.py --f32 third`
  `python3 fable_engineer2_v1/build_e2mem.py fable_porter_v1/stack/stack_S5.py fable_porter_v1/stack/stack_S5_mem.py`
sha256: stack_S5.py b34c2180d91a0258d0d3604f21d19271e44dfef539ecab07fbf9810cacbd9d5f; **stack_S5_mem.py 276455ffc7f203abfa3acb90d87851996de20c40b93551259b411699574ce208**.
engineer2's `s5draft` has other shas: it has no --f32 third (its 6_eng differs by exactly the F32 lines) and uses join v1. He has been told.
| gate | result |
|---|---|
| whest validate | success |
| diet bit-exactness vs stack_S5.py (`stack/parity_S5.txt`) | r35 and r100: MSE bit-identical, flops identical |
| parity vs stack_S5_l15cut_joinv2.py, nets 35 cold then 100 warm in one process | r35: MSE +0.077%, util -1.22%, adjusted -1.14%; r100: MSE +0.142%, util -2.54%, adjusted -2.41% |
| paired VALID40 vs cached bank (`paired_valid40_stack_S5_mem.json`; each net cold) | adj 0.90083, raw 0.94317, 40/40, util 0.180362 |
| vs S4+l15cut per net | adj 0.98405, raw 0.99979 (rounding level), adjusted below on 40/40, raw below on 20/40; median -1.56% (range -2.55 .. -0.97%) |
| vs S4 per net | adj 0.97760 (-2.24%), raw 0.99971, adjusted below on 40/40; median -2.26% (range -3.13 .. -1.07%) |
| 14-net panel (`panel_stack_S5_mem.log`; cold) | panel8 3.476e-9 (0.8998 of bank; 0.9731 of the S3+GN panel); virgin6 2.854e-9 (0.9075; 0.9757); 2% margin passes |
The cold-net VALID40 and panel understate push5's bill cut: warm r100 util is -2.54% against -1.22% cold.

---

## Update 02:45: S5e = S5 + --evict5 (the engineer's frame-eviction fix for S5's server-handle growth) -> fire file `stack/stack_S5e_mem.py`
Built from P2: `build_stack.sh fable_porter_v1/stack/stack_S5e.py --gn=fable_porter_v1/gn_knobs/v3.json --l15cut --join=v2 -- --holdelim --bigarena
--depthrule fable_engineer_v1/depthrule_fragment_push5.py --f32 third --evict5`, then `build_e2mem.py` -> `stack_S5e_mem.py`.
sha256: stack_S5e.py 0c677dbf73e563b7365edef16b1737389bc7f36ab1b56b73e4a48431af9fe2a7; **stack_S5e_mem.py c34bdd791c21bbda3c9a5264a8be6166e1668c1688434c9f68ca75af23f3baf9**
(evict_fragment_push5.py a622a1d1..., build_eng.py 1f96294c...). Against stack_S5_mem.py it adds 44 lines (the evict fragment) and removes nothing.
- Parity vs stack_S5_mem.py, nets 35 cold then 100 warm in one process (`stack/parity_S5e_vs_S5.txt`): MSE bit-identical and flops identical on both.
  The S5 rerun reproduces its 02:07 run exactly.
- whest validate: success. S5's VALID40 (0.90083 vs bank; 0.9776 vs S4, 40/40) and panel (0.8998 / 0.9075) carry over.
- Last gate: engineer2's 20-net handle test (sha handed to him).

---

## Update 02:55: S5f = S5e + --lru (threshold LRU on all frame caches: above 60k arrays at a prediction start, drop to 40k) -> fire file `stack/stack_S5f_mem.py`
Built from P2 with the lead's command: `... -- --holdelim --bigarena --depthrule fable_engineer_v1/depthrule_fragment_push5.py --f32 third --evict5 --lru`, then `build_e2mem.py`.
sha256: stack_S5f.py b4011420862c7131899c26fdeeb84613d7d6c79a376be07965f92312bc686261; **stack_S5f_mem.py c20f790354ea45da25e8036a28d3e15f2e62ed5a6d26c272b5e19686ccf44a0d**
(evict_fragment_lru.py 0a46d5cc..., evict_fragment_push5.py a622a1d1..., build_eng.py 88624978...).
- Regression: the updated build_eng.py rebuilds S5e byte-identically.
- Against stack_S5e_mem.py: it adds 120 lines (the LRU fragment) and removes nothing. Parity on nets 35 then 100 in one process: MSE bit-identical and flops identical
  (the LRU does not trigger within 2 nets; `stack/parity_S5f_vs_S5e.txt`). whest validate: success.
- S5's VALID40 and panel carry over. Last gate: engineer2's 20-net handle test (sha handed to him).
- 03:00, the engineer's check of the S5 files (cached arrays = server handles after r35/r100/r37/r39 in one process): S5e_mem 16,818 / 25,595 / 27,198 /
  34,467; S5_mem 16,818 / 25,595 / 28,405 / 36,056; S4 (stack_S4_joinv2) 15,501 / 23,590 / 26,268 / 33,065. S5e_mem's predictions are bit-identical to
  S5_mem's on all 4 nets. The engineer names S5e as S5's final file and S5f as the hard-bounded variant for S6; the lead's last message named S5f
  the fire configuration. Both are gated with identical predictions; the lead picks.
- 03:05, lead's decision: the FIRE FILE is S5f = `stack/stack_S5f_mem.py` (sha c20f790354ea...), with the LRU hard bound on the handle store. S5e
  (`stack/stack_S5e_mem.py`, sha c34bdd791c21...) is the fallback only if engineer2's 20-net test shows the LRU misbehaving. The lead fires after that verdict.

---

## FIRED 2026-09-23: S5e as #331992 = `stack/stack_S5e_mem.py`, sha256 c34bdd791c21bbda3c9a5264a8be6166e1668c1688434c9f68ca75af23f3baf9
The lead switched from S5f to S5e: engineer2's 20-net test showed S5e below S4 on server handles, and S5f's LRU costs residual time.
Gates behind it: parity vs S5 (MSE and flops bit-identical, r35 and r100 in one process), whest validate, and S5's VALID40 (0.90083 vs bank;
0.9776 vs S4, 40/40) and panel (0.8998 / 0.9075 vs bank), which carry over because the predictions are bit-identical.
**Base for any S6** (run from P2; add new layers as options or kernel flags):
    bash fable_porter_v1/build_stack.sh <out.py> --gn=fable_porter_v1/gn_knobs/v3.json --l15cut --join=v2 -- --holdelim --bigarena \
        --depthrule fable_engineer_v1/depthrule_fragment_push5.py --f32 third --evict5
    python3 fable_engineer2_v1/build_e2mem.py <out.py> <out_mem.py>
Pinned inputs: bank d79463f3..., e2 v7 737bdcc9..., evict_fragment_push5.py a622a1d1..., build_eng.py 88624978... (it rebuilds S5e byte-identically).

---

## Update 04:30: S6 = S5e + the analyst's closure refit (epoch 2) + engineer2's diet stages mem,yright,arena -> `stack/stack_S6_mem.py`
S5e (#331992) graded 3.1763e-9 (-2.64% vs S4). Built from P2:
  `build_stack.sh fable_porter_v1/stack/stack_S6.py --gn=fable_porter_v1/gn_knobs/v3.json --refit=fable_analyst_v1/refit_s2_full_lr2e4_v2/params_epoch2.pt --l15cut --join=v2 -- --holdelim --bigarena --depthrule fable_engineer_v1/depthrule_fragment_push5.py --f32 third --evict5`
  `python3 fable_engineer2_v1/build_e2mem.py fable_porter_v1/stack/stack_S6.py fable_porter_v1/stack/stack_S6_mem.py mem,yright,arena`
sha256: stack_S6.py 8437b32abde2f003be045eca7ed24270d7ac3fffc1350b184e594b21e6df8293; **stack_S6_mem.py 2b4372a7ee3f46c711fc11a19467e2752d8ebec9f06bb08eb7d0897861a0edd7**
(params_epoch2.pt 20f15441...). The refit export changed exactly the 5 learned literal tables (builder postcondition).
| gate | result |
|---|---|
| refit layer vs S5e (nets 35 cold, 100 warm, one process; `stack/parity_S6.txt`) | MSE -0.433% / -0.818%; flops identical |
| diet stages vs stack_S6.py | MSE bit-identical; flops -0.043% / -0.075% |
| whest validate | success |
| paired VALID40 vs cached bank (`paired_valid40_stack_S6_mem.json`) | adj 0.89591, raw 0.93842, 40/40; util 0.180285 |
| **vs S5e per net** (S5e rows = the S5 VALID40, same MSE) | raw 0.99497 (-0.50%), raw below on 33/40; adj 0.99454, adj below on 34/40; raw median -0.44% (range -1.81 .. +1.49%; worst r48 +1.49%, r130 +0.75%, r146 +0.62%) |
Passes the lead's bar (pooled raw <= -0.3% vs S5e, >= 24/40). The sha is with engineer2 for his gate.
- 04:35, S6 14-net panel (`panel_stack_S6_mem.log`): panel8 3.463e-9 (0.9963 of S5e's 3.476e-9; raw 0.9964), virgin6 2.819e-9 (0.9877 of S5e's
  2.854e-9; raw 0.9880). Vs bank 0.8965 / 0.8963. Both halves beat S5e, which is the lead's bar. On the off-distribution virgin nets (fresh MLPs) the refit gains even more.

### S6b = S6 + --batch (the engineer's E-35: each layer's five FLMS 1024^3 products in two batched kernel calls) -> `stack/stack_S6b_mem.py`
Built from P2 with S6's command plus `--batch` at the end of the kernel flags, then `build_e2mem.py ... mem,yright,arena`.
sha256: stack_S6b.py b89d72d1731dfa5977459f73d3bfff25f2037cd3ed07e4b108120d317d3c3d5e; **stack_S6b_mem.py a514c21b23ea209a7a3fd25b778a0e4fa2ed9850fc1a97a8c6bfce2bdc2cafc6**
(build_eng.py 3a3dfb94..., which rebuilds S6 byte-identically). Against stack_S6_mem.py it adds 176 lines and replaces 2 body lines (Cpre via
_flms_cov_star, and star_pre reusing the batched result).
- Parity vs stack_S6_mem.py, nets 35 then 100 in one process (`stack/parity_S6b_vs_S6.txt`): r35 cold MSE bit-identical, flops +0.3803%;
  r100 warm MSE bit-identical, flops identical. Exactly as predicted. whest validate: success.
- S6's VALID40 and panel carry over (bit-identical predictions). Fire file: S6b if engineer2 passes --batch, otherwise S6.
- 04:45, engineer2's "S6-full" (gated by him) = the S5e chain with `--depthrule fable_engineer_v1/depthrule_fragment_push5p6.py --f32 third --evict5 --caps`,
  then `build_e2mem.py ... mem,arena,yright`. The stage order does not matter; his builder applies mem, yright, arena in a fixed order. Reproduced byte-identically with
  build_stack.sh (scratchpad only): stack c82e331f... (= the engineer's candS6_p6caps.py), final 7b609890... (= fable_engineer2_v1/s6full/S6full_mem_arena_yright.py).
  It has no refit and no --batch, and it uses push5p6, which the lead ruled out for S6 at 04:00. The lead's S6 line is S6 (2b4372a7) / S6b (a514c21b).

---

## FIRED and GRADED 2026-09-23: S6b as #332005 = `stack/stack_S6b_mem.py`, sha256 a514c21b23ea209a7a3fd25b778a0e4fa2ed9850fc1a97a8c6bfce2bdc2cafc6
Graded 3.1758e-9, a tie with S5e (#331992, 3.1763e-9). Per the lead, the refit's raw gain (VALID40 -0.50% vs S5e, 33/40; panel 0.9963 / 0.9877)
was offset on the board by the platform restarting every worker once, which makes more networks cold (S6b's --batch costs +0.38% on a cold first net).
Night summary of the porter chain (each fire was one build_stack.sh command plus the gates above):
| fire | file (fable_porter_v1/) | board |
|---|---|---|
| #331961 S4 | stack/stack_S3HEB_gn.py (082f2b50) | 3.2623e-9 |
| #331992 S5e | stack/stack_S5e_mem.py (c34bdd79) | 3.1763e-9 |
| #332005 S6b | stack/stack_S6b_mem.py (a514c21b) | 3.1758e-9 |
Standing down on the lead's instruction; no new builds. Matt picks the next phase in the morning.

---

## Update 05:45 (S7 phase: 3e-9 needs -5.5% from S6b): S6b baseline rows + refit exporter trunk check
- S6b's own VALID40 (`paired_valid40_stack_S6b_mem.json`, the reference rows for every S7 layer): adj 0.89933 vs bank, raw 0.93842, 40/40, util 0.180971;
  MSE bit-identical to S6 on 40/40; util x1.00380 vs S6 (--batch's +0.38% one-time cost on a cold net; the VALID40 runs every net cold).
- build_refit_candidate_v2.py exports the GNN trunk. On S6b's epoch-2 .pt with enc0_w, msg_w and enc4_b perturbed, only _GNN_LITERAL changes and all 15 p keys
  decode bit-equal to the .pt. A .pt must hold the full p/a/c/e groups; a partial p fails loudly.
- S7 base recipe (S6b): `build_stack.sh <out> --gn=fable_porter_v1/gn_knobs/v3.json --refit=<pt> --l15cut --join=v2 -- --holdelim --bigarena
  --depthrule fable_engineer_v1/depthrule_fragment_push5.py --f32 third --evict5 --batch [new kernel flags]`, then `build_e2mem.py <out> <final> mem,yright,arena`.

### S7 base = S6b + --coldbind (the engineer's E-38: no reshapes for unread layout-region views, read5 children bound once) -> `stack/stack_S7base_cb_mem.py`
Built from P2 with the S6b command plus `--coldbind` at the end of the kernel flags, then `build_e2mem.py ... mem,yright,arena`.
sha256: stack_S7base_cb.py b2d4d44ac3d211a5... (= fable_engineer_v1/candS6b_coldbind.py); **stack_S7base_cb_mem.py f9e67942c8688cdbc91c92671d7a271ead07a6c7ff9b8ffffd041b297080b554**
(= the engineer's candS6b_coldbind_mem.py, byte-identical). build_eng.py f0f3eb15... still rebuilds S6b byte-identically.
- Parity vs stack_S6b_mem.py, nets 35 then 100 in one process (`stack/parity_coldbind_vs_S6b.txt`): r35 cold MSE bit-identical, flops -1.275%; r100 warm
  MSE bit-identical, flops -0.250%. whest validate: success. No VALID40 is needed for a bit-exact kernel layer; S6b's rows cover the MSE.
- The engineer may fold a used-slots step in as --coldbind2; asked, and I rebuild if it lands.
- 06:00, **S7 base -> S6b + --coldbind2** (the engineer's: --coldbind plus a used-slots fragment; it implies --coldbind): `stack/stack_S7base_cb2_mem.py`
  sha ecf9b9c105f70359... = the engineer's candS6b_cb2_mem.py (byte-identical). It adds 111 lines to the coldbind base and removes none; the current
  build_eng.py still rebuilds the coldbind base byte-identically. Parity vs stack_S7base_cb_mem.py (`stack/parity_coldbind2_vs_coldbind.txt`): r35 cold MSE
  bit-identical, flops -0.229% (vs S6b -1.502%); r100 warm MSE bit-identical, flops -0.068% (vs S6b -0.318%). whest validate: success. Pending engineer2's gate.
- 06:20, **S7 base -> S6b + --coldbind3** (the engineer's E-40: --coldbind2 plus the unwind's temporaries in dead product slots; it supersedes cb/cb2):
  `stack/stack_S7base_cb3_mem.py` sha 23d4f33906acb57a... = the engineer's candS6b_cb3_mem.py (byte-identical). build_eng.py 4967be49... still rebuilds
  the cb and cb2 bases byte-identically. Parity vs stack_S7base_cb_mem.py (`stack/parity_coldbind3_vs_coldbind.txt`): r35 cold MSE bit-identical, flops
  -0.587% (vs S6b -1.855%); r100 warm MSE bit-identical, flops -0.182% (vs S6b -0.432%). whest validate: success. Pending engineer2's gate; the fallbacks are
  cb2 (ecf9b9c1) and cb (f9e67942).

### S7-full on cb3 (lead's recipe; for engineer2's gate) -> `stack/stack_S7full_cb3_mem.py`, sha 9011916a405b8bb5d17e63bd7c4c5200d31121ba92b0089b9334d0731e6dc158
Built from P2: `build_stack.sh fable_porter_v1/stack/stack_S7full_cb3.py --gn=fable_porter_v1/gn_knobs/v3.json --refit=fable_analyst_v1/refit_s2_full_lr2e4_v2/params_epoch2.pt
--l15cut --join=v2 -- --holdelim --bigarena --depthrule fable_engineer_v1/depthrule_fragment_push5p6.py --f32 third --evict5 --caps --caps2 --batch --bindhold
--coldbind3`, then `build_e2mem.py ... mem,yright,arena,hold2` (hold2 applied last: 217 statements). Pre-diet file stack_S7full_cb3.py aa8269ed...; push5p6 fragment
99b76ad9... (the ORIGINAL, embedded verbatim in the engineer's candS7_mem.py; not the MU variants).
- whest validate: success.
- Parity vs the cb3 base (`stack/parity_S7full_cb3.txt`, nets 35 cold then 100 warm): MSE -0.306% / +0.178%, OUTSIDE the lead's +-0.1% band; flops -0.221% / -0.416%
  vs cb3, -2.072% / -0.846% vs S6b; adjusted vs S6b -2.37% / -0.67%.
- Bisection (`stack/bisect_S7/`): my build_stack reproduces the engineer's candS7_mem.py byte-for-byte (5bf2a07e). Its final MSE is BIT-IDENTICAL to S7-full's on r35 and
  r100, and so is S7 + coldbind3. So --coldbind3, --caps2 and hold2 are MSE-exact, and the -0.306% / +0.178% vs S6b is push5p6's own effect. The engineer's quoted
  "+-0.09% vs S6b" does not reproduce on r35/r100 in this harness. Whether it averages to zero needs VALID40, which is held until engineer2's gate.

## 2026-09-23 06:51 S7 bisection resolved: NO layer breaks bit-identity; the MSE gap is the numerics ENV (porter)
- Lead's order, .venv (numpy 2.2.6, macOS system BLAS), nets 35 cold then 100, one process each: engineer's candS7_mem.py (5bf2a07e, run from
  fable_engineer_v1) = + --coldbind3 = + --caps2 = + hold2 (= stack_S7full_cb3_mem.py 9011916a); MSE identical to 16 digits at every step
  (r35 1.5605794029107084e-08, r100 1.5730179645939266e-08). Record: stack/bisect_S7/bisect_result_v2.txt.
- OpenBLAS runtime env (numpy 2.4.6, scipy-openblas 0.3.31; engineer2's gate env) via parity_e2copy.py (verbatim copy of engineer2's parity.py):
  candS7 reproduces engineer2's par4_S7 MSE exactly (positive control); S7-full-cb3 is BIT-EQUAL to candS7 at all 16 layers on r35 and r100;
  vs S6b there: MSE +0.037% / -0.086%, flops -2.072% / -0.846%, adjusted -2.04% / -0.93% (r35 cold / r100 warm).
- Same pair in the .venv: MSE -0.306% / +0.178%, adjusted -2.37% / -0.67%. FLOPs identical to the unit across envs; S6b's own MSE differs
  between envs by +0.029% (r35) / -0.308% (r100). Record: stack/bisect_S7/bisect_result_v3_envs.txt.
- Law: per-net MSE parity bars (±0.1%) cannot judge a non-bit-exact change like push5p6 across envs; bit-identity = same env, all-layers
  array_equal. VALID40/panel stay like-for-like (candidate and cached bank rows both scored in the .venv, one net per process).
- Still HELD by the lead: engineer2's S7-full gate and my VALID40. Four-net OpenBLAS check (35,100,37,39 vs engineer2 par4_S7.npz) running.
- 2026-09-23 06:54 four-net extension (OpenBLAS env, engineer2's gate order 35 cold,100,37,39 in one process): S7-full-cb3 is BIT-EQUAL to engineer2's
  par4_S7.npz (candS7) at all 16 layers on all four nets. Vs S6b there: adjusted -2.04 / -0.93 / -0.96 / -1.01% (mean -1.24%; warm-only mean
  -0.97%); MSE +0.037 / -0.086 / -0.017 / +0.012%; flops vs candS7 -1.87 / -0.48 / -0.12 / -0.50% (coldbind3 + caps2 + hold2).
  Record: stack/bisect_S7/bisect_result_v4_4nets.txt.

## 2026-09-23 07:18 S7-full-cb3 VALID40 vs S6b rows (lead GO 06:5x) — PASSES both bars on the cold VALID40
- stack/stack_S7full_cb3_mem.py (9011916a), 40 nets, each cold in its own process, .venv; report valid40_vs_ref_ci.py (PT-9, paired bootstrap
  95% CI over nets) -> gate_S7full_cb3_valid40_vs_S6b.txt:
  raw pooled -0.016% [-0.068%, +0.038%], wins 23/40; util mean 0.177080 vs 0.180971 (-2.150%); adjusted pooled -2.163% [-2.225%, -2.102%],
  wins 40/40 (per-net -2.55..-1.68%). Vs bank rows: adjusted 0.87987 (S6b 0.89933), raw 0.93827.
- Warm-dominated figure is borderline: warm FLOP cut on r100/r37/r39 (in-process) -0.85/-0.95/-1.03% (mean -0.94%); with about 1 cold net
  per 20 the platform-weighted adjusted is about -1.0%. A 40-net warm analogue (S6b vs S7-full-cb3, cold lead-in then all VALID40 nets warm)
  follows the cb3-base VALID40.

## 2026-09-23 07:21 S7-full-cb3 + --symout built (engineer E-43, rounding-level; lead: stacks on S7-full-cb3 once it passes)
- stack/stack_S7full_cb3_so_mem.py (879b1d4b1da37a78...; pre-diet stack_S7full_cb3_so.py f6f1097b) = S7-full-cb3 recipe with --symout
  appended to the kernel flags, then build_e2mem.py mem,yright,arena,hold2. Chain stages 1_e2..5b_l15cut have the same shas as S7-full-cb3's;
  6_eng differs (2e4c5ebe). whest validate: success (stack/gate_stack_S7full_cb3_so_mem_whest_validate.log).
- Queued (symout_chain.out): parity r35,100 in one process (.venv) against S7-full-cb3, then VALID40 vs S7-full-cb3 rows and S6b rows.
  Engineer's bar: pooled raw within +-0.05%.
- The 40-net warm pass was dropped before launch: engineer2's paired 20-net run (1 cold in 20, platform mix) measures the same warm util.
- 2026-09-23 07:39 cb3-base VALID40 (stack_S7base_cb3_mem.py 23d4f339) vs S6b rows: MSE IDENTICAL on 40/40 nets (bit-exact over the full set, cold);
  util -1.836% [-1.848%, -1.823%]; adjusted -1.836%, 40/40; vs bank rows adjusted 0.88281. Split of S7-full-cb3's cold VALID40 -2.163%:
  cb3 -1.836% + (push5p6, caps, caps2, bindhold, hold2) -0.333% [-0.392%, -0.273%] (39/40, raw -0.016%). Warm in-process: cb3 -0.43/-0.18/-0.47%,
  S7-full-cb3 -0.85/-0.95/-1.03% (r100/r37/r39), so the push5p6 group is worth ~0.5% warm. Files: gate_S7base_cb3_valid40_vs_S6b.txt,
  gate_S7full_cb3_valid40_vs_cb3base.txt.

## 2026-09-23 07:53 S7 FIRED by the lead as #332026 = stack_S7full_cb3_mem.py (9011916a; fable_fire_v1/fire_candS7/candS7_full_cb3_mem.py same sha)
## 2026-09-23 07:53 S8 = S7 + --symout v2 built: stack/stack_S8_so2_mem.py (4f91e8f44ff612c73eb96ad2a69f3f55af62c5f711a349e497ad4ac76d506953)
- Pre-diet stack_S8_so2.py 69031854; 6_eng c6f0e6d0; embeds fable_engineer_v1/symout_fragment_v2.py verbatim (v1 not present). whest validate: success.
- My earlier symout build (stack_S7full_cb3_so_mem.py 879b1d4b, 07:27) carried v1 (the builder switched to v2 at 07:29); its VALID40 was stopped at
  18/40 nets and archived as paired_valid40_stack_S7full_cb3_so_mem_ABORTED_symout_v1.{json,log}.
- Chain S8_chain.out: parity r35,100 (.venv) then VALID40 vs S7 rows (bar pooled raw within +-0.05%) and S6b rows. Handed to engineer2 (reference S7).

## 2026-09-23 08:14 S8 (4f91e8f4) BLOCKED: --symout joined-core einsum raises flopscope SymmetryError (fire-blocking); tested fix sent to the engineer
- Parity (.venv, r35 cold then r100): r35 ran (MSE +0.357%, flops -1.214% vs S7); r100 raised in _shared_old384_push at
  joined_core = fnp.einsum('ia,a,ja->ij', Fj, weight, Fj): "SymmetryError ... max deviation = inf (atol=1e-06, rtol=1e-05)". The inf is
  hard-coded (non-finite results skip the check): the finite float32 result failed allclose(R, R.T). S8 VALID40 stopped (0/40), archived
  *_ABORTED_symerr.
- Margin diagnostic diag_symout_margin.py (PT-11; ratio = max|R-R^T|/(1e-6+1e-5|R^T|), raises iff > 1), nets 35,100,37,39,42,48:
  L7 join 0.03-0.08, L10 0.18-0.35, L13 (core diagonal 500-750) 0.30-1.53. Raised: .venv (numpy 2.2.6, flopscope 0.10.0) r100 1.53, r42 1.30;
  OpenBLAS env (numpy 2.4.6, flopscope 0.12.0; check code identical) r100 1.01. Logs stack/diag_symout_S8_{venv,openblas}.log.
- Offline test test_symout_gram_forms.py (PT-12) on dumped failing operands (stack/symout_ops_*.npz), both envs: form B
  X = Fj*sqrt(w), einsum('ia,ja->ij', X, X): never raises, asymmetry exactly 0, flops +4,608/call vs current (same half-price billing,
  dense = 2.0x), error vs float64 1.3-4.1e-7 (current 1.2-9.7e-7). Log stack/test_symout_gram_forms.log.
- Awaiting the engineer's builder fix; then rebuild S8, parity, margin diagnostic (must show 0 asymmetry), VALID40 vs S7 rows.
- 2026-09-23 08:15 PRE-FIRE EINSUM GATE einsum_site_diff.py (PT-13, the lead's crash law): multiset diff of einsum call sites (hold wrappers stripped) vs
  the last platform-proven file; FAIL on any NEW repeated-operand einsum (inferred symmetric, runtime-validated) or symmetric-tag call.
  S8 (4f91e8f4) vs S7 #332026: FAIL (new: einsum('ai,aj->ij', X1, X1), ('ai,aj->ij', X2, X2), ('ia,a,ja->ij', Fj, weight, Fj) x3 copies).
  S7 vs S6b #332005: PASS (13 = 13 sites; the 2 old repeated-operand sites are platform-proven). Form B (same-operand Gram) is ALSO a
  repeated-operand einsum, so the crash law excludes it; symout v3 = covariance upper-block/mirror only (engineer to build).

## 2026-09-23 08:17 S8 v3 = S7 + --symout v3 (covariance upper-block/mirror only) built: stack/stack_S8_so3_mem.py (5116cee9adb4f5040ed9548268d567bf7f2a89b4619996e5aab9e697f310221c)
- Byte-identical to the engineer's fable_engineer_v1/candS8so3_mem.py (pre-diet 30411d80 = candS8so3.py). Embeds symout_fragment_v3.py only.
- einsum_site_diff.py vs S7 #332026: PASS (0 new repeated-operand sites). whest validate: success.
- Chain S8v3_chain.out: parity nets 35 cold,100,37,39 (.venv) then VALID40 vs S7 rows (bar pooled raw within +-0.05%) and S6b rows.
- stack_S8_so2_mem.py (4f91e8f4, symout v2) is WITHDRAWN (SymmetryError); engineer2 told to drop it.
- 2026-09-23 08:49 S8 v3 parity (.venv, 35 cold,100,37,39 in one process; no raise): vs S7 flops -1.020/-0.967/-0.982/-0.928%, MSE +0.245/+0.012/+0.061/-0.226%,
  adjusted -0.78/-0.95/-0.92/-1.15% (stack/parity_S8_so3.txt). symout's cut is per-layer compute: cold and warm alike.
- 2026-09-23 08:49 S8 v3 VALID40 (0 errors): vs S7 rows raw +0.018% [-0.035%, +0.070%] (18/40) PASS (bar +-0.05%); util -1.022% [-1.027, -1.016];
  adjusted -1.003% [-1.056, -0.951] (40/40; per-net -1.51..-0.58%). Vs S6b rows adjusted -3.145%; vs bank 0.87104 (raw 0.93844).
  Board: S7 #332026 graded 3.126e-9 (08:43). Open: engineer2's CPU/residual gate (symout adds standalone covariance calls).

## 2026-09-23 08:51 S9 = S7 + B2 epoch-0 refit (lead 08:44: next fire, zero CPU change) built: stack/stack_S9_B2_mem.py (1534b1ddfac6f8fc69ed928e56e48e1f62790fdea88d4ebf80fd48d59c8f1ad1)
- Fired S7 recipe with --refit=fable_physicist_v1/refit6b/s6b_B2_fullnet_lr1e4_trunk5e5/params_epoch0.pt (46f27bdd); pre-diet stack_S9_B2.py a0774c54.
- Diff vs S7 #332026 = exactly 5 literal statements (_GNN, _MESSAGE_EDGE, _EDGE, _BDNH_A, _BD_COEFFICIENT). All 28 parameter tensors decoded from
  the literals are BIT-EQUAL to the .pt (stack/verify_S9_literals.txt), incl. the trunk msg/enc0/enc2/enc4 w,b.
- einsum gate vs S7: PASS; whest validate: success. Chain S9_chain.out: parity 35 cold,100,37,39 (.venv) then VALID40 vs S7 rows.
- 2026-09-23 08:57 S9 parity (.venv, 35 cold,100,37,39): FLOPs IDENTICAL to S7 on all four nets (zero compute change); MSE vs S7 -0.451/-0.776/+1.076/-3.403%
  (torch B2/S6b on 37/39: +0.67/-3.38%, native tracks torch). stack/parity_S9_B2.txt.
- 2026-09-23 08:57 PRE-BUILT (not run) S10 = S7 + symout v3 + B2 refit: stack/stack_S10_so3_B2_mem.py (c7d30c1975e2cde17672f9cd6204e4c5528459735b7d9a7997d2417ba101403e).
  Diff vs S8 v3 = the 5 refit literals; diff vs S9 = the same 39 lines as S7 -> S8 (the symout edit). einsum gate vs S7 PASS; validate OK.
- 2026-09-23 09:25 S9 VALID40 (0 errors) vs S7 rows: raw -1.001% [-1.352%, -0.645%] (30/40); util +0.000% (identical every net); adjusted -0.999%
  [-1.352, -0.641] (30/40; per-net -3.40..+2.35%, median -0.78%). Vs bank rows adjusted 0.87108, raw 0.92888. Native vs torch refit effect:
  -1.001% vs -1.090% pooled, per-net correlation 0.985, per-net diff sd 0.206% (gate_S9_B2_native_vs_torch.txt). Expected ~3.095e-9.

## 2026-09-23 09:28 LAWS + RECORDS (lead's requests)
- BLAS-ENV LAW (one line): compare candidate and reference in ONE numerics env only; the .venv (numpy 2.2.6, macOS BLAS, flopscope 0.10.0) and the
  OpenBLAS env (numpy 2.4.6, scipy-openblas 0.3.31, flopscope 0.12.0) agree on FLOPs to the unit but differ by up to 0.31% per-net MSE on the same file.
- SYMMETRIC-OUTPUT GUARD (lead, adopted): any future symmetric-output layer needs (1) einsum_site_diff.py PASS vs the last fired file, (2) the margin
  diagnostic diag_symout_margin.py showing 0 asymmetry on 6 nets x 2 envs, and (3) a sign/finiteness proof of its inputs, before any fire.
- POSSIBLE LATER (record only): joined core as X = Fj*sqrt(w), einsum('ia,ja->ij', X, X): exactly symmetric in both envs on the dumped failing
  operands, same half-price billing (+4,608 flops/call), error vs float64 1.3-4.1e-7 (test_symout_gram_forms.py, stack/test_symout_gram_forms.log).
  Sign note: at this site the weights are squares (weight = coefficient * coefficient, _shared_old384_push), so w >= 0 and sqrt(w) is finite;
  my first crash report's "mixed-sign weights" was a guess, not a measurement. Rejected anyway: exact symmetry on the platform server is unproven.
- 2026-09-23 09:28 S9 (1534b1dd) FIRED by the lead as #332037 at 09:26 (fable_fire_v1/fire_candS9/candS9_S7_B2refit_mem.py, same sha).
- 2026-09-23 09:28 RESTART document: fable_porter_v1/RESTART_20260923.md.
- 2026-09-23 09:32 S10L = S9 + symout v3 + --lean v3 (engineer E-45, bit-exact round-trip cuts) built: stack/stack_S10L_so3_lean_B2_mem.py
  (da7e8f65945736d33f72c1aebc02a53a90490e4e84a9137ea997d1953a8f5ff3) = PFX R_B2 K_S7 --symout --lean, diet mem,yright,arena,hold2. einsum gate vs
  fired S9: PASS; validate OK. Parity (.venv, 35 cold then 100): MSE BIT-IDENTICAL to S10 (c7d30c19, no lean) and FLOPs identical (lean cuts
  round trips only); vs S9 flops -1.020/-0.967%, MSE -0.001/-0.053%, adjusted -1.02/-1.02% (stack/parity_S10L.txt). VALID40 not run (wall-gated).
- 2026-09-23 09:34 S10L VALID40 vs S9 rows STARTED 09:33:43 (lead GO; runner 6455, log paired_valid40_stack_S10L_so3_lean_B2_mem.log). Restart doc revised (merged with the lead's fallback).
- 2026-09-23 10:00 S10L VALID40 (0 errors) vs S9 rows: raw -0.072% [-0.137%, -0.005%] (25/40; |d| <= 0.1%, CI entirely below 0); util -1.022%;
  adjusted -1.094% [-1.156, -1.030] (40/40; per-net -1.41..-0.66%) PASS (bar <= -0.8%). Vs S7 rows adjusted -2.082% (39/40), raw -1.073%;
  vs bank 0.86155. Fire waits on S9's platform kernel ns/kflop and engineer2's wall projection. Files gate_S10L_valid40_vs_{S9,S7}.txt.

## 2026-09-23 10:04 S10' = push5p6 swapped for symout (lead 10:0x: S9 telemetry kernel ns/kflop 227.7 vs S7 227.0 vs S6b 210.9 -> S7's excess is CODE (push5p6); S10L FAILS the wall rule)
- stack/stack_S10p_p5_so3_lean_B2_mem.py (1deaeb454e5039859dcf2576de9136714cf3f6122e130abbd100aa940225b1ad) = PFX R_B2 + kernel flags
  --holdelim --bigarena --depthrule fable_engineer_v1/depthrule_fragment_push5.py --f32 third --evict5 --caps --caps2 --batch --bindhold --coldbind3
  --symout --lean; diet mem,yright,arena,hold2. Embeds push5 + symout v3 + lean v3 (no push5p6). einsum gate vs fired S9: PASS; validate OK.
- Parity (.venv, 35 cold then 100) vs S9: flops -0.798/-0.549%, MSE +0.025/-0.117%, adjusted -0.77/-0.67% (stack/parity_S10p.txt).
- VALID40 vs S9 rows started 10:03:40 (S10p_chain.out). engineer2: ABBA + backend-time gate vs S9 (wall x2.5).
- (2026-09-23 10:48, recorded 2026-09-24 10:03) S10' VALID40 (0 errors) vs S9 rows: raw -0.082% [-0.142%, -0.019%] (23/40); util -0.700% [-0.725, -0.675];
  adjusted -0.785% [-0.846, -0.722] (40/40; per-net -1.16..-0.27%). Vs S7 -1.777% (38/40); vs S6b -3.902%; vs bank 0.86424.
  Lead (09-24): engineer2 backend pairs S10' vs S9 -3.9% (12 nets), S10L vs S9 +1.4% (7 nets) -> S10L worst ~97.4 s clears the rule; S10L beats S10'.

## 2026-09-24 10:04 S11 = S10L recipe + B2 EPOCH-1 refit (lead: next fire): stack/stack_S11_so3_lean_B2e1_mem.py (acf67b9d59c8b68f6eb7ab0f1b9ce8209f11f5b40b0ac792d223ecce5e89a24a)
- PFX --refit=fable_physicist_v1/refit6b/s6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt K_S7 --symout --lean; diet mem,yright,arena,hold2.
- Diff vs S10L (da7e8f65) = exactly the 5 refit literal statements; all 28 tensors BIT-EQUAL to params_epoch1.pt (stack/verify_S11_literals.txt);
  einsum gate vs fired S9: PASS; validate OK. Chain S11_chain.out: parity 35,100 (FLOPs must equal S10L) then VALID40 vs S9 rows.
  Bar (lead): pooled adjusted <= -1.2% vs S9, >= 24/40 raw wins, util -1.0% +-0.1.
- 2026-09-24 10:23 S11 parity (.venv, 35 cold then 100): FLOPs IDENTICAL to S10L; vs S9 MSE -0.024/-0.457%, adjusted -1.04/-1.42% (stack/parity_S11.txt).
- 2026-09-24 10:23 S11 VALID40 (0 errors) vs S9 rows: raw -0.271% [-0.363%, -0.180%] (30/40); util -1.022% (per net -1.084..-0.981%); adjusted -1.289%
  [-1.382, -1.200] (40/40; per-net -2.02..-0.83%) -> PASSES all three lead bars. Vs S10L rows (ep1 vs ep0, same kernels): raw -0.199%
  [-0.309, -0.087] (26/40; torch -0.23 pt). Vs bank 0.85985. Files gate_S11_valid40_vs_{S9,S10L}.txt. READY TO FIRE (lead).

## 2026-09-24 10:30 S11 FIRED by the lead as #332182 (10:24; fable_fire_v1/fire_candS11/candS11_S10L_B2e1_mem.py, acf67b9d)
## 2026-09-24 10:30 SCOPE (no build) of native structural options (lead's request; sources fable_physicist_v1/refit6b/foldpush.py, fable_covhunter_v1/s6bcard_r320.py)
- RANK R: 3 count-asserted edits in the live file: push 'rank = 384', alive check '< 384', _SLR_RANK = 384 (lazy-rotation shape asserts); the
  envelope width 384.0 stays (conservative for R <= 384). Kernels shape-generic (_mm: zero dims -> zeros, n/k > 8192 tiled, non-/32 -> plain
  matmul). Torch algebra identical (sketch = first R columns of W, QR). ~45 min build + R=384 regression; ~1.5 h fire-ready. Risk low.
- FOLD: schedule fits the join builder's (first, last) ranges (build_stack.sh hard-codes every3 at line 59 -> --sched). BLOCKER: native L15 row
  cut pushes only kept rows (_shared_old384_push(15, W[:, rows])), so a join at L15 would sketch the frame from a column subset (torch masks rows
  after a full push): diverges, and raises the QR-shape check when fewer than r rows are kept. Fix (A) split join-state (full W) from row
  projection at a join L15, or (B) torch fold through L14 only. Zero-width young block safe (_mm zeros); wide joins tiled; per-layer ranks
  (r0 != r) unsupported (+2 h). Effort (A) ~3.5-4 h fire-ready, (B) ~2 h + torch re-run. Risk moderate (L15 split, bill/wall sign, handles).

## 2026-09-24 10:49 NATIVE TIER-RANK OPTION (lead GO): build_tier_rank.py (PT-14) + rank sweep on S11 (no refit)
- Edits: _TIER_RANK = R (one literal) with _SLR_RANK = _TIER_RANK; live push 'rank = _TIER_RANK' and alive guard '< _TIER_RANK'; envelope 384.0 kept
  (conservative, R <= 384 enforced). stack_S11_R{384,352,320,288}_mem.py = caa6b2c4 / 8184574c / ea630fdb / 689998d0; einsum gate PASS; validate OK.
- R384: all-16-layer predictions + FLOPs IDENTICAL to S11 on 35,100,37,39,42,48.
- No-refit sweep (.venv, same 6 nets/order): R352 bill -1.672% raw +1.637% adj -0.06%; R320 bill -3.889% raw +4.258% adj +0.20% (r37 +8.12% vs
  covhunter torch +7.93%); R288 bill -5.933% raw +10.575% adj +4.01%. Report stack/rank_sweep_report.txt. Paired census running (rank_census.out).
- 2026-09-24 11:00 rank census (concurrent pairs vs S11, client/server, nets 35 cold,100,37): R352 requests -2.04%, server CPU -6.5%, flops -1.70%; R320 requests
  -2.91%, server CPU -6.3%, residual -23/-25/+3 ms, flops -3.89%; R288 requests -5.10%, server CPU -8.7%, flops -5.95% (stack/rank_census_report.txt).
- 2026-09-24 11:01 FOLD CLOSED by the lead (physicist no-refit: rank 768 +51% raw at +30% bill; rank 384 16x worse): no A/B L15 decision needed. Rank line open: r320 refit ETA ~12:10 (watch_r320_refit.sh running).
- 2026-09-24 11:05 build_stack.sh PT-8b: new option --rank=R (join -> 7_join.py -> build_tier_rank.py -> out; CHAIN.txt lists 'tier rank R'). Old script kept as
  build_stack_v2_S11.sh. Regressions: no --rank rebuild of S11 + diet = acf67b9d (fired S11, byte-identical); --rank=320 = ea630fdb (=
  stack_S11_R320_mem.py; rank-then-diet == diet-then-rank). Engineer flags still pass after '--'. S12-R320 recipe = S11 recipe with
  --refit=<covhunter r320 .pt> --rank=320, then diet mem,yright,arena,hold2.

## 2026-09-24 11:30 DEADCUT35 (lead GO; analyst AN-50 variant (a), E-48 2b): build_deadcut.py (PT-15); stack/stack_S11_dc35_mem.py (63db1ee2...)
- Live push gains rows=None: join keeps the full W; projection U, old rebuild, young/newborn push on W[:, rows]. Loop push -> _deadcut_push
  (L1-14, kept = select(a + 3.5, 0), W5 off for that call, pp and pre-basis gathered back with exact zero dead rows). einsum gate PASS, validate OK.
- NULL build (stack_S11_dcnull_mem.py dd82c7f7): all-16-layer predictions + flops identical to S11 (r35, r100).
- Dropped rows per layer = torch variant (a) exactly (r37, r39); raw vs S11: r37 -0.53% (torch -0.63%), r39 +0.65% (torch +0.76%).
- REAL BILL vs S11 (.venv, 35 cold,100,37,39): -0.364/-0.687/-0.654/-0.405%, pooled -0.525% (E-48 estimate -2.3%). Adjusted -0.80/-0.92/-1.18/+0.24%.
- Cause (stack/mm_rows_bench.log): only m == 1024 takes the six-level padded push; m < 1024 bills ~4-5.5% more per row (m=992: young push +0.84%,
  old rebuild +2.17% vs m=1024). VALID40 held; options sent to the lead (kernel extension to m % 64 == 0 + kept rounded up to 64; stack as a small layer; drop).
- 2026-09-24 11:38 deadcut35 USER-CPU census (lead gate; run_usercpu_pair.sh = optime_split_v3 pair, OpenBLAS env, 6 nets): flops -0.447%, USER CPU -10.35%
  (276.6 vs 308.5 s), backend -10.76%, major faults 0/0 (stack/usercpu_dc35_vs_S11.txt). Saving = Winograd transforms: add/subtract -25.3 s
  (207 -> 172 GF); matmul +1.3% flops but -5.7 s. The m=1024 six-level padded push is flop-cheap but transform(CPU)-heavy; the kernel
  extension to m%64 may trade the CPU saving back for ~0.5-1% bill. Builder PT-15b: --z=Z --round=R (defaults byte-identical; old build_deadcut_v1.py).
- 2026-09-24 11:48 COLPACK (engineer E-49, post-diet build_colpack.py): my S11 build = engineer's candS11cp_mem.py (c6ca3ebf, byte-identical). Vs S11 on 6 nets:
  all-16-layer predictions IDENTICAL; flops -0.293% cold, -0.345..-0.351% warm (pooled -0.339%). colpack+deadcut35 (stack_S11_cp_dc35_mem.py
  7333dea9) vs deadcut35: predictions identical, dropped rows identical, flops a further -0.17..-0.22%; vs S11 -0.53/-0.89/-0.86/-0.62%.
  ORDER: build_stack [--rank] -> diet -> build_colpack -> build_deadcut (reverse breaks colpack's young-push anchor). stack/colpack_parity.txt.
- 2026-09-24 11:48 stage_s12full_pair.sh (running): S12-full = S12-R320 + colpack + deadcut35 -> gate, validate, user-CPU pair vs S11 (porter_S12full).
  stage_r320dc35_pair.sh stopped before building (superseded).

## 2026-09-24 12:06 R320 OUT: covhunter r320 refit VALID16 valid_epoch0 1.8045e-8 (valid_before 1.8047e-8; bar <= 1.78106e-8) -> FAIL
- Native S12-R320 (stack_S11r320ref_mem.py 19b3fb4c; 5 literals, 28 tensors bit-equal to params_epoch0.pt aa26d6ff, gate + validate OK): vs S11 raw
  r35 +0.86% (no-refit +1.25%), r100 +10.65% (no-refit +10.12%); flops = no-refit R320. One epoch of trunk refit does not absorb rank 320.
- Per plan: S12-lite (stack_S11_cp_dc35_mem.py 7333dea9 = S11 + colpack + deadcut35) VALID40 vs S11 rows running (pregate_s12_valid40.sh, since 11:55).
- S12-full (7e4d0486) built + gated but moot; its CPU pair stopped before running.
- READ6Q (engineer2, build_read6.py q, applied LAST): my S11 build = engineer2's S11_read6q.py (91641643). S12-lite + read6q =
  stack/stack_S11_cp_dc35_r6q_mem.py (c20d67c7...): einsum gate PASS, validate OK. Proposed next VALID40 (fire bar -1.0%, CI excl 0, >= 26/40).
- 2026-09-24 12:11 engineer2: my S12 stack (stack_S11_cp_dc35_r6q_mem.py c20d67c7) = their S12lite_read6q.py byte-identical; read6q vs S12-lite (OpenBLAS, 35/100/37/39): bill -0.363% pooled, MSE +0.04% pooled, adjusted -0.32% (the gain holds on the deadcut base). Their user-CPU pairs queued: e2_read6q vs S11, e2_S12r6q vs S11 (total S12 wall).
- 2026-09-24 12:24 engineer E-51: --m64 (fable_engineer_v1/build_m64.py, post-stage LAST after colpack + deadcut --round=64): bit-exact at m=1024; under deadcut r64 bill a further -0.58/-0.61% (r35/r100) but +3.4-4.6 s local user CPU per net -> engineer recommends NOT using it (CPU finding confirmed). If ever combined with read6q, the order must be settled: both claim 'last' (both touch _mm dispatch).

## 2026-09-24 12:24 S12-LITE VALID40 (S11 + colpack + deadcut35, 7333dea9) vs S11 rows: raw -0.012% [-0.142, +0.122] (22/40); util -0.657% [-0.732, -0.586];
   adjusted -0.666% [-0.824, -0.508] (35/40; per-net -1.71..+0.28%); vs bank 0.85413. Fire bar (-1.0%) FAIL, as expected: a clean base.
- S12 (S12-lite + read6q, c20d67c7) client/server smoke OK: r35 48,469 requests / server CPU 64.98 s; r100 43,061 / 64.15 s (residual 0.55-0.58 s at load ~19).
- S12 VALID40 vs S11 rows running since 12:16 (fire_chain_S12.sh). Expectation ~-0.98% (S12-lite -0.666% + read6q -0.32%): at the bar.
- T640 trunk mid80: VALID16 1.75128e-8 vs B2 ep1 1.74614e-8 = +0.29% (worse): switch rule NOT met. S12T640mid built + checked
  (stack_S12T640mid_mem.py c0630683; 5 literals, 28 tensors bit-equal to params_mid80.pt 6d67547e; gate PASS; validate OK). Watcher's VALID16
  parse used the per-net dict (mean read from the analyst log line valid16_mid_update80).

## 2026-09-24 13:37 S12 VALID40 (c20d67c7) vs S11: raw +0.022% [-0.124, +0.171] (23/40); util -1.022%; adjusted -0.995% [-1.164, -0.824] (39/40): relaxed bar
   (-0.8%) PASS; fire copy staged fable_fire_v1/fire_candS12/ (FIRE_CHECKS.txt all OK). read6q vs S12-lite: adjusted -0.332% (40/40). Lead 12:48: SUPERSEDED by S13.
## 2026-09-24 13:37 S13 = S12 + engineer --ipt v3 (build_ipt_v3.py, post-stage LAST): stack/stack_S13_ipt3_mem.py (2a5dd324 = engineer's candS12ipt3_mem.py)
- einsum gate vs S11 PASS; validate OK; no new imports. In-process (.venv) vs S12: all-16-layer predictions IDENTICAL on r35, r100; flops -2.486/-1.747%;
  wall 25.7 vs 39.6 s / 25.4 vs 36.3 s (stack/S13_vs_S12_parity.txt). fire_chain_S13.sh running (smoke, VALID40, identity vs S12 rows, bar, stage fire_candS13).

## 2026-09-24 14:07 S13 FIRED by the lead as #332195 (13:53; fable_fire_v1/fire_candS13/candS13_S12_ipt3_mem.py, 2a5dd324) = S11 + colpack + deadcut35 + read6q + ipt3
- VALID40 vs S11 rows (0 errors): raw +0.022% (identical to S12's rows), util -3.473%, adjusted -3.446% [-3.615, -3.275] (40/40); final MSE identical
  to S12 on 40/40 nets; relaxed + strict bars PASS. FIRE_CHECKS.txt complete (sha, validate, einsum gate, no new imports, recipe byte-rebuild, smoke,
  identity, VALID40/bar, engineer cs_probe cmp byte-identical r35/r100, mem gate). Expected ~2.96e-9 (first sub-3e-9).
- 14:06 released a STALE /tmp/fable_paired.lock (taken 13:51:52 by engineer2's S13 wall pair, which finished 13:58 without its rmdir; verified no
  holder: 4 runners in sleep loops, no python on lock scripts). Queue resumed (engineer e_ipt4 vs e_ipt3b took it 14:06:34).
- 2026-09-24 14:20 S13 #332195 GRADED 2.9743e-9 = NEW BANK (first sub-3e-9; 50/50 public nets better than S11; roundtrip max 73.2 s; 0 failed). Base for S14 = fable_fire_v1/fire_candS13/candS13_S12_ipt3_mem.py + paired_valid40_stack_S13_mem.json. S12 fire copy kept as history.

## 2026-09-24 14:35 S14 = engineer candS14_mem.py (9003e536) = S13 + tree six-level reads (bit-exact) + deadcut rows rounded to 64 + m64 (lead: next fire)
- Recipe REBUILT byte-identical: fired S11 -> build_colpack -> build_deadcut --round=64 -> build_m64.py (= engineer candS11cp_dc35r64_m64_mem.py af3f3161)
  -> build_read6.py q -> build_ipt_v3.py -> cat + ipt_read6_fragment.py = 9003e536 (scratch_rank_order/s14_recon_143401). Copy: stack/stack_S14_mem.py.
- fire_chain_S14.sh running: smoke -> VALID40 vs S13 rows -> bar (adj <= -0.7%, CI excl 0, >= 26/40; raw within +-0.1%) -> stage fire_candS14.
- 2026-09-24 14:46 ALTBASIS (engineer2, build_altbasis.py, apply after --ipt and any depth fragment): S13_ab = stack/stack_S13_ab_mem.py (97178798 = engineer2's
  S13_ab.py); S14_ab = stack/stack_S14_ab_mem.py (e313ab77). Both: einsum gate vs S13 PASS, validate OK, no new imports. valid40_S13ab.sh queued after
  the S14 chain (VALID40 vs S13 rows, report only). engineer2 parity vs S13: bill -0.995%, MSE -0.099% pooled, adjusted -1.09% (rounding-level).
- 2026-09-24 14:48 lead: S13+altbasis VALID40 SKIPPED (stage stopped before running). S15 = engineer candS14ab_mem.py (9d898520) = S14 + --ipab: cat S14 +
  ipab_fragment.py = 9d898520 byte-identical; einsum gate vs S13 PASS, validate OK, no new imports; stack/stack_S15_mem.py. fire_chain_S15.sh queued
  after S14's chain: smoke -> VALID40 vs S14 rows -> bar (adj <= -0.7%, CI excl 0, raw +-0.15%) -> stage fire_candS15.
- 2026-09-24 14:50 lead HOLD on S14's fire: smoke residual r35 (cold) 0.379 s vs S13's 0.285 s (cap 0.4 s; S13 platform max 0.28 s). residual_gate_S14.sh
  queued after S14's VALID40: 3 concurrent S13/S14 client/server pairs (order alternated), nets 35 cold,100,37; rule: S14 cold <= S13 cold + 0.03 s or
  projected platform max (0.28 s + cold delta) <= 0.33 s. Then fire_chain_S15.sh and residual_gate_S15.sh (same design, tags rz15_*).
- 2026-09-24 14:56 S14 VALID40 vs S13 rows (0 errors): raw -0.013% [-0.102, +0.075] (20/40); util -0.935%; adjusted -0.948% [-1.052, -0.844] (40/40): bar PASS
  (adj <= -0.7%, raw +-0.1%). Fire copy STAGED fable_fire_v1/fire_candS14/candS14_S13_tree_r6_m64_mem.py (9003e536) + FIRE_CHECKS (all OK). Fire HELD on the
  residual gate (running since 14:54).
- 2026-09-24 14:56 lead: S15 SWITCHED to S14 + engineer2 --altbasis (stack/stack_S14_ab_mem.py e313ab77); the ipab S15 (9d898520) dropped for residual (+531 warm/+1,007
  cold ops/net vs altbasis -59, CPU equal). fire_chain_S15ab.sh + residual_gate_S15ab.sh queued after the S14 residual gate (ipab scripts kept, unused).
- 2026-09-24 15:00 S15 residual gate made THREE-WAY (engineer2: add S15 vs S14): residual_gate_S15ab.sh = S13/S14/S15 concurrent per rep (3 reps, order rotated, tags rz3_*); lead's rule unchanged (S15 vs S13); pairwise version kept as residual_gate_S15ab_pairwise_v0.sh. Fixed a bash-3.2 incompatibility (declare -A) before it ran.
- 2026-09-24 15:11 engineer confirmed S14/S15 post-stage commands (plain appends = cat; build_m64 before build_read6); appended to fire_candS14/FIRE_CHECKS.txt.
- 2026-09-24 15:11 S14 RESIDUAL GATE PASS: cold delta vs S13 -0.013 s (S13 0.471 / S14 0.458 s mean under load ~16), projected platform max 0.267 s; the smoke's 0.379 s was a window artifact (absolute residuals under load 0.44-0.47 s for BOTH). Appended to fire_candS14/FIRE_CHECKS.txt.
- 2026-09-24 15:12 S14 FIRED by the lead as #332203 (15:05, from the gate log; fable_fire_v1/fire_candS14/, poll.log there). S15 (altbasis) chain continues.
- 2026-09-24 15:29 S16 = engineer candS16bb6_mem.py (a4a1e0f7) = S15 (e313ab77) + cat bb6_fragment.py (E-59 batched depth-6): append byte-identical; einsum gate vs fired
  S14 PASS, validate OK, no new imports; stack/stack_S16_mem.py. fire_chain_S16.sh armed: waits for S15's residual gate AND fable_porter_v1/S16_GO
  (touched only if engineer2's 4-net bb6 parity is bit-identical), then smoke -> VALID40 vs S15 rows (adj <= -0.7%, raw +-0.15%) -> paired residual vs S15 -> stage.
- 2026-09-24 15:30 S15 (e313ab77) VALID40 vs S14 rows: raw +0.037% [-0.023, +0.094] (17/40); util -1.197%; adjusted -1.161% [-1.218, -1.106] (40/40): bar + raw PASS. Fire copy STAGED fable_fire_v1/fire_candS15/candS15_S14_altbasis_mem.py (FIRE_CHECKS all OK). 3-way residual gate running (fire waits on it).
- 2026-09-24 15:32 S16 RELEASED (S16_GO): engineer2's 4-net parity bit-identical (all layers, maxabs 0.0, bill delta 0; porter-verified from par4_engD6.log). fire_chain_S16.sh starts after S15's residual gate.
- 2026-09-24 15:38 S15 FIRED by the lead as #332207. Its 3-way residual gate PASS (cold: S15-S13 +0.024 s, S15-S14 +0.017 s, S14-S13 +0.007 s; projected 0.304 s).
- 2026-09-24 15:38 lead: S16 residual rule TIGHTENED to cold delta <= +0.010 s vs S15 (projected platform max <= ~0.31 s = S14 actual 0.276 + 0.017 + 0.010); if exceeded,
  stage anyway + FLAG. fire_chain_S16.sh (running, old 0.03 rule, not edited) + post_S16_residual.sh (applies 0.010, stages-anyway on residual-only fail, flags).

## 2026-09-24 17:26 LEAD GO on levers 1-2 (fp16 DOA: flopscope bills float16 GEMMs 1.0x; k4 lever merged into the physicist's)
- Lever 1 RUNNING: star_ablation_s6b.py (torch s6bcard, B2 ep1, no refit, 8 VALID nets, 3 threads): arms base, star term zeroed at L1-4/5-8/9-11/12-14
  (state_out hook zeroes the plane leaving layers 0-3/4-7/8-10/11-13), rank-32/128 SVD truncation of the plane at every layer. base r37 = analyst's
  base exactly (1.7361759331095405e-08). Log stack/star_ablation.log / .jsonl.
- Lever 2 staged: bill_by_function.py on S16 r100 (callers) queued behind the lock; if the pair map >= 6% -> pairmap_f32_s6b.py (torch f32 pair map, 8 nets).
- 2026-09-24 17:39 LEVER 1 (star) torch no-refit, 8 VALID nets (stack/star_ablation_report.txt): zero star L1-4 +3.55%, L5-8 +22.6%, L9-11 +18.7%, L12-14 +11.9% (skip DEAD);
  rank-32 plane +0.37% [-0.72, +1.56], rank-128 +0.49% [+0.19, +0.75] (NOT a cliff). Native: low-rank A makes pushed=W^T A and s2 O(n^2 r); s1 stays full ->
  ~2 of 4 batched products (_bb_product 16.5% of S16's bill) -> ~-8.2% bill. Next falsifier: torch VALID40 at r=32/64.
- 2026-09-24 17:39 LEVER 2 (pair map) DEAD: bill census S16 r100 (stack/billfn_S16_r100.txt): _nonlinear_pair_batch 1.41% (+first_gaussian 0.06%) < 6%.
- 2026-09-24 17:39 S17-j05 built (stack_S17_j05_mem.py ae52f49a; build_j05.py PT-27; chain reproduces S16 + S16r3b byte-exactly from B2 ep1; 28 literals bit-equal; null = S16
  bit-identical r35/r100; native vs torch on r35/r100 agree (+1.18/+0.73% vs +0.84/+0.52%)). fire_chain_S17.sh running.
- 2026-09-24 17:58 S18 SIZING (PT-29/30): _bb_product 16.5% = FULL cost (leaf matmul inline) of 56 dense six-level 1024^3 products/net (14 layers x {cov first, star
  pushed, s1, s2}) at 1.057 GF each; dropping pushed+s2 = -8.26% gross. flopscope: f32 (M,K,N) = 2MKN, f64 2x -> low-rank path in f32.
  Star-plane anatomy (star_plane_anatomy{,_v2}.py, net 37, rel error of the NEXT layer's star term): diagonal <1% of plane energy (diag+LR ~free,
  tiny gain .104->.100); rank-32 exact .101 median (L0-3 planes .24-.37, L6-14 .06-.11); float32 range finder q=0 .217 (x2 worse; p=96 still .136),
  q=1 .104, q=2 .101 -> build with ONE power iteration: ~0.63 GF/layer -> net ~-5.8% bill (+~-0.15% at L15's star diagonal).
  cert_star_diaglr.py: merged native form (2 pw + wt2*d_row)@W + (wt2@Y)@(W^T X)^T == dense on A' (8e-16; negative control fails).
- 2026-09-24 18:05 BILL MAP S16 r100 (bill_by_function_v2.py PT-31: kernel-transparent, by semantic caller + line; stack/billfn2_S16_r100.txt): bank column push
  six-level >4096 cols (_cp_d6_shared) 22.1%; young read 21.9% (full read before the old tier 11.5%, young part after 10.4%); cov+star 20.2% (batch1
  cov-first+star-pushed 8.3%, batch2 s1+s2 8.3%, cov _sym_upper 3.5%); old-tier products (_shared_old384_mm) 13.3%; newborn push 6.9%; push5 6.8%;
  join 1.8%; pair map 1.4%; rest < 1.2% each. CP bank (push + read) ~71%.
- 2026-09-24 18:05 PT-30 partial (stopped at 6/16 for the census slot): nets 37/39/42 pooled dr32 +0.85%, rf8q1 +0.83% vs exact r32 +1.16% (stack/star_diaglr_partial.txt).
- 2026-09-24 18:11 S17 REVISED (lead GO): stack/stack_S17p_j05_prebind_mem.py 9a4ccc1a = stack_S17_j05_mem.py (ae52f49a) + cat engineer prebind_fragment_v2b.py
  (diff vs candS16r3bp2b 579056e9 = 18 lines = 5 literals + j05 lines). einsum PASS, validate OK. fire_chain_S17p.sh (log fire_chain_S17p.out):
  smoke WITH setup() (census_boundary_setup.py = engineer's block; run_census_porter_v2.sh) -> VALID40 vs S16 (prebind inert: harness has no setup)
  -> bar adj <= -1.5%, CI excl 0, >= 26/40 -> native vs torch -> residual vs S16 (setup client, cold <= +0.010 s) -> fire_candS17/candS17_S16r3bp2b_j05_mem.py.
  fire_chain_S17.sh (without prebind, bar -1.0%) STOPPED in its lock wait before any step.
- 2026-09-24 18:18 S18 QUEUED (lead): stack/stack_S18_S17p_lrq1r32_mem.py ab8c8ba8 = S17p 9a4ccc1a + cat engineer lrstar2_k48_q1_r32.py (823c693b; 90 lines).
  einsum PASS, validate OK, recipe dry run byte-exact (S17p + S18). fire_chain_S18.sh (log fire_chain_S18.out): waits S17p staged -> torch pre-gate
  (j05 r32 VALID40 pooled raw <= +0.8%) -> rebuild/einsum/validate -> smoke with setup() -> VALID40 vs S17p (adj <= -3.0%, CI excl 0, >= 30/40, raw
  +-1.0%) -> residual vs S17p (setup client) -> fire_candS18/candS18_S17_lrq1r32_mem.py. Bill (census v2, r100 cold): lrstar on S16r3b 338.51 GF vs S16
  358.49 GF = -5.57% (cov+star 72.25 -> 46.05, _lr_factor +6.75, L15 star diag 0.87 -> 0.10 GF). Omega draw bit-identical numpy 2.2.6/2.4.6.
- 2026-09-24 18:18 S17p smoke with setup(): bound 358 in 2.14 s; r35 util .160752 (cold, prebind), r100 .160447; VALID40 vs S16 started 18:13:56.
- 2026-09-24 18:27 S17p VALID40 vs S16 PASS: raw -2.159% [-3.116, -1.211] 31/40; util +0.015%; adjusted -2.138% [-3.091, -1.196] 30/40 (bar -1.5%/26).
  native vs torch j05 effect: pooled -2.159% vs -2.046%, per-net corr 0.996, per-net diff sd 0.289% (gate_S17p_native_vs_torch.txt). Residual gate running.
  fire_chain_S18.sh relaunched 18:28 (idle before; FIRE_CHECKS now quotes the engineer's usercpu_pair_S18.log + chain_S18gate.log).
- 2026-09-24 18:38 S17 STAGED 18:33 (fire_candS17/candS17_S16r3bp2b_j05_mem.py 9a4ccc1a) and FIRED by the lead as #332224 (18:34). Residual gate vs S16 (setup
  client) PASS: cold -0.033 s (requests 39,841 -> 34,752), r100 -0.023 s, r37 -0.023 s. Expected ~2.79-2.80e-9.
- 2026-09-24 18:38 STAR TORCH VALID40 (PT-29, no refit, 40 nets) ALL PASS the lead's bar: B2 r32 +0.375% [-0.105, +0.845] 13/40; B2 r64 +0.290% [-0.019, +0.598];
  j05 r32 +0.190% [-0.051, +0.437] 18/40 (stack/star_valid40.jsonl, star_valid40_report_S18pregate.txt). S18 pre-gate PASS.
- 2026-09-24 19:01 S18 STAGED 18:58 (fire_candS18/candS18_S17_lrq1r32_mem.py ab8c8ba8): VALID40 vs S17p adj -5.352% [-5.598, -5.097] 40/40; raw +0.232%
  [-0.027, +0.498] 18/40; util -5.569%. Residual vs S17p PASS (cold -0.025 s). Engineer: wall pair user CPU -6.36%, flops -5.60%; identity PASS.
  NEXT: S19 = S18 + covhunter learned-ON dials (literal-only + j2/j3 physical legs in _bd_update); gates VALID40 vs S18 rows adj <= -1.5% + residual.
- 2026-09-24 19:14 S18 FIRED by the lead as #332226. S19 PLUMBING (PT-33): build_s19_dials.py (guarded native edits: _bd_update raw legs X0/X2/X3 +
  _BD_J1_PHYS=X1, b x bscale after the b-fix, lam x s_carry / newborn x s_new at the end-of-layer concatenate, k4 x band L1-7/8-11/12-15 at its
  definition, _K31R_TABLE c0/c1; --gn=torch|shared for the GN dcov's k4 + row table). NULL = S18 bit-identical (all 16 layers + flops, r35/r100:
  stack/S19_null_parity.txt). Old FIT (s6bcard_jfitON1) natively vs S18: torch-GN r35 -3.342% r100 -1.511%; shared-GN -3.476% / -1.013%;
  flops +0.056%. Lead: option (a) = covhunter refits on native-consistent semantics (= --gn=shared). fire_chain_S19.sh <fit.json> <rows.json> ready.
- 2026-09-24 19:17 S19 LAUNCHED (lead: don't wait for the refit): fire_chain_S19.sh stack/s19_fit_jfitON1.json (= covhunter S19_dials_spec.json values, all 12
  checked) + rows ../fable_covhunter_v1/s18_revalidate_n40.json (parity INFORMATIONAL: rows use mirrored semantics). Candidate
  stack/stack_S19_dials_mem.py c333be74 (--gn=shared). einsum PASS, validate OK, recipe rebuild sha equal. Informational old-values torch-GN
  VALID40 stopped after 84 s to free the lock. Covhunter's consistent refit, if better, = S20.
- 2026-09-24 19:19 S19 smoke OK (setup 358 in 2.13 s; r35 util .15175, r100 .151457 = +0.06% vs S18); VALID40 vs S18 started 19:18:28. S19 dials =
  covhunter S19_fit_current.json (value-identical). covhunter: S19_dials_spec.json is a superseded draft (mirrored semantics); their
  s18_revalidate_n40.json fit rows are mirrored semantics (approximate parity), base_s18 exact. S20 = S18 + S19_fit_shared.json (refit on shared
  semantics, rows s18_revalidate_s6bcard_jfitON2_n40.json exact) ~20:25: fire_chain_S20.sh auto-launches when both files exist
  (scratchpad launch_S20_when_ready.sh); gate vs S18 rows (adj <= -1.5%, CI excl 0, >= 26/40) + informational S20 vs S19.
- 2026-09-24 19:52 S19 VALID40 vs S18: adj -1.450% [-2.161, -0.737] 31/40, raw -1.511% [-2.220, -0.799] 32/40, util +0.052%; missed -1.5% by .05 pt -> lead
  RELAXED to <= -1.2% (CI excl 0): PASS; stage_S19_relaxed.sh runs the residual gate vs S18 then stages fire_candS19. Native vs torch (mirrored
  rows, informational): pooled -1.511% vs -2.053%, per-net corr .969, diff sd .614%, mean +.528 pt (= the semantics cost).
  S20 REDEFINED (lead): S18 + covhunter refit dials (--gn=shared) + engineer2 d6x_fragment_v2.py (66647fef; S18_d6x_v2 d3884efb = S18 + it)
  appended last. fire_chain_S20.sh patched (backup scratch_rank_order/fire_chain_S20_pre_d6x.sh.bak) and auto-launch re-armed.
  Stand-in (current dials) composition: dials part == S19 byte-exact; = S19 + cat d6x; einsum PASS, validate OK.
- 2026-09-24 20:23 S19 STAGED 20:03 (fire_candS19/candS19_S18_dials_mem.py c333be74; residual vs S18 PASS cold +0.005 s). S20 auto-launched (covhunter refit
  files landed 19:39/19:50): stack_S20_dials_d6x_mem.py 46821240 (refit dials X1 .5328 X2 .2134 X3 -.2283 bscale 1.0488 s_carry 1.00439 s_new .9932
  c0 .6527 c1 1.6084 k4 .9883/.9618/.9631 + d6x); smoke OK (setup 362 in 1.39 s; util r35 .150627); VALID40 vs S18 running (20:18).
  S21 PREBUILD (PT-37 prebuild_S21.sh, verify_literals.py PT-36): S18ep2 (S18 recipe --refit=s6b_j05_ep2 params_epoch0.pt 3057234a) cff1eea4: ALL 28
  BIT-EQUAL, diff vs S18 = 5 literal statements; S21 stand-in (S19 dials) 29f99876: ALL 28 BIT-EQUAL, diff vs S18+same dials+d6x = 5 literal
  lines; einsum PASS; validate OK.
- 2026-09-24 20:25 fire_chain_S21.sh <round-1 dials.json> <tag> READY (prebuild_S21.sh -> smoke with setup -> VALID40 vs S19 rows, bar adj <= -1.0%, CI excl 0
  -> residual vs S19 (setup client) -> fire_candS21/candS21_S18ep2_dials_d6x_mem.py). Waiting for covhunter's round-1 dials path (~21:10).
- 2026-09-24 20:34 S20 STAGED 20:33 (fire_candS20/candS20_S18_dials2_d6x_mem.py 46821240): VALID40 vs S18 adj -2.337% [-2.974, -1.699] 34/40, raw -1.843% 33/40,
  util -0.514%; vs S19 adj -0.900% [-1.047, -0.757] 39/40 (raw -0.337%). Exact parity vs covhunter jfitON2 rows: native -1.843% vs torch -1.981%,
  corr .986, diff sd .349%. RESIDUAL FLAG: cold +0.011 s vs S18 (rule +0.010; requests +1,187: d6x ~+1,040, dials ~+150), r100 +0.004, r37 +0.007.
- 2026-09-24 20:34 S21 (round1) LAUNCHED: stack_S21_round1_mem.py eddbcfcd (ep2 literals ALL 28 BIT-EQUAL; round-1 dials covhunter dials_r1_ep2.json 31110fbb,
  shared). Stand-in (ep2 + S19 dials + d6x) vs S19 in-process: r35 MSE -0.649% flops -0.735%; r100 MSE -1.010% flops -0.493%.
- 2026-09-24 21:16 S21 BILL CENSUS written: census_S21.json (per layer x category, U and %, column counts, unit costs). Tools: bill_census_layers.py (PT-38,
  per-layer attribution via the live _predict_binary frame) + census_S21_report.py (PT-38b categories). Headline: 78.7% per-birth / 21.3% fixed.
- 2026-09-24 21:20 S20 GRADED #332230 2.5937e-9 = BANK (lead); S21 fired #332231 (20:56). S22 PRE-STAGED: prebuild_dials_stack.sh (PT-39; reproduces fired S21
  eddbcfcd byte-exactly), fire_chain_S22.sh (PT-39b: VALID40 vs S21 rows, bar adj <= -0.8% CI excl 0; residual vs S21; fire_candS22),
  auto_launch_S22.sh RUNNING (waits: physicist s6b_loopR1 VALID16 <= 1.657977e-8 + params; covhunter dials_r2_loopR1.json).
- 2026-09-24 21:25 fire_chain_S22.sh: added an informational per-net parity step vs covhunter's r2_loopR1dials_vs_r1.json (bounded 10-min wait, before staging). Covhunter's round2_chain.sh writes dials_r2_loopR1.json only if their fit clears -0.3% on VALID16.
- 2026-09-24 21:53 JOIN SKETCH TEST (PT-40, torch no refit VALID16; RESULT_join_sketch_PT40.txt): B_lit +3.46%, B +3.59%, C (Nystrom) +0.49% [-0.21, +1.20]
  raw vs current join; all FAIL the +0.1% bar. Pricing (21:24): L15 old part already in-frame (0 U); joins already randomized (Gram form, Omega = W
  cols). MACHINE RULE (lead 21:5x): machine_gate.sh gates auto_launch_S22_v2.sh (v1 stopped) + S22 smoke/VALID40/residual. loopR1 VALID16 PASS
  1.65054e-8 (params f18a30b9); S22 waits on covhunter dials_r2_loopR1.json.
- 2026-09-24 21:58 S22+s1 VARIANT READY (lead): prebuild_dials_stack.sh 4th arg EXTRA (after d6x); (ep2 + round-1 dials + engineer2 s1lr_fragment_v2.py 67bc7260)
  reproduces S21_s1lr_mem.py 30aea66e byte-exactly (no-EXTRA path still = fired S21). fire_chain_S22s1.sh (VALID40 vs paired_valid40_e2_S21s1lr.json
  once complete, bar -0.8% CI excl 0; residual vs S21_s1lr). auto_launch_S22_v3.sh (v2 stopped idle) waits: dials -> stack/S22_recipe.flag (S21|S21s1,
  written by the porter on the lead's s1 verdict) -> machine gate -> chain.
- 2026-09-24 22:01 S21jn (lead GO: Nystrom join C natively on S21): build_join_nystrom.py (PT-42; edits only the LIVE _shared_old384_push join block; switch
  _JOIN_NYSTROM appended; null 99a029fd, variant stack/stack_S21jn_mem.py 23c3abc2; einsum PASS, validate OK). run_S21jn_parity.sh (null must be S21
  bit-identical) then fire_chain_S21jn.sh (below S22: yields while 'bash fire_chain_S22*' runs + machine gate; VALID40 vs S21 rows bar adj <= -1.2%
  CI excl 0 + raw <= +0.8%; residual vs S21; fire_candS21jn). Torch mirror for refits: s2card.make_push join block = join_sketch_test.py arm C.
  NOTE: yield pattern must be "bash fire_chain_S2[2]" (a bare "fire_chain_S22" matched my own monitor's command line and stalled the job 5 min).
- 2026-09-24 22:04 S22 flag = S21s1 (lead: s1 VALID40 PASS). fire_candS21s1 PRE-STAGED (stage_S21s1.sh; finalizes FIRE_CHECKS.txt on engineer2's residual verdict).
  S21jn: null bit-identical (r35/r100); Nystrom in-process vs S21 flops -1.72/-1.82%, MSE +0.31/+2.30%; chain launched (smoke waiting on the lock).
- 2026-09-24 22:11 S21s1 FIRED #332236 (lead, 22:07; from my staged fire_candS21s1, engineer2 residual PASS cold +0.003 s). Covhunter round-2 dials FAILED (dials
  converged at round 1). S22 LAUNCHED 22:10 = S21s1 + loopR1 params (f18a30b9) + ROUND-1 dials + d6x + s1 (fire_chain_S22s1.sh, log fire_chain_S22.out;
  byte-check ep2+round-1+s1 = 30aea66e; bar adj <= -0.5% vs S21+s1 rows CI excl 0; parity vs stack/loopR1_physicist_rows.json). S21jn stopped at its
  VALID40 start (partial rows archived *_STOPPED_2210_for_S22.*) and relaunched behind S22 (log fire_chain_S21jn_run2.out).
- 2026-09-24 22:24 S22 (22:10 run) DIED at the smoke assertion: prebind wall guard under load (setup 3.004 s, 284/362 bound). RELAUNCHED 22:25 as
  fire_chain_S22s1_v2.sh (log fire_chain_S22_v2.out): smoke accepts a guarded stop (no restore, bound present, wall <= 3.1 s, 16384 finite outputs/net),
  census_boundary_setup_v2.py + run_census_porter_v3.sh record n_out/nonfinite, EXIT trap logs a death to TEAM_LOG; watcher prints ALERT on chain exit.
- 2026-09-24 22:40 S21jn VALID40 vs S21 PASS: adj -1.816% [-2.212, -1.414] 35/40, raw -0.026% [-0.429, +0.385], util -1.792% (lead: don't fire alone; composes).
  S21s1 GRADED 2.5332e-9 = BANK. S22 VALID40 vs S21+s1 PASS: adj -1.259% [-1.708, -0.816] 34/40 (raw -1.256%); residual running.
  S23 PREPARED (prebuild_S23.sh; record stack/prebuild_S23_record.txt): stack/stack_S23_mem.py 9247a212 = S21s1 recipe + loopR1 + round-1 + d6x + s1
  + s1=0 + Nystrom join; partials rebuild 30aea66e / a49da660 / 23c3abc2; ALL 28 BIT-EQUAL; einsum/validate PASS. In-process vs S21s1: r35 flops
  -4.239% MSE -1.145%; r100 flops -4.371% MSE +1.280%. fire_chain_S23.sh [BAR=-3.0] awaits the lead's GO.
- 2026-09-24 22:42 S23 LAUNCHED (lead GO, bar -3.0): fire_chain_S23.sh (log fire_chain_S23.out). Order: S22 residual + staging, engineer2 s1=0 VALID40 (VALID40_DONE in
  lrroute/run_after_S22.out), then S23; HOLD before its VALID40 if engineer2's valid40_e2_S21s1z_bar.json is not PASS (release: stack/S23_go.flag GO|ABORT).
  S21jn chain STOPPED (own residual gate skipped per lead). Watcher now a script file (scratchpad watch_chains.sh) so no command line holds a chain name.
- 2026-09-24 22:46 S22 STAGED fire_candS22/candS22_S21s1_r2_mem.py (33d4f0d0): VALID40 vs S21+s1 adj -1.259% [-1.708, -0.816] 34/40; residual PASS cold -0.003 s
  (in-chain analysis FAILED on a rename typo rz22 vs rz22s1 after the reps; recomputed from the census files -> ADDENDUM in FIRE_CHECKS). S23 relaunched
  22:46 with engineer2's queue pattern fixed (run_after_S2[2] matches run_after_S22_v2.sh).
- 2026-09-24 23:28 S23 STAGED 23:14 and FIRED by the lead #332246 (S22 fired #332241): VALID40 vs S21s1 adj -5.897% [-6.495, -5.290] 40/40, raw -1.645%, util
  -4.315%; residual PASS cold -0.009 s (requests -2,051). Attribution (same base): s1=0 -2.911% (engineer2), loopR1 -1.259% (S22), join -1.816% (vs S21).
  prebuild_S23struct.sh (PT-46) builds any S23-structure candidate; reproduces S23 9247a212 byte-exactly (for covhunter's round-3 S24, file-gated).
- 2026-09-24 23:29 S24 STANDS DOWN: loopR2 VALID16 -0.33% (bar -0.5%) FAILED; covhunter round 3 stopped. prebuild_S23struct.sh kept ready.
- 2026-09-25 00:20 S23 GRADED 2.3951e-9 (bank, 4th). S24 (round-3 dials) DOES NOT EXIST: covhunter's fit VALID16 +0.032%; launcher stood down itself 00:07.
- 2026-09-25 00:20 STACK LEVER (keep; rides with the next real change, lead 00:0x): STAR NYSTROM START + power step = stack/lrnys_fragment_q1.py appended
  after S23 (the star's range finder takes the 48 largest-norm columns of A instead of A Omega; the power step and rank-32 truncation stay). Torch VALID16
  raw +0.08% [-0.15, +0.31]; native VALID40 vs the S23 rows (stack/S23_lrnys_q1_TEST.py, paired_valid40_S23_lrnys_q1.json): adjusted -0.320%
  [-0.488, -0.156] 30/40, raw +0.137% [-0.035, +0.309] 18/40, util -0.455%; 0 errors. Lead's keep bar (adj <= -0.25%, raw <= +0.3%) MET. No power
  step = DEAD (+33.8% raw). Residual gate not run yet (adds ~8 small requests per star call, drops one product).
- 2026-09-25 00:20 JOIN SCHEDULE re-check (lead, with covhunter's torch side): prebuild_S23struct_sched.sh (env JOIN_SCHED = name or dict literal;
  every3 rebuilds S23 byte-exactly). even vs every3, in process: bill -0.07% r35 / -0.47% r100; native MSE -1.06% / +0.25%. 4-join pick pending.
- 2026-09-25 00:31 JOIN SCHEDULES KILLED (covhunter torch: even +0.57% raw [+0.04, +1.05] vs native bill -0.07/-0.47% -> adj ~+0.3%; four7 +1.18% raw,
  not built). every3 stays. Next S24 = next real change (E-71 group 1 / loopR3 / depth-routing) + star-Nystrom fragment appended last + residual gate.
- 2026-09-25 01:45 [engineer2] GROUP 5 SHELVED, READY (lead): E-71 group 5 body diet ready, -974 requests/net, bit-identical except signed zero (einsum merges: -0.0 -> +0.0; the 4-function prototype showed 0 flips). Not shipped: no residual purchase needs it (routing needs +33..55 ms; depth 7 closed). Prototype fable_engineer2_v1/diet5/proto/S23_g5e_mem.py f288a88a (einsum part only: gate PASS, bitwise parity 35/100, -264 requests). Pricing and full-build plan: TEAM_LOG 2026-09-25 01:44.
- 2026-09-25 01:40 S24 STAGED 01:37 + SUBMITTED by the lead 01:40 = fire_candS24/candS24_S23_loopR3_lvf2_lrnys_mem.py (30810da3) = S23 structure + loopR3
  params/dials + lvfuse2 + star-Nystrom (fire_chain_S24c.sh). VALID40 vs S23: adj -0.854% [-1.094, -0.618] 34/40, raw -0.402%, util -0.452%; residual PASS
  cold -0.004 s, requests -3.2k/net; setup 1.808 s. j3cut composes onto S24 (same _bd_update body).
