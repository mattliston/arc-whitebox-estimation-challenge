# [opus-roster] roster-alignment lever: priced, partial measured, verdict (2026-10-16, teammate fork)

Files: census_nz_v1.py (roster + product census), curve_v1.py (kernel bill curve), diff_census.py, cand_dc32.py (003d0706),
census_{s2tr,sel64,dc32}_7103256.json, curve_s2tr.json, wj/dc32_7103256.json (native warm_judge vs opus_solo_v1/wj/s2tr_7103256.json).

## 1. Kernel bill curve (production stack, in-process after setup; ratio = billed flops / 2mkn)
- ANY dim not a multiple of 32 -> plain dense matmul: ratio 1.000, 1 request (no tree path exists).
- 64-aligned dims: 0.448 (depth-7 colpack push) / 0.483-0.497 (depth-6 pushes, singles) / 0.507-0.514 (ks reads).
- odd multiples of 32: 0.457-0.502 via the half-zero strip kernels (+0.8-1% per element over aligned) BUT several classes fall to depth 5:
  reads m=992 or 864 x 6144 x 384 -> 0.535 (vs 0.507-0.509 aligned), m x 384 x 5920 with m odd-32 -> 0.537-0.540 (vs 0.513-0.514):
  +4-5% per element, i.e. 992 rows cost MORE than 1024 rows there.

## 2. Roster census (seed 7103256, S2tr): rosters per layer and their padding this net
- select (3/layer, 32-rounded): mean padding 12.7 rows (64-rounding would be 31.1).  _cp_read_young_sq / _rtt_read (32): 11.2 / 16.4.
- 64-rounded sites: _astra_bcc_deadcut_impl 42.2 (!), _dcv_roster 35.6, _push_bank_shared_newborn 34.1, _slr_reconstruct 38.0, _astra_bcc_recon_impl 29.0.
- Which product dims they feed (by in-situ census diffs): select rosters -> pushes' k and packet widths n (sums of 32-rounded cohorts), reads' m/k,
  singles; dead-cut-type rosters -> ONLY the pushes' m (14 _mm, 42.8 GF) and 15 old-tier m dims (10.9 GF).

## 3. Measured in-situ responses (native warm_judge, seeds 7103248 first / 7103256 warm)
- select 32 -> 64 (sel64, parent): bill +2.81% (+7.24 GF) for +18.4 rows/roster, requests -1,210, MSE +0.03%.
  Linear ceiling for EXACT select rosters (remove 12.7 rows avg): about -1.9% of the bill, if non-32 dims could run at the aligned tree rate.
- dead-cut-type 64 -> 32 (dc32, this lane): bill +0.17% first / +0.26% warm, requests -98, MSE -0.02%. Fewer rows, HIGHER bill: odd-32 m on
  the big pushes (n up to 10048 > hzg's 8192 cap; d7 dims 896/1024 only) falls to padded/depth-5 paths (+2..+5% per push). The cheapest
  partial is NEGATIVE. Exact dead-cut rosters ideal ceiling (35 rows of ~900 on 53.7 GF): about -0.7%.
- Total ideal ceiling (exact rosters priced at the aligned rate): about -2.6% of the bill. Not reachable with the present kernels.

## 4. Feasibility
- The kernels need m, k, n divisible by 2^depth (32 for depth 5, 64 for depth 6); zero strips are skipped only at 32-granularity (one tree
  level: 4 of 7 deepest chunks). A finer strip (16/8/exact) needs a multi-level strip kernel: compute views 2^t per strip per k-chunk,
  fills +1..+7, i.e. +4 (t=2) .. +12 (t=3) requests per strip and NEW view patterns per (shape, residue) that setup cannot prebind
  (lazy binds on cold public nets = the #336506 failure mode). Clean-build request headroom is ~580 (S5 median .3155 vs .32 gate). NOT feasible
  under zero-failure in the time left.
- Dense remainders (aligned main part through the trees + the leftover rows as a plain matmul, no new keys): pays only when the remainder
  is < ~15 rows (dense costs 2.2x a tree row); captures ~55% of the select ceiling on paper (~1.0% bill) at +2-3 requests per product
  (~+300-400/net), BUT exact roster sizes also reach the Smirnov kernel (17% of bill) whose view families are bound per shape at setup,
  and every elementwise path. New shape families on cold public nets = residual risk. Not buildable + verifiable (release gate, 6-net,
  store MSE) safely tonight.
- Small, safe-looking positives seen (each < 0.2%): bank width 5920 -> 5952 aligned costs LESS (-0.13..-0.5% on ~29 GF of products,
  ~0.1 GF); odd-32 reads that fall to depth 5 (~12 GF at +4-5%) would save ~0.4 GF if routed through hz. Together ~0.2% of the bill.

VERDICT: the roster-alignment tax is real (~2.6% of the bill ideal) but it is the price of the fixed-shape Winograd/prebind design; the only
safe partial (dc32) measured +0.26% bill. No >= 1% zero-MSE part from this lever without a new multi-level strip kernel.

## 5. STAGE 1 BUILD (lead assignment 22:0xZ): dense-remainder family on S2tr — MEASURED, STOPPED (not MSE-neutral, not request-neutral)
Files: fragment_rem1{,b,c,d,e,f}_v1.py.txt, cand_rem1c.py 86eef776, cand_rem1d.py b520f916, cand_rem1e.py a5d22940, cand_rem1f.py 69dc73d0,
cand_sm40off.py ffcf2809; store arms K/cand_decomp_rem1{,b,c,d}_v1.py (not run: single-net MSE already disqualifying); wj/*.json; census_rem1*.json.
Design as built: select() exact for the response and/or births selections (detected by the layer loop's response_score / _e2_gate_min objects, so
no line numbers; the live roster stays 32-rounded because it is the W5 port's previous roster and the pushes' k); _mm/_shared_old384_mm wrapped:
aligned part (dim - dim % 32) through today's kernels, leftover <= 14 rows/cols (10 inner) as one plain matmul joined into the exact output,
larger leftovers zero-padded to the next 32; W5 port calls with unaligned right columns zero-padded under the grant; Smirnov (SM40) reads with
exact shapes get the plan of the nearest dominating table key (the kernel pads to its own grid): sm40 calls 12 vs 10 on S2tr.
Native warm_judge, first 7103248 / warm 7103256, vs wj/s2tr:
| build | exact rosters | bill first | bill warm | requests first/warm | MSE 7103256 |
| rem1c | response+births, no SM40 fix | +1.22% | +0.41% | +1054 / +436 | +2.28% |
| sm40off (diagnostic, S2tr rosters) | none | +0.71% | +0.76% | -376 / -456 | +0.09% |
| rem1d | response+births + SM40 synthesis | +0.41% | -0.47% | +1197 / +517 | +2.39% |
| rem1e | response only (+SM40 synth) | +0.05% | +0.16% | +217 / +208 | +0.69% |
| rem1f | births only (+SM40 synth) | +0.32% | -0.44% | +1192 / +476 | +1.55% |
Reading: the saving is real but small (-0.44% warm, births/bank side; first-net +0.3..0.4% = lazy binds of the new aligned shapes), the requests
exceed the budget (+476 warm, +1192 first), and the MSE cost is large and ADDITIVE (response +0.69%, births +1.55%): the units that the
32-rounding adds (the best-scoring units just below the 0.01 floor, ~15 per roster) carry about 2.3% of final accuracy, while the next 16
(sel64) carry none. The padding is a tuned accuracy budget, not waste; exact rosters at the same floor are a floor raise (the birth-floor cliff).
Stage 2 (dead-cut rosters) is dead the same way: dc32 (64 -> 32) was bill +0.26%, and exact counts would drop marginal units too.
Stage 3 positives re-priced in situ: odd-32 reads already run hz in situ (class ratio 0.523 vs 0.519); 5920 -> 5952 is ~0.1 GF (0.04%). Not worth building.
