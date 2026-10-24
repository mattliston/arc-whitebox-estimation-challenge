# Physicist lane, 2026-10-01 (UTC). Newest first.

All paths relative to `P2/opus_reexam_20261001_v1/physicist/` unless they start with `P2/` or `O/`.
"Card" = the current torch research card (`pcard.py`: `s6bcard_loop5js` built as `train_segment.py` builds it, research-parent parameters and
dials, final-layer `k4_15` hook = identity). Parity: network 35 final-layer MSE 14.107e-9 (analyst: 14.107e-9), learned mean correction
off 16.417e-9 (analyst A6: 16.42e-9). "Kink block" of layer L = response rows x alive columns of the card at L, off-diagonal.
Truths: organizers' 1e9 post means (final error), covhunter 32M all-layer mean/covariance, 8M planes, and this lane's 16M mechanism
Monte Carlo `t2mc_rank0035.npz` (two independent 8M blocks; block 0 = the samples of the 8M plane files).

## STATE OF THE LANE (2026-10-02 16:40Z; update this block first)

- Board: S40 = S39 + the pair-carried law graded 1.7590e-9 (#333559, 3rd; J2W 1.5995e-9, marius_binner 1.6407e-9). S41 = S40 + the lead's K31L4
  (fourth-chaos part of the young pairs' local quartic, ages 1-2) + the trainer's first dials is being fired (expected about 1.71e-9).
- What this lane established today (T27): the remaining variance error is plane error; the plane error is ONE object, the network-specific
  fourth cumulant beyond its second chaos (the law) and the young local quartics (K4L, K31L4, F4); it is made about evenly by all earlier
  layers; the first-order theory (births + transport + conversion of the third cumulant) explains 96-97% of the true K31 plane and 99.9% of
  the true K21 plane; every exact piece beyond what is shipped costs a second push of the bank columns (27% of bill). Closed on cost.
- Closed today with numbers (T27): per-unit k4 companions; the true K31 row coefficient (hurts); the b-form (the truth planes ask for 1.4-7.8%
  more, the loop loses); a band split of bscale; K21 truth as a target (its 8M noise is several points).
- Open and free: K31L4 strengths by cohort age (ages 1-3 beat 1-2 by 1.2 / 0.9 points on 35 / 100; the trainer solves them); the C^2 form of
  the law without the matrix C^2 (`t4_run_ora46.sh`, running: (C^2)_ij = (v_i + v_j) C_ij + (Co^2)_ij, first part elementwise); a split of the
  K21 knob by row class (`t4_run_ora44.sh`, running).
- Instruments built today (all in this directory): truth-plane oracle in the loop (e16b options tk21 / tk31 / tk22 / tkrows / tkm / tkdraw /
  tkdump / tkb / tks21, frozen s3o); `t22_planes.py`, `t23_k31l_vs_truth.py` (card planes against truth planes); `t24_restart_planes.py` +
  `t25_restart_tab.py` (Gaussian-restart Monte Carlo: age of plane content); `t26_conversion_mc.py` + `t27_conversion_tab.py` (first-order
  theory term by term); `t28_age_profile.py`; `t29_k21_forms.py`, `t29b_bform.py`.
- How variants are judged: never on networks 35 / 100 / 37 / 39 alone. Held-out ranks of `O/trainer/cert/runB_val.txt` (first six:
  651 378 424 328 287 641; 1-24 for tuning; 25-48 as the check set).
- Machine rules: one torch process of <= 4 threads; CPU gate = bash O/idle_gate.sh 30; nothing at all inside a quiet window, numpy included
  (every launch of mine now checks the window first; at 14:23Z I ran a 30-second numpy job one minute after a window had opened, reported
  to the lead 15:36Z); GPU through O/gpu_request (the truth producer was stopped during my two GPU jobs 15:38-16:14Z). Never SIGCONT a
  stopped process. Time stamps in TEAM_LOG.md come from `date -u`, not from memory (one line of mine was stamped 9 minutes ahead).

## T145-T148 (10-13 15:4xZ-): co-adaptation screens null; c0 failed out of sample; the lead's bigger bets and pe3 running

- **T145** (the lead's co-adaptation arms on top of C, 8 networks): (a) K4L age-2 x0.75 = +0.004%; (b) x1.25 slope +0.045% (se 0.163), so no slope. The lead stopped (c)/(d) (radial k4) before they ran; the unrun requests are archived in t145_coadapt/archived_unrun_20261013T160325Z.
- **T146 FAILED out of sample.** c0 +0.05 / +0.10 at L8-14 gave -0.204% each on 8. At 24, +0.10 = +0.044% (OOS +0.168%) and +0.15 = +0.376%. The bowl goes the wrong way: +0.15 vs +0.10 = +0.33% (se 0.08). c0 hurts L11-14 by +0.2-0.3%.
  - The S56 gradient (t -4.74) overstated the gain by about 0.5% per +0.10.
  - Likely cause: gradient runs treat the carrier modules as layer constants (otr_m_v7 REPLAY), so a dial that feeds the carriers' state gets a biased gradient. Screen by evals.
- **T147** (lead 17:0xZ, pre-declared 17:10:15Z), on top of C.
  - **(h) K31L4 age-4 cohort at 0.6 PASSED at 24:** -0.157% (OOS -0.115%; screen 8 -0.242%, se 0.098). Natively full-layer age 4 netted about 0 on 10-12/13 (rounding |d|^2 plus +0.12% bill); the L11-14 band (-0.05% at n 3) is with scout-native.
  - **(i) Redistribution null:** -0.046%. Age 4 pays only ON TOP of age 3.
  - **(j)/(k) law dfit 4 / 3 errored on every request.** The cutsim law clone n6y_cutsim_v2 supports dfit 2 only, and decide crashed on the errored arms, which blocked (h)'s extension for 70 minutes. Fixed; the lesson is in memory chain-generation-pitfalls #10.
- **T148** (Matt's call): pe3 -0.2 / -0.1 at L8-11 FAILED at 24: +0.36% / +0.03%, OOS +0.66% / +0.18%. The screen-8 gain (-0.27% for -0.1) did not hold.
- **T150:** the dfit re-run with law_response_fit false and a matched dfit 2 control, `t150_*`.
- **T151:** the carrier-dial scan on FRESH 7100608-623, confirmed on 592-607 (nfs +/-0.15 at L7-14, nfd +/-0.15, K31L4 x1.2). The law's and K31L4's leaves get no gradient in RAD runs and have not moved since S48-S50.
- **T152:** the pbase flips so2m 1.0 and x half, same protocol, in T151's queue. Smoke requests sit at the head of T150's queue.
- **Pattern today:** three gradient-derived or screen-8 gains (c0, pe3, and (h)'s screen size) shrank or vanished out of sample. 7100584-591 has hosted about a dozen screens, so screens now run on fresh networks (lead 18:5xZ).

## T143-T146 (10-13 12:0xZ-): per-unit k4 sources closed; age caps closed; co-adaptation to C read off the RAD gradients (c0 found)

**C fired.** s54k1 (#335782) carries C and graded 1.52120e-9, -0.498% vs #335709. The stage-2 baseline carry is now k4la3 = k4la4 = -0.5 at L8-15.

**Closed (TEAM_LOG):**
- T143: what C still misses in per-unit k4. Leave-one-network-out energy explained: ages >= 5 3.8%, joined hubs 2.4%, (Cpre^2)_ii 0.5%, all three 5.3%. Bar 20%: stand down.
- T144 (age cap, my half): ages 5-6 at 0.5 where unjoined, on top of C. -0.028% mean log (se 0.096); the control matched C bit for bit. Scout's S1 / S2 were also null, so the cap stays at 4.

**Gradient read of the co-adaptation (read-only, 15:54Z).** The trainer's S55 and S56 RAD gradients share the same 64 networks (7100264-327), so S56 minus S55 is a paired measurement of what C (+ gk + k4lf34h) changed. First-order d log(final MSE) = pl_grads / (scale x mse).
- The lead's (a) K4L age-2 x0.75: +0.015% (se 0.025); before C it was +0.074% (t 3).
- The lead's (c) radial k4 x0.9 at L8-15: -0.51% (se 0.58). Per-network responses span -11% .. +11%, so 8 or 24 networks cannot resolve (c)/(d).
- **c0, the K31 row factor (pair map), has no RAD route.** Summed over L8-14 it went from -0.0114 per unit (t -1.3) to -0.0493 (t -4.74). This is T146: c0 +0.05 / +0.10 at L8-14 on top of C.
- k31 points the same way (r 0.98).
- Two other route gaps: k4la1 at L15 (t +4.0, but only 0.2% per unit) and pe3 at L8-11 (t +4.6).
- Tools: the T145 / T146 makers and tables, and `t145_chain_v2.sh` (one queue, auto-extension).

## T136-T142 (10-13 03:2xZ-11:0xZ): K4L ages 3-4 at INNER layers = -0.513% (C, n 24); gated ds / gated beta / F4 remainder / ages 5-6 closed

**T139 / T141 / T142: the result of the day.** K4L hub content from pairs aged 3-4, at strength 0.5. The fired k4lf34h had it at the final layer only.
- T139 screens, on top of k4lf34h at the S51 + gk point, 7100584-591: A (L12-14 + final) -0.290% mean log; B (L8-11 + final) -0.176%.
- T141, 24 networks (mine; scout-card's duplicate run was stopped before any eval): A -0.218%, B -0.293%, and C (L8-15) -0.513% (se 0.117), 19 of 24 lower. The out-of-selection 16 give -0.535%. The parts add up. C goes into the s54k1 stack.
- T142 (strength 0.75 / 1.0, plus L4-7, final 0.75): all null vs C. C sits at the optimum.
- The trainer's worker v16 reproduces my T141 C record bit for bit.

**Closed today (logged in TEAM_LOG):**
- T136: kink-gated ds; per-network gradient vs kink |r| <= 0.12 at n 64.
- T137: signed gated beta, -0.040% at n 24.
- T138: F4's nonlinear remainder N is 2-3% of L in the K21 slice, under the 5% bar.
- T140: K4L ages 5-6 at the final layer, +0.058%.

**Stage-2 loops:**
- v10 crashed at S54's STARTED line, a latent format bug, before any chain ran. It is fixed in v12 and v14.
- S53 was cancelled as moot.
- v14 / v15 carry the FIRED per-layer K4L form into baselines from S56 (`t127_baseline_carry.json`).
- S54's stage 2 started 11:04Z on worker v15.

## T134-T135 (10-12 23:4xZ - 10-13 03:2xZ): refit base v2 screen (close); learned gain grows with kink; stage-2 loops v10/v11

**T134 refit-base screen.** Learned mean correction OFF (request pl ds = -1 - stored), S51 point + gk, A41 7100584-591, each arm paired with base_off. Run dir `t134_refit_screen`, table `t134_table.py`.
- so2m: -0.677% pooled (mean log -0.625%, se 0.766), 4 of 8 lower.
- so2t: -0.134% (se 0.372), 5 of 8.
- Neither reaches the lead's >= 1% learned-off bar. Recommendation: CLOSE the refit base.
- Learned ON vs OFF (the control): -11.87% final (sd 3.15, range -15.9..-8.4). It grows with the kink fraction (kink 8-11 r -0.62, kink 12-15 r -0.56, n = 8), so the learned correction helps MORE on kink-rich networks.

**T135 / stage-2 loops v10-v11.** The judge's worker is chosen per round: launcher chain -> new families (pe3 / pe4 -> v14, k21k -> v15) -> capped by `t127_certified_workers.json` (12-15 since 00:08Z).
- v13 inside the judge reproduced the S51 stage 2 bit for bit (`t135_v13_judge_smoke.py`).
- Long shot #2 (per-column law transport): closed on derivation. nlaw's gbar averages over the SUMMED unit, which W spreads democratically.

## T127-T134 (10-12 08:08-22:10Z): stage-2 loops armed (S52-S66); late capacity closed (floors, ss8); S51 sealed -0.051%; theory hunt after gk

**T127 stage-2 loops.** `t127_stage2_loop_v6.py` (S52-S63) and `v7` (S64-S66; it waits on v6) have run since 18:31Z with numpy side work at BLAS 1.
- Per round: a judge re-pinned from the v14 template (`t85_pinned_judge_S<N>.py`); start on the verdict FILES only; 32 of 32 COMPLETE before any table.
- Baseline = the frozen-point request of the LAST FIRED round (`O/fire/<tag>` without 'twin' and with a receipt.json); fallback = the round's own start point.
- Each round posts one fixed line: `S<N> STAGE 2 TABLE: sealed <x>% pooled (se), <k> of 32 lower ...`.
- S53+ requests carry gk (pbase gsk / gk4) through config patch.ncar. The engineer's pbase_cutsim_v1.py already reads them, so the pinned worker v12 needs no change. Worker v13 = v12 numerics (weights regenerated in memory).

**T126 S51 sealed stage 2:** -0.051% pooled (se 0.108), 20 of 32 lower. Measured with judge v14 on sealed A42 7100680-711 against the S50 frozen point. Pooled with the trainer's check: -0.162% (se 0.081, 40 of 64).

**T128 idea sprint:** no candidate reaches -1%. Networks 16 / 35 against the 32M per-unit truths.
- L13-15 per-unit excess kurtosis is real structure (true sd 0.014-0.017), but the card carries one constant per layer.
- Cheap predictors explain R^2 <= 0.27 in sample and about 0 across networks.
- L15 per-unit variance error has no predictable structure (R^2 about 0 across networks).

**T129 late birth floor raise: CLOSED.** S50 point, 7100136-139, against legacy truths.
- Results: 0.02 at L8-14 +4.93%; 0.03 at L8-14 +13.75%; 0.05 at L8-14 +37.66%; 0.03 at L11-14 +3.98%. All 4 networks are worse in every arm.
- The bill falls only 1.0-1.6%. The young bank sits at a cliff.

**T131 star rank 8 at L11-14: CLOSED.** +0.150% pooled (se 0.272, 2 of 4). The control reproduced the base within 6.0e-5.

**T132 top-mode variance dial at L13-15 (v_i += d lam1 u1_i^2): CLOSED (10-13 00:31Z).** 16 networks, each paired with its T129 base.
- d = +0.02: final +130.3%, 0 of 16 lower. d = -0.02: final +124.2%, 0 of 16 lower.
- Finite differences put the best d at -0.0002, predicted -0.018%: a steep bowl centred at the card's own variance.
- Controls: d = 0 vs the smoke (d 1e-9) 1.8e-6; the environment offset vs the 16:3xZ base 6.7e-5. The t133d smoke stop came from its too-strict 1e-6 tolerance; t133e reran the remaining phases.

**T130 disk:** physicist/ = 49.7 GiB. Weight caches are 32.1 GiB and regenerable (about 19 GiB are duplicates); npz dumps are 17.4 GiB. Nothing was deleted (`t130_disk_inventory.tsv`).

**T134 theory hunt (lead 21:5xZ; code reading only).** These places still use the Gaussian Phi(a) or phi/sigma next to the corrected mean / k3 / k4. All arms run on top of gk; the exact edit strings went to scout-card, who tests on 1 worker.
- (1) lq: the n6y law computes its own Gaussian g and d (n6y_cutsim_v2.py:186), giving its per-layer carry gb = mean(g - m d), V = 2m(1-g), S2, rv and D. This is the only remaining site that compounds over ages.
- (2) gbd: b's radial source with the Edgeworth kink density dq.
- (3) gbeta alone.
  - CORRECTED 23:11Z: the trainer's card cuts `_bd_update`'s j3 (`for j in range(3)`), so gkpB's gbp did not duplicate a fitted direction. That direction's original fable coefficients were noise (-0.65 .. +0.53) and the trainer cut it.
  - beta only re-splits a fixed layer K21 total between the newborn pair and the star, so its best strength is empirical.
- (4) gbu: `_bd_update` with dq and gq.
- (5) gom: the omitted-radial aa and beta.
- Below the top 5: Sq (the F4/K4L/K31L birth data S1-S4) and bk4 (local k4 only, since aux['pre_k4'] includes radial v^2).
- Already consistent: the pair map (its linear e3/e4 terms equal gq), the star, trace_post.

## T121-T126 (10-12 00:41-05:35Z): rdc scan closed; pair-split young read closed; S50 sealed stage 2 PASSES (-0.379%)

**T122 rdc scan** (S48 card in torch, trainer's worker v11 pinned, 16 networks): vs rdc 2.0: 2.2 -0.463% (se 0.28, 9 of 16), no cut -0.340%, 2.4 = no cut bit
for bit, 1.8 +0.83%; engineer's native bills 2.2 +0.74% / 2.4 +1.16% / 1.8 -0.41% -> 2.2 scores +0.28%: CLOSED. **T123 pair-split read** (lead's idea: young pairs
p+- = c +- h split exactly, K21 = (L+ + L-) lam h' + (L+ - L-) lam c'; `t123_worker_pairsplit.py`, controls bit for bit, pair formula 2.4e-8): A384 (oracle basis, c in
the right factor only, L8-14) +0.247% on 16 (se 0.21); B384 (c compressed everywhere) +13.5% (0 of 15): c's per-unit values matter in the left operands / state;
A128 +18.4% (0 of 12; c energy 0.982-0.995): no r-128 basis can pass. T124 pushed basis (engineer's scheme): v1 control failed (F4C in-place shifts of the youngest
pairs), v2 (shifts projected) passes; R32 +1.44% on 4 (dropped). Native psrN128 (engineer): bill only -1.64%, error +91% as built: CLOSED. **T121 S50 sealed
stage 2** (judge v13 on A40 7100552-583; 4 seeds cut by the release memory guard at 04:50 / 04:55Z re-served after archiving each cut attempt): S50 frozen point vs the
MS start: -0.379% pooled (se 0.109, -3.46 SE, 25 of 32 lower), all-layer -0.081%; pooled with the trainer's check 64 networks -0.233% (se 0.077): fire rule met.
**T126** S51 stage-2 trigger armed on the verdict FILES (judge v14 re-pinned to worker v12, sealed A42 7100680-711).

## T118-T120 (10-11 20:45-23:01Z): the rank cut at the joins costs 2.15%; it is the pairs' differences h, which are high-rank: join-basis family CLOSED

**SUMMARY (lead 23:0xZ: join family closed for good).** (1) Join TIMING sits at a cliff between absorption ages 4 and 5 (lead's count; the fired 7/10/13
absorbs at 5-7): absorbing at age 4 (6/9/12) +10.67%; 7/10/- -0.23% for +0.65% bill (native 2 networks: -1.1 / -1.3% for +1.4 / +1.6%); 8/11/14 -1.31% for
+4.18%: the fired schedule is at the optimum. (2) The rank CUT costs 2.15% (exact old tier, 15 of 15). (3) That loss is the +- pairs' differences
h = (gap/2) M e_a (89-98% of the first-order third-cumulant loss), one pushed unit vector per born unit: high-rank, so no fixed-rank basis holds it.
(4) Re-aimed bases are null (pairk3 -0.15%; native per-column +0.39 / +0.41%; 10-10 exact eigenvectors -0.80%); carries cost a young push or equal rank;
rank trades ~1:1 with bill. Memory: p2-1011-join-schedule-bet.

Plain torch card (S47e / Nd learned parts fixed, one-direction star, no cut simulator), fired joins 7/10/13, 15 networks 7100136-150 vs legacy truths, paired
with T115's base; `t116_table_v2.py`. **T118 exact old tier** (`t118_worker_fullrank.py`: every join keeps the full live space; positive control: kept rank =
live units at every join): final-layer MSE **-2.151%** (mean log -2.167%, se 0.362, 15 of 15 lower), all-layer -1.00%. Absolute gain by layer: 12% of the final
gain by L9 (the L7 join), the L10 join adds -6.9 / -7.0 / -5.9e-11 at L10 / 11 / 12 (75% by L12): the dropped content costs at every later layer.
NB the joins compress the BANK's columns (third-cumulant CP columns behind the K21 read and the final k3), not the covariance (carried in full);
'captured energy' = the joined columns' lambda^2-weighted energy kept (fired L7 / L10 0.99970, L13 0.99996).
**Carries priced (no runs)**: an exact read of the dropped part, a per-unit diagonal carry, a residual basis all need the dropped residual pushed (= a young
layer of the joined columns, ~5% of bill) or equal a rank increase: none is cheaper than rank (rank trades ~1:1, T36).
**T120 pair-aware sketch** (`t120_worker_pairsketch.py`; controls: copy of the card's join push bit for bit = base; the pair transform with the card's Gram
within 2.8e-4 per layer = rounding): the bank's newborn columns come in +- pairs p+- = s + x+- e_a (weights +-lam), c = (p+ + p-)/2, h = (p+ - p-)/2; a pair's
third cumulant 2 lam (cch + chc + hcc + hhh). Dump (`t120_dump_table.py`, 2 networks, card's basis): |h|/|c| median 0.30 / 0.18 / 0.14 at L7 / L10 / L13;
kept lam^2-energy c 0.9998-0.99999, h 0.9972 / 0.9929 / 0.9988; first-order third-cumulant loss in h: 89% / 96-97% / 98%. pairk3 (sketch on [c, h] with
those loss weights): **-0.147%** (mean log -0.172%, se 0.352, 8 of 15; per network -2.7..+1.8%): null. Under its basis the first-order loss falls only
10-25% (h kept 0.9929 -> 0.9942 at L10): h is one pushed unit vector per born unit (2,784 at L10 in 877 dims), intrinsically high-rank. Engineer's native
per-column weights lam^2 |p|^4 / |p|^3: +0.39 / +0.41%. **T113 trimmed** (cut simulator, 16 networks): births floor 0.005 / 0.003 at all layers -0.77 / -0.99%
(se 0.48 / 0.49) for +5-8% / +7-14% births; final-layer row cut z 3.0 +0.07% (neutral), z 2.5 +3.4%.

## T111-T116 (10-11 18:35-19:40Z): K21 error is spread; k4 not a lever; EARLY joins +10.7%; join schedule = the lead's big bet (T116 queued)

**T111 K21 error by carrier** (`t111_age_error_s47e.py`; S47e, network 35, L12, 8M truth planes, covariance-weighted): the card holds 99.68% of the truth's
energy; per-age coefficients 0.989 / 0.988 / 0.985 at ages 1-3 falling to 0.977 at age 11; a joint per-age fit removes 11.3% of the error energy (young
ages 0.3% of it): the error is spread, no carrier scale to fit. **T114 k4** (`t114_k4_probe.py`; net 35, L12-14): weighted as the mean formula uses k4 the
card / truth scale is 1.020 / 1.033: not a lever.
**T115 join schedule on the plain torch card** (`t115_worker_joins.py`: the trainer's worker v4 through runpy, PINNED otr_m, star v2 ss -1, no cut simulator,
s2card.EVERY3 replaced per process; Nd learned parts fixed; seeds 7100136-150 vs legacy truths; `t116_table_v2.py`): EARLY joins 6/9/12 vs the fired
7/10/13 on 13 paired networks: final-layer MSE +10.67% pooled (mean log +10.15%, se 0.77), higher on 13 of 13, all-layer +6.42%; per layer L6 +1.8,
L7 +3.5, L9 +6.4, L10 +9.4, L12 +11.6, L13 +12.3, L15 +10.7%: the error enters at each join. Young column-layers a network 110,191 -> 93,922 (-14.8%;
engineer's bill model -4.51%): CLOSED. Engineer's prices of the alternatives (19:37Z): 7/10/- (no third join) +0.65%, 7/10/14 +0.82%, 7/11/14 +1.04%,
8/11/14 +4.18% bill. **T116 (19:49-20:16Z, CLOSED by the lead 20:2xZ)** (`t116e_chain.sh`, worker v2 `t116_worker_joins_v2.py` = v1 + the card's
own join log), 15 networks vs 7/10/13: 7/10/- (no third join) -0.232% (mean log -0.207%, se 0.129, 10 of 15 lower; L13 -0.09 / L14 -0.11 / L15 -0.23%)
for +12.8% young column-layers (+0.65% bill): net loss. 7/11/- (7:0-2, 11:3-6) +1.125% (se 0.53, 4 of 15). 8/11/14 (triples) -1.305% (se 0.35,
12 of 15; L12 -1.50%) for +4.18% bill. Join energy capture at the kept rank (fraction of the joined covariance's trace): fired L7 / L10 0.99970,
one layer later 0.99990, L14 0.99999+: each layer of delay cuts the lost trace about 3x. The fired schedule sits just past the cliff (early +10.7%).
Parity pair for the engineer: `t115_nolast` + `t115_basekf` (7100136-137, final_prediction; base re-run bit-identical).
**T113 trimmed** (births floor add 0.005 / 0.003 at all layers, L15 row cut z 3.0 / 2.5; `t113_floors_l15/`): 2 networks per arm so far; resumes after T116.
**RAD-S48 stage 2**: `t117_stage_rads48.py` copies the trainer's two check requests (id / rank / truth_plan removed), lists the differing fields and runs the
v11 judge's prepare (dry test PASS: `t117_DRYTEST_unused_run`); chain `t112_chain.sh`, table `t112_table.py`.

## T101-T110 (10-11 14:55-18:32Z): module audit at S47e; star rank and r_u found; young-bank floors at a knee; S48T sealed look; star pair-closure killed

All on S47e = the trainer's frozen Nd point on the v11 cut simulator, 16 fresh seeds 7100136-151 (fit set), final-layer error vs the fired point (t93 arm A),
pooled / se of mean log / networks lower; workers on the PINNED otr_m copy `otr_frozen_bba0dea0` after the trainer's 15:30Z live edit broke 283 runs.

**Module audit (whole network, off)** (`t101_build.py`, `t101_table.py`, `t101_audit_whole/`): pbase bsk +0.05%, F4C exact S2 +0.40%, kappa_C +1.08%, b6 +1.37%, edge
+1.45%, K31L4 +1.66%, K4L +2.72%, head x0.5 +4.2%, F4 rule +4.4%, head columns +6.1%, covariance knobs +6.4%, law K31 +7.4%, law fold +20%, K31 row factor +29%,
b source +46%, K22 off-diagonal +2500%. With the engineer's bills (17:01Z / 15:16Z) no drop pays. Bands (`t101_audit_b2/`): nothing qualitatively new.
**Star** (`t102_worker_star_v2.py`, `t104_ssrank/`): star off +260%; ss 2 / 4 / 8 / 16 / 32 = -0.23 / -0.30 / -0.43 / -0.49 / -0.49%; exact -0.62%. Native prices
(engineer): ss 8 +0.60%, 16 +1.18%, 32 +2.37% bill: none pays; the one-direction star ss -1 (+0.035% bill) gave natively -0.26% (se 0.11, 11/16) -> in S48.
**r_u** (`t105_ru/`): 272 -0.39, 288 -0.60, 320 -0.89 (se 0.26, 13/16), 384 = 512 = -0.89%; + ss 8 -1.31%, + ss 16 -1.37% (additive). Native: bill +0.96..+1.34%,
net +0.10% on the gate seeds: OUT.
**Young-bank floors** (`t103_worker_floors_v2.py`, `t103_floors/`): births / response-row floor 0.03 at 7-10 +10.2 / +10.8%, at 11-14 +3.9 / +6.8%; 0.10 at 11-14
+49 / +81% (13-40% fewer rows): steep knee. **Other cuts** (`t106_cuts2/`): l15 row cut exact 0.00%, push dead cut exact +0.02%: not levers.
**Sealed look** (`t108_worker_ss1.py`, `t109_sealed200_s48t/`, `t109_table.py`): S48T = MS point + ss -1 vs S47e on SEALED 7100200-231: -0.260% (SE 0.130, -2.0 SE),
mean log -0.282% (se 0.131), 18 of 32 lower, all-layer -0.149%. Parity of ss -1 on 7100136: torch vs native change correlation 0.84 (base-state gap 3.7e-6;
the engineer confirmed identical forms).
**Star pair-closure gate** (lead's idea; `t110_worker_stardump.py`, `t110_gate.py`, networks 35 / 100 vs 8M-sample planes): the star's dropped triple-distinct
hub part, transported and pushed two layers, is 10-12% of the card's K21 error at l+2 (l = 10-12) and removes only 0.1-1.2% of its energy: KILLED.

## T98-T100 (10-11 11:06-11:12Z): research gates #1 (scale mode in the covariance channel) and #2 ((3,1) per-row model) BOTH FAIL

**#1** (lead GO): S47 / Nd on the v11 simulator, 7100136-151, the pair map's covariance output recomputed with K21_C = K21 + kappa_C (v mu' + 2 C o mu 1'),
kappa_C = max(gamma radial / 6 - b_pre . mu / mu . mu, 0); mean, target, radial from the card's own call (`t98_worker_k21c.py`, `t98_k21c_gate/`).
Certificate gamma 0 = the fired card bit for bit. gamma 0.5 at 10-14: +0.20% (se 0.22, 3 of 16 lower, all-layer +0.06%); gamma 1 at 10-14: +287% (4 networks);
gamma 1 at 2-14: +2303% (2). **Why** (`t99_worker_pindump_v2.py` dumps of the pair map's K21 input at 10-14 on 35 / 100 vs the 8M-sample true planes,
`t100_k21_decomp.py`, weight (pdf_a g_b / s_a)^2): the scale-mode form explains 80-87% of the TRUE plane, and the card carries 97-99% of the true
scale-mode coefficient (theta 0.0095-0.0112 vs truth 0.0096-0.0115); the Oct 2 figure 0.68 predates the law / bfix / NCAR modules and the refits. The
K21c oracle's headroom is plane-SPECIFIC content: fitted on 35 and judged on 100, the scale-mode direction removes 10-18% of the weighted K21 error at 13-14,
a global factor 7-15%, per-row / per-column factors about 1 point more.

**#2** (offline, `t99_k31_gate.py`): the card's (3,1) plane misses 47-53% of the true energy at 10-14; per-row models fitted on 35 remove 12-17% of the
error on 100 (6-8% weighted by pdf_i g_j), nearly all of it one global factor = the trainer's existing k31 dial (Nd: dk31 +0.96). Closed.

## T95-T96 (10-11 08:00-08:42Z): Ne stage 2 staged; the per-layer predictability gate at layers 12-14 FAILS (all-layer-loss refit closed)

**T95 staging**: `t85_pinned_judge_v8.py` = v7 except the judge block = A30 plan_v2's sealed final 7100200-231 (sha b35e8a1c, 16M, 32/32 ready), no
two-look protocol / scorer pins (numbers by `t92_table.py`); dry prepare + ready PASS (`t95_v8_DRYTEST_unused`, never run); chain `t95_chain.sh`. Baseline =
Nd's frozen request (`t92_Nd_candidate_request_v1.json`); candidate + decision receipt = Ne's freeze.

**T96 (lead 08:10Z, P2)**: S47 physics (the trainer's frozen Nd point, v11 simulator); dump worker `t96_worker_dumpl.py` (= worker_generated_v4 + read-only
capture of the head's inputs at 12-14; replica's correction = card's bit for bit; final MSE = t93 arm A); A27 truths (fit 7100136-151 at 64M, held-out 152-167 at
32M; halves A / B -> truth noise). Target (truth - card mean) / sigma per layer; sklearn HistGradientBoosting, iterations / shrink by 5-fold CV over the fit
networks; % of the per-layer residual power removed on the held-out networks (noise-free in brackets). `t96_ceiling.py` -> `t96_out.txt`.

| layer | held-out MSE | truth noise | head's 128 inputs | + top-mode shares, b_pre / sigma | + 30 raw features |
|---|---|---|---|---|---|
| 12 | 1.26e-8 | 23% | 0.04% (0.06%) | -0.12% (-0.16%) | -0.23% (-0.30%) |
| 13 | 1.36e-8 | 20% | 0.05% (0.06%) | 0.04% (0.05%) | -0.08% (-0.10%) |
| 14 | 1.49e-8 | 16% | 0.02% (0.02%) | 0.03% (0.04%) | -0.05% (-0.06%) |

**Verdict** (gate 5%): closed. As at layer 15 (T84), what each layer's correction leaves is not predictable from the information that layer holds; per-layer
supervision cannot find it. The remaining error needs new moment information (covariance / plane content).

## T94 (10-11 07:42-07:53Z): fewer tail rows cut: NONE PAYS (thresholds at their optimum both ways)

Lead 07:50Z: the other direction of T93. Same setup (S47 / Nd point, v11 simulator, 7100136-151, paired vs the fired thresholds; fired arm reused from
`t93_tailscan`, same request and worker). Table `t94_table.py t93_tailscan t94_tailscan_fewer`; row counts in each run's `cut_counts.jsonl`.

| setting | final change | se | lower | all-layer | rows cut per layer 7-14 (mean) | bill by the audit's whole-tail prices |
|---|---|---|---|---|---|---|
| old dead tail -2.5 -> -3.0 | -0.16% | 0.21 | 10 / 16 | -0.14% | old tail 164-264 -> 116-240 | about +0.5% |
| young dead tail -2.5 -> -3.0 | -0.50% | 0.27 | 9 / 16 | -0.35% | young tail 36-60 -> 0-16 | about +1.4% |
| old positive tail 2.5 -> 3.0 | -0.10% | 0.25 | 8 / 16 | -0.10% | old tail -> 116-228 | about +0.6% |

Error gains are 3-6x below the bill they cost; with T93 the thresholds sit at their optimum (rows just beyond are cheap in error when cut, rows just inside
are expensive). The three un-cuts sum to -0.76% of the -1.06% all-tails-exact ceiling (T90).

## T93 (10-11 07:28-07:41Z): more aggressive dead-tail cut thresholds: BOTH LOSE

Lead 07:30Z: re-optimise the dead-tail thresholds on 16 seeds. S47 physics = the trainer's frozen Nd point request on the v11 cut simulator
(`t93_worker_tailscan.py` = worker_generated_v4 + the simulator's own per-layer row-cut log -> `t93_tailscan/cut_counts.jsonl`), fresh seeds 7100136-151,
final-layer error paired vs the fired thresholds (neg = ynt = 2.5). A dead-tail row is cut only if it is also not alive (gate < .01, rounded to 32), so any
threshold above about -2.33 means 'every non-alive row'.

| setting | networks | final change | se | lower | all-layer | rows cut per layer 7-14 (mean) |
|---|---|---|---|---|---|---|
| old dead tail neg 2.0 | 16 | +0.49% | 0.21 | 5 / 16 | +0.27% | old tail 164-264 -> 172-272 |
| young dead tail ynt 2.0 | 16 | +1.49% | 0.45 | 2 / 16 | +1.27% | young tail 36-60 -> 60-84 |

ynt 1.5 = ynt 2.0 bit for bit; neg 1.5 cuts as many rows as neg 2.0 but not the same ones (the 64-row rounding is filled in score order), which moves a
network by about 0.5%. Against the audit's bill of the whole tails (young 1.65%, old 2.9%) the extra rows save at most about 0.6% / 0.1%: neither passes the
lead's rule (bill cut > error increase + 0.2%). The fired thresholds stand.

## T92 (10-11 06:22-07:25Z): Nd stage 2 on the sealed final 32 (lead's 06:06Z pre-registered rule)

Pinned judge `t85_pinned_judge_v7.py` (purpose judge, manifest `t92_judge_Nd_v1/manifest.json`, identity 829ae949): baseline = the S46 v11 request of the T85
bridge, candidate = the trainer's own frozen-point request a_0mNdC_pt (jmNd_p1.pt 4f066ac5 + hu 6a2d7817) with only truth_plan removed
(`t92_Nd_candidate_request_v1.json`), decision receipt trainer/cert/Nd/FROZEN_POINT_v1.json, A27 plan 71f455d1 (16M truths), protocol a80bbcec, scorer score_v2
fa65d1e3; two chains of 16 seeds (`t92_chain.sh`, 3 threads, release guard), 32/32 lanes, no errors. Tables: `t92_table.py`, `t92_score_early_v1/report.json`.

| reading | networks | final change | SE of per-network % | lower | all-layer |
|---|---|---|---|---|---|
| stage 1 (trainer, 7100232-263) | 32 | -0.482% | 0.205 (-2.35 SE) | 19 / 32 | -0.218% |
| stage 2 (sealed 7100168-199) | 32 | -0.155% | 0.172 (-0.90 SE) | 19 / 32 | -0.123% |
| pool | 64 | -0.323% | 0.135 (-2.40 SE) | 38 / 64 | -0.171% |

score_v2 early: pooled ratio 0.99845, paired bootstrap 99.5% upper 1.0028 (its accuracy_pass false). The lead's rule (pool < -2 SE and stage 2 negative) puts Nd in
the SAFE file. Stage 2 alone is the unbiased size (stage 1 was the gate): about -0.15..-0.3% expected on the board.

## T90-T91 (10-11 03:55-06:21Z): Astra's T48 repair heads and the native tail estimate's scale: BOTH CLOSED

Lead 03:5xZ: screen the T48 repair heads (HU31/32 = old / young tail repair / (v s) at layers 7-14) before any native or refit time; bar -0.4% held-out.
What the columns are (O/engineer/a22_cutsim/torch_rowcuts_v15.py 88-134): at layers 7-14 the native card zeroes the pushed old-tier columns of the tail
rows (alpha > 2.5, or alpha < -2.5 and dead; count rounded up to 64s) and the young columns of the dead tail (alpha < -2.5), and adds back a mean-field
estimate of those rows' k3 diagonal ('repair'); the columns are that estimate, nonzero only on the cut rows. Card: S46 on the certified v11 cut simulator
(trainer/worker_generated_v4.py with the pinned S46 v11 request of the T85 bridge; all 2539 pinned sources unchanged since the bridge), eval, 3 threads;
final-layer error against the fired cut, paired per network; fresh seeds 7100136-151 (64M truths). Runs: `t90_repair_ceiling/` (arms by rowcuts options),
`t91_scale16/` (`t91_worker_rscale.py` = worker_generated_v4 + per-layer repair scales; certificate: its uniform x1.25 = the global option x1.25 bit for bit
on 7100137). Tables `t90_table.py`, `t91_table.py`.

| arm | networks | final change | se (mean log) | lower | all-layer |
|---|---|---|---|---|---|
| tails exact (no cut, no estimate) | 16 | -1.08% | 0.36 | 13 / 16 | -0.62% |
| cut kept, estimate removed | 5 | +5.90% | 1.52 | 0 / 5 | +3.07% |
| estimate x1.25 | 16 | +0.94% | 0.37 | 6 / 16 | +0.46% |
| estimate x1.5 | 16 | +2.64% | 0.70 | 3 / 16 | +1.39% |

The first 4 seeds read x1.25 at -0.20% (3 of 4): unrepresentative. The parabola through x1 / x1.25 / x1.5 puts the best common scale at 0.82 (-0.18% in-sample,
noise); per-network best scales scatter from -4.9 to 1.6. **Verdict:** the estimate's scale is at its optimum and no common scale exists; the head columns carry
only this estimate (a head corrects means at 7-14, while the skewness error the estimate leaves travels on through births), so they cannot plausibly clear
-0.4%; closed without the 100-run least-squares screen. Computing the tails exactly is worth -1.06% but costs the tails' bill (engineer's audit: old-tier tails
+2.9%, young dead tail +1.65%): does not pay as is.

## T40 (10-10 21:59-23:24Z): the ten masked layer-15 inputs of the learned correction: CLOSED

Lead 20:49Z: the final layer's correction masks inputs 13-15 (pre-correlation row mean / rms / mean |rho|), 22-25 (K21 plane row / column norms and sums over v_i s_j),
26 (rho @ alpha / n) and 27-28 (post-correlation row rms / mean), because the card computes only the diagonal of C_pre15. Card: jmIv (trainer label a_0mIJv = S46),
torch replay on frozen d61b46d1 (`t80_shadow.py`). Inputs formed at layer 15 as the step forms them at layers 1-14: C_pre15 = W15' C_post14 W15 in full; the K21 plane at 15 =
the card's own bank read (response rows x live columns, its diagonal) + the full star + the b-form; the post covariance = the plain pair map on (mu, C_pre15, K21).
Arm `dump15` saves the head's first-layer pre-activation, weights, the ten inputs, host, final prediction, truth (`l15j_dump/`); with nothing opened it is the card exactly
(534: 13.85646e-9). Pricing is exact offline (`t83_l15_unmask.py`): delta recomputed from the dump to 4.6e-6, fixed-parts groups match the direct replay
(`t80_unmask.jsonl`, 12 rows) to 4.4e-4 of the error. Networks: 24 trainer networks (training) + the 32 untouched fresh seeds 7100104-7100135 (held-out, 64M-sample truths,
noise about 1.15e-9 per network, paired). Final-layer error change vs the card, pooled; se of the mean log ratio over networks.

| | inputs | change | se | lower |
|---|---|---|---|---|
| mask opened, existing weights (56 networks) | 13-15, 26 | +1.60% | 0.20 | 7 / 56 |
| | 13-15 | +0.70% | 0.11 | 10 / 56 |
| | 26 | +0.47% | 0.15 | 20 / 56 |
| | 13, 24, 25, 26 (natively free) | +0.58% | 0.14 | 20 / 56 |
| | 22-25 | +5.74% | 0.32 | 1 / 56 |
| | 27-28 | +0.02% | 0.02 | 27 / 56 |
| | all ten | +7.49% | 0.39 | 0 / 56 |
| new layer-15 weights from 0, held-out 32 | best single (14) | -0.017% | 0.011 | 20 / 32 |
| | 13-15 + 26 | -0.013% | 0.028 | 17 / 32 |
| | 13, 24, 25, 26 | +0.354% (in-sample -1.01%) | 0.155 | 12 / 32 |
| | all ten | +0.144% (in-sample -1.13%) | 0.116 | 15 / 32 |
| | 22-25, standardised per network | -0.099% | 0.129 | 19 / 32 |
| gradient-boosting ceiling, held-out 32 | the head's own inputs | 0.000% | 0.044 | 17 / 32 |
| | + the ten inputs | -0.012% | 0.034 | 18 / 32 |

Weights: Gauss-Newton with weight decay chosen by 5-fold cross-validation over the training networks, step on the training networks (`t83_U_a_0mIJv.npz`); ceiling:
sklearn HistGradientBoosting on (truth - final) / sigma, iterations and shrink by 5-fold cross-validation over training networks (`t84_l15_ceiling.py`).
**Verdict:** the residual the layer-15 correction leaves is not predictable from layer-15 information (its own inputs or the ten masked ones); opening the mask with
the existing weights costs +0.6..+7.5%. The engineer's U15a (mask opened natively) stays off. The rivals' final-layer edge (final 0.95 at layers 0-14 1.0) is not here.

## T39 (2026-10-10 20:50-21:55Z). S46 on my untouched seeds; the stale-constant audit

**S46 (jmIv) against S45 (jmH) on my assigned seeds 7100104-7100135** (32, never used by anyone; `t79_fresh_judge.py s46` through the chain `t4_run_t79_s46.sh`): pooled
-0.66% (truth noise out -0.72%), mean log -0.67% (se 0.15), 25 of 32 lower. The judging-only 32 said -0.53%, the trainer's 48 -0.74%: S46 holds at about -0.7%.

**Stale constants** (lead: every table / constant fitted once on an older card). Inventory: almost every physics strength of the card is a per-layer dial of the trainer's
otr_pl (F4C f4a / x1a / x2a, K4L k4la, the law n6b / n6r / n6d / nfs / nfd, K31L4 nkl, the gn3 knobs iso / k21n / k31 / edge / rc, c0 / c1, bscale, bsrc, s_carry / s_new,
k4 bands, kappa_C, ds, e2); his zero-dial screen at jmG (13:29Z) put all at noise except X1 / bscale / c0 / k31, which jmH moved. The law's per-layer literal RHO1 (fitted 10-02
on 651 / 35) rides on nfd per layer (noise). The head's feature normalisation (frozen centers / scales) is not stale in effect: no feature clipped on more than 1% of units at
any layer (12 S42 adumps), and enc0 absorbs affine drift. Without a dial, finite differences at jmIv, fixed parts, 11 sz16 networks (`t80_shadow.py` arms knob / ncar,
explicit base values reproduce the card exactly): b-fix gamma 0.1 -> 0.05 +0.88% (2 of 11), -> 0.15 -0.21% (se 0.34, 8 of 11); the law's ca 0.75 -> 0.6 +0.41%, -> 0.9 +1.40%;
the law's removed forms 3 -> 2 +0.32%, 4 = 3. Nothing stale beyond about 0.2%. Not probed: the births' gap floor / u clamp (hard-coded) and the star's Nystrom rules.

## T38 (2026-10-10 20:10-20:50Z). The bold bet (lead): a cheaper carrier for the young bank (53% of bill). NO-GO, measured

Harness `t80_shadow.py` (the trainer's request a_0mIJv = jmIv replayed on his module d61b46d1; new arms: joinsched = any join schedule of the cohorts into the old tier,
certified (the card's own schedule given explicitly = the card on 534); readsvd / readeig = the young read's plane (response rows x live columns) projected on k column
directions at chosen layers). Bill proxy `t82_joins_table.py` (any schedule; census rates; rough). Learned parts fixed.
(a) EARLIER JOINS: every 2 layers from 4 (ranks 384) +47.5% error / about -19% bill (534); every 3 from 5 +40.6% / -19%; every 2 from 3 breaks the young-pair modules
(K4L / K31L4 assert their pairs aged 1-3 are young); cohorts 9-11 joined at 14 +3.6% (0 of 4) / about -2.9% bill.
(b) READ ON FEWER COLUMN DIRECTIONS at layers 8-14 (4 networks): top-256 / top-512 eigenvectors of C +131% / +10.5% (plane energy kept 97.3% / 99.8%); the plane's own
best rank-256 / rank-512 SVD +36% / +2.0% (99.2% / 99.98% kept).
Reading: the accuracy lives in the young bank's exact content and in the plane's last fraction of a percent of energy; the card is hypersensitive to any approximation of
either (like every capacity cut, T36). No-go for the 10-12 12:00Z decision.

## T37 (2026-10-10 13:30-14:00Z). Clean fresh-network judgement of the trainer's torch points: S45's base against S44 on 32 untouched generated networks

**Tool** (`t79_fresh_judge.py <tag> <labels> <seeds|parity>`, table `t79b_table.py`): each candidate is one of the trainer's own requests (run_M/claimed/<label>_r*.json),
replayed exactly as his worker applies it on his current card module (frozen copy `otr_frozen_d61b46d1/otr_m.py` = O/trainer/otr_m.py of 13:15Z), on the analyst's
GENERATED networks (O/analyst/fresh/loader.py: seed -> weights; 64,094,208-sample truths, two halves; final-layer truth noise about 1.15e-9, taken out in the table),
which no refit batch, sizing, check or judging set has touched (seeds 7100060-7100091; the record shows no other use). Parity on the trainer's stored results:
a_0mGJn (jmG without b7) rank 1 to 7e-8; a_0mCJq (S44's torch point, jmC without qold) to 4e-5 (his number was made with an older module).
**S45 base against S44** (jmG without b7 against jmC without qold): 16 seeds -0.32% pooled (-0.35% noise out), mean log -0.30% (se 0.29), 9 of 16 lower; all 32 seeds
-0.39% pooled (-0.43% noise out), mean log -0.40% (se 0.20), 21 of 32 lower. The trainer's 48 clean judging ranks: -0.69% (se 0.16). Reading: within 1.2 se; the untouched
estimate of S45's refit gain is about -0.4%.
**jmH** (the trainer's request a_0mHJ, params jmH_p2.pt; parity on his rank 1 to 2e-5), same 32 seeds: against S44's torch point -1.33% pooled (-1.45% noise out), mean log
-1.35% (se 0.31), 25 of 32 lower; against jmG without b7 -0.94% pooled (-1.02% noise out), mean log -0.95% (se 0.35), 23 of 32 lower. The trainer's 48 judging ranks:
-1.75% / -1.07%; the untouched reading keeps 76% / 88% of them (within about 1 se): jmH holds.
**jmI** (request a_0mIJ, params jmI_p2.pt, parity on rank 1 exact): FIRST reading on the 12 never-used seeds 7100092-7100103 (table tag jmI): against jmH -0.45% pooled
(-0.49% noise out), mean log -0.45% (se 0.24), 9 of 12 lower; SECOND reading on the 32 seeds 7100060-7100091 (judging only, never fitted): against jmH -0.61% (mean log -0.60%,
se 0.17, 22 of 32), against S44's torch point -1.93% (se 0.35, 27 of 32). The trainer's 48: -0.82% vs jmH: jmI holds (55-75% of it).
**jmIf / jmIv** (jmI's native forms: v = X1 / bscale / s_carry back to jmH's values, f = also iso back; parity 1e-6 / exact), same seeds, against jmH: first reading
jmIf -0.14% (se 0.23, 7 of 12), jmIv -0.35% (se 0.26, 9 of 12); second reading jmIf -0.36% (se 0.19, 18 of 32), jmIv -0.53% (se 0.20, 22 of 32). The natively free jmIf keeps
55-60% of jmI's untouched gain, jmIv 80-85% (the iso knob). From 17:44Z my untouched judging uses seeds 7100168 onward (lead's assignment; the trainer's sizing 7100104-7100167).

## T36 (2026-10-10 10:40-12:20Z). Bill-neutral reallocation of the bank's capacity (lead): early capacity is the most valuable; the old tier's ranks trade 1:1; no bill-neutral gain

**Card**: the trainer's CURRENT torch card at jmG's start point (`otr_frozen_482e996f/otr_m.py` = his otr_m.py of 10:16Z; request a_0mG1_base: params jmD_p1.pt, b6 / b7 at zero,
k31ly ages 1-3, pbase bsk 0.34), replayed as his worker applies a request (`t76_realloc.py`; parity on rank 5: 12.477094e-9 = his). Knobs: the floors of the card's three
selections per layer (live units, the read's response rows, births; 0.01 each, applied by scaling the score handed to the card's select), the old tier's ranks at the joins of
layers 7 / 10 / 13 (384 / 320 / 384) and the sketch size that caps them (covstep.TIER_RANK 384). Counts per layer (rank 5): births 1024 at layers 0-2 falling to 512 at 14;
columns pushed 2,048 (L1) -> 23,040 (L15). **Bill proxy** (`t77_realloc_table.py`): the engineer's S40 census of 651 (push 43%, young read 36% of bill) fitted per layer on
live_prev x young columns, response rows x young x alive, and old-tier rank (8,100 flops per live_prev x rank unit per layer, the join's sketch prorated by rank); 25%
residuals: ROUGH. Networks: cert/refit/sz16_ranks.txt (16; 4-12 used).

| arm | final-layer MSE (pooled) | networks lower | bill proxy |
|---|---|---|---|
| births floor 0.05 at L1-5 + 0.001 at L8-11 | +17.2% (se 3.0) | 0 of 5 | -1.4% |
| births floor 0.05 at L1-5 alone | +18.0% (se 2.7) | 0 of 6 | -3.3% |
| births floor 0.001 at L8-11 alone | -0.05% (se 0.37) | 3 of 6 | +3.4% |
| response rows 0.05 at L1-5 + 0.003 at L8-12 | +11.6% (se 3.9) | 0 of 4 | -0.1% |
| response rows 0.003 at L8-12 alone | -0.12% (se 0.65) | 3 of 6 | +1.4% |
| ranks 256 / 384 / 448 (L7 cut) | +4.5% (se 2.9) | 1 of 4 | (448 capped) |
| ranks 384 / 384 / 384 (L10 320 -> 384) | **-1.26% (se 0.31)** | **11 of 12** | +0.9% |
| ranks 384 / 320 / 320 (L13 cut) | +1.02% (se 0.30) | 2 of 12 | -0.9% |
| ranks 384 / 384 / 320 (64 moved L13 -> L10) | -0.19% (se 0.37) | 5 of 12 | +0.0% |
| ranks 384 / 384 / 256 | +5.5% (se 0.80) | 0 of 12 | -0.8% |
| ranks 384 / 448 / 448, cap 512 | -1.98% (se 0.47) | 10 of 12 | +2.6% |
| ranks 384 / 512 / 512, cap 512 | -2.32% (se 0.47) | 12 of 12 | +4.4% |
| ranks 448 / 448 / 448, cap 512 | -2.08% (se 0.46) | 11 of 12 | +3.5% |

(ranks above 384 without the cap raised are exactly the 384 arm: the sketch caps them; the cap alone changes nothing.) Reading: the early births are the most valuable capacity
(the premise 'errors born early are damped' does not make early capacity cheap: the bank's early cohorts carry the plane content that later layers read); births or response
rows added at 8-12 buy nothing; the old tier's rank at the layer-10 and layer-13 joins trades about 1:1 with bill (64 rank units ~ 1.0-1.3% error ~ 0.9% bill by the proxy).
The one candidate: the L10 join's rank back to 384 (September's R320 cut undone), -1.26% error, net about -0.4% if the proxy holds; the engineer prices it natively.
**Native prices (engineer 11:45Z, on jmD, network 651)**: L10 rank 384 +1.675% bill (+739 requests), 448 +3.38%, 512 +4.98%: about +1.65% per 64 rank units, twice my
proxy; so L10 384 (-1.26%) is about +0.4% worse on score. **L13 only 448** (cap 512; lead 12:2xZ), +0.62% bill natively: -0.36% (se 0.12, 11 of 12 lower): net about +0.26% worse.
Above 384 the return falls (384 -> 320 at L13 +1.02%, 384 -> 448 -0.36%). LINE CLOSED (lead).
**More early births** (lead 12:2xZ, same 12 networks): floor 0.005 at L1-5 -0.43% (se 0.21, 10 of 12) for about +0.66% bill (proxy; native about 2x); 0.002 at L1-5 -0.50% (se 0.23,
10 of 12) for +1.26%; 0.002 at L1-3 only +0.06% (5 of 12); cohorts 0-2 joining at layer 8 instead of 7 -0.13% (7 of 12) for +2.8%; 0.002 at L1-5 + join at 8 -0.65% (10 of 12) for
+4.0%. The cut is steep, the addition flat: at the 0.01 floor layers 1-2 already bear all 1,024 units; the births a lower floor adds at 3-5 have |alpha| > 2.33 and carry little.
**Old-tier sketch quality at fixed rank** (lead 12:5xZ; the join = energy-ordered column sketch of M = J diag(w) J' + Qp core Qp', QR, truncation in QR order;
re-implemented in `t76_realloc.py` below the modules' wrappers, certified bit-identical on 534): Rayleigh-Ritz truncation at L10 -0.59% (se 0.32, 9 of 12); 512-column
sketch + Rayleigh-Ritz at all joins -0.79% (9 of 12); one power iteration at L10 / all -0.46% / -0.67% (10 of 12 each); exact top eigenvectors at L10 / all -0.64% / -0.80%
(the ceiling of any basis at fixed rank). Extra products (my count): Rayleigh-Ritz at L10 ~1.9 Gflop (~0.7% of bill), a power iteration ~9 Gflop (~3.4%), exact ~9%.
CAPACITY FAMILY CLOSED.
**Final-layer skew, old-tier part** (`t80_shadow.py` arm k3old15, jmIv's point, 6 sz16 networks; the L15 k3 product is 3.6% of bill, about 1.6% its old-tier
reconstruction): dropping the old tier's cubic part of k3 at layer 15 (44% of its norm) +167% (0 of 6 lower), at half strength +41%. **|lam|-weighted old-tier sketch**
(`t80_shadow.py` arm sketch:w<p>, join re-implemented in the shadow harness, certified bit-identical at jmIv on 534; 12 sz16 networks, learned parts fixed): |lam| at all
joins -0.17% (se 0.17, 7 of 12), |lam|^1.5 -0.08%, |lam|^3 +0.15%, |lam| at L10 only -0.09%: nothing (lead's bar -0.4%). PHYSICS SHOTS CLOSED.
**Plane oracle on the current card** (`t78_plane_oracle.py`, jmD point, learned on, 35 / 100 with 8M truth planes; off-diagonal planes set to card + eps (truth - card)):
K21 replaced everywhere at 13-14 +188% / +256% (the card's transport breaks); K21 in the map's COVARIANCE output only: 13-14 eps 0.5 -2.7% / -4.1%, eps 1 -0.6% / -4.1%;
10-14 eps 0.5 -10.8% / -11.7%; K31 at 13-14 -2.4% / -2.1%; K22 at 13-14 +1.0% / +1.6%; K21 (covariance output) + K31 + K22 at 13-14: eps 0.5 -6.3% / -6.4%, eps 1 -7.1% /
-7.4%; at 10-14 eps 0.5 -19.5% / -17.5%. The ceiling is real; its cheap part is not identified (the card holds 99.7% of the true K21 energy; the rest is unit-specific; the
capacity knobs that buy plane accuracy trade ~1:1 with bill).

## T35 (2026-10-10 05:30-10:05Z). The adjoint screen: exact zero-weight gradients of every candidate column on the S42 card, partial to the head's inputs; batch 7

**Instrument** (`t63_adjoint_dump.py`): the S42 torch card in the trainer's gradient mode (backward of 1e8 x final-layer MSE against the 1e9 truth) with a zero per-unit bias leaf on
the head's first layer at every layer 1-15; G_l = dL/d(bias) (1,024 x 128) gives the exact zero-weight gradient of any column x as sum_l x_l' G_l. Certified on every network:
sum_l z_l' G_l = the trainer's own HEADU gradient to 5e-7..9e-7, sum_l G_l' H_l = the enc0_w gradient to 6e-7..1e-6; final MSE = his (rank 17: 13.2936e-9). 36 of his gradient
ranks (chain at 2 threads, lowest priority, 05:33-06:51Z). Rank-one share of G per layer 0.41-0.71: the 128 weights matter.
**Screen** (`t64_adjoint_screen.py`, tables `t64_screen_36.txt` (top 40) and `t64_screen_36_full.txt` (all)): per candidate the trainer's statistics over the 36 networks (weights of
128 beyond 2.5 se, chance 2.2 with 35 dof, his rule >= 6.6; mean t^2; halves cosine; first-order change at step 0.05; cosine to column 6's remaining gradient) for the RAW
column and for its PARTIAL part (residual after least squares on the head's 126 inputs + column 6 + constant, per network and layer). Positive control: the module's own
columns reproduce his tables (col 17 mean t^2 30.89, col 9 13.36, col 16 5.51, col 6 3.86).
PARTIAL, layers 1-14 (all-units convention unless noted): fold 102 / 27.7 / -0.455%; qold 90 / 17.9 / -0.33%; f4 92 / 17.2 / -0.34%; f3 84 / 9.9 / -0.29%; q4 76 / 10.7 / -0.26%;
f2 (response rows) 77 / 11.3 / -0.20%; e56 77 / 11.0 / -0.18%; e7 76 / 9.4 / -0.19%; eold 69 / 9.0 / -0.15%; kmu 109 / 48.4 / -0.083%; kg 109 / 41.7 / -0.058%; qk 105 / 41.6 /
-0.073%; qb 104 / 36.5 / -0.063%; e34 40 / 4.4 / -0.14%; s1 37 / 4.1 / -0.14%; mld6 (6th mode loading) 36 / 4.5 / -0.15%. NO signal (mean t^2 0.9-2.7): m2, gk, gb, k3d2, rg,
the head's correction pushed (pdel) and its change of the covariance pushed (hc), the other mode loadings and shares, t3, alpha, m1k4. Every passing column's mean gradient has
|cosine| 0.86-0.94 to column 6's remaining gradient. Not covered: hub membership counts (the births' unit lists are not in the dumps).
**Batch 7** (`b7cols.py`, handed to the trainer 10:00Z for jmF): fold, kmu, kg, qk, e34, s1 (raw; standardise over all units). Certificate (`t75_b7cert.py`, network 17): equal to the
adjoint dump's values to 8e-15, the card's final MSE unchanged (read-only).

## T34 (2026-10-10 05:05-05:35Z, final week). The post-map change that overshoots is the head's mean correction moved into C (dial kappa_C); the scale-mode consistency is refuted per unit; the lead's best-bet question

**Split of the post-map change** (`t71_v15_dump.py` version 3 with the learned edge captured, `t72_v15_regress.py`, table `t72_v15_12nets_v3.txt`, 12 networks with paired truths).
In the base step (model_bank_bfix.Card.step) the post covariance after the pair map receives the learned EDGE (layers 8-14, `_message_edge_source`) and, after the head,
C = C - host d' - d host' - d d' (the second moment kept, the mean changed); the b-update and the star do not touch C. The dumped remainder (after minus edge) correlates
0.97 with -(host d' + d host' + d d') pushed (ratio 0.86-0.88). Regression of the next layer's per-unit variance error on each piece pushed through W:
head's change into C: theta +0.95 at L15 / +0.81 at L14 (12 of 12 signs), R^2 4.3 / 4.8% raw, 1.1 / 1.4% after the head's inputs; edge: theta +0.26 / +0.24 (9 of 12), R^2 0.2-0.4%.
**Closed-loop arm** (`t74_kappa_c.py`, the dial kappa_C = the fraction of that change kept, layers 1-14, learned parts as fitted, six of the trainer's line-search ranks
22 685 476 367 261 397; log `t74_kappa_c.log`): kappa 0.5: 0.0% pooled (se 0.40, 3 of 6 lower; per network -1.5 / +0.5 / -0.5 / -0.1 / +1.3 / +0.6%); kappa 0: +1.9% (1 of 6 lower).
The learned parts are fitted around kappa = 1: handed to the trainer as a jmB arm (only a refit can release it).

**Scale-mode consistency (lead's option a) refuted per unit** (`t73_channel_interaction.py`, table `t73_channel_interaction.txt`, the 7 T60 networks with paired truths,
layers 6-15, kept rows): the scale mode's shapes hold 0-4% of dv / v ((1 + alpha^2)), 1-11% of dk3 / (v sigma) (6 alpha) and 3-5% of dk4 / v^2 (3 (1 + alpha^2)^2), with fitted
amplitudes near zero; the variance, skew and kurtosis channels compensate only mildly in the mean (cross terms -0.02..-0.10 of the energy of the sum each; sum of energies /
energy of the sum 1.13-1.22). A20's joint-greater-than-sum at L15 is a loop effect, not a broken per-unit cancellation.
**Per-network scaling of the final head's correction** (75 layer-15 dumps): the per-network optimal extra scale has mean +0.07, sd 0.27, and buys 0.23% of the final MSE
in-sample (max 2.4% on one network); with a second per-network number (the raw mean's shape) 0.9%: network-level adaptation of the head is not a lever.

**Answer to the lead's best-bet question (05:32Z):** none of (a) scale-mode consistency (<= 1%), (b) K31 conversions at layers 10-12 only (T27: the missing K31 at depth is
two thirds the transported full fourth-cumulant tensor; the conversions are 20-30% of it and made evenly by all layers, so layers 10-12 bring about a quarter: about -0.5..-1%
for +5-6% bill), (c) the planes at 8-14 (ceiling -8..-24%, but no carrier cheaper than the second push, +27% for -2..-3%) reaches 3% on today's data; (d) the adjoint screen
(partial zero-weight gradients at layers 1-14 for jmB) is the work with a positive expected value (0.5-1.5% with kappa_C and column 17), running.

## T33 (2026-10-10 04:45-05:05Z, final week). V15: the per-unit variance error at layer 15 is carried off-diagonal covariance error; no term of layer 14's pair map predicts it

Lead's task V15 (04:40Z): the analyst's A20 prices the true per-unit variance at layer 15 alone at -9.7% of the final error; what predicts the card's dv_15?
**Dump** (`t71_v15_dump.py`, S42 torch card, eval mode): a wrapper on the pair map (`_nonlinear_pair_batch`) at layers 13 and 14 recomputes its post covariance from its own
inputs, split into the terms the card computes (Gaussian Mehler part r = 1..6, K21 term, K22 term, K31 term from t4 rho, the modules' remainder31 (zero here), per-unit skew
and kurtosis terms) and terms it drops (Gaussian orders 7-12, the rho^3 continuation of the K21 and skew / kurtosis terms, second-order T21 o T21' F3 G3 / 4 and rho t3 t3 F4 G4 / 36),
each times s_i s_j off the diagonal, plus the post variances on the diagonal and 'after' = the handed-on C minus the map's (b-update / edge / star). Each pushed as
diag(W' X W) into the next layer. Certificates: the computed terms add up to the map's covariance to 7-8e-4 of its largest element (float32 card); the pushed handed-on
C = the card's v_15 to 3e-7, v_14 to 2e-6. **Regression** (`t72_v15_regress.py`, table `t72_v15_12nets.txt`; 12 networks with paired per-unit truths, 2 x 32M A13b halves or
the g19 pair for 100): target dv / v (kept rows at 15), truth noise removed with the halves (3% of dv after the baseline). dv / v rms 1.3-1.5e-3 at layers 14 and 15.

- Diagonal against off-diagonal: the card's post-variance errors of the previous layer pushed through W o W (exact per unit from the true mc_post_var) are 0-6% of the dv energy
  on every network; the off-diagonal part is 97-103%, at both layers.
- Pair-map terms, shares of dv (raw / after the head's 126 inputs): 'after' 3.5% / 0.6% (theta +0.67, 12 of 12 signs: two thirds of the post-map change points the wrong way),
  K31 term 1.0% (theta -0.29), skew terms 0.7%, Gaussian part 0.4%, K21 term 0.4%, K22 under 0.1%; JOINT, leave one network out, 4.1% / 0.7% -> implied final-error gain
  -0.40% / -0.07%. Layer 14 from 13: the same (off-diagonal 98-102%, 'after' 3.7%, joint 4.8% / 1.5%).
- Dropped terms are 100-1000 x smaller than dv: Gaussian orders 7-12 3e-8 of v, rho^3 continuations 2-4e-6, second-order products 5e-7 .. 7e-6.

So the variance error at layer 15 is the off-diagonal covariance error carried in through C_pre14 (and injected by the planes' errors at 14), without a per-unit or low-rank
signature (T69: top post-covariance modes 4%, column 17's numerator 2%). As a head input V15 is worth at most -0.4%; the -9.7% needs a more exact off-diagonal covariance at
layers <= 14 (the plane / carrier line).

## T32 (2026-10-10 03:00-04:20Z, final week). The final layer's own correction fitted OFFLINE (lead 03:05Z / 03:22Z): column 17 in the kept-rows convention pays -0.42% held out; the candidate set's joint ceiling

**Joint ceiling of the screened candidates** (`t65_joint_cv.py`, numpy, the eight T60 dumps with per-unit truths; table `t65_joint_cv.txt`). All 48 candidates together (ridge,
leave one network out, baseline = the head's 30 features + 96 messages + column 6 + its 64 state numbers, truth noise removed by the two draws) predict on a held-out network
2.2 / 1.3 / 1.2 / 0.7% of the card's per-unit VARIANCE-error effect at layers 5-8 / 9-11 / 12-14 / 15, 1.8-2.0% of the skew effect, 2.2-3.5% of the kurtosis effect, and 0%
of the post-mean error after the head (-0.4..+0.3%). So per-unit candidates of these families carry little that the head does not already see; an em screen cannot rank them.
Network-level common modes of the final error (per network: constant, gate x sigma, pdf x sigma, gate x mu, the prediction, variance and kurtosis shapes; 6 numbers fitted
in-sample) hold 1.7% of the final-layer MSE (39 networks): the final error is unit-specific.

**Layer-15 dumps** (`t66_l15_dump.py`, eval mode, S42 torch card = the trainer's request a_0hS41_u2_r0287 on the frozen otr_m f2dfccba; one process for many networks,
self-gated): the head's 126 inputs at 15, sigma, state, delta, raw mean, final prediction, the 1e9 truth, and raw candidates: c17 / c18 / c19 = module columns 17-19
recomputed on the KEPT rows (|mu| <= 3.5 sqrt(v) at 15; 17 = law diagonal dB = sum_r pp_ir (w_r pp_ir + x_r pp_i,partner(r)), residual on (v, 1, mu^2) over the kept rows,
over (sum v) v_i), f15 / qold / eold, per-age cohort sums, (W o W) pushes, top-8 modes of layer 14's post covariance. Order `t66_ranks_v2.txt`: 25 runB_train, runM_val50,
175 more runB_train. About 45-80 s a network (machine load).

**Offline fit** (`t67_l15_fit.py`): final prediction + kappa (delta(U) - delta(0)), kappa = 0.7844 (the card applies 0.784 x delta at layer 15, exactly: m - host =
0.784380 delta, residual 3e-12); delta recomputed from the dumped inputs = the card's to 4e-6 relative. Gauss-Newton weights with exact per-unit sensitivities g_i =
d delta_i / d h1_i (rank-one share of g only 0.64: the 128 weights matter), ridge by 5-fold cross-validation over the training networks, exact head along the step.
25 training / 27 held-out networks (`t67_both_25tr.txt`), held-out change of the pooled final-layer MSE:

| columns at layer 15 (own 128 weights each) | 5-fold cv (training) | held-out (27 of runM_val50) |
|---|---|---|
| c17 alone | -0.42% | -0.42% (se 0.07, 24 of 27 lower) |
| c17 + qold | -0.61% | -0.43% |
| c17 + qold + f56 + f3 | -0.68% | -0.50% (21 of 27) |
| c17 + qold + f56 + f3 + q56 + e44 | -0.91% | -0.57% (se 0.17, 22 of 27) |
| c17 + c18 + c19 | -0.42% | -0.49% (22 of 27) |
| qold alone / f3 / f56 / q56 | | -0.22 / -0.16 / -0.13 / -0.10% |

One-number physics terms added after the head (`t68_l15_physics.py`; the law diagonal as a direct variance change, old quartics as kurtosis changes, f15, e34, e44, the
card's own variance / skew / kurtosis terms rescaled): all within 0.25% held out, most nothing: column 17's information works only through the head.
Weights of column 17 for the trainer: `t67_U_17.npy` (1 x 128, step included).

**Judged on all 50** (`t67b_judge17.py`, the handed-over weights fixed): pooled final-layer MSE 1.4177e-8 (= the trainer's S42 number on the same 50) -> 1.4118e-8 = -0.418%
(mean log ratio -0.413%, se 0.071%, 41 of 50 lower). Parity file for the native build: `t67b_parity_c17.npz` (per network z17 on the kept rows, the change of the final
prediction, S42's and the new prediction). Dumps stopped at 75 (lead 04:15Z: the CPU goes to the trainer's joint refit).

**Error anatomy by layer** (`t70_layer_anatomy.py` on the T60 dumps, 7 networks with the analyst's paired 2 x 32M per-unit truths; `t69_l15_anatomy.py` at layer 15 on the
T66 dumps of 287 / 378 / 424 / 651). Each unit's post-mean error after the head split into first-order channels: carried mean = gate x (mu_card - mu_true), variance, skew,
kurtosis (truth noise removed with the two halves). The four TOGETHER explain 93-97% of the error energy at every layer 5-15 (coefficients ~1.0 / 0.95 / 0.8 / 0.75):

| layer | MSE | carried mean | variance | skew | kurtosis | joint |
|---|---|---|---|---|---|---|
| 6 | 3.3e-9 | 0.63 | 0.20 | 0.08 | 0.06 | 0.99 |
| 10 | 7.6e-9 | 0.70 | 0.13 | 0.06 | 0.05 | 0.96 |
| 13 | 1.22e-8 | 0.77 | 0.10 | 0.04 | 0.03 | 0.97 |
| 15 | 1.49e-8 | 0.78 | 0.09 | 0.04 | 0.04 | 0.97 |

(layers 2-4: the joint share is 0.2-0.5, the 1e9 truth's own noise is comparable to the error there). So the final error is 78% CARRIED mean error (the 'half' of T31 was
the share after the head's inputs are regressed out); the raw carry gain is 0.78 MSE_l / MSE_(l-1) = 0.85-0.91 per layer; the local injection is 22-25% of each layer's error,
and the per-unit VARIANCE error is about half of it. Final error = local_15 + 0.86 local_14 + 0.74 local_13 + 0.64 local_12 + 0.55 local_11 + ...: layers 8-15 all count;
a layer-15-only lever is capped at about 22% of the final error. The variance error at 15 has no simple structure: top mode of layer 14's post covariance 4%, column 17's
numerator 2%, old-cohort sums 1-2% of it.

## T31 (2026-10-10 02:12-02:40Z, final week). Open-loop screen of new head inputs on the S42 card against truth; batch 6 as one certified module; the closed-loop row-gain and all-learned-off runs of 10-02

**Card and dumps.** `t60_s42_dump.py` = the trainer's S42 torch card exactly as he judged it (his request `O/trainer/run_M/claimed/a_0hS41_u2_r0287.w0.json`: first dials
`c3q4zd6_optimum.pt`, f4c 1, ss 0, hf 2, law module n6y with the table, k31ly ages 1 / 2 / 3 at 0 / 1 / 0, head column 6 at the step-2 weights; frozen card module
`otr_frozen_f2dfccba/otr_m.py`, head weights padded with zero rows to its 21 columns), run by his engine with read-only hooks. Parity: network 287 final-layer MSE 14.568104e-9 =
his stored 14.568104e-9. Dumps `s42dump_rank0{035,100,150,200,250,300,400,500}.npz` (the first version without batch 6: `*_v1.npz`): per layer the pair map's mu / v / k3 / k4,
the head's 30 normalised features, its 96 message columns (formed as the head forms them), the module's 21 columns, the head state in / out, delta, the post mean, and the
candidates (per-age local quartic q, fourth-chaos diagonal f, k3 c, |lam| energy e and signed energy s of the bank's cohorts; row-gain inputs; (W o W) pushes of the previous
layer's post v / k3 / k4 / kink density / delta^2; K21 . mu, K21 . g; the batch-6 columns through `b6cols.py`).

**Screen** (`t61_input_screen.py`; tables `t61_screen_base.txt`, `t61_screen_state.txt`, `t61_screen_state_resp.txt`, `t61_screen_b6_*.txt`). Targets per unit and layer 5-15: the
first-order effect on the layer's own mean of the card's per-unit variance / skew / kurtosis error against two independent per-unit truths (32M + 32M for 35 / 100; 8M + 4M for
the others, k3 and k4 from one draw there), and the post-mean error after the head (1e9 truth). Candidate and target are regressed on the head's 30 features + 96 messages +
column 6 (+ the 64 state numbers); partial R^2 with the truth noise removed by the two draws; the sign agreement over networks x layers. Calibration: column 6 itself, out of the
baseline, reads 0.016-0.022 on the variance error with 24/24 signs (and it paid -2.6% on the board).

| candidate | error | layers 5-8 | 9-11 | 12-14 | 15 |
|---|---|---|---|---|---|
| f15 = m2 at layer 15 (batch 6.3) | kurtosis | | | | 0.031 (8/8) |
| m2 = t3^2 He4 pdf / 72 (batch 6.1) | kurtosis | 0.012 (31/32) | 0.015 (23/24) | 0.016 (24/24) | |
| qold = local quartic of the pairs aged 3+ (new) | variance | 0.019 (30/32) | 0.008 (24/24) | 0.005 (22/24) | 0.010 (8/8) |
| qold | skew / kurtosis | 0.006 / 0.008 | 0.011 / 0.007 | 0.011 / 0.009 | 0.011 / 0.002 |
| qb = the read's row sum against the scale mode's column | kurtosis | 0.012 (32/32) | 0.009 (24/24) | 0.006 (22/24) | |
| eold = |lam| energy of the cohorts aged 3+ | variance / skew | 0.001 / 0.005 | 0.004 / 0.006 | 0.006 / 0.007 | 0.009 / 0.013 (8/8) |

(strongest baseline incl. the head state, response-rows convention.) Weak (under 0.005 or mixed signs): batch 6.2 (k3d2), 6.4 (gk, gb), 6.5 (xk4), 6.6 (row gain); the per-age
parts of the local quartic separately (0.008-0.011 at layers 5-8 against 0.019 for their sum); (W o W) pushes; K21 . mu and K21 . g. Against the post-mean error after the head
every candidate reads 0.001-0.002 (the trainer's columns 9 and 16, which passed his gradient test, too): this screen ranks, it does not price.

**Ceiling** (`t62_error_ceiling.py`): of the post-mean error left after the head (after its own inputs are regressed out), the card's TRUE per-unit input errors of the same layer
explain 18-23% at every layer (variance 6-10%, skew 4-6%, kurtosis 3-7%); the error of the layer's pre-activation mean (the mean error carried in from the layer before, times the
gate) explains 3% at layer 5, 26% at 10, 38% at 12, 45% at 14 and 50% at 15. So half of the final-layer error after the head is mean error carried from earlier layers, a fifth is the
final layer's own input errors.

**Batch 6 as one module** (`b6cols.py`, handed to the trainer 02:30Z): install after otr_m.install; read-only hooks fill cd.b6cols[l] with raw per-unit vectors (m2, k3d2, f15, gk, gb,
xk4, rg, qold, eold, qb, resp). Certificate: m2 x sigma equals the host change that pbase so2m makes when it acts on one layer only (2, 9, 14), to 2e-12 (cosine 1.00000000, size
ratio 1.00000000; `s42dump_rank0035_cert{2,9,14}.npz`).

**Closed-loop runs of 10-02 evening** (old base F4 + K4L + law + K31L4; torch, no refit). Row-gain law (`t4_ora75.log`, 35 / 651): mean + sigma term +8.4 / +3.4%, mean alone +6.3 /
+1.9%, sigma term alone +2.2 / +2.2%, half strength +3.1 / +0.8%, the joint three-input fit +10.3 / +4.6%, on the plane for every consumer +18.5 / +12.6%, with the mean correction off
+6.8 / +3.5%, with every learned part off +8.2 / +4.4%. The lead's joint test (`t4_ora76.log`, 35, budget arms): with the true per-unit skew + kurtosis in the mean the law still
costs +7.6%, the true K21 column +4.2%; the true K21 plane without its column -6.8% alone and -15.7% once the skew is exact. Every learned part off (`t4_ora74.log`, 35 / 651 / 100):
the learned edge alone is worth 0.9 / 2.8 / 2.1%; the cross form x -3.6 / -4.6 / -5.9%, x half -3.0 / -3.2 / -3.9%, t +0.2 / -1.8 / -3.3%, m -0.1 / 0.0 / -3.4%; the K31 row level
at the truth still +8.3 / +7.7 / +7.5%; the K21 column at the truth still +2.3% (35); the true second-chaos kurtosis for the mean at half strength -1.2 / -1.4%, at full +6.9 / +6.2%.

## T29 (2026-10-02 16:51-20:10Z). The refit's physics base (lead's orders of 16:51Z / 18:01Z): one table on two cards, one module; the lead closed the base as a lever at 19:51Z

**What was asked.** Each physically right term that the present learned parts reject, on top of the law + K31L4, with the learned mean correction off and on, six held-out ranks
(651 378 424 328 287 641), final layer and all layers, on the arithmetic of the file on the board (first dials + table); one recommended base; every term a keyword of one module.

**The module.** `pbase_frozen_s4a.py` (working copy `pbase.py`; `pbase_frozen_s3x.py` = the same without so2k / so2p / so2td), interface of `O/ncar6.py`:
`install(cd, torch, st, k31=1.0, so2m, so2c, so2t, so2f, so2k=1, so2p=123456, so2td=0, bsk, gsk, gk4, gbk, x1, x2, lo=1, hi=14)`; all keywords 0 = the card bit for bit.
so2m / so2c / so2t = `so2_frozen_s2b.py` (second-order mean term inside the pair map, covariance list, third-cumulant target), so2f = the mean term at layer 15 after the learned
correction, bsk = birth weight with the skew-corrected kink density, gsk / gk4 / gbk = the transport gate with skew / kurtosis / b-mode corrections (single lines of the step's
source; on the trainer's per-layer card the two lines `otr_pl` has already replaced are kept), x1 / x2 = the cross form (`bk3.py`, specific planes). Certificates: on the screen
harness every keyword gives the harness's own arm to the last printed digit (network 651, ten arms, `t4_ora59.log`); on the trainer's engine (`t41_s41_base.py`, frozen
`otr_frozen_cbe85587/otr_m.py`, parameter file `O/trainer/cert/fold/c3q4zd6_optimum.pt`, law = module ncar6 forms 3 dfit 2, + k31l) the law-only row of 651 is the trainer's stored
13.05005e-9 / 85.32188e-9 exactly, the module with every keyword at 0 changes nothing (arm N1 = S1), and s4a = s3x on the old keywords (12.874282934e-9 through both).
In a request: `{"module": "pbase", "so2m": 1, "so2f": 1, "bsk": 1}`; the trainer registers `MODULES['pbase'] = ('pbase_frozen_s4a', pbase.ARGS)`.

**Table A, old base** (F4 0.6 + K4L 0.75, default dials, law through the extra plane with the constant regressor + K31L4; torch, no refit; final-layer MSE against the base of the
same switch; mean over the six networks (standard error; networks lower); `python t38_refit_table.py t4_ora52.log QL`):

| term | learned correction off | on | all layers, off |
|---|---|---|---|
| m: second-order mean term inside the pair map | -1.61% (0.89; 5/6) | +0.90% (1/6) | -0.56% |
| c: second-order covariance list | +0.65% (0.34; 1/6) | +1.10% (0/6) | +0.97% |
| t: second-order part of the third-cumulant target | -1.03% (0.49; 6/6) | -0.02% (2/6) | -0.47% |
| m + c + t | -1.55% (1.46; 4/6) | +2.09% (1/6) | +0.09% |
| bsk | -0.40% (0.30; 5/6) | -0.34% (4/6) | -0.10% |
| gsk / gk (gate, skew / skew + kurtosis) | -0.54% (5/6) / -0.76% (0.33; 5/6) | -0.27% / -0.34% | -0.40 / -0.53% |
| gb (gate + 1.5 d bpre) | -0.62% (0.35; 5/6) | -0.40% (5/6) | -0.37% |
| x / x half (cross form 3.2, 10 / 1.6, 5) | -1.98% (0.89; 5/6) / -1.82% (0.45; 6/6) | +1.56% / -0.07% | -1.72 / -1.52% (6/6) |
| b-form at the truth's size | +1.34% (1/6) | +4.50% (0/6) | +0.52% |
| K31 row level at the truth | +11.05% (0/6) | +7.80% (0/6) | +6.83% |
| m + c + t + gk + gb + x | +0.41% (2.99; 2/6) | +8.46% | +1.63% |

The base with the learned correction off is 11.5 points above the base with it on (3 / 12 / 8 / 17 / 20 / 9 on the six). The terms do not add (the union: -1.7 +1.6 +11.0 0.0 -11.6 +3.2).

**Table B, the S41 torch card** (first dials + table + k31l ages 1-2; learned off = the trainer's leaf ds at -1; the first dials hold the learned correction at 0.784 of its trained
size; `python t38_refit_table.py t4_ora60.log S`, also `t4_ora60_table.txt`; base with the correction off 5.4 / 12.8 / 9.4 / 15.4 / 17.3 / 8.8% above the base with it on, mean 11.5%):

| term | off: per network 651 378 424 328 287 641 | mean (se; lower) | on: mean (se; lower) |
|---|---|---|---|
| m | -1.96 -0.62 +0.21 +0.23 -5.43 +1.00 | -1.10% (0.96; 3/6) | +0.85% (0.76; 1/6) |
| c | -0.01 +0.61 +0.13 +0.65 +0.32 +2.40 | +0.68% (0.36; 1/6) | +0.97% (0/6) |
| t | -1.88 -0.53 +1.01 -0.19 -2.26 -0.13 | -0.66% (0.49; 5/6) | +0.13% (3/6) |
| f: mean term at the final layer | -0.96 +0.42 -0.99 +0.17 +1.23 -0.93 | -0.18% (0.38; 3/6) | +0.59% (2/6) |
| bsk | -0.32 +0.02 -1.12 +0.57 -1.40 -1.22 | -0.58% (0.32; 4/6) | -0.57% (0.33; 4/6) |
| gk | -0.67 -0.44 +0.82 -1.11 -1.20 -0.67 | -0.55% (0.30; 5/6) | -0.30% (4/6) |
| gb | -0.78 +0.03 +0.06 -0.44 -0.56 -0.13 | -0.30% (0.14; 4/6) | -0.24% (4/6) |
| x | -2.78 -0.99 +4.08 -0.69 -4.77 -0.13 | -0.88% (1.21; 5/6) | +1.79% (1/6) |
| x half | -2.51 -1.28 +0.62 -1.16 -3.11 -0.77 | -1.37% (0.54; 5/6); all layers -0.98% (6/6) | -0.12% (0.54; 4/6); all layers -0.49% (5/6) |
| m + f + bsk | -3.38 -0.11 -1.39 +1.15 -5.36 -1.14 | -1.70% (0.95; 5/6) | (m + t-diagonal + f + bsk: +0.99%) |
| m + t + f + bsk | -4.69 -0.04 +0.67 +1.65 -7.01 -0.99 | -1.73% (1.38; 4/6) | +1.95% (1.06; 1/6) |
| the same, t without rho terms | -4.79 -0.07 +0.67 +1.66 -7.18 -1.00 | -1.78% (1.41; 4/6) | |
| lead's proposal m + x half + gk + gb + bsk | -3.93 +0.29 +4.78 +0.17 -9.47 -0.90 | -1.51% (1.96; 3/6) | +2.13% (1.72; 1/6) |
| m + x half | -3.45 -1.28 +2.40 -0.04 -7.52 +0.93 | -1.49% (1.45; 4/6) | |
| m + t + f + bsk + gk + gb + x half | -4.28 +2.02 +7.68 +1.60 -8.83 -0.78 | -0.43% (2.32; 3/6) | +5.05% (2.02) |
| the same with x at full strength | -1.87 +4.09 +14.65 +4.13 -8.48 +1.34 | +2.31% (3.13; 2/6) | +9.30% |

t's cheaper forms (engineer 18:44Z), learned off / on: without rho terms -0.86% (0.53; 5/6) / +0.12%; parts a b only, no rho -0.10% / +0.29%; diagonal only -0.03% / +0.02%.

**Reading.** (1) No set is better than about -1.7% with the learned correction off, and the standard errors are as large as the differences between the sets; the steadiest are
m + f + bsk (-1.70%, 5/6) and x half alone (-1.37%, se 0.54; the only term that improves every layer, by 0.5-1.2% at layers 5-15). (2) With the learned correction on only bsk
(-0.57%), gk (-0.30%) and gb (-0.24%) help; natively bsk on S42's arithmetic reads -0.34% (se 0.32, 8 of 10 fresh networks; engineer 20:06Z). (3) The terms are worth less on the
first dials than on the default dials (m -1.1 against -1.6%, x half -1.4 against -1.8%, t -0.7 against -1.0%): each dial solve takes up part of what a term would give.
(4) PER LAYER (`t4_ora60.log`, three networks): m makes layers 6-11 WORSE by 1-2% with the correction off or on and improves only layers 12-15 with it off; x half improves every
layer; the K31 row level at the truth hurts from layer 5 on (old base). (5) 'Learned off' in both tables switches off the learned MEAN correction; the learned EDGE (a learned
K21-like term in the covariance output of layers 8-14, on per-unit features) stays on. Arms with the edge off as well: `t4_ora74.log`.
The lead closed the base as a lever at 19:51Z (S42 = S41 + the head column; bsk tested natively for S43).

## T30 (2026-10-02 18:05-19:10Z). The scale mode: three quarters of the true K21 plane is one form; the card splits it between b and the bank; its column at the truth HURTS; per-unit kurtosis facts

Networks 35 and 100 (8M truth planes, two draws on layers 2 / 5 / 9 / 12 of network 35; 32M truth mean / covariance; two independent 32M per-unit truths). Card planes = the dump
`tkd_rank00{35,100}_ora41_Qd.npz` (base F4 0.6 + K4L 0.75 + the law, learned parts on). All plane comparisons in the metric the covariance uses (rows x pdf / (2 v), columns x gate / sigma,
off the diagonal). numpy tools, 2 threads: `t40`-`t50b`; oracles `t4_ora61.log`, `t4_ora62.log` (e16b frozen s3y / s3z, options `tkcol`, `tkm21`, `tkm31`).

**1. The mode.** A bias-free ReLU layer commutes with a positive scale, relu(r y) = r relu(y): a fluctuation of the activations' overall size passes every later layer exactly. Measured
(`t45_scale_mode.py`, and the table printed at 18:27Z): the fluctuation along the mean direction has variance s2 = mu' C mu / |mu|^4 = 0.0045 at layers 1-3, 0.0067 at layer 9, 0.0076 at
layer 14 (sd 6.7% -> 8.7%); it is the covariance's top eigenvector only from layer 5 on (cosine 0.81 at 5, 0.95-0.97 at 9-14; top eigenvalue 14 against a trace of 213 at layer 12).
The input radius alone, the only part exactly independent of direction, is 1 / (2n) = 0.0005.

**2. The true K21 plane is mostly that mode's form.** For z = (1 + S) y with S independent of y: K21 = (2 s2 + k3S)(v_i mu_j + 2 C_ij mu_i) + (1 + 3 s2) K21^y + k3S mu_i^2 mu_j.
Least squares of the TRUE plane on v_i mu_j + 2 C_ij mu_i: share of the plane's weighted energy 55% (layer 2), 74% (5), 80% (9), 80% (12), 81% (14); coefficient / (2 s2) = 0.50, 0.62, 0.68,
0.68, 0.73 (a pure independent scale mode gives 1; the mu_i^2 mu_j term's coefficient is -1 to -3 s2^2, negligible). The card's whole plane gives the same projection to 0.4% (layer 2),
1.3% (9), 1.9% (12), 2.9% (14) (short), but its b vector holds only 0.10 (layer 2), 0.26 (9), 0.29 (12), 0.32 (14) of 2 s2: the rest of the form sits in the bank columns (52-58% of the
bank plane's weighted energy is its column along mu; the b-form 88-95%).

**3. A fifth to a half of the card's K21 error is one vector per layer.** Residual truth - card (no knobs): 0.12 / 0.22 / 0.33% of the plane's energy at layers 5 / 9 / 12. Its top
singular direction (fitted on draw 1, scored on draw 2) holds 18 / 33 / 41% of it; the column vector is mu (cosine 0.95 / 0.97 / 0.98). `t43`: the row vector is unit-specific: v, mu^2,
mu (C mu), the card plane's own contraction with mu, k3, k4 / v, alpha, He2, the kink flag explain at most 10% of the residual each (12.5% with the b-form). `t44`: the card's three parts'
own columns along mu as regressors: the truth wants the BANK's column -6 / -15 / -23 / -31% and the B-FORM's +0 / +14 / +23 / +44% at layers 5 / 9 / 12 / 14 (network 100: -6 / -15 / -20 /
-22% and -2 / +8 / +25 / +18%); joint share 3 / 15 / 23 / 29% of the residual. `t47`: a free b vector (1,024 numbers, cross-validated) could hold 4 / 11 / 13% of the residual at layers
5 / 9 / 12; a free row vector with the column mu 15 / 29 / 38%; both 18 / 34 / 43%.

**4. The two K21 knobs are not physics of K21** (`t42_k21_knob_audit.py`): the truth wants the iso form (kink_i (k3_i / v_i) C_ij) with coefficient -0.002 to -0.037 beside the b-form
(about 0 alone) where the knob is -0.094, and the k21n form with 0 to +0.014 where the knob is -0.028; with the knobs the plane is 3-9% further from the truth.

**5. Oracle: the column at the truth hurts** (covariance output only, kink rows, 8M planes; network 35, base Q = law, learned on, against Q 12.3387e-9):
whole difference -5.78% (the old arm: -10.87% against B); ONLY its part with the column vector mu (one number per row) **+5.22%**; the difference without that part **-10.77%**; the
column part on all rows +14.09%. On the four layers with two draws: column part +2.17 / +2.09 / +2.12% (draw 1 / draw 2 / mean: the column is a clean signal); whole difference
+0.48 / +7.09 / +1.48% (the rest of the plane carries the oracle's noise). The collective pieces together (`t4_ora62.log`, against Q, learned on / off): K21 column +5.22 / +3.26%;
K31 row part (kink rows) -0.63 / +3.01%; K22 whole -1.53 / -1.29%; K21 column + K31 row +5.65 / +6.81%; + K22 +4.21 / +5.81%; all three planes whole -10.50 / -6.52%;
K21 without its column -10.77 / -7.97%. So the card's split of the scale mode is something the loop is fitted around: every collective correction alone or together moves it off the fit.
(My oracle replaces the plane as its knobs make it; the arm that keeps the knobs is `t4_ora64.log`. The knob's own column is under 0.2% of the plane's collective form, so the
+5.2% is not the knob.)

**6. The cross form's strength is the scale mode's.** A scale mode gives K31 the form 1.5 (b_j k3^y_i + 3 b_i K21^y_ij) with b the form's TRUE size; with the card's b (0.29 of 2 s2
where the truth is 0.68) the fitted 3.2 / 10 of T28 is 1.5 x 0.68 / 0.29 = 3.5.

**7. An exact treatment of the mode buys nothing by itself** (`t46_scale_truncation.py`, exactly solvable model z = (1 + S) y, both Gaussian, mean and covariance of layer 5 / 12 of
network 35): a first-order pair map fed the model's EXACT third and fourth cumulants gives the next layer's per-unit variance to 1.6e-5 / 3.6e-5 relative rms (4 times less at half the
mode's variance: second order); without the fourth-cumulant terms 8e-4 / 6e-4; Gaussian closure 5e-3 / 8e-3. So a "scale-factored" pair map is not a lever; the fourth-order collective
forms are (6-8e-4 of the variance), and those are the trained ones (K31 row level 0.53, radial).

**8. The feed of the true second-chaos matrix into K21** (`t40`): coefficients 1.0-1.1 and 0.5-0.7 on its two forms (the derivation is right), 0.7-1.8% of the K21 error.

**9. Per-unit inputs** (`t48`, `t49`, `t50`, `t50b`; network 35). The card's per-unit VARIANCE error is 0.25e-3 relative rms at layer 2 (the truth's noise) and 1.1e-3 / 1.6e-3 at
layers 12 / 14, unbiased (mean under 0.1e-3), and no per-unit candidate predicts it (R^2 at most 0.05 alone, 0.08 together: alpha, alpha^2, kink, skew, kurtosis, the three parts'
responses to the mode, the bank's row norm). Per-unit SKEW: the truth is 70-86% the collective line t3 = c alpha with c = 0.49-0.74 of 6 s2; the card's error is 3% of the truth's rms
at layer 2, 5.6% at 14, unit-specific (the collective line explains 1-12% of it; the feed of the second-chaos matrix 1-4% with coefficient 0.5-1.1). Per-unit KURTOSIS: its level is
0.29 (layer 2) to 0.61 (layer 14) of 12 s2; the spread across units (sd 0.0017 -> 0.015 of k4 / v^2) is 27-37% the second-chaos form (6 / n) R_ii with the true matrix (coefficient
0.85-1.0), the card's unit-specific part (K4L, pairs aged 1-2) correlates with the truth's at R^2 0.10-0.22 (coefficient 1.05-1.37), and 20-30% of the card's kurtosis error is that
second-chaos form (coefficient 0.63-0.85). The earlier arms that fed a matrix diagonal into k4 (the old oracle K4: +3.2% / -1.0% on 35 / 651; the lead's dial; my companions) fed
EVERY consumer of k4: the pair map's K31 row term and its per-unit covariance term e4 as well. The arms that feed the per-unit mean and post variance only are `t4_ora67.log` (true
matrix), `t4_ora68.log` (K4L, older pairs), `t4_ora69.log` (the card's own diagonal).

## T28 (2026-10-02 16:30-18:00Z). The one cheap coherent piece of the missing K31 content: the cross form of the collective mode and the specific third cumulant (truth-validated, free, no gain with the learned parts on); closures; the covariance is nearly low-rank at depth

Base and conventions as in T27 (lead's base F4 0.6 + K4L 0.75, law through remainder31 = arm Q; truth planes of networks 35 / 100; planes compared "as the
covariance uses them": standardised, K31 rows x alpha pdf / 6, columns x gate, row-by-row multiple of rho out). Pooled tables: `python t30_pool.py <log> <ref arm>`.

**(a) The missing K31 plane is low-rank (`t31_rest_spectrum.py`, 35, two truth draws, singular vectors from one draw scored on the other).** Share of the
missing energy (truth minus card with the law) in the top k directions, layers 5 / 9 / 12: k = 1: 9 / 15 / 19%; k = 8: 23 / 34 / 41%; k = 16: 35 / 48 / 54%;
k = 32: 51 / 64 / 70% (a random plane: k / n = 0.1 / 0.8 / 1.6 / 3.1%). The card's own carried plane is as low-rank (top 16: 41 / 52 / 55% of itself).

**(b) The top direction is k3_i x bpre_j (`t32_top_direction.py`).** Layers 9 / 12: the column vector correlates 0.91 / 0.93 with the card's bpre (0.90 / 0.88 with
mu; joint R2 on the candidates 0.97 / 0.96), the row vector 0.91 / 0.91 with the per-unit third cumulant (joint R2 0.84 / 0.85). At layer 5 the top direction is
not yet this one (the form's share is 8-11% there).

**(c) The form and its coefficients (`t33_bk3_form.py`, `t33c_bk3_split.py`).** A collective fluctuation S of the layer's scale changes the covariance (1 + S),
shifts the means (beta_j S) and scales the specific third cumulant (1 + gamma S). The law of total cumulance then gives, besides the card's b-form of the third
cumulant (v_i b_j + 2 C_ij b_i, b = beta Var S = bpre), a fourth cumulant gamma Sym4(b (x) kappa3_specific): in the (3,1) plane x1 k3s_i b_j + x2 b_i Ks_ij,
x1 = gamma, x2 = 3 gamma (Ks = bank + star planes, k3s its diagonal); a common multiplicative fluctuation has gamma = 3/2. Least squares of the missing plane on
(k3s_i b_j, b_i Ks_ij, v_i b_i b_j), layers 8-14: network 35: 3.1-3.4 / 9.8-11.0 / -1 to -5; network 100: 2.9-3.4 / 9.6-11.2 / -1 to -7 (the second-order form
would be 9 if the whole K21 plane belonged in the form: it does not). Share of the missing energy: 17 / 21 / 23 / 22% at layers 9 / 12 / 13 / 14 on 35, 17 / 20 /
23 / 27% on 100 (one truth draw: lower bounds); 4-6% at layers 2-4. With the whole K21 plane and k3 instead of the specific parts the fit reads 2.2 / 4.5 and
explains 15-24%. The same strength shows in the true K22 plane (coefficient 2.0-4.2 on b_i K21_ji + b_j K21_ij, expected 2 gamma = 3 at gamma 3/2, but under 2%
of the K22 error) and in the per-unit k4 (coefficient 4-16 on b_i k3_i, correlation 0.07-0.23) (`t33b_bk3_k22_k4.py`).
Equivalent reading: x2 b_i Ks_ij in the K31 slot = the K21 term with the row unit's kink density taken at the mean mu_i + (x2 / 3) b_i; natively a row factor on
the bank and star planes and one outer product. Module: `bk3.py` (level 'r31' or 'pair').

**(d) In the loop with the learned parts ON it does not pay (`t4_ora51.log` whole plane, `t4_ora53.log` specific parts; final-layer MSE against Q).** Networks
35 651 378 100 424: both forms at 3.2 / 10: -1.64 +3.12 +1.29 -0.40 +5.76, pooled +1.70% (se 1.31); half strength +0.02% (se 0.48); only b_i Ks_ij at 10: -0.12%
(se 0.63); only k3s_i b_j at 3.2: +0.46%; 1.5 / 4.5: -0.03%; on top of K31L4 (-2.22%, 5 of 5): -0.52%. The same form in the K22 plane: +124% (5 of 5); per unit
(whole-plane version, 35): +3.8%. Its feed into the carried third cumulant, the transport gate g + kappa d bpre (option gbk; (d_c / 2) g_a g_b kappa4(a, b, c, c)
contains kappa d_c b_c g_a g_b kappa3(a, b, c)): kappa 1.5: -0.68 -0.75 -0.78 +1.71 +1.17 (pooled +0.18%); 3.2: +0.56%; and it does not repair the card's
shortfall on old K21 content (age profile with the corrected gate: 0.984 against 0.983 at age 4, 0.977 against 0.976 at age 8; `t28_age_profile_rank0035_eff_Gd15.log`).
With the learned mean correction OFF the form gains (first network of the lead's table, 651: -15.01% against -12.62% for law + K31L4, all layers -4.46 against
-2.98%; full table in T29).

**(e) What is left after the cross form (`t35_rest_after_forms.py`, `t37_mode_rows.py`, `t36_column_subspace.py`).** Fitting the two forms out takes 11 / 17 / 21%
of the missing energy at layers 5 / 9 / 12; the rest is still low-rank (top 16: 30 / 40 / 45%), and its column vectors are the 2nd, 3rd, ... eigenvectors of the
covariance. Per mode m, the row vector R_m[i] = the missing cum(z_i, z_i, z_i, u_m . z) is a clean signal (agreement between truth draws 0.92-0.98 for the top
eight modes) but for modes 2-8 it correlates with none of twenty per-unit candidates (below 0.15 after the row term and the cross form are out; the cross form
takes 76-78% of the mode-1 vector's energy). 75% of the missing K31 content at layer 12 has its columns inside the top-64 eigenspace of the covariance (70% at
layer 9, 54% at layer 5; top 16: 46 / 36 / 20%).

**(f) The covariance spectrum (true pre-activation covariance, 35).** Share of the trace in the top 16 / 64 / 256 eigenvalues: layer 1: 8 / 28 / 71%; layer 5:
21 / 53 / 90%; layer 9: 31 / 67 / 94%; layer 12: 37 / 72 / 95%; layer 14: 41 / 75 / 96%. Eigenvalues at layer 12: 14.1, 6.0, 5.6, 5.3, 5.1, 4.8, ..., median
0.013. The card holds the top eigenvalue to -0.23 ... +0.05% at every layer and its per-unit variance error is not along that mode (R2 0-4%;
`t34_collective_mode.py`): the collective sector is balanced by the trained constants, which is why every collective form at "its true size" (b-form, K22
form, K31 row coefficient) loses without a refit.

**(g) Closed.** The C^2 form of the law without the matrix (`t4_ora46.log`, 287 641 651 35, against the constant regressor without a C^2 form): exact b C^2 (b from
the diagonal) -0.83% pooled; only the elementwise part (v_i + v_j) C_ij: +6.72%; plus the top mode of Co^2 from two power steps: +3.77%; plus the top 4 exact
eigenpairs: +1.92% (on 287: 8 / 16 eigenpairs -0.4 / -1.2 points against exact -1.6). The b-form (`t4_ora43.log`, eight networks, against Q): added in the
covariance at 0.0025 / 0.005 / 0.0075 x layer +1.93 / +5.16 / +9.69% (0 of 8 lower); bscale x 1.005 / 1.01 / 1.015: +0.13 / +0.27 / +0.63%. The true K31 rest one
layer at a time (`t4_ora48.log`, `t4_ora50.log`): on 35 the thirteen single-layer gains add to 10.7 points (all layers at once 10.6), layer 5 / 6 / 13: 4.2 / 2.3 / 2.6,
the others between -1.3 and +1.4; layer groups 5-7 / 8-10 / 11-14: 5.4 / 1.3 / 5.1 points on 35, 4.2 / 3.8 / 6.2 on 100; layers 2-4 read -1.7 / -3.1 with the 8M planes
(30-58% noise there); 64M-sample planes of layers 2-7 exist now for both networks (`t24c_merge.py`: K31 noise 9 / 5 / 3 / 2 / 1 / 1%), oracle `t4_run_ora54.sh` queued.

## T27 (2026-10-02 14:40-16:30Z). What the remaining plane error is made of: the network-specific fourth cumulant, made over ALL earlier layers; the first-order theory is complete; nothing of what is left is cheap

Base everywhere: the lead's base F4 0.6 + K4L 0.75 on the research card (learned on), torch, no refit, final-layer MSE against the organizers'
truth: network 35 13.0429e-9, network 100 15.1398e-9. "Law" = the pair-carried law through remainder31, three forms, constant regressor
(arm Q: -5.40% / -4.47%). Truth planes = `P2/public_g19_attribution_v1/mps_planes_rank00{35,100}.npz` (8M samples, layers 2-14; second 8M
draw for 35 at layers 2, 5, 9, 12). Tables: `python t4_tab.py <log> <rank>`.

**(a) Budget with the law in (`t4_ora38.log`; per-unit inputs of the mean formula exact at every layer, 32M truths).** Network 35 (law -6.18%
in that run): exact variances -26.3%, skew +13.2%, kurtosis +0.5%, all three -55.2% (without the law: -30.2 / +7.5 / -4.8 / -57.9%).
Network 100 (law -2.40%): -41.3 / +3.8 / -8.8 / -69.1% (without: -42.5 / +1.0 / -14.3 / -69.7%). With exact variances the cards with and
without the law give the same error (35: 9.1 against 9.0e-9): the law works through the variance and has taken a fifth (35) / a
sixteenth (100) of that lever. Per-unit k4 companions of the law (`t4_ora39.log`, 35): -3.09 / -4.94 / -1.36 / +1.83% against the law's -6.18%:
closed.

**(b) Truth planes in the closed loop (`t4_ora40.log`, `t4_ora41.log`, `t4_ora45.log`; options tk21 / tk31 / tk22 / tkrows / tkm / tkdraw of
`e16b_f4b.py`, frozen s3l-s3o).** The truth replaces the card's plane AS ITS KNOBS MAKE IT, off the diagonal, only in the covariance output of
the pair map (the post K21 target, the star and the births keep the card's planes; the first version through pair_in changed the target
too and read +373%: `t4_ora40_v1_pin_route.log`, invalid). Controls: the rebuilt arm repeats the first one (35: -10.87 against -10.85%,
-13.92 against -13.92%). Final-layer MSE against the base, 35 / 100 (all layers in brackets):

| plane at truth (on top of the law) | 35 | 100 |
|---|---|---|
| nothing (law alone) | -5.40% (-6.36) | -4.47% (-5.18) |
| K31 slot, whole | -13.92% (-11.71) | -10.12% (-6.77) |
| K31 slot, only the part proportional row by row to rho (the "row part") | -2.90% (-4.62) | -0.04% (-0.24) |
| K31 slot, only the rest (card's own row coefficients) | **-15.96% (-14.23)** | **-15.70% (-11.54)** |
| K22 | -6.43% (-6.57) | -7.32% (-9.67) |
| K21, kink rows (abs(mu) < 1.5 sigma) | -10.87% (+5.17) | -12.89% (+5.45) |
| K21, the card's response rows | -8.09% (+5.76) | -13.56% (+6.12) |
| K21, rows with mu >= 1.5 sigma | +3.97% (-2.88) | +1.38% (-1.51) |
| K21, rows with mu <= -1.5 sigma | -6.35% (-8.06) | -5.29% (-5.97) |
| K21 all rows + K31 + K22 | -11.43% (-1.26) | -19.24% (+0.91) |

Noise of the 8M planes (35, truth only at layers 2, 5, 9, 12; first draw / second draw / mean of both): K31 -10.29 / -12.51 / -11.79%
(all layers -9.58 / -10.05 / -10.62): about one point per four layers, so the K31 numbers are underestimates. K21 kink rows -4.95 / +1.31 /
-4.00% (all layers +2.26 / +3.35 / -3.31): several points of noise, and the "all layers worse" of the K21 rows above is mostly this
noise. The K21 arms are therefore not built on. The split of K21 into a row part and a rest (tkm) is not a valid split for K21 (each part
alone reads +34 to +60%): the K21 plane's large forms are column forms, not row forms.
The card's K31 row coefficients are 0.47-0.57 of the true ones (`t22_planes.py`; the true row coefficient is 0.85-0.92 x the unit's own
true kurtosis, correlation 0.90; the card's per-unit kurtosis has the right mean, 0.91-1.00 of the truth, but only 0.30-0.45 correlation
with it across units), and still the true row coefficients lose 2.5 / 4.4 points: the trained constants c0 and the k31 knob hold the row
term at about half its theoretical size on purpose.

**(c) Audit of the lead's K31L4 (`k31l_audit.py` = copy of O/k31l.py sha 2ccd70d0e195 with dump / agemin / rowrm; `t4_ora42.log`;
`t23_k31l_vs_truth.py`).** The subtracted projection is the exact 3 x (1/n) sum_k Q(k,k,i,j) (re-derived slot by slot). Against the truth's
rest on 35, least-squares coefficients of (law, age 1, age 2): 1.01 / 1.00 / 1.02 (layer 2), 1.03 / 0.99 / 0.98 (5), 1.00 / 0.92 / 0.92 (9),
0.85 / 0.86 / 0.86 (14); energy explained: law 9-17%, K31L4 ages 1-2 6.5% (layers 2-6) to 1.7% (14), all 15-24%. Loop, 35 / 100:
ages 1-2 -5.89 / -6.81%, age 1 -5.34 / -5.54%, age 2 -6.37 / -5.69%, ages 1-3 -7.11 / -7.67%, full local form +0.71 / -2.13%, row part
removed -5.79 / -5.81%, row part only -5.93 / -5.32%, strength 0.5 / 1.5 on 100 -6.00 / -6.79%; on top of K31L4 the truth's rest reads
-16.76% on 100. F4 puts nothing into the K31 slot (its plane content is removed by the anchor).

**(d) How old the content is: Gaussian-restart Monte Carlo (`t24_restart_planes.py`, GPU, 11 runs of 2 x 4M samples on 35; `t25_restart_tab.py`,
log `t25_restart_tab_rank0035.log`).** The pre-activations of layer r are drawn Gaussian with the true mean and covariance and the network
is run exactly from there: the plane at layer l then holds what the last l - r layers made, every diagram. Controls: a measure layer equal
to the restart layer gives zero planes (cross-block cosine 0.002); products are taken across independent blocks and against the stored
truth draws. Share of the truth's energy made by the last d layers:

| plane | layer | d=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K31 rest | 5 | .16 | .44 | .69 | .86 | | | | | | | |
| K31 rest | 9 | .07 | .24 | .40 | .56 | .70 | .80 | .88 | .95 | | | |
| K31 rest | 12 | .05 | .18 | .31 | .45 | .57 | .67 | .76 | .83 | .89 | .93 | .96 |
| K21 (off the diagonal) | 12 | .16 | .33 | .44 | .54 | .64 | .72 | .79 | .85 | .89 | .92 | .96 |
| per-unit k4, unit-specific part | 12 | .07 | .21 | .27 | .43 | .53 | .60 | .69 | .77 | .85 | .90 | .96 |

No layer makes most of it: at layer 12 every one of the eleven earlier layers adds 4-14 points of the K31 rest.

**(e) The first-order theory, term by term, with exact inputs (`t26_conversion_mc.py`, GPU, 2 x 4M samples on the true stream, layers 5 / 9 /
12, networks 35 and 100; `t27_conversion_tab.py`, logs `t27_conversion_tab_rank00{35,100}.log`).** With zeta_i = sum_a w_ai g_a (z'_a - mu'_a)
the linear-response image of unit i, the K31 plane is births (z' Gaussian: the restart plane one layer back) + TRANSPORT
cum(zeta_i, zeta_i, zeta_i, zeta_j) + CONVERSION of the third cumulant of z', where one unit d of the previous layer carries two of the
four slots, either both through its weights (coefficient V_d = 2 m (1 - g)) or one through its weight and one through a covariance edge
(coefficient d_d = pdf / sigma):
  CONV_G = 3 sum_d G_id [V_d w_di w_dj + d_d (w_di c_jd + c_id w_dj)],  G_id = cum(zeta_i, zeta_i, z'_d),  c_id = sum_{e != d} C'_de g_e w_ei;
  CONV_H = 3 cum(zeta_i, zeta_j, eta_i),  eta_i = sum_d [V_d w_di^2 + 2 d_d w_di c_id] (z'_d - mu'_d).
Share of the truth's rest energy on 35 (truth across its two draws), layers 5 / 9 / 12: transport 0.44 / 0.62 / 0.67; CONV_H 0.22 / 0.18 /
0.14; CONV_G covariance-edge part 0.07 / 0.07 / 0.06, weight part 0.04 / 0.02 / 0.01; births 0.16 / 0.07 / 0.05; ALL TERMS AT STRENGTH 1:
0.957 / 0.964 / 0.965 (joint fit 0.97, coefficients 0.99 / 0.92 / 0.86 / 0.97 / 1.15-1.18). Network 100 (one truth draw, so the shares carry
its noise): 0.75 / 0.90 / - at layers 5 / 9. Per-unit k4 (unit-specific part): 0.93-0.96. The card's carried plane holds 0.21 / 0.18 / 0.14.
For the third cumulant the same run gives, as the covariance uses the plane (rows x pdf / 2, columns x gate): transport + births 0.985 /
0.992 / 0.993, + the feed of the fourth cumulant (sum_c (d_c / 2) w_cj cum(zeta_i, zeta_i, z'_c, z'_c) + sum_a d_a w_ai cum(z'_a, z'_a,
zeta_i, zeta_j)) 0.9989 / 0.9992 / 0.9991.
Reading: nothing is missing from the theory. Two thirds of the K31 rest at depth is the fourth cumulant of the previous layer carried
through the gates, i.e. the whole tensor, of which the law carries the second-chaos part only. The conversion terms are the lead's mixed
terms (his MX1 = the covariance-edge part of CONV_H; the weight part V_d w_di^2 is the coincident-unit term with its own coefficient).
Every one of them needs a second push of the bank columns (theta_ir = sum_d pp_dr [V_d w_di^2 + 2 d_d w_di c_id]) or the read's result
before its push: 27% of the bill each by the engineer's census (`O/engineer/census/s40_651.json`: young read 27.6%, young push 27.1%).

**(f) Which ages the card holds (`t28_age_profile.py`, logs `t28_age_profile_rank0035{,_eff}.log`).** As the covariance uses the planes, layer
12 on 35: K21: the card holds 99.66% of the truth's energy (99.87% at layer 5, 99.78% at 9); by age its coefficient is 0.989 of the truth's
for content made one layer back, falling smoothly to 0.977 for content 7-11 layers old. K31 rest: the card holds 15.5% (21% at layer 5); by
age 0.47 / 0.28 / 0.21 / 0.14 / 0.16 / 0.13 ... of the truth's coefficient: half of the youngest content, an eighth of the old. The
unweighted comparison on kink rows (card 86% of the truth) is dominated by columns the covariance does not use (dead units).

**(g) Is the K21 error a form? (`t29_k21_forms.py`, `t29b_bform.py`).** The residual truth - card (covariance-weighted) is 0.25-0.48% of the
truth's energy at layers 5-14. The card's own b-form v_i bpre_j + 2 C_ij bpre_i explains 9-16% (35) / 4-27% (100) of it at layers 10-14 with
a positive coefficient: the card's b is 1.4% (layer 2) to 6.5-7.8% (layer 14) too small; the card's bpre is 0.96-0.98 parallel to mu and
grows from 1.06 mu / n (layer 2, the exact value of the input's radial mode) to 5.1 mu / n (layer 14). IN THE LOOP IT IS NOT A LEVER
(`t4_ora43.log`): the b-form added in the covariance at 0.0025 / 0.005 / 0.0075 x layer: 35: -2.79 / +0.97 / +5.83%, 100: -1.34 / +3.31 /
+9.43% (law alone -5.40 / -4.47%); the dial bscale x 1.005 / 1.01 / 1.015: 35 -5.89 / -6.51 / -6.54%, 100 -4.10 / -3.42 / -3.36% (opposite
signs on the two networks). The card's learned covariance parts already stand in for it.

**What follows.** The variance error is plane error (analyst A20), the plane error is the network-specific fourth cumulant beyond its second
chaos and the young local quartics (K31 directly, K21 through the feed), that content is made over all earlier layers, and each exact
piece costs a push. Closed on cost with the lead's numbers (his mixed terms: -2 to -3% for +27% of bill). What stays open is free: the
strengths by age of K31L4 (ages 1-3 beat 1-2 on 35 and 100), and dials.

## T26 (2026-10-02 13:30-14:40Z). AUDIT of the lead's pair-carried law in numbers; what it lacks (the gain term) and why that does not matter

All on the lead's base card (F4 0.6 + K4L 0.75, default dials; 35: 13.0429e-9, 651: 16.0081e-9), torch, no refit, strength 1, final-layer MSE /
all layers. My copy `ncar6_audit.py` (= O/ncar6.py of 14:00Z + dumps and options; parity: its numbers equal the lead's and the trainer's to the
digit). Logs `t4_ora28.log` .. `t4_ora36.log`; tables `t20_audit_tab.py <log> <rank> <arm>`.

**The pair matrix against my full-matrix carrier** (both passive on the same card, four forms removed), layers 1-14:

| | cosine | share of the Monte Carlo oracle: pair / full matrix |
|---|---|---|
| network 35, against the complete certified law | 1.000 0.992 0.983 0.975 0.965 0.956 0.939 0.921 0.900 0.877 0.844 0.796 0.739 0.685 | 0.87 (layer 2) .. 0.55 (14) / 0.89-0.92 .. 0.82 |
| network 651, the same | 0.999 0.993 0.984 0.973 0.964 0.951 0.938 0.917 0.890 0.855 0.812 0.756 0.723 0.711 | 0.87 .. 0.58 / 0.89-0.92 .. 0.82 |
| network 35, against my lean law WITHOUT the gain term | 1.000 0.999 0.996 0.995 0.994 0.994 0.994 0.993 0.992 0.991 0.990 0.989 0.989 0.988 | equal |

So pairs, hub selection (1024 hubs with a pair at layers 1-3, 608 at layers 13-14), the card's birth directions (cosine with g o Co[:, h]
1.000 .. 0.992) and the joins lose about 1%; the whole drift is the missing GAIN (homogeneity) term.

**Closed loop** (through remainder31, four forms removed): pair law 35 651 287 641: -4.96 -14.43 -4.84 -3.47% (pooled -6.90%); my full matrix,
complete certified law: -5.02 -15.06 -4.14 -2.61% (pooled -6.65%). Lean full matrix on 35 with / without the gain term: -4.79% / -4.19%. The gain
term and the K21 x covariance products are worth nothing in closed loop: nothing is left in the law itself.

**The gain term cannot ride on the pairs.** It is kappa g g' o Co = sum over ALL units h of (kappa g_h / 2)(e_h c_h' + c_h e_h'), largest at the
always-on units, which have no pair. As a row form on the newborn pairs: cosine with my lean law 0.998-1.000 at layers 1-3, then 0.955 (4), 0.70 (9),
0.47 (14), size twice the truth; closed loop +3.6% on 35 (kappa from the diagonals: -3.95% through remainder31, +1.35% in the read).
The gain correction as a matrix (`t21_gain_spectrum.py`, networks 35 and 287, one form removed): low rank (top direction 68-81% of it at layers 4-14,
75-80% of the one-sided form m q' + q m' on the mean direction), but taking the mean direction out of the plane at the read does not help the final
layer (287 -1.29%, 641 -1.59% against -0.58% / -0.72% without and -5.71% / -4.21% with four exact forms).
The forms belong to the law's matrix: my full matrix (gain term in) with ONLY the multiple of C removed: 287 -4.12% (four forms -4.14%) but 641 +1.87%
(four forms -2.61%).

**The multiple of C without a table.** Exact off-diagonal one-form coefficient over the coefficient of tr C v_i in the diagonal regression, layers 1-14:
regressors (tr C v, mu^2, 2 mu C mu): 1.04 falling to 0.59 / 0.61 (the lead's RHO1); regressors (tr C v, 1): 0.95-1.03 at every layer on 35 and 651.
(The D e e' forms push to W' diag(g^2 D) W, an identity-shaped part of the diagonal.) Fold in the read, seven networks 35 651 378 424 328 287 641:

| removal | per network | pooled |
|---|---|---|
| one form, table RHO1 | -5.20 -12.36 -5.16 -9.09 -6.16 -1.89 -0.04% | -5.60% |
| one form, regression on (tr C v, 1) | -6.14 -12.72 -5.53 -9.26 -6.59 -0.58 -0.72% | -5.81% |
| three forms, regression on (tr C v, mu^2, 2 mu C mu, 1) | -6.18 -12.97 -5.79 -8.96 -7.09 -1.80 -0.92% | -6.13% |
| four exact forms (lead E51) | | -7.46% |

Trainer, first 25 judging ranks on candidate 3: three forms with the constant regressor -6.43%, with the table -6.58% (difference 0.16, se 0.16).
Lead 14:14Z: with the (C^2)_ii regressor all four coefficients come from the diagonal to the digit of the exact fit; the matrix C^2 is the cost.

**K22 companion on the pair law** (K22_ij += s (r_i + r_j) / n, r = residual of the diagonal regression on the response rows): 35: s = 0.5 -6.18%,
1 -3.59%, 2 +9.05% (s = 0: -6.14%); 651: 0.5 -13.43%, 1 -11.21% (0: -12.72%). A dial at most.

## T25 (2026-10-02 12:55-13:25Z). Native details of the fold; a K22 companion; audit of the lead's pair-carried law (O/ncar6.py) by reading

**Native details** (`ncar_fold_frozen_s2x.py`; `t4_ora25.log`; torch, no refit, S38 physics, strength 1, final-layer MSE / all layers, network 35 / held-out 651):

| arm | 35 | 651 |
|---|---|---|
| frozen form (T24) | -3.81% / -3.54% | -4.51% / -3.54% |
| conversion weights from the card's beta = 2 (gaussian_mean - host gate) alive | -3.81% | -4.51% |
| diagonal regression on the response rows only | -3.79% | -4.50% |
| regression on tr C v_i alone (factor 1.41 kept) | -3.76% | -4.52% |
| transport factor 0.9 / 1.1 | -3.35% / -4.06% | -4.23% / -4.52% |
| strength 1.25 | -4.17% | -4.86% |
| + the diagonal's per-unit part in K22: K22_ij += (B_ii + B_jj) / n | **-4.50%** / -3.64% | **-6.26%** / -4.89% |
| + the diagonal's per-unit part in k4: k4_i += (6 / n) B_ii | -3.24% / -4.07% | -4.00% / -3.93% |

The three native shortcuts are free (identical results). With the one-regressor fit the off-diagonal / diagonal ratio is 1.29-1.33 (35) and 1.39-1.42
(651): factor about 1.35. All three together with factor 1.31, network 35: -3.79%; with the K22 term at strength 1 / 0.5: -3.95% / -4.10%; with the
three-regressor fit and the K22 term: -4.30% (`t4_ora26.log`; the K22 term here uses the residual on the response rows only).
The K22 term is the second chaos of cum(z_i, z_i, z_j, z_j) = (1/n)(Ntilde_ii + Ntilde_jj) + fourth chaos: a per-unit vector, free natively.
Held-out ranks and the oracle ceiling of that use: queued (`t4_run_ora26.sh`, `t4_run_ora27.sh`).

**Audit of the lead's pair-carried law by reading** (O/ncar6.py of 14:00Z; spec O/NCAR6_SPEC.md). The lean law's fresh sources are sums over hubs h of
forms on (c_h, e_h), c_h = g o Co[:, h]: births sum_h S2_h c_h c_h', row forms sum_h a_h (e_h c_h' + c_h e_h'), diagonal sum_h D_h e_h e_h'; the card's
newborn columns of hub h are p+- = s_h + t+- e_h, so each hub's 2 x 2 form M_h = [[S2, a], [a, D]] rides on its own pair with weights
w+ = 2 tau1 M tau1', w- = 2 tau2 M tau2', x = 2 tau1 M tau2', tau1 = (-t-, 1) / gap, tau2 = (t+, -1) / gap. Checked: the change of basis, the factor 2
(Ntilde = 2 W' N W), the cross term counted once, the partner bookkeeping; the row factor (corrected form, Co V); the diagonal D = my law's diagonal
minus what the pair forms put there themselves (g^2 (Co2 S2 + gbar diag B + diag K3V)); the dropped gain term (what is left out is
2 kappa W' offdiag(Cpost - g g' o C) W, second order and form-shaped); the isotropic rest a_rest = 0.75 (radial / 3) tr C - a_B replacing what the gain
term produced (variance tilt and diagonal need the whole matrix's diagonal; its sandwich gbar a_rest (gamma v - VarG) is my -kappa VarG). All consistent
with `nlaw.law` (lean). Not decidable by reading, queued as one passive run of both carriers on the lead's base (`t4_run_ora28.sh`, `ncar6_audit.py` =
a copy that dumps its matrices): the level 0.75 radial / 3; s_h = stored[:, h] (after _bd_update's changes) against c_h; the per-layer cosine and size
against my full-matrix carrier and the oracle; a constant regressor in the diagonal regression (the D e e' terms push to W' diag(g^2 D) W, an
identity-shaped form the three regressors cannot absorb) as a replacement of the lead's 14-literal table.
The K21 x covariance products of the complete law do not ride on the pairs (each needs g o K21[h, :] as a third vector per hub).

## T24 (2026-10-02 12:30Z). S40 SPEC: the fold, OP LIST for the native port (lead's decision of 12:36Z; reference module `ncar_fold_frozen_s2w.py`, level 'read')

**State added to the card:** one vector `w` of the bank's length, aligned with `lam` (same roster: extended with zeros wherever `lam` is extended at the
births; it rides through the push, the alive mask, the dead cut and the joins untouched, because those act on the columns). Before layer 1 it is all zero.

**In every layer l = 1..14, in the card's own order** (n = 1024; pp = the R columns pushed into the layer, n x R; lam = their weights as given to the
push; mu, v = diag(Cpre), C = Cpre of the layer; response rows / live columns = the read's own block):

1. READ (has an effect from layer 2 on; at layer 1 w = 0). Per response row i: f_i = strength x mu_i / (n v_i). Left factor of the read:
   left2_ir = pp_ir^2 - f_i pp_ir (w_r / lam_r), i.e. plane_ij = sum_r lam_r pp_ir^2 pp_jr - f_i sum_r w_r pp_ir pp_jr. Young columns and old tier alike
   (the old tier's weights are the leading entries of w). The diagonal of the plane is overwritten by the card afterwards (cpdiag), as now.
2. FORM, on the read's block (response rows i, live columns j): plane_ij += f_i x theta x a_d x tr(C) x C_ij.
   a_d = the coefficient of the first regressor in the least-squares regression, over the units i, of B_ii = sum_r w_r pp_ir^2 on the three per-unit
   vectors (tr(C) v_i, mu_i^2, 2 mu_i (C mu)_i). theta = 1.41. (Only the multiple of C is removed; the two mean-form regressors only serve the fit.)
3. END OF THE LAYER (after the pair map and the learned correction; m = the layer's final post mean, the card's out[0]): g_i = Phi(mu_i / sigma_i),
   pdf_i = phi(mu_i / sigma_i), sigma_i = sqrt(v_i); gbar = mean over all n units of (g_i - m_i pdf_i / sigma_i); V_i = 2 m_i (1 - g_i);
   c_r = lam_r x sum_i V_i pp_ir for the R columns of step 1; w_r <- 2 (gbar w_r + c_r) for those; the columns born in this layer get w = 0.

**Literals:** strength (1.0; the trainer's line search will set it, 1.25 was better than 1 on 35 and 651), theta (1.41), first / last layer of step 1 (2 / 14).
Step 3 runs at layers 1..14.

**Why these choices** (torch, no refit, S38 physics, strength 1, final-layer MSE; `t4_ora21/23/24.log`):

| rank | base (1e-9) | step 2 as specified (diagonal x 1.41, multiple of C) | exact removal of the multiple of C | three forms, diagonal x 1.41 | three forms exact | diagonal without the factor (three forms) |
|---|---|---|---|---|---|---|
| 35 (not held-out) | 13.0215 | -3.81% | -3.68% | -4.06% | -4.03% | -3.47% |
| 651 | 15.6222 | -4.51% | -4.53% | -3.98% | -4.16% | -3.27% |
| 378 | 13.7539 | -2.54% | -2.56% | -2.14% | -2.28% | -1.60% |
| 424 | 15.3993 | -1.62% | -1.71% | -2.18% | -2.04% | -1.43% |
| 328 | 15.9541 | -2.73% | -3.02% | -2.40% | -2.50% | -1.59% |
| 287 | 16.7235 | -4.36% | -4.45% | -3.80% | -3.93% | -3.08% |
| 641 | 17.5767 | -1.52% | -1.47% | -1.36% | -1.38% | -1.21% |
| six held-out, pooled | 95.030 | **-2.88%** (se 0.53, 6 of 6) | -2.96% | -2.64% | -2.71% | -2.03% |

(The bank channel as a separate remainder31 plane, four forms, all rows, on the same six: -3.27%.) Ratio of the off-diagonal to the diagonal
coefficient of tr C x C on the seven networks at layers 5 / 9 / 13: means 1.38 / 1.41 / 1.44, range 1.31-1.51; exact coefficient 0.0017-0.0020 /
0.0032-0.0039 / 0.0044-0.0054.

Nothing removed: +0.01% / +0.54% (35 / 651). Off-diagonal over diagonal coefficient of tr C x C at layers 5 / 9 / 13: 1.42 / 1.39 / 1.42 (35), 1.39 / 1.44 /
1.50 (651); the exact coefficient itself differs by 6-23% between the two networks, so a per-layer constant is not safe. The fold only into the pair
map's K21 (not the read) loses half the gain (the K21 target carries it into the star).

**Open details, being measured** (`t4_run_ora25.sh`, networks 35 and 651; each is a one-keyword change of the reference module, frozen s2x):
(i) step 2's regression over the response rows only instead of all units (the native left factor exists on the response rows); (ii) V replaced by the
card's own beta = 2 (gaussian_mean - host gate) alive (then step 3's dot products are the ones the omitted-radial source already computes);
(iii) the regression on tr(C) v_i alone; (iv) transport factor 0.9 / 1.1; (v) the same matrix's diagonal also in K22 and in the per-unit k4.

**Parity material:** per-layer predictions of the reference arms, `k4l_pred_rank{0035,0651,0378,...}_ora24.npz` (keys S38 = base, FRd1x14 = this spec,
FRd3x14, truth; arrays 16 x 1024). The engineer's native base differs from the torch base by its own parity gap (651: 15.240e-9 native against
15.622e-9 torch on the S38 file), so the target is the per-layer CHANGE of the prediction, as in the remainder31 parity test (a34_r31).

## T23 (2026-10-02 11:40-12:25Z). The corrected law in closed loop; the bank channel in native form; THE FOLD into the K21 bank read

All: torch, no refit, S38 physics (35: 13.0215e-9, 651: 15.6222e-9), final-layer MSE against the organizers' truth / all layers, one network per
number (single-network scatter between similar arms about 1 point on the final layer).

**Corrected law** (`ncarry.py` frozen s2r, dial ncown = the old own term; full matrix, strength 1, conversion from the bank columns; `t4_ora17.log`;
unit checks `t19v_unit.py` ALL PASS):

| arm | 35 | 651 | share of the oracle held, layers 2 / 5 / 9 / 13 / 14 (35) |
|---|---|---|---|
| complete law, certified row form | -7.30% / -8.59% | -16.46% / -8.52% | 0.89 0.92 0.89 0.85 0.81 |
| old own term at 0.5 | -6.65% / -8.56% | -16.84% / -8.87% | 0.86 0.91 0.88 0.85 0.81 |
| old own term at 1 (as run before 11:05Z) | -6.09% / -8.72% | -17.06% / -9.04% | 0.76 0.88 0.87 0.84 0.80 |
| lean law, certified | -7.01% / -8.26% | -16.22% / -8.41% | 0.89 0.91 0.87 0.81 0.75 |
| lean, state cut to rank 32 | -5.11% / -4.83% | -12.02% / -6.88% | 0.36 0.47 0.57 0.58 0.56 |
| lean, no Gaussian births | -8.32% / -8.29% | -14.40% / -6.65% | 0.83 0.84 0.82 0.79 0.76 |
| Monte Carlo oracle | -7.99% | -15.60% | 1 |

The fix is worth +1.2 points on 35 and -0.6 on 651 (pooled over the two: 0.25% better); dropping the births is +1.3 on 35 and -1.8 on 651: keep them.

**The bank channel in native form** (`ncar_bank.py`, the lead's carrier interface; frozen s2s; `t4_ora20.log`): a second weight per bank column,
w_r <- 2 (gbar w_r + lam_r (V . pp_r)) on the columns pushed into the layer, new columns at 0, matrix of a layer = sum_r w_r pp_r pp_r' on ITS
pushed columns. All columns: 35 -4.34% (matrix recursion -4.32%), 651 -5.64% (-5.70%): the bank's own transport, alive mask and joins change nothing.
Plane from the m columns with the largest |w_r| |pp_r|^2: m = 16 / 64 / 256 / 1024: 35 -0.96 / -1.14 / -1.13 / -3.01%; 651 +0.16 / +1.04 / -1.87%:
the value is spread over the columns (about 6,000 young + 9,000 old-tier coefficient columns), so no plane from a few columns.

**The fold** (`ncar_fold.py`, frozen s2t / s2v / s2w; `t4_ora21.log`, `t4_ora23.log`). To first order in the pair correlation the K31 term of the pair
map is (1/6) K31_ij e_i g_j, e_i = E relu''' = - alpha_i pdf_i / v_i, and the K21 term is (1/2) K21_ij d_i g_j, d_i = pdf_i / sigma_i; so the plane
(3/n) B_ij is an effective K21: K21_eff[i,j] = K21[i,j] - (mu_i / (n v_i)) B_ij. Certificate on the card's own pair map (layers 5 / 9 / 13, network
35): the two changes of the post covariance have cosine 0.998 and size ratio 0.997. Inside the bank read this is the left factor
pp_ir^2 lam_r -> pp_ir (pp_ir lam_r - (mu_i / (n v_i)) w_r): no product over the columns.

| arm (strength 1, all columns, response rows) | 35 | 651 |
|---|---|---|
| reference: plane through remainder31, four forms removed | -4.76% / -4.05% | -5.47% / -4.04% |
| fold INSIDE the read, four forms removed (exact off-diagonal least squares) | -4.54% / -4.02% | -4.48% / -3.54% |
| three forms (tr C x C, mu mu', mu (C mu)') | -4.03% / -3.86% | -4.16% / -3.48% |
| the multiple of C only | -3.68% / -3.48% | -4.53% / -3.48% |
| nothing removed | +0.01% / -1.86% | +0.54% / -1.35% |
| three forms, strength 1.25 / 0.75 | -4.26% / -3.42% | -4.46% / -3.57% |
| three forms, coefficients from the DIAGONAL of B | -3.47% / -3.74% | -3.27% / -3.34% |
| multiple of C only, coefficient from the diagonal | -3.33% / -3.46% | -3.76% / -3.42% |
| diagonal regression with a C^2 regressor | -2.43% | -2.46% |
| fold only into the K21 the pair map receives: four forms / multiple of C / nothing | -2.52 / -1.41 / +5.05% | -3.10 / -1.76 / +9.66% |

Readings. (1) The response rows are enough (on all rows the remainder31 route gives -4.34% / -5.64%). (2) The isotropic part must come out: the
card already carries it in `radial` (omitted-radial source). (3) The fold belongs in the READ: put only into the pair map's K21 the term reaches
the post K21 target and the star; inside the read it cancels against the inherited plane. (4) Coefficient of tr C x C in B: exact off-diagonal fit
0.0017 / 0.0032 / 0.0044 (35) and 0.0018 / 0.0039 / 0.0054 (651) at layers 5 / 9 / 13; the regression of the diagonal B_ii = sum_r w_r pp_ir^2 on
(tr C v_i, mu_i^2, 2 mu_i (C mu)_i) gives 1/1.39 .. 1/1.50 of it on both networks, so 'diagonal x 1.41' is the native candidate and a per-layer
constant is not (the two networks differ by 6-23%). (5) The dial sits above 1.

## T22 (2026-10-02 10:25-11:20Z). Held-out ranks, what a port must hold, the bank channel; and an ERROR in the law's row form

**The error (found 11:05Z, `t19u_cvown.py`, log `t19u_cvown_35.log`).** The mean-tilt row form of the law used (C V)_a = sum over ALL c of
C_ac V_c. The c = a part is already the own (3,1) term S3, so it was counted twice. The layer 0 -> 1 certificate of T19k was written with the sum
over c != a; the law rebuilt at depth (nlaw.py v1 / v2, ncarry.py up to frozen s2q, nlaw_block.py v1, and the lead's ncar3.py / ncar4.py, which
transcribe it) took the measured tilt, which includes c = a, and the layer 0 -> 1 check was not re-run on the rebuilt function. Numpy, network 35,
true auxiliary inputs, share of the true network-specific part held (cross-block):

| layer | 1 | 2 | 3 | 5 | 7 | 9 | 11 | 13 | 15 |
|---|---|---|---|---|---|---|---|---|---|
| free run, law as shipped | 0.60 | 0.83 | 0.86 | 0.91 | 0.91 | 0.91 | 0.92 | 0.89 | 0.81 |
| free run, row form over c != a | 0.98 | 0.97 | 0.96 | 0.96 | 0.94 | 0.94 | 0.93 | 0.90 | 0.82 |
| the same, lean law | 0.98 | 0.97 | 0.96 | 0.95 | 0.94 | 0.92 | 0.92 | 0.87 | 0.75 |
| the same, lean, Gaussian births dropped | 0.92 | 0.90 | 0.88 | 0.88 | 0.87 | 0.87 | 0.88 | 0.85 | 0.77 |
| one step from the true matrix, as shipped | | 0.92 | 0.95 | 0.96 | 0.96 | 0.97 | 0.97 | 0.96 | 0.93 |
| one step from the true matrix, fixed | | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.96 | 0.93 |

At layer 0 -> 1 the shipped law has 1.48 times the true energy with projection 1.04: the extra piece is orthogonal to the truth. Fixed as the
default in `nlaw.py` (keyword ownrow, 1 = old), `ncarry.py` (install(..., ownrow=0.0); frozen s2r), `nlaw_block.py`; the old files are kept
(`nlaw_v2.py`, `nlaw_v3.py` = the flag version t19u uses, `ncarry_v6.py`, `nlaw_block_v1.py`). EVERY closed-loop number below was measured
WITH the double count (ownrow = 1). The closed-loop value of the fixed law is the first thing to measure after the quiet window.

**Held-out ranks** (torch, no refit, final-layer MSE against the organizers' truth, base = the S38 physics; strength 1; `t4_ora10.log`,
`t4_ora15.log`, frozen s2n / s2p; oracle = Monte Carlo 2 x 4M of the same network, `t19_oracle_rank*.npz`):

| rank | base (1e-9) | bank channel alone | full-matrix carrier (ownrow 1) | all layers | Monte Carlo oracle |
|---|---|---|---|---|---|
| 651 | 15.6222 | -5.70% | -17.06% | -9.04% | -15.60% |
| 378 | 13.7539 | -2.81% | -6.03% | -5.89% | -4.88% |
| 424 | 15.3993 | -2.60% | -8.73% | -7.89% | -9.18% |
| 328 | 15.9541 | -3.29% | -3.95% | -6.28% | -4.65% |
| 287 | 16.7235 | -4.38% | -3.81% | -5.19% | -1.41% |
| 641 | 17.5767 | -0.98% | -4.48% | -8.07% | -3.61% |
| pooled (sum of MSE) | 95.030 | -3.27% | -7.25% | | -6.46% |
| per-network mean (se) | | -3.29% (0.66) | -7.34% (2.1) | | -6.56% (2.1) |

(Network 35, not held-out: bank channel -4.32%, full matrix -6.09%, oracle -7.99%.) The carried matrix holds 0.75-0.89 of the oracle's energy at
layers 2-14 on every network; the bank channel holds 0.24-0.28 at layers 2-5 and 0.01-0.13 at layers 13-14, and still delivers a third to all of
the full gain: the value is not proportional to the share of the matrix held.

**The bank channel** (`ncarry.py` skip 'convonly', frozen s2o+): N = g g' o (gbar Nt + K3V), K3V = sum_r lam_r (V . pp_r) pp_r pp_r' on the
pushed bank columns, and nothing else of the law (no births, tilts, own (3,1), K21 terms, gain). Natively it is a second weight per bank column,
w_r <- 2 (gbar w_r + c_r), with the old tier as a second r x r core; the card already carries the isotropic part of the same source in `radial`
(the omitted-radial source of covstep). Network 35: strength 1 -4.32% (all layers -3.87%), strength 0.75 -3.64%, plane with only the multiple of C
removed -3.47%. Variants (no transport, young tier only, cohort age, strength 1.5, state rank 32) are in `t4_run_ora16.sh`, not run yet.

**Port pricing, network 35** (`t4_ora11.log`, `t4_ora12.log`; strength 0.75; conversion from the bank columns; ownrow 1; base 13.0215e-9):

| arm | final layer | all layers | share of the oracle held, layers 2 / 5 / 9 / 13 / 14 |
|---|---|---|---|
| complete law, full matrix | -6.57% | -8.07% | 0.76 0.88 0.87 0.84 0.80 |
| LEAN law (no block product with K21), full matrix | -6.37% | -7.80% | 0.76 0.88 0.85 0.80 0.74 |
| lean, state = multiple of C + rank 48 by eigenvalue | -5.35% | -5.52% | 0.45 0.57 0.64 0.64 0.60 |
| lean, rank 32 | -5.05% | -4.56% | 0.35 0.47 0.56 0.57 0.55 |
| lean, rank 16 | -4.80% | -3.59% | 0.22 0.31 0.42 0.46 0.44 |
| complete law, rank 32 | -5.47% | -4.53% | 0.35 0.47 0.59 0.64 0.63 |
| lean, rank 32, plane with only the multiple of C removed | -5.42% | -4.69% | 0.35 0.45 0.53 0.48 0.45 |
| lean, law and plane from layer 4 | -2.93% | -3.49% | -- 0.39 0.77 0.76 0.72 |
| lean, from layer 6 | -1.57% | -1.62% | -- -- 0.60 0.71 0.68 |
| lean, rank 32, from layer 4 | -1.99% | -2.04% | -- 0.25 0.53 0.56 0.54 |
| lean without the Gaussian births | -7.60% | -7.99% | 0.80 0.86 0.83 0.79 0.76 |

Readings: the lean law costs 0.2 points; the 16 largest directions carry three quarters of the final-layer gain; the C^2 and mean forms need not be
removed in the plane; a late start loses the gain although the state recovers (layers 3-5 gain 3-9% each in the full-matrix arm and the final layer
inherits it). The 'no births' line is suspect: with the row form fixed the births are worth 4-8 points of fidelity at layers 1-11 (table above), so
the -7.60% is most likely a cancellation against the double-counted term. Rerun with ownrow 0.

**What the true matrix looks like** (numpy, networks 35 and 651 agree within 0.02; `t19r_fresh_rank.py`, `t19s_subspace.py`, tables by
`t19r_tab.py`): one step from the true matrix with the lean law, the FRESH part (law applied to the forms alone) is 85 / 64 / 53 / 46% of the next
layer's network-specific energy at layers 2-5 and 30-37% at layers 6-15; the exactly transported part 2 gbar W' (g g' o R) W is 57-66% at depth; the
rows driven by the carried diagonal are 0.0%. The fresh part is not low rank: its 16 / 32 / 48 largest directions hold 27 / 43 / 55% of its energy at
layer 2, 45-57 / 64-74 / 74-83% at layers 6-10, 64-73 / 79-85 / 86-90% at layers 12-15 (the truth itself: 27 / 43 / 55% -> 74 / 86 / 92%).
Free run with a sketched state (lean law, shares of the true network-specific part at layers 2 / 5 / 9 / 13): full matrix 0.83 / 0.91 / 0.90 / 0.85;
exact rank 32 0.38 / 0.50 / 0.62 / 0.63; two passes with 32 Gaussian probes 0.07 / 0.15 / 0.31 / 0.38; with 16 probes 0.02 / 0.08 / 0.18 / 0.26;
single-pass Nystrom diverges. One-sided form P R + R P - P R P on the TRUE matrix of layer l+1 (share of off-diagonal energy, layers 4 / 8 / 13,
network 35): best rank 16 0.39 / 0.58 / 0.71; partner form on 16 covariance eigenvectors 0.41 / 0.63 / 0.76; on the pushed top-16 directions of the
layer before 0.32 / 0.53 / 0.69; on the union 0.55 / 0.74 / 0.84, cut back to rank 16 0.36 / 0.56 / 0.70 (rank 32: best 0.57 / 0.75 / 0.85, union
cut 0.53 / 0.73 / 0.84). So a rank-r state can be computed with r + 16 applications of the law per layer and no random probes; its closed-loop
arms (`ncarry.py` option oneside, `t4_run_ora13.sh`) are queued.

**Cost notes** (bill 2.6e11 flops; one n^3 product about 0.3% of it; one application of the lean law to one vector about 5.75 block products =
1.2e7 flops, 0.065% of the bill over 14 layers): full-matrix lean carrier 2.4-4.4 products per layer = 10-18% of the bill; one-sided rank 16 with
16 eigenvectors (32 applications per layer) about 2.1%; the bank channel well under 1% if the plane is formed from the old tier's core and the
young columns.

## T21 (2026-10-02 09:55-10:15Z). A CARRIER of the second-chaos matrix inside the card, no Monte Carlo: -6% on network 35 (held-out ranks running)

**Free run of the law on true auxiliary inputs** (`nlaw.py`, `t19p_freerun.py 35`, log `t19p_freerun_35.log`; `t19m_mc_law.py` = an independent
Monte Carlo stream of the K21 planes and of the conversion matrix sum_c V_c kappa3(a,b,c); the matrix itself starts at 0 at layer 0 and is never
taken from Monte Carlo). Share of the true network-specific part held (truth cross-block energy), network 35:

| layer | 2 | 5 | 9 | 13 | 15 |
|---|---|---|---|---|---|
| all auxiliary inputs | 0.83 | 0.91 | 0.91 | 0.88 | 0.80 |
| without the conversion (and the K31 contraction) | 0.64 | 0.70 | 0.71 | 0.75 | 0.67 |
| without K21 as well | 0.32 | 0.31 | 0.31 | 0.40 | 0.30 |
| one step from the true matrix (cross-block) | 0.92 | 0.96 | 0.97 | 0.96 | 0.92 |

The isotropic level comes out too (coefficient of tr C x C 0.0036 -> 0.0207 against 0.0032 -> 0.0222). The V-contraction of K31 adds nothing
(conversion alone 0.85 / 0.81 at layers 14 / 15). The matrix closure of the transport (gbar x Ntilde for the gamma-weighted contraction) holds
99.3-99.6% of the exact term at every layer (`t19n_law_depth.py 35`); the mean-direction matrix adds nothing. No error accumulation.

**In the card** (`ncarry.py`, frozen `ncarry_frozen_s2k.py`; arms nc / ncv / ncr / ncskip of `e16b_f4b_frozen_s2{j,k}.py`; `t4_ora6.log`, `t4_ora7.log`):
the same law on the card's own covariance, means and K21; conversion = sum_r lam_r (V . pp_r) pp_r pp_r' on the pushed bank columns
(+ optionally the b mode C_ab (b.V) + (C V)_a b_b + (C V)_b b_a and the star pushed exactly); state = the full n x n matrix, 0 at layer 0; use =
the K31 extra plane (3/n) R (four forms removed) through remainder31 at layers 1-14. Network 35, S38 physics 13.0215e-9, torch, no refit:

| arm | final layer | all layers | share of the oracle matrix held, layers 3-14 |
|---|---|---|---|
| conversion from the bank columns, strength 1 | -6.09% | -8.72% | 0.82-0.88 (0.76 at layer 2, 0.80 at 14) |
| + b mode / + star | -5.69% / -5.71% | -8.72% / -8.76% | 0.81-0.87 |
| all three, rank 32 / strength 0.75 | -5.74% / -6.37% | -6.51% / -8.14% | |
| no conversion, strength 1 / 0.75 / rank 32 | -2.08% / -3.12% / -2.65% | -4.93% / -4.75% / -3.65% | 0.63-0.70 |
| no conversion, no K21 terms | +1.65% | -1.19% | 0.22-0.40 |
| no conversion, no homogeneity term | -1.47% | -4.18% | 0.59 falling to 0.37 |
| Monte Carlo oracle | -7.99% | -9.01% | 1 |

Value against fidelity (substitutes built from the truth with the old incomplete law, `t4_ora3.log`, network 35 / 651): one step from truth
(0.33-0.67 held) -8.1% / -14.0%; two steps (0.21-0.37) -5.2% / -12.3%; the old free run (negative share) +3.3% / -2.6%.
OPEN: held-out ranks (queued: 651 with the same arms, then 378 424 328 287 641); the cheap core for a native form (`t4_run_ora8.sh`).

## T20 (2026-10-02 09:05-09:55Z). CLOSED-LOOP CEILING of the second-chaos matrix: the K31 extra plane is worth -8% / -16%; the one-layer law that makes it

**Oracle arms** (lead's order of 09:01Z). `t19g_mc_all.py` (GPU, true Ntilde at layers 1-15, 2 x 4M samples, exact-mean centring) ->
`t19h_oracle.py` (network-specific part R_l by the lead's E32 definition; block agreement 0.55 at layer 1, 0.82-0.90 at layers 2-4,
0.92-0.98 at layers 5-15) -> `ntor.py` injected into the torch card (`e16b_f4b.py` options nt4, nt31, ntf, ntr, nto, ntx). Base = the
S38 physics on the research card (F4 0.656 / 0.6, K4L 0.75, learned x 0.8, k21n -0.048, b-source 0.965, X1 0.59158, bscale 1.03414):
13.0215e-9 on network 35, 15.6222e-9 on network 651. Final-layer MSE against the base, network 35 / network 651
(`t4_ora.log`, `t4_ora2.log`, `t4_ora4.log`; `k4l_pred_rank00{35,651}_ora*.npz`):

| use of the true matrix | full | rank 64 | rank 32 | rank 16 | rank 8 |
|---|---|---|---|---|---|
| K31 extra plane (3/n) R through remainder31 | -7.99% / -15.60% | -7.27% / -14.72% | -6.07% / -15.03% | -3.40% / -14.12% | -4.73% / -8.59% |
| K31 + feed plane | -8.67% / -14.80% | -8.02% / -14.20% | -7.37% / -14.08% | | |
| feed plane alone (into the next layer's K21) | -1.09% / +2.15% | -1.52% / +1.41% | -1.77% / +1.99% | | |
| per-unit k4 alone, on top of K4L | +3.23% / -0.95% | +2.50% / +1.88% | +1.36% / -1.40% | | |
| all three | -2.26% / -14.80% | -3.97% / -12.00% | -5.15% / -14.46% | | |

- K31 strength 0.5 / 0.75 / 1.25 (full rank): -7.73% / -12.08%; -8.89% / -14.79%; -5.56% / -14.45%. Half strength keeps most of it.
- k31 knob at 0 under the oracle: -7.91% / -15.25% (the knob does not matter once the plane is there). Learned correction off
  (network 35): -12.13% full, -10.73% rank 64, against the base with learned off.
- Per-unit k4 with K4L off: -0.59% / -2.71%, where K4L itself is worth -4.0% / -0.5%: do not use the matrix per unit (lead's E33 agrees).
- FORMS closed: the card's isotropic K31 replaced by (3/n) x the four true forms: +10.8% / +10.3% (nine forms +9.8% / +9.4%); the
  full true matrix in its place -2.3% / -7.8%; k31 knob at 0 alone +1.1% / +1.0%. Add R to the card's own form, never replace it.
- VALUE AGAINST FIDELITY (network 35, `t4_ora3.log`; substitutes of R built by `t19i_recursion.py` with an incomplete law): one step
  from truth, 0.33-0.67 of R held: -7.9..-8.1% (the full value); two steps, 0.21-0.37 held: -5.2%; a free run that goes negative: +3.3%.
  So half fidelity is enough; a wrong matrix hurts.

**The one-layer law** (`nlaw.py`; `t19k_law_cert.py`, `t19n0_law_depth_partial.py`). Ntilde(l+1) = 2 W' N W + 2 kappa C(l+1), N built at
layer l from the covariance C, the mean, the post mean m, the K21 plane and the matrix itself (g, d, gamma = g - m d, V = 2 m (1 - g),
S2 = 2 (gamma - g^2), S3, S4, e = -alpha phi / v as in the K4L spec):
  P = gbar Ntilde + 2 C Gamma C (tilt of the covariance by the norm of the outputs), kappa = <P, C> / <C, C> (its pure-gain part)
  N_ab = g_a g_b [gbar Ntilde + C S2 C]_ab + C_ab [d_a g_b (C V)_a + (e_a/2) g_b Pv_a] + transpose + C_ab (g_b S3_a + g_a S3_b)
         + (third cumulant x covariance terms from K21) - kappa g_a g_b C_ab,   Pv_a = gbar Ntilde_aa + 2 sum_c gamma_c C_ac^2
  N_aa = S4_a + V_a ((C V)_a - V_a v_a) + sum_c (2 gamma_a gamma_c - 2 g_a^2 g_c^2) C_ac^2 + gamma_a gbar Ntilde_aa + (K21 terms) - kappa VarG_a
- Layer 0 -> 1 (Gaussian inputs, so only births, row forms, own (3,1), S4 and the (2,2) row sums): 99.4% of the whole matrix and
  96.9% of its network-specific part on network 35 (99.3% / 96.3% on 651), nothing fitted. Cumulative on the network-specific part:
  5%, 35%, -78%, 28%, 97%: the terms cancel, a subset is worse than nothing.
- At depth, one step from truth, nothing fitted (transitions 5->6 / 9->10 / 11->12 / 13->14, network 35): 0.88 / 0.93 / 0.95 / 0.92;
  all nine coefficients fitted 0.89 / 0.94 / 0.95 / 0.93 (transport by the closure gbar x Ntilde: fitted 1.00). Without the homogeneity
  term (- kappa g C g: the pure-gain part of the tilt maps to kappa x post covariance, a form of the next layer) 0.75 / 0.75 / 0.72 /
  0.67; without the variance tilt 0.59 / 0.66 / 0.70 / 0.70; without the diagonal terms 0.0-0.2.

## T19 (2026-10-02 08:43-09:00Z). The SECOND-CHAOS matrix of the fourth cumulant (lead's 08:25Z diagnostic): predicted coefficients right, share a third to a half

For iid Gaussian weights the fourth cumulant of the next layer splits by Wiener chaos in the weight rows. Its second-chaos part is one
matrix, in pre-activation space Ntilde_ij = sum_k kappa4(z_i, z_j, z_k, z_k) = Cov(z_i z_j, |z|^2) - 2 (C^2)_ij, with the predictions
k4_i = const + (6/n) Ntilde_ii, K31_ij = (3/n) Ntilde_ij, feed kappa4(z_i, z_j, z_k, z_k) = Ntilde_ij / n for every hub k (isotropic model:
Ntilde = (radial/3)(tr C x C + 2 C^2)). Measured on network 35: `t19_second_chaos_mc.py` (GPU through the sharing rule, 249 s, 2 x 8M
samples = the stream and blocks of `t2_mc.py`; full matrices at layers 5, 9, 11, 13, 15 in `t19mc_rank0035.npz`, also the mean-direction
version kappa4(z_i, z_j, p, p)). Every score pairs a predictor from one block with a truth from the other block (K31: `t2mc` M31; feed:
`t2_budget` planes; per unit: the independent 32M draw). Analysis `t19b_analysis.py`, `t19c_feed_combined.py`, `t19d_spectrum.py`,
`t19e_perunit_combined.py`, `t19f_vector_form.py` (numpy; logs of the same names).

| quantity | layer 5 | 9 | 11 | 13 | 15 |
|---|---|---|---|---|---|
| K31 on kink rows, R2 (T12 convention): card's isotropic v_i C_ij | 0.41 | 0.38 | 0.35 | 0.31 | |
| K31: (3/n) Ntilde_ij, nothing fitted (fitted coefficient / (3/n)) | 0.52 (0.97) | 0.53 (0.92) | 0.50 (0.90) | 0.50 (0.92) | |
| K31: noise ceiling | 0.87 | 0.96 | 0.97 | 0.98 | |
| feed share: isotropic C_ij v_k | 0.27 | 0.21 | 0.23 | 0.21 | |
| feed share: Ntilde_ij v_k / tr C, nothing fitted (fitted coefficient) | 0.34 (0.97) | 0.30 (0.93) | 0.33 (0.93) | 0.33 (0.94) | |
| per-unit k4: share of the unit-specific part explained by Q_i at 6/n (fitted coefficient x n) | 39% (5.3) | 42% (5.9) | 39% (5.5) | 36% (5.4) | 28% (5.1) |

- Per unit: "unit-specific part" = residual of the true k4 after 1, v, v^2 (relative rms 0.12 / 0.18 / 0.19 / 0.23 / 0.27 of k4). Against
  v^2 alone Q reads 22-27% with a fitted coefficient of 1.5-1.9 / n: the isotropic content's own second chaos is linear in v, so v^2 alone
  is the wrong reference. The mean-direction contraction adds 0-2 points; the mean-direction matrix adds 0.00-0.01 to K31 and the feed.
- With the hub-local terms (layers 9 / 13). Feed: isotropic + exact hub-local (all ages) 0.34 / 0.34; matrix + hub-local 0.40 / 0.41
  (hub-local coefficient 0.72-0.77 instead of 0.89; the lead's collective scalar adds 0.00 once the matrix is in). So the exact matrix
  is worth +6-7 points of feed beyond F4-type terms. Per-unit kurtosis (layers 5 / 9 / 13 / 15): hub-local (all hubs, age 1) 12 / 5 /
  10 / 1%, Q 39 / 42 / 36 / 29%, both 40 / 43 / 39 / 29%: Q contains the hub-local part.
- The free column weights (T4 ceiling, one weight per bank column, fitted with the isotropic term) hold MORE of the feed than matrix +
  hub-local: 0.49 / 0.60 against 0.40 / 0.41, and the gap grows with depth. Their feed plane is nearly orthogonal to the matrix's
  (cosine 0.05-0.06). The largest unexplained part of the feed is carried content along the bank's own columns, not the matrix.
- Spectrum. Ntilde = a tr C x C + b C^2 + D; D holds 16-20% of the energy (block agreement 0.97-0.99; at layers 9-15 the fitted C^2 coefficient is ten
  times the isotropic model's, i.e. the top mode of C is over-weighted in the true matrix). D is not diagonal plus low rank (2-3% of its
  energy on the diagonal; off-diagonal energy kept by rank 1 / 4 / 16 / 64: 23-27% / 33-58% / 55-78% / 85-95%, lower rank at depth; top
  eigenvector = the mean direction, cosine 0.75-0.90) and not a unit-loading form ((r_i + r_j) C_ij explains 5-15% of D).
  Rank r at layer 13: K31 R2 0.26 (isotropic fit) / 0.39 (r = 4) / 0.44 (16) / 0.48 (64) / 0.50 (full); feed 0.21 / 0.27 / 0.30 / 0.33 / 0.33.
- Reading. The theory is right (all three coefficients within 3-10% of the prediction with nothing fitted), but the matrix is a third
  to a half of the non-isotropic fourth-order content, not most of it. Ceiling of an exact full-matrix carrier: K31 0.35 -> 0.5, feed
  +6-7 points, a third of the unit-specific kurtosis, for the price of a covariance push. A rank-16 piece keeps two thirds of the K31 and
  feed gains for 16 pushed columns but needs sources nobody has (lead's E31c: the one-layer conversion explains 0.2-10% of Q).

## T17 (2026-10-02 08:16Z). What the knob constant k21n stands for: the K21 term with the skew-corrected density at the kink

No card run; per-unit truth of networks 35 and 100 (32M samples), `t17_k21n_reading.py 35 100`, `t17_k21n_reading.log`.
The covariance knob hook multiplies K21 by (1 + k21n) on the rows with |alpha| >= 1.5 (parent -0.028, the trainer's line search -0.048)
and leaves the kink rows alone. The card's K21 term of the covariance is (1/2) K21[i,j] E_G relu''(x_i) E_G relu'(x_j) with GAUSSIAN
expectations. At the next order (parts a and b' of T15) the two expectations take their skew corrections:
  E relu''(x_i) = (phi_i / sigma_i) (1 - (t3_i / 6) He3(alpha_i)),   E relu'(x_j) = gate_j + (t3_j / 6) He2(alpha_j) phi_j,   t3 = k3 / sigma^3.
Size of the row factor -(t3/6) He3(alpha), weighted by the row's weight phi/sigma in the K21 term (network 35 / 100):

| layer | t3 rms | corr(t3, alpha) over units | rows |alpha| >= 1.5: mean, spread | kink rows: mean, spread | column factor: mean, spread |
|---|---|---|---|---|---|
| 2 | 0.016 / 0.015 | 0.84 / 0.84 | -0.3% / -0.2%, 1.2% | +0.2%, 0.3% | 0.0%, 0.2% |
| 5 | 0.037 / 0.038 | 0.91 / 0.92 | -1.1% / -1.1%, 3.3% | +0.3%, 0.4% | 0.0%, 0.5% |
| 8 | 0.060 / 0.063 | 0.94 / 0.95 | -2.0% / -2.5%, 5-6% | +0.4%, 0.6% | 0.0%, 0.8% |
| 11 | 0.078 / 0.080 | 0.94 / 0.95 | -2.4% / -2.3%, 6.6% | +0.5%, 0.7% | 0.0%, 0.9% |
| 13 | 0.090 / 0.097 | 0.94 / 0.96 | -2.6% / -2.5%, 7-7.5% | +0.5%, 0.9% | 0.0%, 0.9% |
| 15 | 0.102 / 0.110 | 0.94 / 0.95 | -3.1% / -3.4%, 8.1-8.5% | +0.5%, 0.9-1.0% | 0.0%, 1.0% |

Reading. (1) The constant is this term's mean: same rows, same sign, same size (-2..-3% at layers 8-15, where the knob acts; nothing
to speak of on the kink rows, which the knob leaves alone). The trainer's -0.048 is larger than the term's mean at every layer.
(2) The term is not a constant: around its mean it varies from row to row by 5-8% (more than twice its mean), and it grows with depth
(-0.3% at layer 2, -3% at layer 15), following t3^2. A constant can hold only the mean. (3) The per-unit skewness is almost a
multiple of alpha (correlation 0.94 over the 1,024 units from layer 8 on: the skewness is the collective mode, k3_i = 3 v_i (W'b)_i with
W'b nearly parallel to the pre-activation mean), so the row factor is close to -(c/6)(alpha^4 - 3 alpha^2) with one number c per layer
and network (c = t3 rms / alpha rms): negative beyond |alpha| = 1.73, positive inside. That is what "computed per layer and network
instead of a constant" would be; the closed-loop arms SCl (term on, knob kept) and SCn (term on, knob 0) of T15 test it.
(4) The other two constants are NOT second-order terms: iso (-0.094 x (k3_i / v_i) C_ij on kink rows) changes a first-order
coefficient by 14-28% ((alpha^2 + 1)/6 -> (alpha^2 + 0.72)/6), where every second-order term of that shape is below 1%; k31 (-0.212) is the
level of the isotropic K31 model (E17 / T12: the fitted isotropic coefficient is 0.76-0.81 x radial).

## T16 (2026-10-02 08:13Z). Open loop, true inputs: the per-unit mean FORMULA is exact to 0.2e-9; every term beyond k4 is worth under 1%

`t16_mean_orders.py 35` (numpy only), `t16_mean_orders_rank0035.log`. The mean formula evaluated with the TRUE per-unit pre-activation
mean, variance and cumulants (two independent 32M-sample draws; 5th and 6th cumulants from the stored raw moments), against the
organizers' post means, network 35. Mean squared error over the 1,024 units, x 1e-9:

| layer | Gaussian | + k3 | + k4 = the card's formula (one draw / two-draw product) | + k3^2 | + k5 | + k3 k4 | + k3^3 | + k6 | card's own error |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 259 | 30.9 | 4.83 / 0.09 | -0.017 | +0.024 | -0.025 | -0.001 | -0.009 | 8 |
| 11 | 220 | 27.0 | 3.29 / 0.16 | -0.051 | +0.008 | -0.031 | -0.003 | +0.005 | 11 |
| 13 | 239 | 24.8 | 2.82 / 0.21 | -0.102 | +0.048 | -0.092 | +0.001 | -0.027 | 12 |
| 14 | 194 | 20.2 | 2.30 / 0.02 | -0.036 | +0.040 | -0.071 | -0.001 | -0.004 | 13 |
| 15 | 177 | 19.7 | 2.03 / 0.00 | -0.047 | +0.036 | -0.069 | -0.002 | +0.008 | 14.1 |

(columns after the card's formula: paired change of the one-draw error when that term is added on top of all terms to its left; the
one-draw value of the card's formula, 2-5e-9, is the noise of the 32M inputs; the two-draw product removes it and is itself uncertain
by about 0.1e-9 at layers 8-15 and by 0.5-1e-9 at layers 1-7, where it comes out between -1.5 and +0.6e-9: the noise of the sample
means lives in a few collective modes, so a product over 1,024 units averages much less than 1,024 independent numbers.)
- With true inputs the card's formula (Gaussian + k3 + k4) is as good as the direct Monte Carlo mean of the same samples: nothing
  measurable is left at layers 14-15 (0.00-0.02e-9), 0.1-0.2e-9 at layers 8-13. The whole final-layer error of the card (14e-9) is
  INPUT error: the per-unit mean of the layer before, variance, third and fourth cumulant.
- The missing second-order term k3^2 He4(alpha) phi / (72 sigma^5) is worth 0.02-0.10e-9 with true inputs (0.047e-9 at layer 15 = 0.3% of the
  card's error; rms of the term 10.5e-6 against an error rms of 119e-6). The third-order cross term -k3 k4 He5(alpha) phi / (144 sigma^6) is
  worth as much (0.03-0.09e-9; 0.069e-9 at layer 15) and is also elementwise on what the card holds; k5 would be needed with it for
  consistency and makes things slightly worse on its own. Rms of the single terms at layer 15, x 1e-6: k3 426, k4 145, k3^2 10.5,
  k5 9.3, k3 k4 8.1, k6 4.1, k4^2 2.9, k3^3 0.8.
- Consequence for the second-order mean term as a board lever: its ceiling is about 0.3-0.7% per layer where it acts, before the learned
  correction (which sees alpha and t3 and can hold a function of them) has taken its share. The lead's -2.2% (learned off) / +0.8% (on)
  on rank 651 is a closed-loop number over 15 layers; see T15 for six held-out ranks.

## T14 (2026-10-02 05:35-06:15Z). CLEAN CONFIRMATION of the mixture on 24 held-out ranks not used for any fit: -0.46%; 48 ranks: -0.48% [-0.83, -0.12]

Ranks 25-48 of `O/trainer/cert/runB_val.txt` (474 648 633 30 1 588 673 302 303 11 324 539 579 272 480 321 439 593 505 652 455 438 599 410),
torch, learned on, no refit, final-layer MSE against the organizers' truth, real closed-loop runs, paired, bootstrap over networks
(`t4_run_fr48.sh`, `k4l_pred_rank*_fr48.npz`, `t4_fr48.py fr48 MBK FK`). Mixture = F4-rule 0.378 + exact S1 0.858 / S2 0.48 in the
birth-constant form of T13, K4L 0.71; reference = F4 0.58 + K4L 0.73 (the best strengths without the exact terms).

| arm | pooled | 95% | better | per-network mean (se) | all-layer |
|---|---|---|---|---|---|
| F4 0.58 + K4L 0.73 | -5.77% | -6.79, -4.73 | 24/24 | -5.64% (0.53) | -5.75% |
| mixture + K4L 0.71 | -6.20% | -7.35, -5.06 | 24/24 | -6.10% (0.58) | -5.92% |
| mixture against F4 + K4L, these 24 ranks | -0.46% | -0.92, +0.04 | 17/24 | -0.49% (0.24) | -0.18% |
| mixture against F4 + K4L, first 24 ranks (T7; fitted there) | -0.51% | -1.02, +0.06 | 17/24 | -0.49% (0.28) | |
| both sets, 48 ranks | **-0.48%** | -0.83, -0.12 | 34/48 | -0.49% (0.18) | |

## T13 (2026-10-02 04:30-05:45Z). F4c = the mixture in its cheap native form; ages, own-square form, certificates

**What the mixture is.** F4's rule at theta = 0.378 (unchanged step) plus the exact S1 and S2 of T6 at 0.48 each (so S1 0.86 in
total), K4L at 0.71. `f4b.py`: mode='mix', cs=(0.48, 0.48, 0, 0), ths=(0.378,) * 4 [the literal mixture; the held-out runs of T7 used
cs=(0.858, 0.48, 0, 0), ths=(0.378,) * 3, which differs by 0.2% in the S1 strength].
**Birth metric = pair constants** (`f4b.py` metric='birth'; `t4_parity_fr24g.log`, `k4l_pred_rank*_fr24g.npz`). The 2x2 least-squares solve
of T6 needs the ratio |gY|^2 : |gWt|^2 of the pair's two gated legs; replacing it by |s|^2 : 1 (s = the pair's common leg at its birth,
|s|^2 = sum of squares of the + column minus t+^2) and dropping the cross term makes the six numbers that turn (Y^2, Y Wt, Wt^2) into the
two shifts constants of the pair. Network 35: -7.12% against -7.14% (cosine of the prediction changes 0.9999, rms difference 1.2% of the
change). Eight held-out ranks (651 378 424 328 287 641 548 557): pooled -5.79% against -5.79%, cosine 0.9997-1.0000 on every rank.
**Native reference** `f4c_native_ref.py 35 2` (plain array operations; `exact_constants` at birth, `f4c_step` per layer): predictions
bit-identical to `f4b.py` (max abs difference 0.0; 13.1036e-9 on network 35). Per layer and cohort, with p, q the pushed + and - columns
on the live rows and D = p - q:
  shift_p = (d/2) [theta A p^2 + (n_p0 p^2 + n_p1 p D + n_p2 D^2) / lam_p],  shift_q = (d/2) [theta A q^2 + (n_m0 p^2 + n_m1 p D + n_m2 D^2) / lam_q]
  anchor as in F4 with X = lam o shift:  beta = [X @ (v @ (u o u)) + 2 u @ (v @ (u o X))] / sum v^2,  diag = 3 sum_r u_r^2 X_r.
My bill count: since q^2 = p^2 - 2 p D + D^2, rule and exact part are ONE quadratic form per column: about 41 flops per pair element
(18 for the shifts, 23 for the generic anchor passes) against F4's 32, i.e. about +0.06% of the bill if the F4 step is rewritten this way,
about +0.25% if the exact part is bolted on beside the existing F4 step. The older general form (`f4b_native_ref.py`, per-layer metric)
also matches f4b.py (2.4e-7 max, rounding level).
**Ages and the own-square form** (24 held-out ranks, real runs of single arms + the leave-one-network-out mixture model that matched
five real runs within 0.2%; `k4l_pred_rank*_fr24e.npz`, `t9_mix.py`):

| arms in the mixture (fitted scales) | left out | 95% |
|---|---|---|
| F4 x 1.46 + K4L x 0.73 | -5.46% | -6.24, -4.67 |
| F4 x 1.03 + exact S1+S2, ages 1-2, x 0.45 + K4L x 0.71 | -6.00% | -6.76, -5.24 |
| F4 x 1.02 + exact S1+S2, ages 1-4, x 0.44 + K4L x 0.71 | -6.36% | -7.13, -5.59 |
| without K4L: F4 x 1.41 / + exact ages 1-2 / + exact ages 1-4 | -3.84% / -4.45% / -4.82% | |
| without K4L: F4 x 0.56 + exact in OWN-SQUARE form x 0.57 | -4.29% | -5.03, -3.53 |

Exact S1+S2 at ages 1-4, strength 1, alone: real run -3.17% (ages 1-2: -3.36%). Own-square form (each column shifts along its own square
only, two scalars per pair by weighted least squares: the native F4 step unchanged; parity with f4.py 13.5072 against 13.5076e-9) carries
about three quarters of the exact terms' gain in the mixture but needs the per-layer metric; the birth-constant general form is the better
port. Ages 3-4 add about 0.36% for about twice the step's bill.
**Gradients** (`f4b_grad_cert.py 35 2`, `f4b_fd_steps.py 35 2`): with c1, c2 and the rule strength as float64 leaves the checkpointed
backward pass runs in 30 s and gives finite gradients (-0.20, -0.34, +0.90 e-9 per unit at the mixture point). A finite difference is
NOT a usable reference on one network: at steps 0.005 / 0.02 / 0.08 it gives -0.55 / -0.47 / -0.32 (c1), +0.01 / -0.08 / -0.35 (c2),
+0.01 / +0.70 / -0.06 (rule strength): the final-layer MSE of one network jumps with the card's discrete selections. A refit has to
check the module's gradients the way the trainer checks its own (pooled over networks). The module detaches the birth data (S1..S4,
t+-, A, |s|^2) as f4.py does.

## T12 (2026-10-02 04:10Z). Open loop: the isotropic K31 is NOT improved by taking the gain mode out of the covariance. CLOSED

`t12_k31_gain.py 35 5,9,12`. If the pre-activations were a scale mixture z = (1 + eps) h with a collective gain fluctuation eps
(variance s2) independent of h, then Cov z = (1 + s2) C_h + s2 mu mu' and the isotropic fourth cumulant would be 4 s2 Sym3(C_h C_h)
+ 3 s2 sum_a mu_a K3_h(rest): built from the covariance WITHOUT its gain-mode part, plus mean x third-cumulant terms (derivation by
the law of total cumulance; the mu mu C terms cancel at order s2). The card builds K31 = radial v_i C_ij from the full C. Facts on
the 8M truth planes, kink rows, off-diagonal, fitted on one draw and scored on the other (network 35, layers 5 / 9 / 12):
top covariance eigenvector = the mean direction (cosine 0.81 / 0.95 / 0.96), s2 = lambda_1 / |mu|^2 = 13.6 / 14.8 / 16.3 in units of
1/(2n), the rank-one part s2 mu mu' holds 7 / 22 / 33% of the off-diagonal energy of C.
R2 of the true K31: card form v_i C_ij 0.409 / 0.386 / 0.338; top mode and the rest of C with separate coefficients 0.411 / 0.395 /
0.343 (the top-mode part wants a LARGER coefficient, 1.3-1.7x, not zero); + v_i mu_i mu_j, mu_i^2 C_ij, mu_i^3 mu_j 0.417 / 0.400 /
0.354; + mu_i K21[i,i,j] and mu_j k3_i 0.438 / 0.405 / 0.360 (noise ceiling 0.87 / 0.96 / 0.98). So 60% of the true K31 at kink
rows is network-specific fourth-order content that no rearrangement of (mu, C, K21) reaches. Also: the card's top covariance
eigenvalue is 0.14-0.23% below the 32M truth at layers 5-14, and its per-unit variance error is not a gain-amplitude error
(regression on [1, v, mu^2]: R2 4-8%).

## T11 (2026-10-02 03:15Z). Open loop: the tree-level NON-LOCAL part of the per-unit kurtosis is small. CLOSED

`t11_k4_nonlocal.py 35 0,4,8,12,14`. For z'_i = sum_a W_ia relu(z_a), z Gaussian (the card's mu, C at layer l), the fourth cumulant of
z'_i at tree level is  local (single hub: S4 W^4 + 4 S3 W^3 Y + 6 S2 W^2 Y^2 + 4 S1 W Y^3, all hubs) + 3 sum_{e != f} zeta_i(e) C_ef zeta_i(f),
zeta_i(e) = W_ie^2 beta_e + 2 d_e W_ie Y_e(i) = d v'_i / d mu_e (the three two-hub classes add up to this one square: the kurtosis made by
the fluctuation of the conditional variance; costs one dense product). Certificate = layer 0 -> 1, where the inputs are exactly
Gaussian: isotropic + local + non-local explains 100% of the true per-unit k4 of layer 1 (32M truth, two draws), fitted coefficients
1.018 (local) and 1.031 (non-local), isotropic coefficient 0.0003: the theory is exact there, and k4 at layer 1 is 98% local.
At later layers (weights = the kurtosis weight of the mean formula; share of the best-isotropic residual energy explained):

| k4 at layer | card rel. rms error | isotropic + local (age 1, all hubs) | + non-local | fitted coefficients (local, non-local) |
|---|---|---|---|---|
| 5 | 12.8% | 0.195 | 0.256 | 1.04, 0.90 |
| 9 | 19.0% | 0.107 | 0.142 | 0.99, 1.14 |
| 13 | 24.7% | 0.169 | 0.178 | 1.47, 0.61 |
| 15 | 29.4% | 0.044 | 0.051 | 0.80, 0.81 |

The non-local term is right (coefficient about 1) but small (0.03-0.04 of the rms of k4): at late layers the unit-specific kurtosis is
content made many layers earlier and carried, not content made by the last layer. Not worth a dense product, at any layer.

## T8 (2026-10-02 03:20-04:00Z). WHERE the final-layer error is made: budget by layer and per-unit input of the mean formula

`t8_budget.py 35 3 off on=0` (and `offF`: with F4b S1+S2; `on`), `t8_analysis.py 35 <tag>`, data `t8_budget_rank0035_<tag>.npz`. The
analyst's A6 oracle (an input of the per-unit mean formula replaced by 32M truth) done at ONE layer at a time, keeping the final
prediction vector, so that contributions can be added and projected. Network 35, learned mean correction OFF (16.417e-9):

| layers | share of the squared final error: variance | skew | kurtosis | all three |
|---|---|---|---|---|
| 15 | 7.9% | 6.8% | 7.6% | 22.3% |
| 14 | 4.2% | 5.5% | 4.3% | 14.0% |
| 13 | 7.7% | 4.9% | 1.0% | 13.7% |
| 12 | 4.4% | 0.3% | 5.4% | 10.1% |
| 9-11 | 4.6% | 6.3% | 2.6% | 13.4% |
| 1-8 | 3.3% | 2.7% | 0.6% | 7.0% |
| all | 32.1% | 26.6% | 21.8% | 80.5% |

(share = minus the contribution projected on the error, over the squared error; the variance arms use two independent truths.)
- Half of the squared final error is made by the three inputs at layers 13-15, 22% by the last layer alone. Variance, skew and kurtosis
  errors weigh about the same with the learned correction off.
- Contributions of neighbouring layers are uncorrelated (cosines -0.15..+0.17), so they add in quadrature; across inputs the totals
  partly cancel (2 v.s -12%, 2 v.k -7%, 2 s.k -10% of the squared error).
- Single-layer contributions add up to the every-layer arm for skew and kurtosis (cosine 0.996 / 0.997, size ratio 1.01) and to 90%
  for the variance (cosine 0.90). All three exact at every layer: 1.02e-9 (noise removed), as the analyst found.
- Made exact alone at layer 15: variance -4.7%, skew -5.6%, kurtosis -6.4% (learned off).
- Learned ON (14.107e-9; `t8_analysis.py 35 on`): shares variance 39.9%, skew 19.3%, kurtosis 19.9%; layer 15 alone 23.5% (variance
  10.2, skew 5.4, kurtosis 7.9), layers 13-15 51.4%; made exact alone at layer 15: variance -6.7%, skew -1.6%, kurtosis -5.5%.
- F4b (S1+S2 at 1, learned off, 15.015e-9 = -8.54% on this network): its gain splits, in projection on the error, into skew inputs
  -3.2% (layers 10-15: -0.25, -0.42, -0.46, -0.44, -0.70, -0.84), variance inputs -2.3% (layer 15 -0.92, layer 9 -0.44, layer 12
  -0.30), kurtosis -0.3%; the summed change of the 45 contributions has cosine 0.83 with the prediction change. So the shift works
  through the per-unit skew and variance of many layers; there is no single channel.

## T7 (2026-10-02 02:47-03:20Z). F4b ON THE 24 HELD-OUT RANKS: NOT BETTER THAN F4 (the four screen networks misled)

First 24 of `O/trainer/cert/runB_val.txt`, torch, learned on, no refit, final-layer MSE against the organizers' truth, bootstrap over the
24 networks. Runner `t4_run_chain*.sh`, script `e16b_f4b.py` (frozen copy `e16b_f4b_frozen_fr24a.py`; parity with `t3_screen.py` on network
35: 0.0), predictions `k4l_pred_rank*_fr24a.npz`, table `t4_fr24.py fr24a`. F4b = exact S1 + S2, strength 1, age <= 2, star anchor.

| arm | pooled | 95% | better | per-network mean (se) | worst | all-layer |
|---|---|---|---|---|---|---|
| F4, theta 0.4 (mine = the lead's to the digit) | -3.54% | -4.10, -2.95 | 24/24 | -3.53% (0.29) | -0.49% | -3.00% |
| F4b alone | -3.36% | -4.29, -2.42 | 21/24 | -3.37% (0.49) | +2.15% | -2.38% |
| F4b + K4L (c 1, age 2) | -4.40% | -5.46, -3.25 | 23/24 | -4.35% (0.59) | +2.73% | -3.96% |
| lead: F4 0.4 + K4L | -4.90% | -5.64, -4.16 | 24/24 | -4.84% (0.37) | -0.60% | |
| lead: F4, theta 0.6 | -3.87% | -4.70, -3.01 | 24/24 | | -0.06% | |

F4b against F4 on the same networks: +0.19% [-0.84, +1.27]; F4b + K4L against F4 + K4L: +0.53% [-0.53, +1.57].
Why T6 misled: F4 is weak on two of the four screen networks (100: +0.95%, 37: -0.56%; held-out mean -3.5%) and the F4b variant was
chosen on those four. Four networks cannot rank two changes that are each worth 3-4% (per-network spread 1.5-2.5%).
**Mixture model** (`t9_mix.py`): final prediction = base + sum_k a_k (prediction of arm k - base), coefficients by pooled least squares,
scored leave-one-network-out on the 24. Cosines of the changes: F4.F4b +0.62, F4.K4L -0.06, F4b.K4L -0.01; sizes |change|^2 / |error|^2:
F4 1.95%, F4b 4.68%, K4L 3.12%.

| mixture (fitted scales) | in sample | left out | 95% | better |
|---|---|---|---|---|
| F4 x 1.41 (theta 0.56) | -3.86% | -3.84% | -4.62, -3.02 | 22/24 |
| F4b x 0.86 | -3.45% | -3.42% | -4.23, -2.61 | 23/24 |
| K4L (on F4b) x 0.66 | -1.35% | -1.31% | -1.85, -0.75 | 18/24 |
| F4 x 0.94 + F4b x 0.48 | -4.52% | -4.45% | -5.19, -3.72 | 24/24 |
| F4 x 1.46 + K4L x 0.73 | -5.51% | -5.46% | -6.23, -4.70 | 24/24 |
| F4 x 1.03 + F4b x 0.45 + K4L x 0.71 | -6.10% | -6.00% | -6.74, -5.25 | 24/24 |

The model reproduces a real run (F4 at theta 0.6 measured by the lead: -3.87%). It says the exact terms add about 0.5% on top of
F4 + K4L, at S1 0.86, exact S2 0.45, F4's implied S2 / S3 / S4 0.41, K4L 0.7. To be confirmed by a real run of the mixture.
**REAL RUNS of the fitted mixtures (04:30Z; `k4l_pred_rank*_fr24d.npz`, `t4_fr24.py fr24d`; f4b.py mode 'mix', cs = (0.858, 0.48, 0, 0),
ths = (0.378, 0.378, 0.378)).** The model is confirmed to 0.07%:

| arm (real closed-loop run, 24 held-out ranks) | pooled | 95% | better | all-layer | linear model had said |
|---|---|---|---|---|---|
| F4 0.58 + K4L 0.73 | -5.58% | -6.37, -4.78 | 24/24 | -5.37% | -5.51% |
| mixture (F4-rule 0.378 + exact S1+S2 0.48) | -4.46% | -5.20, -3.71 | 24/24 | -3.68% | -4.52% |
| mixture + K4L 0.71 | **-6.06%** | -6.82, -5.26 | 24/24 | -5.57% | -6.09% |

Mixture + K4L against F4 0.58 + K4L 0.73 on the same networks: -0.51% [-1.02, +0.06], better on 17/24, per-network mean -0.49% (se 0.28).
Mean cosine between the real prediction change and the modelled one: 0.976-0.988. For S37's point (F4 0.6 + K4L 1.0) the model says
-5.28%: lowering K4L to about 0.7 is worth about 0.3% and is free; the exact terms add about 0.5% (bill about +0.17% if the F4 step and
the exact step share one anchor pass).
**Which exact piece** (`k4l_pred_rank*_fr24c.npz`; same 24 ranks, strength 1): exact S1 alone -2.79% (best scale 1.01); S1 + the part of
S2 that is ODD in alpha (-alpha pdf; F4's rule has exactly this odd part) -3.74% (best scale 0.93); S1 + all of S2 -3.36%. The even
part of the exact S2 (+0.18 at alpha 0) costs 0.4% alone, yet in a mixture with F4 it is wanted at about half strength: F4 x 0.83 +
odd x 0.50 -4.23% left out, F4 x 0.94 + (S1 + S2) x 0.48 -4.45%. F4's rule has a NEGATIVE even part (0 at alpha 0, -0.07 at |alpha| 0.75)
plus small S3 / S4; those extras are worth about 0.7% at theta 0.4-0.56 (they are what the trainer's lower b source would do).
**The gain does not go through the isotropic level** (`t10_radial.py 35 3`, network 35): F4 and F4b change the radial by < 0.07% at every
layer; the base card with the radial profile of the F4 / F4b run: -0.02% / -0.01%; F4b with the radial forced back to the base profile:
-8.20% (F4b: -8.22%). The card's radial is 1.4% (layer 5), 2.5% (8), 4.2% (11), 6.4% (14) below the true projection of k4 on v^2.

## T6 (2026-10-02 01:10-03:30Z). F4b: the exact local fourth cumulant (S1 + S2) of the young hubs. FOUR SCREEN NETWORKS ONLY; see T7 for the held-out verdict

**Rule** (the lead's E15 "RIDE" form; module `f4b.py`, plain-array reference `f4b_native_ref.py`, screen option `f4b=<c>,f4bterms=12` in
`t3_screen.py`, and `e16b_f4b.py` = my copy of the lead's K4L screen with the same option). A hub h born at layer j left the pair
p+- = s + t+- e_h (weights lam+-; s_h = 0, checked to 1e-7). Kept from birth per pair: t+, t-, S1 = -alpha pdf / v, S2 = 2 (g - g^2 - m pdf / sigma)
(alpha, sigma, pdf, g of the hub; m = the pair map's mean of the hub). At a later layer, with the pair's pushed columns pp+-:
  Wt = (pp+ - pp-) / (t+ - t-),  Y = pp+ - t+ Wt                          (transported hub unit vector and covariance column)
  feed (d_k/2) g_i g_j Q_ijkk = (gY)(gY) h1 + [(gY)(gWt) + (gWt)(gY)] h2 + (gWt)(gWt) h3,
  h1 = (d/2)(2 S1 Y Wt + S2 Wt^2),  h2 = (d/2)(S1 Y^2 + 2 S2 Y Wt),  h3 = (d/2) S2 Y^2
  the two carried columns u+- = g o pp+- shift by x+- / lam+-, where (x+, x-) is the least-squares solution of
  (h1, h2, h3) = x+ (1, t+, t+^2) + x- (1, t-, t-^2) in the Gram metric of the three (i, j) matrix types (a 2x2 solve per pair);
  the plane of the added tensor is taken out exactly as in F4 (cheap anchor: beta_k = sum_i v_i P_ik / sum_i v_i^2 and the exact diagonal).
F4 is the special case S_m = A sum+- lam+- t+-^m (S1 right; S2, S3, S4 set by the pair's geometry and odd in alpha).
Certificate `f4b_cert.py 35 3`: strength 0 = the unchanged card exactly; F4's rule pushed through the same projection, shift and anchor
code = `f4.py` to 1.0e-7 rms at the final layer (F4's effect: 1.8e-5 rms); newborn columns differ only on the hub row, by t+ - t-.
Exact against rule-A coefficients (network 35, layer 8, mean per alpha bin from -1 to +1): S2 exact +0.27 / +0.27 / +0.18 / +0.03 / -0.17,
rule A +0.15 / +0.10 / 0.00 / -0.13 / -0.29: the exact S2 has an EVEN part of about +0.15 that rule A lacks.

**Closed loop, current torch card, no refit, final-layer MSE against the organizers' truth, networks 35 / 100 / 37 / 39, columns at most
2 layers old, cheap star anchor** (runners `t4_run_f4b{2..6}.sh`, logs `t4_f4b*.log`, predictions `t3_pred_rank*_f4b*.npz`; tools
`t4_compare.py`, `t4_layers.py`, `t3_shadow.py`). Intervals: bootstrap over the four networks. F4 (theta 0.4) on the same networks: -2.33%.

| terms, strength | net 35 | net 100 | net 37 | net 39 | pooled, learned ON | all-layer | learned OFF | shadow (best scale) |
|---|---|---|---|---|---|---|---|---|
| S1+S2, 1.0 | -8.22% | -3.80% | -2.63% | -3.13% | **-4.36%** [-6.94, -2.86] | -2.62% | -4.54% | -4.31% (scale 0.94: -4.32%) |
| S1+S2, 0.75 | | | | | -4.34% [-6.41, -3.13] | -2.79% | | |
| S1+S2, 0.5 | -5.78% | -3.44% | -2.50% | -2.66% | -3.55% [-4.99, -2.58] | -2.40% | -3.50% | -3.50% (scale 1.88: -4.49%) |
| S1 alone, 0.5 | -3.93% | -1.61% | -1.08% | -2.95% | -2.33% [-3.44, -1.34] | -1.48% | | |
| S2 alone, 0.5 | -1.28% | -1.99% | -1.53% | +0.52% | -1.10% [-1.82, -0.05] | -0.85% | | |
| S3+S4, 0.5 | +0.28% | -2.11% | -0.95% | -0.49% | -0.85% [-1.72, -0.06] | -0.05% | | |
| S1..S4, 0.5 | -5.31% | -4.74% | -2.95% | -2.23% | -3.79% [-5.02, -2.61] | -1.51% | | |
| S1..S4, 0.75 | | | | | -4.00% [-5.44, -2.61] | -0.77% | | |
| S1..S4, 1.0 | -6.72% | -4.40% | -2.46% | +0.46% | -3.26% [-5.54, -0.82] | +1.12% | -3.65% | -3.28% (scale 0.70: -3.99%) |
| S1+S2, 1.0, b anchor | | | | | -4.49% [-6.47, -3.15] | -3.00% | | |
| S1+S2, 0.5, columns up to 4 layers old | | | | | -4.11% [-5.13, -3.49] | -2.47% | | |
| S1..S4, 0.5, age 1 / age 4 | | | | | -2.73% / -3.64% | | | |

Pooled change per layer 2..15 (%), S1+S2 at strength 1: +0.4 -0.2 -0.7 -1.8 -2.3 -2.3 -2.6 -2.2 -2.6 -2.2 -2.2 -1.6 -3.9 -4.4.
S1..S4 at strength 1: +5.8 +6.6 +7.8 +4.4 +4.7 +3.3 +2.7 +1.9 +1.2 +2.0 +1.8 +2.2 -1.5 -3.3 (layers 2-13 worse).
- **The physical strength is the optimum for S1+S2, and learned on = learned off = shadow.** First fix of this card without the
  "overshoot": F4's best strength was 0.4 against 0.73 open loop, and its gain halved with the learned correction on.
- **S3 and S4 damage the early layers.** Measured on network 35: the added tensor has a part of the form Sym(C x beta); the projection
  of beta on the card's isotropic b source (radial d v / 6) is 0.77 / 0.54 / 0.20 / 0.08 at layers 1 / 2 / 3 / 4 and 0 +- 0.07 later
  (S1..S4, strength 1). The card's isotropic term already holds the sum of those hub-heavy terms over all hubs. Rule A (F4) has a
  b-like part of -6..-11% of the b source at every layer (theta 0.4), because its S2 is odd in alpha.
- anchor='b' (beta taken out of the carried b vector, plane and distinct content, instead of out of the star): S1..S4 at 0.5 -2.94%
  (all-layer -2.16%), at 1.0 -2.42%; b anchor at steps 1-4 then star: -2.64% / -2.09%; no shifts at steps 1-4: -2.86% (0.5), -3.26% (1.0).
- Layer ranges (S1..S4, strength 1): shifts at the end of step 14 only +0.1%; steps 13-14 -0.2%; 11-14 -1.5%; 8-14 -1.8%; all -3.3%.
  Strength 0.5: steps 1-12 -3.53% against -3.79% for all. The gain comes through the covariance of later layers (two-layer lag), not
  through the last layer's skew; steps 13-14 can be dropped from the bill.
- Isotropic b source x 0.85 (the trainer's dial): alone -0.31% [-2.25, +1.64]; on top of S1..S4 at 0.5: -3.74% against -3.79%.
- 14 and 96 pairs (networks 100, 37) have two identical columns (dead hubs); the 2x2 solve is singular there and the pair is skipped.
- State errors do NOT track the gain (network 35, layer 14, base / F4 / S1..S4 at 1 star / S1..S4 at 1 b anchor): K21 kink block
  7.42 / 6.98 / 7.30 / 6.78%, per-unit k3 5.62 / 5.28 / 5.65 / 4.94%, per-unit variance 0.157 / 0.151 / 0.173 / 0.163%, final MSE
  0 / -4.25 / -6.72 / -4.61%. The rms plane and variance errors are not the judge of a change; the final-layer error is.
Bill estimate: 2 + 2 (Wt, Y) + 10 (gated legs and three dot products) + 3 (Y^2, Y Wt, Wt^2) + 13 (two shifts) + 4 + 24 (anchor) = 58 flops
per pair element = 29 per column element against 16 for F4: about +0.39% of the bill (F4 measured +0.225%); steps 13-14 dropped: x 12/14;
with K4L in the build Wt, Y and the squares are shared.

## T5 (2026-10-02 00:10-02:15Z). Dial fit with F4 on: NOTHING (closed)

`t5_gn_run.sh`, `t5_gn_fit.py gn1 1e-3`: one finite-difference run per dial (theta, b-source scale, X1, X2, X0, bscale, s_new, s_carry,
c0, three k4 bands) around F4 at theta 0.4 on 9 newer-rank TRAINING networks (240-248), Gauss-Newton fit of the final-layer error,
leave-one-network-out scoring. Single-dial changes of the pooled MSE: theta +0.1 -0.29%, X1 +0.05 -0.21%, b source +0.06 -0.04%,
the others +0.02..+0.56%. Fits: theta and b source (+0.14, +0.04) in sample -0.48%, left out -0.33% (5 of 9 better); + X1 left out
-0.26%; all 12 dials in sample -1.02%, left out +1.35% (3 of 9). No dial change without a refit is worth shipping.

## T4 (22:30Z). CANDIDATE F4: fourth-cumulant star on the bank's young columns, fed into births by an in-place column shift. PASSES on four networks

**Rule.** Every column pair born at hub c keeps A = E f'''(z_c)/birth_c = -alpha_c/(sigma_c (1 + radial/3)) (the three-leg star of the
fourth cumulant made by the ReLU at c; fourth-order weights lam4 = lam A on the existing +- pair give S1 = E f''' and no four-leg term).
At each later layer, for the carried columns born 1 or 2 layers earlier (pushed column pp_r, weight lam_r, post column u_r = g o pp_r):
  u_r -> u_r + theta A_r w_r,  w_r = alive (d/2) o pp_r o pp_r        (adds lam4 Sym3(u,u,w) = the feed (d_c/2) g_i g_j Q_ijcc of that K4)
  star -= 3 dP' - 2 diag(dP),  dP_ik = v_i beta_k + 2 C_ik beta_i (i != k),  dP_ii = 3 sum_r c_r u_ri^2 w_ri,  c_r = theta lam_r A_r,
  beta_k = sum_r c_r ((v.u_r^2) w_rk + 2 (v.(u_r o w_r)) u_rk) / sum_i v_i^2,  v = diag C, C = carried post covariance   ("cheap anchor")
Code: `t3_screen.py` (function cp4; options cp4=theta, cp4anchor=2 (cheap) or 1 (exact plane), cp4age=2). Certificates: `t4_cert_shift.py`
(shift = first-order tensor, plane, anchor sums, feed identity: 1e-15), `t4_cp4_ceiling.py cert` (pushed feed operator against brute force).

**Open loop (network 35, true feed from t2_mc, kink block of the next layer, cross-block).** Scripts `t4_cp4_ceiling.py`, `t4_rule_screen.py`,
`t4_young.py`, `t4_collective.py`; logs `t4_*_rank0035*.log`.

| share of the true feed's energy explained | layer 5 | layer 9 | layer 11 | layer 13 |
|---|---|---|---|---|
| the card now (b source radial d v/6 x bscale) | .25 | .19 | .20 | .19 |
| isotropic form at its best coefficient | .265 | .214 | .228 | .212 |
| isotropic + rule A on all columns (2 fitted numbers) | .334 (theta .71) | .312 (.73) | .336 (.75) | .331 (.75) |
| isotropic + rule A on the unjoined columns only | .334 (.71) | .314 (.75) | .342 (.84) | .328 (.91) |
| coefficients fitted at another layer | .32-.33 | .31 | .33-.34 | .32-.33 |
| ceiling: FREE fourth-order weight per column (CGLS, 250 iterations) + isotropic | | .49 | | .60 |

Other birth coefficients on the same columns add nothing (four-leg star, (2,1,1) weight w22: +0.00-0.01; a generic lam/sigma term +0.01-0.03).
By age the fitted theta is 0.85-0.92 (1 layer old), 0.81-0.86 (2), 0.68-0.78 (3-4), 0.27-0.46 (5+). With free weights the isotropic coefficient
returns to radial/3 (0.0188 against 0.019 at layer 13). The free weights fitted on the feed predict the true non-isotropic K31 with
cosine 0.41-0.43; rule A's own K31 / K22 planes are poor (cosine 0.18-0.27 / 0.06-0.11, K22 norm 1.3-2.8x), and the plane of the added
tensor pushes 2.8-4.4% of the plane norm against 1.6-1.9% for its distinct part: it must be anchored out. The b-carrier form absorbs
71-84% of that plane's energy and leaves a push of 0.35-0.49%. A collective version (columns projected on the mean direction or on the
top 1 / 4 / 16 covariance eigenvectors) holds only 1-3 / 17-24 / 44-60% of the rule's distinct content and gains nothing: the full
columns are needed.

**Closed loop, current torch card, no refit, final-layer MSE (organizers' truth).** `t3_screen_rank*_cp4*.jsonl`, `t3_shadow.py`.

| arm | net 35 | net 100 | net 37 | net 39 | pooled ON | pooled OFF | shadow (best scale) |
|---|---|---|---|---|---|---|---|
| cheap anchor, age <= 2, theta 0.2 | -2.49% | -0.27% | -1.04% | -3.76% | -1.83% | -2.57% | -1.99% (-2.98%) |
| cheap anchor, age <= 2, theta 0.4 | -4.25% | +0.95% | -0.56% | -6.00% | -2.33% | -4.07% | -2.85% (-2.98%) |
| cheap anchor, age <= 2, theta 0.6 | -4.53% | +2.12% | +2.21% | -7.91% | -1.80% | -4.88% | -2.79% (-2.87%) |
| exact anchor, all unjoined, theta 0.4 | -3.70% | +0.88% | -0.69% | -7.21% | -2.54% | -4.04% | -3.29% (-3.29%) |
| exact anchor, all unjoined, theta 0.8 | -0.39% | +6.61% | +3.20% | -7.47% | +0.69% | -3.56% | -1.49% (-3.90%) |
| exact anchor, all unjoined, theta 1.2 | +9.41% | +18.2% | +14.4% | -0.42% | +10.7% | +1.67% | +6.23% (-4.08%) |

Network 35, learned on: un-anchored theta 0.4 / 0.8: 19.3 / 32.6e-9 (from 14.1); exact anchor age <= 1: -3.1%, age <= 2: -3.7%.
State errors, network 35, theta 0.4 cheap anchor age <= 2 (base in brackets): K21 kink block 4.21 / 5.81 / 6.31 / 6.98% at layers
6/10/12/14 (4.45 / 6.15 / 6.72 / 7.42); per-unit k3 5.28% at layer 14 (5.62); per-unit variance 0.151% at layer 14 (0.157).
Bill: 17 flops per element of the two youngest cohorts (4 for the shift, 13 for the anchor), about 44,400 columns x 1024 rows over layers
1-14 = 7.7e8 = +0.30% of 2.66e11 (about +0.25% on alive rows), plus 6 n^2 per layer.

**VALID40 (23:10Z; 40 validation networks of `P2/fable_analyst_v1/split_train640.json`, learned on, no refit, cheap anchor, age <= 2;
`t4_run_valid40.sh`, `t4_valid40_v40a.log`, predictions `t3_pred_rank*_v40a.npz`, summary `t4_summary.py v40a on_0`).** Base pooled
final-layer MSE 1.6955e-8; paired bootstrap over networks.

| strength | pooled final-layer MSE | 95% interval | networks better | median | best / worst | all-layer MSE |
|---|---|---|---|---|---|---|
| theta 0.2 | -1.67% | -1.99 .. -1.34 | 39/40 | -1.82% | -3.81% / +0.31% | -1.46% |
| theta 0.4 | -2.46% | -3.03 .. -1.85 | 35/40 | -2.30% | -6.73% / +1.10% | -2.22% |

A quadratic through the two strengths puts the no-refit optimum at theta 0.48 (-2.5%).
**Held-out newer ranks (23:45Z; first 24 of `O/trainer/cert/runB_val.txt`: 651 378 424 328 287 641 548 557 316 390 293 277 320 312 550 672
333 310 530 259 623 459 16 473; nobody trained or selected on them; `t4_ranks_fresh24.log`, `t4_summary.py fresh24 on_0`).** theta 0.4:
final-layer MSE **-3.54%** [95% -4.10, -2.95], better on 24/24, median -3.62%, best -5.93%, worst -0.49%; all-layer -3.00%
(base pooled 1.7020e-8). Pooled change per layer 0..15 (%): 0 0 0 -0.5 -1.0 -1.3 -2.0 -2.0 -2.4 -2.8 -3.1 -3.6 -3.5 -3.7 -4.1 -4.0
(first 12 networks; VALID40: ... -2.2 -2.1 -2.3 -2.5 -2.5 -2.6 -2.4 -2.5 at layers 8-15).
**Module (`f4.py`, certificate `f4_cert.py 35 3`).** theta 0 reproduces the card exactly (max abs difference 0.0); theta 0.4 reproduces the
screen arm exactly (0.0; 13.5076e-9 on network 35); a checkpointed backward pass runs (87 s under load); d(final MSE)/d theta at 0.4 on
network 35: autograd -0.40e-9 per unit theta, central finite difference (+-0.02) -0.69e-9. The two differ because the card's star basis
is detached in the backward pass (the trainer's convention) while the anchor moves the star: sign and order agree, the size is 0.58.

**Controls and strength x b-source scan (00:15Z; networks 35/39/37/100, cheap anchor, age <= 2; `t4_scan1.log`, `t3_shadow.py scan1 0 ...`).**

| arm | net 35 | net 39 | net 37 | net 100 | pooled ON | pooled OFF | shadow |
|---|---|---|---|---|---|---|---|
| F4 theta 0.4 | -4.25% | -6.00% | -0.56% | +0.95% | -2.33% | -4.07% | -2.85% |
| CONTROL sign flipped (theta -0.4) | +8.45% | +9.13% | +4.35% | +2.61% | +6.01% (0/4) | | |
| CONTROL birth scalar A shuffled across the hubs of each cohort | +1.80% | +2.40% | +4.67% | +1.65% | +2.68% (0/4) | | |
| theta 0.6 with the b source x1.15 | -4.66% | -5.92% | -2.15% | +0.65% | -2.92% | -3.97% | -3.53% |
| theta 0.8 with the b source x1.3 | -5.97% | -3.63% | -2.30% | +4.23% | -1.81% | -2.11% | -2.32% |
| theta 0.4 with j1 at 0.7 | -1.46% | -7.81% | +4.93% | +3.33% | -0.01% | -4.89% | -2.13% |
| theta 0.3 at layers 0-7, 0.6 at layers 8-14 | -3.50% | -6.14% | +0.25% | +1.91% | -1.72% | -3.98% | -2.29% |

The gain needs the physical sign and the hub-specific value of A. The strength/source variants are within the noise of four networks of each
other (+-2.5%); a Gauss-Newton fit of theta, the b-source scale and the old dials on 12 newer-rank training networks is running (`t5_gn_*`).
**The free fit rediscovers the rule (`t4_lam4_rank0035_L{9,13}_none.npz`).** The free per-column weights (fitted to the true feed only)
correlate with the rule's lam x A at +0.83 / +0.75 / +0.64 / +0.53 for columns 1 / 2 / 3 / 4 layers old at layer 9 (+0.77 / +0.71 / +0.64 /
+0.51 at layer 13; slope 0.8 and 0.55), falling to 0.1-0.25 beyond 6 layers; and the two partners of a hub pair get the same lam4/lam
(correlation 0.90-0.98 up to 6 layers old), which is the rule's "pure star, no four-leg term". Columns 1-2 layers old hold 60% (layer 9)
and 38% (layer 13) of the free weights' energy, columns up to 4 layers old 80% and 61%. Features meant to model a drift of the ratio
with age (self-projection (d o pp . g o pp)/|g o pp|^2, projections on the top covariance direction or the mean direction) add at most
+0.03 to the explained share (`t4_drift_screen.py`).

Per hub pair the free fit's tied ratio is explained by A alone at (weighted R2 about zero) 0.57 / 0.38 / 0.22 / 0.17 / 0.07 / 0.02 for pairs
1..6 layers old at layer 9 and 0.50 / 0.46 / 0.33 / 0.25 / 0.12 / 0.05 at layer 13, coefficient 0.7-0.9 throughout; a positive offset
(+0.12 to +0.30 / sigma_c) or alpha_c^2 add 0.5-5 points (`t4_oracle_regress.py`). **No better per-unit law for the isotropic source
exists**: a FREE vector w in Q_ijcc = C_ij w_c explains 0.275 / 0.232 / 0.236 of the true feed at layers 5 / 9 / 13 against 0.268 / 0.215 /
0.212 for w = c v (`t4_freew.py`).

**The same column fourth cumulant against the pair map's planes (`t7_k31_star.py`, network 35, columns <= 2 layers old, covariance-read
weighting, both sides with the isotropic form removed, cross-block).** 42-60% of the rule's K31 plane and 85-96% of its K22 plane are
isotropic-like, i.e. already counted in radial (this is why the un-anchored shift ruins the card). The rest is real: against the true
non-isotropic K31 it has cosine 0.32-0.35, best coefficient 0.74 / 0.87 / 0.89 / 1.01 at layers 5 / 9 / 11 / 13 and explains 9-12% at
coefficient 1; against the true non-isotropic K22: cosine 0.37-0.41, best coefficient 0.86-1.19, explains 13-17%. So one consistent
object (isotropic radial + the star on young columns) describes a part of all three places where the fourth cumulant enters. Delivering
the plane parts to the pair map needs two plane reads over the young columns (about 2.6e9 flops per layer, 6-13% of the bill) for
12-17% of a lever priced at -10..-12% in September: not worth it. Only the feed is free. A per-unit kurtosis from the same weights
(sum_r lam_r A_r pp_ra^4) has cosine 0.09-0.18 with the true per-unit spread: no use.

**Port specification (order of operations at the end of step l, 0 <= l <= 14; reference = `f4.py`).** Inputs available in the step:
pp (pushed bank columns), alive, g = gate, d = pdf/sigma, the carried post covariance C (after recentring), the outgoing column block
p = [alive g o pp | newborn + | newborn -], lam (after the lam dials), b, and the star the card has just built.
1. Columns touched: the carried columns of the cohorts born at l-1 and l-2 (never part of the old-tier frame, which takes cohorts 5-7
   layers old). Each column carries A (float32), set at its birth to -alpha_c / (sigma_c (1 + radial/3)) with the hub's pre-activation
   alpha, sigma and the incoming radial of that layer; same value for the + and the - column of the pair.
2. w = (alive d / 2)[:, None] o pp o pp on those columns; u = their outgoing columns (alive g o pp); c = theta lam A (per column).
3. shift: outgoing columns <- u + w (theta A)[None, :].
4. beta = [ w @ (c o (v @ (u o u))) + 2 u @ (c o (v @ (u o w))) ] / sum(v^2),  v = diag(C).
5. star anchor: the star is built from rest = target - cp_post - trace_post with trace_post = v_i b_k + 2 C_ik b_i. The anchor is the same
   expression with b + beta in place of b (for the star only; the carried b is unchanged), and on the diagonal the exact value:
   rest_ii -= 3 sum_r c_r u_ri^2 w_ri instead of 3 v_i beta_i. (f4.py writes it as star -= 3 dP' - 2 diag dP.)
6. A for the newborn pairs of this layer is appended.
Where it lands in the native source (research parent `H/hill_climb_pair_rr_inverse_v1/candidate.py`, step body of `_predict_binary`
defined at line 3214; read-only inspection, not a tested patch): the gated young columns are formed at line ~3449 as
`p = einsum('i,ij->ij', _e2_ga[live_indices], right)` (live rows only; the cohort born at this layer is still the `newborn` tuple and
is materialised by the next push, so at layer l the last 2 K_{l-1} + 2 K_{l-2} columns of `right` / `p` are the two cohorts F4 shifts).
The shift is three elementwise operations on that trailing block; beta needs four matrix-vector products with it; `trace_post`
(line ~3421) is used twice: for the innovation (keep b) and in `rest = target - cp_post - trace_post` (line ~3470; use b + beta there
and correct the diagonal). A per column is appended where `lam` is appended (line ~3456), from `aux` alpha, sigma and `radial`.
A plain-array reference of the step with the same inputs and outputs: `f4_native_ref.py` (`f4_step`, `birth_scalar`).
Flops per network: step 2-4 touch n x n_young elements about 16 times (n_young = 2 cohorts = 2,400-4,100 columns; sum over layers 1-14
about 44,400 columns): 7.3e8 = +0.27% of 2.66e11; requests: about 15 elementwise or matrix-vector operations per layer.

## T3 (21:40Z). Zero-cost completions named by the budget: all FAIL

`t3_screen.py`, `t3_shadow.py`, logs `t3_bsrc_rank*.log`, `t3_screen_rank0035_v1.jsonl`. Current card, no refit.
- b source radial d v/6 scaled by x (the feed's isotropic coefficient is 0.75 x radial/3): network 35 plane error 7.42 -> 6.60% (x .75)
  -> 6.20% (x .5) at layer 14, k3 5.62 -> 4.67%; final-layer MSE on 35 / 100 / 37 / 39: x .875: ON -0.13 / -0.38 / +2.14 / -1.91%
  (pooled 0.00%), OFF pooled -1.65%, shadow -0.17% (best scale -0.36%); x .75: ON +0.23 / +1.91 / +4.36 / -4.51% (pooled +0.65%),
  OFF -2.12%, shadow +0.67% (-0.58%); x .625: shadow at best scale -0.96%. FAIL (bar -1.5%).
- network 35 only, final-layer MSE learned ON / OFF: birth weight radial/3 off +0.14 / -0.15%; K22 leg -(d_i/4) eta_c K22_ic
  +0.92 / +0.62% (plane error 7.42 -> 7.25%); vertex skew +0.59 / +0.98%; j1 at 0.8: +3.5 / -0.05% (plane 6.27%); j1 at 1.0: OFF +4.5%.

## T2c (21:30Z). The non-isotropic fourth-order content is not young: no two-layer closed form

`t2_mc_restart.py` + `t2_restart_analysis.py`, network 35, Gaussian restarts (true mean and covariance) 1, 2, 4 layers before layers 9 and
13, true network in between, 2 x 4M samples; compared with the truth at the same layer (cross-block products).
Share of the truth's energy reproduced by the restart (coefficient 1), layer 9 / layer 13:

| content made in the last k layers | k = 1 | k = 2 | k = 4 |
|---|---|---|---|
| non-isotropic part of the fourth-cumulant feed into births (kink block of the next layer) | 10% / 8% | 29% / 20% | 65% / 50% |
| non-isotropic part of K31 (covariance-read weighting) | 8% / 7% | 26% / 18% | 61% / 46% |
| the (2,1) plane K21 on the kink block | 23% / 19% | 45% / 32% | 70% / 56% |
| per-unit k3 | 21% / 19% | 43% / 31% | 67% / 52% |
| per-unit k4 spread around the isotropic value | 8% / 5% | 22% / 19% | 49% / 36% |

The isotropic (gain) coefficient of the restarts is 0.03-0.2 of the truth's (radial .0013 / .004 / .010 against .044 at layer 9): the gain
mode is older still. Reading: half of the non-isotropic (2,1,1) and (3,1) fourth-order content needs exact physics over more than four
layers. It is closed for the same reason as the per-unit kurtosis (September restart law): it needs a carrier, not a formula.

## T2b (21:15Z). Anatomy of the true fourth-cumulant feed

`t2_feed_anatomy.py`, network 35, layers 5/9/11/13.
- The kernel V[a,c] = kappa4(y_a, y_a, z_c, z_c) (y = gate-linear next pre-activation) is 94-98% of the isotropic form
  c22 Var(y_a) v_c + c11 Cov(y_a,z_c)^2 with c22 = 0.94-0.98 x radial/3. But the DISTINCT-index part that feeds births is not:
  least squares on one truth block, scored on the other, kink block of the next layer:

| model fourth cumulant Q_ijcc | share of the true feed's energy explained, layers 5 / 9 / 11 / 13 |
|---|---|
| C_ij v_c (isotropic, free coefficient) | .265 / .214 / .228 / .212 (coefficient .0078 / .0109 / .0128 / .0139 = 0.80 / 0.75 / 0.74 / 0.73 x radial/3) |
| + C_ic C_jc | +.000 (coefficient 0.37 / 0.15 / 0.08 / -0.06 x 2 radial/3) |
| + mu_i mu_j mu_c^2, mu_i mu_j v_c, C_ij mu_c^2, (C_ic mu_j + C_jc mu_i) mu_c | .273 / .232 / .262 / .244 |
| + mu_j K21[c,i] + mu_i K21[c,j] and C_ic K21[c,j]/v_c (gain on skew) | .305 / .273 / .305 / .303 |

- The residual of V (d-weighted, kink rows) is not low-rank: 2 / 16 / 64 singular directions hold 22-26% / 32-44% / 53-70% of it
  (cross-block validated).
- So 70% of the feed is network-specific (2,1,1) fourth-order content.

## T2 (21:10Z). Mechanism budget of the K21 kink-row error: it is the fourth-cumulant feed into births

`t2_mc.py` (GPU, 630 s), `t2_capture.py`, `t2_lib.py` (plane algebra), `t2_cert.py` (brute-force certificate, n = 7, all identities to
1e-14), `t2_budget.py` -> `t2_budget_rank0035.json`, `t2_budget_rank0035_planes.npz`.
The pre plane at L = l+1 is the push of the post plane at l plus the push of the post distinct-index content D. Truth:
D = gate-linear transport of the true pre-activation tensor + births from a Gaussian with the true mean and covariance + third-cumulant
insertions on a tree birth (j0, j1, gate skew, vertex skew, physical strength, from the true planes) + fourth-cumulant feed
(d_c/2) g_i g_j Q_ijcc (exact Monte Carlo statistic) and its legs + remainder. Sizes, % of the true plane norm on the kink block
(layers 6 / 10 / 12 / 14): transport 39 / 43 / 44 / 43; Gaussian births 26 / 24 / 22 / 21 (tree + doubled bond reproduces them to
0.1 point); third-cumulant insertions 2.6 / 2.9 / 2.9 / 3.0 (j1 2.4-2.95, j0 0.8-1.0, vertex skew 0.14-0.23, gate skew 0.05);
fourth-cumulant feed 2.75 / 2.98 / 3.21 / 3.23 (feed proper 2.85-3.3, K22 leg 0.31-0.37, K31 legs 0.23-0.35, vertex 0.08-0.12);
remainder 0.83 / 0.81 / 0.82 / 0.83. The first-order theory is complete to 0.8%: second-order birth terms are closed.

Card error, % of the true plane norm (signed share of the squared error; shares add to 1):

| piece (card minus truth) | L6 | L10 | L12 | L14 |
|---|---|---|---|---|
| total | 4.45 | 6.15 | 6.72 | 7.42 |
| transported content (carried bank + transported b against the true transport) | 3.05 (.50) | 5.04 (.68) | 5.87 (.73) | 6.82 (.79) |
| fourth-cumulant feed (b source + birth weight radial/3 against the true feed) | 2.55 (.31) | 2.83 (.18) | 3.02 (.17) | 3.10 (.15) |
| third-cumulant insertions (learned j legs against physical) | 1.09 (.08) | 1.27 (.06) | 1.26 (.06) | 1.35 (.06) |
| Gaussian births | 0.51 (.00) | 0.62 (.00) | 0.63 (.00) | 0.67 (.00) |
| remainder (terms the card does not have) | 0.83 (.03) | 0.81 (.01) | 0.82 (.01) | 0.83 (.01) |
| pushed post-plane error | 1.03 (.05) | 1.40 (.05) | 1.78 (.02) | 2.04 (-.01) |
| star read (rank 32, one term) against the exact push of the star | 0.84 (.03) | 0.75 (.01) | 0.68 (.01) | 0.63 (.01) |

The same shares hold in the covariance-read metric (next-layer per-unit variance: total 0.025 / 0.047 / 0.059 / 0.070%).
Reading: the distinct content keeps 0.88 of itself per layer (true transport 43% of a 49% total), so 3.3% of fresh error per layer
accumulates to the measured 5-7% "transport" error. The fresh error energy per layer is 60-75% the feed, 12% the j legs. The card's feed
removes only 8-14% of the true feed's energy. Certificates in the json: block 0 reproduces the 8M plane files (K21 0.1%); the card's
pieces add up to its own plane exactly; the card's newborn columns equal the hub algebra exactly.
Side finding: the 8M truth K31 planes (`mps_planes_rank0035.npz`) differ from the centred recomputation on the same samples by 3-5%
(K22 1-2%): uncentred float32 sums. Small against their sampling noise.

## T1 (20:48Z). One-step injection budget on the current card

`t1_onestep.py`, `t1_table.py`, `t1_rank0035.json`, `t1_rank0100.json`. Step l re-run from the card's state with the 32M true mean and
covariance at l-1; pair-map covariance output recomputed with all inputs at truth except one taken from the card (including the card's
covariance knobs). Next-layer per-unit variance change, relative rms in %, network 35 / 100 (floor of the all-truth arm 0.016-0.018%):

| layer | all card inputs | K21 kink rows | K21 other rows | K31 | K22 | k4 per unit | k3 per unit |
|---|---|---|---|---|---|---|---|
| 4 | .030 / .030 | .025 / .025 | .006 / .008 | .015 / .016 | .008 / .008 | .001 | .001 |
| 7 | .039 / .041 | .033 / .033 | .010 / .011 | .023 / .023 | .009 / .009 | .003 | .002 |
| 10 | .058 / .055 | .048 / .046 | .016 / .014 | .029 / .029 | .012 / .010 | .004 | .004 |
| 13 | .079 / .071 | .069 / .062 | .021 / .017 | .041 / .038 | .014 / .011 | .005 | .005-.010 |
| 14 | .082 / .083 | .070 / .073 | .018 / .022 | .041 / .042 | .013 / .012 | .006 | .006-.010 |

Share of the card's own one-step injection removed when one input alone is set to truth: K21 kink rows 0.63-0.74, K31 0.18-0.26,
K21 other rows 0.02-0.08, K22 0.02-0.07 (sum 1.0: no cancellation at one step). September: K21 all rows .038/.056/.073 at layers
7/10/13 (now .035/.051/.072), K31 .024/.031/.039 (unchanged), K22 .02 (now .009-.014).
