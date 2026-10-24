# Engineer lane: bill reduction on RRPI (2026-10-01), then native ports of the physics levers (2026-10-02)

**State 2026-10-10 08:20 UTC (section 33).**
- **S44 (native sha 67dfcf71) fires on the trainer's 50 (lead 08:05Z)**: jmC without qold = refitted arrays, kappa_C 0.697 at
  layers 1-14, head columns 6 (new weights) / 20 / m2 / eold / qb at layers 1-14, column 17 at layer 15, BSK 0.34, the plane's
  age 3 at 0.64, the deeper diet, the padded dead cut. One command: `stack/s43_build.sh jmC jmCnq s43c17`.
  Ten fresh networks (list a) against the two S42 draws: **-0.78% pooled (se 0.61), 8 of 10 lower, bill +0.05%**; torch on the
  same ten -1.00% (on the 50: -2.0%); native follows torch network by network, 0.21% weaker. Parity by the change from each
  side's own S42: cosine 0.92-0.98 and size ratio 1.01-1.07 at layers 8-15, final change -1.14 / -2.95% native against -1.07 /
  -2.77% torch on 651 / 378.
- **S43 (sha 4011f35f)**: -0.16% (6 of 10) against S42 on the same ten (torch -0.42% there); **S43 + column 17 (sha f03a044f)**:
  -0.43% (8 of 10), column 17 alone -0.28% (se 0.07).
- **New native levers**: `KC@<name>` (kappa_C: the rank-2 covariance update after the head's correction scaled by 1 + rc - kc;
  off byte-identical, on free), batch-6 head columns m2 / qb / eold / qold, the converter of the trainer's 31 x 128 table
  (layers 1-14 and layer 15 apart), column 18 pinned to ages 1-2.

**State 2026-10-10 02:55 UTC (the restart after the weekly pause; section 32).**
- **S42 was graded: #333607 = 1.6666e-9, fifth on the board** (final-layer MSE 1.424e-8; the board moved during the pause:
  J2W 1.4425e-9, marius_binner 1.5678e-9, Luna 1.5692e-9, oliver_chen 1.6097e-9; third place needs -5.8%).
- **S43 skeleton (sha 0df2ba07)** = S42's arithmetic with the step-2 head column, on the deeper request diet, + the
  skew-corrected birth: **-0.28% pooled final-layer error against the two S42 draws on ten fresh networks (8 of 10 lower),
  bill +0.080%, 305 requests fewer on every network**: 635 / 432 requests of room under the caps 43,800 / 42,700. The base
  for the trainer's hand-overs, not a fire.
- **Final-layer head columns** (the trainer's 17-19 at layer 15, own 128 weights u15; lead 02:25Z) run natively in the
  kept-rows convention (`a42_head/fragment_hcf_v2.py`, lever `HCF@<weights>`): +132 requests, +0.10% of bill for all three
  (19 alone: +25 requests, free). The trainer's first table (02:51Z) used the layer-14 alive mask, which the native file
  cannot reproduce (no pushed columns off the kept rows at layer 15); asked for u15 in the kept-rows convention.
- **State readouts (carried b, radial)** native (`a43_readout/fragment_ro_v1.py`, lever `SRO@<weights>`): +254 requests,
  0.003% of bill; the trainer's line searches reach -0.1..-0.3% only (batch 4: nothing passes).
- **The covariance sandwich with the dead units of the layer below cut: IN S43 AT GATE FLOOR 1e-3** (section 32; lead 03:24Z):
  ten fresh networks against the skeleton: **final-layer error +0.03% (se 0.06), bill -0.250%, 1,397 requests fewer** (lever
  `DCV64f0.001`, sha b7d33806). At the live roster's floor 0.01 the same cut cost +16.3% / +6.7% (651 / 378): the units with
  gates between 1e-3 and 1e-2 carry covariance with the live ones (1-2e-3 of C_pre, not low-rank). **S43 takes the padded form
  `DCV64f0.001p` (sha 9e45e268; lead 03:27Z): ten networks against the skeleton, bill -0.370% (-0.21 .. -0.49%), error +0.06%
  (se 0.07, 7 of 10 lower), charged ops -58**; against the two S42 draws -0.21% error (7 of 10 lower), bill -0.289%.
- **S43 build in one command**: `stack/s43_build.sh <hand-over> [<head weights> [<base>]]` (base `s43m` = BSK at 0.34 + the
  padded cut; the deeper diet); waiting for O/trainer/candidates/s43_mix/ (lead 04:05Z); S44 = the same with jmA's hand-over.
- **Batch-6 head columns native** (lead 03:17Z, for the trainer's joint refit): m2, qb, eold, qold at layers 1-14 as the
  physicist's b6cols.py defines them (`a46_b6/fragment_b6_v1.py`, columns of the HC lever; unit check PASS).
- **Layer-15 copy of the learned correction**: an OFF switch (`a45_l15/fragment_l15_v1.py`, lever `L15@<name>`), not run
  (the layer-15 dump was cancelled, lead 03:17Z).

**State 2026-10-02 20:14 UTC.**
- **S42 was fired as #333607 (19:49 UTC; grade pending)**: my step-2 file (sha 0fccf396) + the release fragment Rl3_bind128
  (variant sha 471a9841). The lead's stage C on the exact file: requests 43,470 / 42,573, projected wall 82.1 / 88.6 s,
  residual 0.320 / 0.356 s, bill x0.9995 of S41; the release lane's stage D on the step-1 variant: PASS. Expected on the
  board 1.675e-9 to 1.69e-9.
- **For S43 (section 31)**: the skew-corrected birth on S42's arithmetic reads **-0.34% pooled on ten fresh networks against
  two rounding draws of S42 (se 0.32, 8 of 10 lower)**, the size torch gave and the same sign network by network; +105
  requests. The deeper request diet as S42's second draw: **+0.10% (se 0.14) in error, +0.078% bill, 410 requests fewer** on
  every network: the lead made it the base of S43. Several head columns at once are written and run natively (the one-column
  case reproduces the fired S42 byte for byte), but the trainer's joint third step over columns 6, 9, 16 gave nothing beyond
  its noise (20:13 UTC), so S42's weights stand.
- **S41 was graded: #333587 = 1.7110e-9, the bank, third place** (S40 1.7590e-9: -2.73%; final-layer MSE 1.4620e-8, -2.99%;
  budget share 0.117035; 100 of 100 networks). My ten fresh networks had said -3.26% in error.
- **S42 is defined by the lead (19:13 UTC) as S41 + one new input column of the learned mean correction, and its native file
  meets the lead's fire rule on ten fresh networks** (section 30): the trainer's column (the unit-specific part of the pair
  law's diagonal on the read's response rows, 128 first-layer weights) is computed inside the fold from arrays it already
  holds. Two weight sets were run, each on the same ten networks against the mean of the two stored S41 draws:
  **step-2 weights (sha 0fccf396): final-layer MSE -2.15% pooled (mean of the per-network changes -2.09%, se 0.28), lower on
  10 of 10, bill -0.03%**; step-1 weights (sha 33582663): -1.55% (se 0.38), lower on 8 of 10. The files are S41's arithmetic
  with the free request diet + the column: 294 requests more than without the column and fewer than the fired S41 on every
  network. Parity with the trainer's torch column: correlation 0.995 at layer 1 on both check networks with the same size;
  later layers 0.86-0.96, limited by the native rounding draw.
- **The refit's base terms are complete as native switches** (sections 28-29): beside the four of section 28, the second-order
  part of the third-cumulant target in a lean exact form (0.30% of bill and 658 requests a network in full; 0.15% / 434
  without the rho terms; 0.09% / 308 with the K21 terms only; no bill / 196 for the diagonal alone) and the final layer's
  mean term (7 requests). Unit-checked against transcriptions of the physicist's module; none of these has run natively
  (the lead's hold of 18:01 UTC stands for them until the base is named).

**State 2026-10-02 18:11 UTC (kept for the record).**
- **The lead cancelled S42 as a dials file (18:01 UTC); the next file is the refit file.** Four native switches for the refit's
  candidate base terms are written, unit-checked and, for three of them, compared with the physicist's torch arms on one network
  (section 28). Nothing native runs until the base is named.
- **S41 was fired as #333587 (16:51 UTC; grade pending)**: S40 + the fourth-chaos plane K31L4 + the trainer's first dial values
  (with the table) + my leaf-view lever HV + the release lane's two request fragments (section 25). Its arithmetic natively on
  ten fresh networks against the fired S40 file: **final-layer MSE -3.26% [95% -4.97, -1.66], 9 of 10 lower, bill +0.304%,
  MSE x bill -2.97% [-4.66, -1.36]**.
- **For the next file, four request levers are built and measured (section 26); together S41's arithmetic runs at 921
  requests fewer on a first network and 445 fewer on every other, byte for byte the same predictions (sha d464c12c).** (1) The once-per-process constants at the head
  of the binding phase (lever CS): predictions byte-identical, the first network of a worker makes 452 requests fewer by my
  count. (2) A diet that keeps every bit (F4c v3, K4L v7, pair law v9, edge cut v2): predictions byte-identical, **323
  requests fewer on every network and the bill 0.023% lower**. (3) The strips' leaf views cached per tree (HV2): byte-identical,
  146 / 122 index requests fewer on the first / second network. (4) A deeper diet (F4c v2, K4L v6) saves 350 more but changes
  the rounding and costs 0.073% of bill; on ten fresh networks its MSE reads +0.26% [-0.16, +0.60]: not a default.
- **One command builds the next file from a trainer hand-over**: `stack/law_file_from_trainer.sh <name> <2 | 3> [same | deep | s41]`.
- **The second dial set (`law_dials_d7_ages12`) buys nothing measurable over S41 natively** (section 27): on ten fresh networks
  against two rounding draws of S41, table form -0.18% (se 0.20), constant regressor +0.14% (se 0.30). The solve itself is
  confirmed (second against first dials on the constant regressor: native -1.15%, torch -0.84%), but that form with the first
  dials is 1.31% worse than S41. A native run carries a rounding draw of about 0.5% of a network's MSE; effects below 1% are
  judged on ten networks against two draws.

**State 2026-10-02 15:35 UTC (kept for the record).**
- **S40 was graded: #333559 = 1.7590e-9, the bank, THIRD PLACE** (S39 1.8470e-9: -4.76%; final-layer MSE 1.5070e-8, -5.64%;
  budget share 0.116720, +0.90%; 100 of 100 networks; platform wall 79.8 s mean / 91.2 s worst, residual 0.311 s mean /
  0.365 s worst). My native session on ten fresh networks had said -7.93% in error and -7.16% adjusted; the trainer's 50
  torch ranks -6.24%: the fresh list flatters this lever by about 2 points, the board followed the 50 ranks.
- **The next term is ported and has native parity** (section 24): the lead's fourth-chaos plane of the young pairs' local
  quartic (K31L4), as a switch of the same fragment. Networks 651 / 378 against the same file without it: **final-layer MSE
  -2.33% / -2.36%, bill +0.29%, +477 logged calls** (the lead's torch: -2.09% / -2.65%; the trainer's 25 ranks: -1.87%, se 0.39).
- **The S41 stack so far** (file sha 8d110449): S40 + that term + the pair law's faster kernels + the K4L birth by tables.
  Against the fired S40 file on 651 / 378: MSE -2.12% / -2.36%, bill +0.30%, +253 / +250 logged calls. Waiting for the
  trainer's dial values and the lead's spec.

**State 2026-10-02 15:05 UTC (kept for the record).**
- **S40 was fired as #333559 (14:52 UTC; grade pending).** The lead's chain passed all three stages on the file of section 23
  (sha e2be4c75). **Natively the fired file reads, on ten fresh networks against S39: final-layer MSE -7.93% [95% -9.97,
  -5.95], lower on 10 of 10, bill +0.876%, MSE x bill -7.16% [-9.23, -5.14]** (14:53-15:03 UTC). At the boundary it makes
  43,628 requests a network (+1,004) and x1.025 kernel instructions; projected platform wall 79.1 s mean / 89.7 s worst,
  residual 0.309 / 0.355 s.
- **A faster form of the same arithmetic is built** (section 23, last part): the exchange of the pair columns by split +
  concatenate instead of put_along_axis, which alone was 5.2 of the pair law's 8.4 G kernel instructions a network; with
  the K4L birth coefficients from two polynomial tables (238 requests fewer, coefficients bit-identical in the unit check)
  that is file sha cf1f735c, a candidate for the designated variant. Not yet measured natively.

**State 2026-10-02 14:25 UTC (kept for the record).**
- **S39 was fired as #333528 and is the bank: 1.8470e-9** (S38 1.9070e-9: -3.15%; final-layer MSE 1.597e-8; budget share
  0.115675; 100 of 100 networks; platform wall 77.3 s mean / 87.6 s worst, residual 0.302 s mean / 0.347 s worst). Fourth
  place, 0.18% behind the third (1.8437e-9). My native session had said -2.27% [95% -2.98, -1.57] (section 19).
- **The S40 file is built and in the lead's qualification** (section 23): S39 + the pair law of the fourth-order norm coupling
  (lever NP; the lead's law O/NCAR6_SPEC.md, with the physicist's bank channel as its first half), sha e2be4c75. What was
  measured natively, all against the stored S39 rows:
  - the parity version (one form removed) on ten fresh networks: **final-layer MSE -7.50% [95% -9.45, -5.49], lower on 10 of
    10, bill +1.17%, MSE x bill -6.46% [-8.45, -4.40]**; on the six ranks the trainer's torch card also ran, native and torch
    agree network by network (-3.2 to -13.8%);
  - the lean version: the same predictions at rounding level (4.7e-6 rms on two networks) with **+871 / +885 logged calls
    instead of +3,568** and bill +0.98%;
  - the spec file (three forms removed, the card's beta for the carried weights): networks 651 / 378 **-14.24% / -6.54%**,
    bill +0.875%, logged calls +929 / +913, about +995 requests as the platform counts them (my count).
- **The bank channel folded into the bank read** (lever NB, section 22; the pair law's first half as a lever of its own): ten
  fresh networks -2.49% [-3.87, -1.34], bill +0.54%, MSE x bill -1.95%; parity with the physicist's torch module on network
  651 (-4.46% against -4.51%). Superseded by the pair law.
- Closed this morning (section 21): the join repair JR (exact, no gain), wider joins, a wider common part of the old pairs.

**State 2026-10-02 11:20 UTC.**
- **The S39 file is built, measured natively and in the lead's qualification** (section 19): the trainer's candidate 3 (five
  changes) on the S38 lever stack, with the star carried by its row sums alone. Against the fired S38 file on 10 fresh
  ranks: **final-layer MSE -0.04% [95% -0.75, +0.67], bill -2.24%, adjusted -2.27% [-2.98, -1.57]**, 659 more logged calls
  per network. Sha a158b9ab. With the trainer's 50-rank torch number for the dials (-0.70%) the board should be expected
  near -2.4% (about 1.86e-9).
- **New bill lever, the star by its row sums (SS, section 17): bill -2.37%, 150 calls fewer, MSE +0.50% [-0.06, +0.97] on
  10 fresh ranks, adjusted -1.88%.** What every layer takes from the star plane is 99.4-99.8% one vector (the plane's row
  sums pushed through the weights); the fired rank-32 form missed 1-2% of those row sums and spent 2.4% of the bill on the
  rest. No rank in between pays: rank 8 is -1.77% of bill for +0.31% MSE, a one-direction form measures worse than rank 0.
- **The native pair map takes an extra (3,1) plane with parity to the torch hook** (section 18; oracle matrix on network 651:
  native -15.31%, torch -15.60%, left-over at rounding level). The fourth-order carrier (S40) only has to produce its plane;
  nothing is ported yet, its design is being priced by the physicist and the lead.
- Candidate 3-prelim (three dial shrinks) measured nothing natively (+0.02% [-0.41, +0.52]) and was not fired.

**State 2026-10-02 08:36 UTC (kept for the record).**
- **S38 was fired as #333481 and is the bank: 1.907e-9** (S37 1.951e-9: -2.26%; final-layer MSE 1.612e-8, -1.65%; budget
  share 0.118301, -0.60%; 100 of 100 networks; platform wall 77.1 s mean / 88.3 s max, residual 0.29 s mean / 0.33 s max;
  section 14). My native session on 10 fresh ranks had said MSE -2.03%, bill -0.624%, adjusted -2.66%; with the trainer's 50
  torch ranks (-1.44%) I had expected about -2.1% on the board. At the boundary the file made 41,862 requests per network
  on a session's first two networks and 40,784 over a worker's 20 (S37: 42,864 and 42,186). It is the trainer's second dial-only candidate (learned mean correction x 0.8,
  k21n -0.048, b-source scale 0.965, bscale 1.03414, X1 0.59158) + the S37 lever stack with F4 strength 0.656 / 0.6 per age
  and K4L 0.75 / 0.75 + the edge cut with the tanh stand-in (EcutS) + both request levers (L3, L3o):
  `stack/candidate_T_C_B_F_D_Hfull_F4v4t0.656,0.6_K4Lv3c0.75,0.75_EcutS_L3_L3o_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_dials2.py`,
  sha 7c1a67f4. Against the S37 file on the same 10 fresh ranks, same order: **final-layer MSE -2.03% [95% -3.12, -0.99],
  bill -0.62%, adjusted -2.66% [-3.73, -1.63]**, 1,373 fewer logged calls per network. The trainer's torch number for the
  dials is -1.44% (se 0.26) on 50 held-out ranks, so the board should be expected near -2.1% adjusted.
- An earlier build on the trainer's first dial candidate (sha 0d51ccb9; with L3o 882dd7d1) measured -1.06% MSE and -1.70%
  adjusted on the same ranks; it is superseded and was not fired.
- **F4c, the physicist's exact-term mixture, is ported** (section 15): unit check and native parity on four networks pass; it
  costs +0.13% of bill and about 850 requests per network. **At his literal strengths it does not beat the fired S38 file
  natively: +0.18% MSE [-0.40, +0.91] on 10 fresh ranks**; the trainer's line search at candidate 2 has to show a gain
  before it goes into a file. Levers for candidate 3 are ready (F4C, K4L v4, the second-order skew term E2s) and the one
  command reads the trainer's new leaves; rehearsed end to end on a parameter file of mine.
- **7 levels of recursion for more products: priced on S38's own census and closed** (section 6): -0.43% of bill for about
  +4.3 s of wall and about +1,000 requests; S38 came back at 88.3 s on its worst network.
- **Every dial run M trains now has native parity** (section 10): ten dials moved at once, native change against the torch
  card's on networks 35 and 100: cosine 0.97-0.99 per layer, leftover at rounding level, no dial missing, doubled or swapped.
- **S36 was fired as #333429** from this lane's stack: `stack/candidate_T_C_B_F_D_Hfull_Phfull_v1_Rlvfuse_sized32_Rtsr_off.py`
  (sha 87b3d6ef: float32 fixes + full strips + regenerated prebind list + the release lane's two fragments). Board: 2.031e-9
  (S35 2.0672e-9, -1.74%), final-layer MSE unchanged, budget share -1.78%.
- **S37 was fired as #333435 and is the bank: 1.951e-9** (-3.94% against S36, -5.62% against S35; final-layer MSE 1.639e-8,
  -4.26% against S36; budget share 0.119015; 100 of 100 networks; platform wall 78.0 s mean / 87.6 s max). File (the lead's
  constants: F4 strength 0.6, K4L c 1, both on pairs at most 2 layers old):
  `stack/candidate_T_C_B_F_D_Hfull_F4v2t0.6_K4Lv2_Phfull_v1_Rlvfuse_sized32_Rtsr_off.py`, sha 2a25a074. My native session on 10
  fresh ranks had said, against RRPI: final-layer MSE -6.00% [95% -7.51, -4.35], bill -1.32%, adjusted -7.21% [-8.65, -5.67]
  (about -7.3% against S35). The board gave less: -5.62% against S35 and MSE -4.26% against S36, both just outside the upper
  end of those 10-network intervals (-5.77% and -4.35%). My 10 fresh ranks overstated the gain by about 1.7 points; a bootstrap
  over 10 networks is too narrow, and the lead's 24 held-out ranks (-5.2%) were closer.
- **Run M export path** (section 10): native forms for the five dial families of the run M card the exporter refuses (F4 strength
  per age, K4L strength per age, k22), one command from a parameter file to the candidate
  (`stack/from_params.sh <params.pt> <tag> '<extra levers>' <the trainer's exported candidate.py>`), rehearsed on parameter files
  of my own and checked natively against the torch card (the parity session, 06:35 UTC).
- **F4 port** (the physicist's fourth-order column shift): parity with the torch card per layer and per network; on 20 fresh ranks
  at strength 0.4 the full-strip stack with F4 is at MSE -3.83% [-4.46, -3.15], bill -1.70%, adjusted -5.47% (torch F4: -3.77%).
- **K4L port** (the lead's per-unit fourth cumulant): on 10 fresh ranks its increment on top of F4 is -0.86% natively against
  -0.69% in torch, correlation 0.990 per network.
- Both ports cost bill (+0.225% and +0.115%), about 1,775 charged ops and about half a second of kernel time per network (section 8).
- **Requests** (section 12): lever L3 loads every operand of the tree kernel with three requests instead of six, lever L3o writes
  two output quadrants with one addition. Both are byte-identical natively (networks 651 and 378, as first and second network
  of a process). Measured: L3 -1,002 / -1,227 logged calls for +0.127% / +0.075% of bill (about +0.065% over a worker's 20
  networks; my estimate was +0.049%); L3o -254 / -364 calls for exactly 0 flops.
- **Edge cut** (lever EcutS, in S38): bill -0.659% / -0.703% natively on the same two networks, for +0.016 to +0.026% of MSE in
  the torch simulation of the same mask.
- **Why the native file is 1.2% worse than the torch card** (sections 11, 13): its row cuts, shown by applying the same cut
  rules to the card (the simulation reproduces the native-minus-card prediction difference with cosine 0.80-0.98). The cuts
  cost about 1.3-1.5% of MSE and save about 5% of bill; the one threshold that looked wrong on networks 35 / 100 is at
  break-even on fresh ranks. No change proposed.
- Native sessions are allowed again since 06:05 UTC (one at a time, one thread; no torch runs while run M trains). Run since
  then: S38 on fresh list a, the L3 and L3o byte checks, the ten-dial parity session. Not run: the no-refit builds on RRPI
  (superseded by S38, which holds the same dials through the trainer's export), the all-cuts-off audit build (section 13
  answered its question in torch), YNT 3.0 (not recommended).

Goal set by the lead: a native candidate billed at least 3.5% below S35 on the same networks, final-layer MSE unchanged
(at most +0.3% pooled for an approximation, bill saving at least 5x the MSE loss), inside the platform limits.

**Where it stands: the 3.5% target is NOT reached. The best stacked candidate is billed 1.86-1.96% below RRPI (about 2.0-2.1%
below S35) and its error is unchanged: on 10 fresh networks the pooled MSE moves -0.11% (standard error 0.07), on 10 validation
networks -0.20% (section 3). Three deliverables, each with a byte-identical add-on that cuts requests by 2-3% (section 5):
no kernel change (bill -0.67%, kernel CPU -0.5%), pad-only strips (-1.3%, kernel CPU +0.8%), full strips (-1.9..-2.0%,
kernel CPU +3.6%).
What is left is small: about 0.5% in all, the largest item 0.25% and CPU-expensive (section 6).**

Parent: RRPI = `H/hill_climb_pair_rr_inverse_v1/candidate.py` (sha256 a3efb3a1...). S35 = `H/hill_climb_prefix128_fusion_v1/candidate.py`:
same predictions as RRPI (sha256 of `pred_35.npy` identical), bill +0.104% (release lane's paired census).
Sets: screen = 35,100,36,45; check = 52,58,64,70,76,80,84,88. "Bill change" = sum of candidate flops / sum of parent flops - 1 over
the set; "MSE change" = mean candidate final-layer MSE / mean parent final-layer MSE - 1.
Two ways of running, both native (in-process flopscope on numpy/OpenBLAS, one thread, under the guard), parent and candidate on the
same rank list in the same order, each in its own process:
- **harness** = `H/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py` through `run_native_v2.sh`; rows in `H/opus_engineer_*/rows.jsonl`;
  compare with `compare.py <parent tag> <candidate tag>`. RRPI baselines: screen bill 1051.716 Gflop, mean MSE 1.541461e-8
  (`H/opus_engineer_rrpi_screen_a`); check 2138.212 Gflop, 1.706161e-8 (`H/opus_engineer_rrpi_check_c`).
- **quick** = `quick.py` through `run_quick.sh` (same estimator calls, plus optional counters); rows in `logs/quick_<name>.json`;
  compare with `qcompare.py <reference json> <candidate json>`. A network's bill depends on its position in the process (the
  first networks pay more lazy tree binds: network 100 bills 264,557,131,454 flops as first network and 263,630,360,720 as
  second), so quick comparisons are always against an RRPI run with the same list in the same order.

## 1. Table of attempts

| attempt (directory) | what changed | bill change % (set) | MSE change % | byte-identical | CPU / requests | verdict |
|---|---|---|---|---|---|---|
| a01_response_floor | response-set floor 0.01 -> 0.0075 (live set and births unchanged) | +0.705 (screen, harness) | -0.805 | no | not comparable (load) | closed: bill x MSE -0.09%, the floor sits at its optimum |
| a02_select64 | selection counts rounded up to 64 with real units instead of 32 | +0.110 (screen, harness) | +0.644 (per net -2.2/+4.9/+3.1/-3.7) | no | - | closed |
| a03_hzs_ks | half-zero strips on plane reads only (first version), request price 750k / 250k / 0 flops | -0.777 / -0.951 / -0.963 (screen, harness) | +0.203 / +0.175 / +0.183 | no | - | superseded by a07 |
| a04_hzs_all | strips on reads (price 250k) + pushes and single products (750k) | -1.217 (screen), -1.219 (check), harness | +0.177, +0.226 | no | - | superseded by a07 |
| a05_hzs_v4 | same products, contiguous strip buffers (a strided vector operand drops numpy's matmul off BLAS: 8x slower) | -1.015 / -1.176 (screen, harness; read price 750k / 250k) | +0.151 / +0.177 | no | release instrument: kernel calls +614 / +1072, projected platform kernel +5.4% / +3.9% | superseded by a07 |
| a06_hzs_v5 | switch "pad-only": only products the parent already zero-pads | -0.594 (networks 35,100, release instrument) | - | no | kernel calls +264, projected platform kernel +1.37% | superseded by a07 |
| a07_hzs_v6 full (`candidate_k250000_g750000_d51.py`, sha 47ec2df4) | strips with a chunk-group leaf stage (4-5 batched matmuls per packet, no tiny products, no buffers); also converts products the parent keeps at 5 levels | -1.310 (screen), -1.294 (check), harness | +0.015, +0.148 | no | release instrument, networks 35,100: kernel calls +1046 (+2.5%), local CPU +4.2%, projected platform kernel +4.3% | kept: lever Hfull of the stack |
| a07_hzs_v6 pad-only (`candidate_k750000_g750000_d50.py`, sha a53b1735) | as above, only products the parent already pads | -0.701 (screen), -0.755 (check), harness | +0.223, +0.145 | no | kernel calls +232, local CPU +1.1%, projected platform kernel +1.35% (matmul time unchanged; inside that instrument's run-to-run spread) | kept: lever Hpad |
| a08_f32_core, T | the two first-chaos tail traces (`_ott_trace`, `_ynt_trace`) and the tail part of `omitted_radial_source` in float32: they widened float32 operands to float64 and paid the 2x rate on 943 Mflop per network | T + C: -0.441 (network 35), -0.438 (network 100), quick | +0.085, +0.472 | no (rounding level; only rows beyond the row cuts) | fewer float64 passes | kept |
| a08_f32_core, C | the join's Nystrom core R K^-1 R' (float64 solve + float64 product) by one float32 solve: K = Y[rows] = Q[rows] R, so the core is Q[rows]^-1 R'. cond(Q[rows]) = 4.6 and 6.2; the new matrix equals the parent's to 4-5e-8 relative in float64 and in float32 (`probe_core_100.json`) | (in the row above) | | no (rounding level) | 2 requests fewer per join | kept |
| a10_bd32, B | the birth update `_bd_update` in float32: 27 n x n passes per layer at the 2x rate (850 Mflop per network) although both outputs are stored in float32. Unit check `unit_bd32.py`: base equal to 5e-8, plane delta to 2e-7, billed 0.53-0.54 of the parent | B + F: -0.21 (quick, by difference) | (in the stack) | no (rounding level) | fewer float64 passes | kept |
| stack, F | the mean-correction net's nine plane reductions per layer accumulate float32 planes in float64 and are then stacked into float32 features; float32 accumulation | (in the row above) | | no (rounding level) | - | kept |
| a12_live_slice, D | where the dead-side row cut is active, the live rows are the leading rows of the pushed array: a slice instead of a scatter of the young columns of the full plane (rosters compared element by element first) | -0.017 (network 35), -0.020 (network 100), quick | 0.000 | **yes** (max abs difference 0.0 on 35 and 100) | 18-21 charged ops fewer per network | kept |
| **stack `T+C+B+F+D`** (`stack/candidate_T_C_B_F_D.py`, sha 7ef39d94) | the five levers above, no kernel change | **-0.669 (35), -0.669 (100)**, quick | +0.515, +1.015 on these two networks (see section 3) | no | charged ops +36 / +33 per network; release instrument: kernel calls +44, local CPU -0.5%, projected platform kernel **-0.5%** | **deliverable 1** |
| **stack `T+C+B+F+D+Hfull`** (`stack/candidate_T_C_B_F_D_Hfull.py`, sha e2007766) | plus half-zero strips, full | **-1.957 over a 12-network session (screen four -1.979, check eight -1.947; per network -1.66 to -2.32); -1.863 on 10 fresh networks; -2.113 against S35 on validation batch 1**, quick | 12-network session +0.205 (screen +0.070, check +0.266); fresh **-0.107**; validation **-0.201**; noise power unchanged (section 3) | no | charged ops +2.9% (38,118 -> 39,231 per network); release instrument, networks 35,100: kernel calls +1090, local CPU +3.5%, projected platform kernel **+3.6%** (3.5..3.7) | **deliverable 3** |
| **stack `T+C+B+F+D+Hpad`** (`stack/candidate_T_C_B_F_D_Hpad.py`, sha 5ac5d264) | plus strips, pad-only | **-1.316 (35), -1.275 (100)**, quick | +0.310, +0.227 on these two networks (section 3) | no | charged ops +271 / +262 per network; release instrument: kernel calls +276, local CPU +0.6%, projected platform kernel **+0.8%** (0.6..0.9) | **deliverable 2** |
| a13_prebind, P | the prebind list regenerated from tree-key censuses of fresh networks (the parent's list is fitted to the old census networks; section 5) | -0.006 (10 validation networks, quick, holdout: list fitted on other networks); 0.05-0.07 expected on the platform where the setup guard cuts the list | 0.000 | **yes** (10 of 10 networks) | charged ops per network 39,179 -> 38,328 (-2.2%); setup 24-30 Gflop instead of 35 | kept: add-on for every deliverable (`...+Prrpi_v1`, `...+Phfull_v1`) |
| **final builds with the regenerated lists** | `stack/candidate_T_C_B_F_D_Prrpi_v1.py` (sha 9830f644), `stack/candidate_T_C_B_F_D_Hpad_Prrpi_v1.py` (sha b1d16de6), `stack/candidate_T_C_B_F_D_Hfull_Phfull_v1.py` (sha 11c0e574) | networks 35 / 100 as first two of a process, quick, against RRPI: no-kernel build -0.783 / -0.788; full-strip build -2.062 / -2.466 (the pad-only build with the list was built and setup-tested, not run) | identical to the same build without the list | **yes** against the build without the list (35 and 100, both builds) | charged ops against RRPI's 38,406 / 38,330: no-kernel build 37,215 / 37,294 (-3%), full-strip build 38,551 / 38,138 (+0.4% / -0.5%) | **what I would hand to release** |
| **a14_f4, F4** (`stack/candidate_T_C_B_F_D_Hfull_F4v2.py`, sha 604d9a43; with the release fusion fragment: `..._F4v2_Rlvfuse_sized32.py`, sha 4dc76e11, built, not run) | NOT a bill lever: the physicist's F4 (`O/physicist/f4.py`), strength 0.4, columns at most 2 layers old, cheap anchor, ported into the live loop (section 8) | F4 itself **+0.225 (35), +0.224 (100)** of the build's bill; the build against RRPI on 20 fresh ranks **-1.695** [95% -1.77, -1.61], quick | **-3.83 on 20 fresh ranks** [95% -4.46, -3.15], lower on 19 of 20; torch card on the same networks -3.77; networks 35 / 100: -4.48 / +0.73 (torch -4.25 / +0.95) | no, by design; strength 0 with the step running is byte-identical to the build without F4 (35 and 100) | +690 charged ops per network (+1.8%); CPU anatomy pending | **passes: adjusted -5.47% [95% -6.08, -4.81] against RRPI on 20 fresh ranks** |
| **a15_k4l, K4L** (lever `K4Lv2`; **S37 file** = the lead's S36 file plus F4 at strength 0.6 and K4L: `stack/candidate_T_C_B_F_D_Hfull_F4v2t0.6_K4Lv2_Phfull_v1_Rlvfuse_sized32_Rtsr_off.py`, sha 2a25a074; native on 10 fresh ranks: MSE -6.00%, bill -1.32%, adjusted -7.21% against RRPI, section 8) | NOT a bill lever: the lead's K4L (`O/K4L_SPEC.md`), strength 1, pairs at most 2 layers old (section 8) | K4L **+0.115** of the build's bill (35: +0.1155, 100: +0.1146, 37: +0.1152, 39: +0.1127); F4 + K4L **+0.340** | networks 35 / 100 / 37 / 39, native against torch: K4L alone -2.34 / -3.31 / -2.53 / -2.23 against -2.69 / -3.65 / -2.94 / -1.96 (pooled **-2.61 against -2.83**); F4 + K4L -7.64 / -3.04 / -3.47 / -8.03 against -7.43 / -3.16 / -3.65 / -8.23 (pooled **-5.43 against -5.50**) | no, by design | K4L +1,082 charged ops per network, F4 + K4L +1,775 (+4.6%); CPU anatomy pending | parity holds: on 10 fresh ranks K4L's increment on top of F4 0.4 is -0.86% natively against -0.69% in torch, correlation 0.990 per network; K4L alone on the screen four about 0.35 points weaker than torch (rows the native row cuts do not push get no correction) |
| a17_tanh, E (`stack/candidate_T_C_B_F_D_E.py`, built, NOT RUN yet) | the learned covariance edge (layers 8-14) is 1.20% of the bill; its tanh on an (n, n, 16) tensor is 0.715% at 16 flops per element. Stand-in on pointwise ops, tanh(x) = S / sqrt(S^2 + K) with S a quintic in x (sinh series) and x^2 clamped at 81: 10 flops per element, largest error 2.0e-4 (`a17_tanh/probe_tanh.py`); plus the edge's bias added before the (n, n, 16) broadcast (one tensor pass fewer) | expected -0.27 (tanh) and -0.045 (bias) | expected neutral: the edge output moves by about 1e-3 of itself | no (2e-4 approximation of one activation; the bias fold is rounding level) | numpy's tanh is already vectorised: 46 ms per layer against 78 ms for the ten passes on this machine (`a17_tanh/bench_tanh.py`), about +0.2 s per network; +70 requests | to be measured on 35 / 100 after the quiet window |
| **a19_edge_cut, Ecut / EcutS** (EcutS is in the S38 file; `stack/candidate_T_C_B_F_D_Ecut.py`, `..._EcutS.py` are the screen builds, not run) | the learned covariance edge evaluated only on the layer's response rows x alive columns (its raw output is T_ij pdf_i gate_j with T bounded; the live loop's own rosters, about 800 x 912 of 1024 x 1024), 0 elsewhere; EcutS adds the tanh stand-in on the kept block | **EcutS, native: -0.659 (network 651, first of a process), -0.703 (378, second)** (the S38 file without L3 against the S37 file; the dials in it bill nothing). The cut kept 0.51-0.53 of the grid. Ecut without the stand-in: expected about -0.6, not run | torch simulation of the same mask on the card's own edge (`a24_edgecut/torch_edgecut.py`, networks 35 / 100 / 651): **+0.026 / +0.021 / +0.016% of MSE** (power of the prediction change over the MSE); a tighter mask (0.42-0.44 of the grid) +0.06-0.08%; no edge +2.2 to +2.9%. Unit check (`a19_edge_cut/unit_ecut.py`): kept block equal to the full tensor's to 1e-7 (tanh) / 2e-4 (stand-in). Natively the cut is measured only together with the dials (section 14) | no (approximation of one learned part) | +197 logged calls per network; local CPU of the S38 session 72.6 s per network against S37's 74.0 s | **kept: in S38.** Passes the 5x rule by a wide margin (0.68 points of bill for about 0.02 of MSE) |
| a18_f4b, F4B (draft, SHELVED by the lead 03:17Z: on 8 fresh ranks torch F4b is -4.47% where F4 0.6 is -5.36%; `stack/candidate_T_C_B_F_D_F4Bv3c1.0t12*.py`, built, NOT RUN) | the physicist's F4b (`O/physicist/f4b.py`): two shifts per young pair from the hub's exact fourth cumulant. Native form through a closed form of its per-pair least squares (section 9) | expected +0.32 in place of F4's +0.225 (22.9 flops per column element, unit check) | torch, 4 networks (physicist): -4.36% for exact S1 + S2 at strength 1, against F4 -2.33% | no, by design | 117 charged ops per layer in the unit check (F4: 46) | unit-checked against f4b.py's arithmetic (shifts 9e-8, beta 2e-7, diagonal 1e-7); native parity runs wait for the quiet window to end |
| **a21_load3, L3** (in the S38 file; on S37 alone: `stack/candidate_T_C_B_F_D_Hfull_F4v2t0.6_K4Lv2_L3_Phfull_v1_Rlvfuse_sized32_Rtsr_off.py`, sha ab626d8c, not run) | a REQUEST lever, not a bill lever: every operand load of the tree kernel in three requests instead of six (the key's four level-1 data slots bound as one view; one copy of the whole operand; the alt-basis sum taken in place; section 12) | **native: +0.127 (network 651, first of a process), +0.075 (378, second)** (the S38 file against the same file without L3, sha e544d59a): each load copies one quarter more (the steady +0.05) and every tree key pays one reshape at its first load (160 keys on the first network, 53 on the second). About +0.065 over a worker's 20 networks; my estimate was +0.049 | 0 | **yes, natively**: predictions byte-identical on 651 and 378; and in the unit check (22 product calls) | **native: -1,002 / -1,227 logged calls** on the two networks (expected about -1,260 on S37); setup makes fewer calls (7,074 against 7,269) for the same flops | **kept: in S38.** The release lane's boundary session must count the requests and time the residual |
| **a21_load3, L3o** (`stack/candidate_T_C_B_F_D_Hfull_EcutS_F4v2t0.6_K4Lv3c0.75,0.75_L3_L3o_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_dials_ds08.py`, sha 882dd7d1 = the S38 file + L3o; on S37: sha 042bf49a, not run) | a REQUEST lever: the upper two output quadrants of a tree product written by one pair addition (section 12) | **native: exactly 0 flops** on both networks (265,884,291,670 and 256,481,897,563 with and without) | 0 | **yes, natively**: predictions byte-identical on 651 and 378; and in the unit check (22 product calls, k-split and strip kernels included) | **native: -254 / -364 logged calls** on the two networks (the first networks also bind the pair views: 85 keys on the first network, 34 more on the second; about -440 in steady state) | **ready: S38 + L3o is the file I would send**; the lead's S38 recipe names L3 only, so it is his choice |
| a23_ds, DS (`stack/candidate_T_C_B_F_D_Hfull_F4v2t0.6_K4Lv2c0.73_DS0.85_Phfull_v1_Rlvfuse_sized32_Rtsr_off.py`, sha 79c0d033 = the lead's no-refit list: S37 with K4L strength 0.73 and the learned mean correction x 0.85; built, unit-checked, NOT RUN on a network) | NOT a bill lever: a global scale on the learned mean correction, layers 1-15 (the lead's 05:32 UTC ask; torch, fresh ranks: x 0.85 about -0.8% MSE, K4L 0.73 about -0.3%). The scale multiplies the per-layer gain constant of the correction net where it is formed (once per layer and process) | 0 | by design | no, by design; unit check `a23_ds/unit_ds.py` (the correction net alone, layers 1 / 7 / 15, synthetic inputs): scale 1.0 byte-identical, scale 0.85 = 0.85 x the correction to 5-9e-8, hidden state byte-identical | no operation added per prediction | the lever itself has not run on a network (parity builds for 35 / 100: `..._F4v2_K4Lv2_DS1.0.py` sha c7c351c9, `..._DS0.85.py` sha 1e498b7a). The same scale ran natively in S38 in the trainer's form (x 0.8 folded into the correction net's output weights by the exporter); the lever stays for parents the trainer did not export |
| a20_runm, GN (lever `GN<knob><value>`) | NOT a bill lever: the parent's covariance-output knobs iso / k21n / k31 at other values (an appended override of `_GN_KNOBS`), for a refit that moves them (the trainer's k21n lead) | 0 | by design | no, by design | - | **parity holds** (ten-dial session, section 10): iso / k21n / k31 within +0.001 / -0.014 / +0.040 of the card on network 35 and +0.155 / +0.006 / +0.107 on network 100 (0 = as in the card) |
| **a20_runm, K22 + export path** (`stack/from_params.sh`; rehearsal files sha 67d1b145 (run natively), 83d9cfdf, 09fbe869) | NOT a bill lever: the run M card's five dial families the exporter refuses, as levers (F4 strength per age, K4L strength per age, k22), and one command from a parameter file to the candidate (section 10) | 0 (one scalar multiplication per layer for k22) | by design | no, by design | - | **parity holds** (networks 35 / 100, ten dials at once): native change against the card's: cosine 0.970-0.992 / 0.969-0.990 per layer, size ratio 1.01-1.04, leftover 4.96 / 5.77e-6 rms at the final layer = two rounding-level native variants; every dial within +-0.055 (35) and +-0.16 (100) of "as in the card" |
| S38 first build, dials 1 (`stack/candidate_T_C_B_F_D_Hfull_EcutS_F4v2t0.6_K4Lv3c0.75,0.75_L3_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_dials_ds08.py`, sha 0d51ccb9; with L3o: sha 882dd7d1; section 14) | the lead's 06:05 UTC recipe: the trainer's first dial-only candidate (learned mean correction x 0.8, k21n -0.048) + the S37 levers with K4L strength 0.75 on both ages + EcutS + L3 | -0.624 against the S37 file [95% -0.652, -0.591], 10 fresh ranks (list a), quick | -1.062 [95% -1.878, -0.267]; per-network mean -1.007 (se 0.428), lower on 7 of 10; the trainer's torch number for these dials: -0.98 (se 0.23) on 50 held-out ranks | no, by design | logged calls 36,173 per network against 37,206 (-1,033); local CPU 72.6 s against 74.0 s | superseded by the next row before any fire (adjusted -1.70% [95% -2.50, -0.91] against S37) |
| **S38 = the next board file, dials 2** (`stack/candidate_T_C_B_F_D_Hfull_F4v4t0.656,0.6_K4Lv3c0.75,0.75_EcutS_L3_L3o_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_dials2.py`, sha 7c1a67f4; section 14) | the lead's 07:03 UTC recipe: the trainer's second dial-only candidate (as dials 1, plus b-source scale 0.965, bscale 1.03414, X1 0.59158) + the S37 levers with F4 strength 0.656 / 0.6 per age and K4L 0.75 / 0.75 + EcutS + L3 + L3o | **-0.624 against the S37 file** [95% -0.652, -0.591], 10 fresh ranks (list a), quick | **-2.034** [95% -3.122, -0.986]; per-network mean -1.982 (se 0.531), lower on 9 of 10; against the dials-1 build -0.98 [-1.56, -0.35]; the trainer's torch number for these dials: -1.44 (se 0.26) on 50 held-out ranks | no, by design | logged calls 35,833 per network against 37,206 (-1,373) | **passes: adjusted -2.66% [95% -3.73, -1.63] against S37; in the lead's qualification since 07:25 UTC** |
| a20_runm, BR (lever `BR<x>`; `stack/candidate_T_C_B_F_D_Hfull_BR0.5_EcutS_..._runM_dials_ds08.py`, sha 7d70ac05, built, NOT RUN) | NOT a bill lever: the lead's birth dial (the newborn pairs' coefficient d (1 + x radial / 3)); one loop line | 0 | torch (lead, 6 held-out ranks): x = 0.5 +0.10% (se 0.31), x = 0 +0.84% | no, by design | - | closed by the lead (06:56 UTC) before any native run |
| **a27_f4c, F4C** (lever `F4Ct<theta>c<c1>,<c2>`; `a27_f4c/fragment_f4c_v1.py`; parity builds on RRPI sha 6cefab2e / d692d8cf; section 15) | NOT a bill lever: the physicist's exact-term mixture (F4's rule + the hub's exact S1, S2), six coefficients per pair and age fixed at the pair's birth; replaces F4 | **+0.130 / +0.127 over F4** (networks 474 / 648, native; the cohort born at layer 0 is shifted too) | torch (physicist, 48 held-out ranks): **-0.48** [95% -0.83, -0.12] against F4 + K4L. Native against his stored arms on 474 / 648: -0.44 / -0.36 where torch has +0.13 / -0.80 (pooled -0.40 against -0.35) | no, by design; unit check against his reference: 2e-8 to 4e-7 | **+856 / +846 logged calls per network**; local CPU unchanged | **parity holds on four networks** (projection of the native change on the torch change 0.97-1.04, leftover 4.7-5.6e-6 = two rounding-level native builds). **On the fired S38 stack at his literal strengths: MSE +0.18% [95% -0.40, +0.91] on 10 fresh ranks, adjusted +0.33%: no gain**; not for a file until the trainer's line search at candidate 2 shows one |
| a29_e2, E2s (lever `E2s<x>`; `a29_e2/fragment_e2_v1.py`; unit-checked, in the rehearsal file) | NOT a bill lever: the lead's second-order skew term of the mean formula, x k3^2 (alpha^4 - 6 alpha^2 + 3) pdf / (72 sigma^5), layers 1-15, through the correction slot | about 0.0003 (50 flops per unit and layer) | torch (lead, network 651, x = 1): +0.8% with the learned correction on; the trainer's line search decides | no, by design; unit check: equal to the lead's formula to 5e-16 | 10 charged ops per layer | ready for candidate 3 if the trainer's e2 is not 0 |
| a15_k4l, K4Lv4 (`a15_k4l/fragment_k4l_v4.py`) | K4L v3 with the per-age multiplication always made (v3 skipped a strength of exactly 1): operations independent of the values, as the release lane asked for its literal-only fast path | 0 | 0 | same arithmetic as v3 unless a strength is exactly 1 | +1 call per cohort only in that case | in the F4C parity build and the rehearsal file |
| S38 dials 3 = the trainer's candidate 3-prelim (`stack/candidate_T_C_B_F_D_Hfull_F4v4t0.656,0.6_K4Lv3c0.75,0.75_EcutS_L3_L3o_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_dials3.py`, sha 65c199aa; literal-only against the fired S38 file) | three more dial shrinks of the trainer on S38's stack | 0 (identical to the flop on every network) | **+0.02 against S38** [95% -0.41, +0.52], 10 fresh ranks (list a); per-network mean +0.04 (se 0.26), lower on 7 of 10; the trainer's torch number: -0.25 (se 0.09) on 50 ranks | no, by design | logged calls identical on every network | not fired: nothing measurable natively; superseded by candidate 3 with five changes (the S39 row) |
| **a33_starsum, SS** (lever `SS<form>`; `a33_starsum/fragment_ss_v3.py`; rank 0 is in the S39 file; section 17) | the star plane moved to the next layer by its ROW SUMS (one vector through W), exactly, with the parent's factors at a chosen rank on top (32, 16, 8), one direction, or nothing (rank 0: no factorization, no QR, no thin product) | against S38 on 10 fresh ranks: **rank 0 -2.372** [-2.41, -2.33]; one direction -2.353; rank 8 -1.765; on networks 651 / 378: rank 16 -1.14 / -1.18, rank 32 + row sums +0.036 / +0.037 | rank 0 **+0.50** [95% -0.06, +0.97]; one direction +0.89 [+0.36, +1.47]; rank 8 +0.31 [-0.28, +0.84]; rank 16 +0.20 / -0.19 and rank 32 + row sums -0.24 / -0.52 (two networks each) | no (a change of what is carried); unit check against float64: 7e-7 (rank 16), 1e-6 (rank 0) | rank 0: **-150 logged calls** per network; ranks 8-32: +135 | **kept, rank 0: adjusted -1.88% [-2.44, -1.43] against S38.** 4.7x on ten networks, a little under the 5x rule; no rank can earn its bill back (everything beyond the row sums moves the prediction by 0.6% of the MSE's power and costs 2.4% of bill) |
| **S39 file = candidate 3 (five changes) + SS0** (`stack/candidate_T_C_B_F_D_Hfull_F4Ct0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv4c0.8597296897041633,0.8597296897041633_EcutS_L3_L3o_SS0_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_cand3_five.py`, sha a158b9ab; section 19) | the lead's 10:40 UTC recipe: the trainer's candidate 3 file (five changes) + the S38 lever stack with the exact-term module (F4C: rule 0.656 / 0.6, exact S2 0.444 / 0.171) and K4L 0.860 + the star by its row sums | **-2.239 against S38** [95% -2.27, -2.20], 10 fresh ranks (list a), quick | **-0.04** [95% -0.75, +0.67]; per-network mean +0.01 (se 0.39), lower on 4 of 10. Parts: star +0.50, candidate 3 on top -0.54 (torch on the same ten -0.31, on all 50 -0.70; native against torch per network: correlation 0.81) | no, by design | logged calls 36,492 per network against 35,833 (+659: module +809, star -150); peak RSS 4.62 GB | **passes: adjusted -2.27% [95% -2.98, -1.57] against S38; handed to the lead's qualification 11:17 UTC** |
| a34_r31 (no lever: the parent's `extra31` input of the pair map; section 18) | NOT a bill lever: the input a fourth-order carrier needs. The native pair map takes an extra (3,1) plane; fed with the physicist's oracle matrix on network 651 (file unchanged, two names set on the loaded module) | not a price (a full float64 plane built outside the budget) | native -15.31% against torch -15.60% (the physicist's arm K31f) | per layer cosine 0.94-0.995 from layer 3 on, left over 1.4-5.1e-6 rms = rounding level | - | **parity holds**; nothing ported yet (the carrier is S40, its design is being priced) |
| a32_c2k, C2K (lever `C2K<b>`; `a32_c2k/fragment_c2k_v1.py`; unit check exact, 2.06 flops per covariance element; built, NOT RUN on a network) | NOT a bill lever: the lead's row-energy term of the per-unit fourth cumulant, k4 += (6 b / n) (C^2)_ii with the isotropic level projected out | about +0.01 | torch (lead, E33, four held-out ranks): b = 0.125 +1.2%, 0.25 +3.3%, 0.5 +10.4% | no, by design | - | closed by the lead (09:26 UTC) before any native run |
| a35_join, JR / JK / LU (levers; section 21) | NOT bill levers: the norm coupling a join's projection drops, put back through the transported star (JR); the rank of the old tier's frame at the three joins (JK); the rank of the old pairs' common part (LU) | JR +0.044; JK 384,384,384 +1.70; JR + JK 384,256,384 -1.22; LU 384 +1.33 (against S39) | JR, 10 fresh ranks: -0.11 [95% -0.74, +0.53] (all layers -0.28); JK 384,384,384 on 651 / 378: -2.79 / -1.56; JR + JK 384,256,384: +2.55 / +5.66; LU 384: -0.60 / -0.52 | no, by design | JR: about +50 logged calls | closed as file changes: the repair is exact and gains nothing at the final layer; wider joins break even at best |
| **a37_fold, NB** (lever `NB<forms>[d<dfit>][x<dscale>][b]`; `a37_fold/fragment_nb_v1.py` (the ten-network session), `_v3.py` (parity), `_v4.py` (lean, unit-checked, not run); section 22) | NOT a bill lever: the physicist's bank channel of the fourth-order norm coupling (`O/physicist/ncar_fold.py`, level 'read'): one more weight per bank column, folded into the left factor of the bank read | **+0.539** against S39 on 10 fresh ranks (v1); +0.52 with the 1.41 factor; +0.37 with the card's beta | **-2.49** [95% -3.87, -1.34] on 10 fresh ranks, lower on 10 of 10, all layers -3.25; on 651 / 378 with the factor 1.41: -4.74 / -1.92 (three forms -4.46 / -1.04; beta -4.99 / -1.90; one regressor -4.72 / -1.42) | no, by design; parity with the torch module on 651: cosine 0.94-0.97 per layer, left over 0.8-4.9e-6, MSE -4.46% against -4.51% | +727 logged calls (v1), +670 (v3), +605 (one regressor) | superseded by the pair law, which contains it |
| **a38_pairlaw, NP** (lever `NP[f<forms>][r<dfit>][b][g<diag>][a]v<n>`; `a38_pairlaw/fragment_np_v1.py` = the parity version, `_v2.py` = lean, `_v4.py` = lean + switches; section 23) | NOT a bill lever: the lead's pair-carried law of the fourth-order norm coupling (`O/ncar6.py`, level 'read'): a second weight per bank column and a cross weight per pair, their sources from a per-unit law at the birth hubs, the fold in the bank read | parity version **+1.168** against S39 (10 fresh ranks); lean one form +0.98; three forms +1.01; three forms + beta **+0.875** | parity version, 10 fresh ranks: **-7.50** [95% -9.45, -5.49], lower on 10 of 10, all layers -6.95 [-7.73, -6.13]; on 651 / 378: parity -13.04 / -6.51, lean one form -13.60 / -6.40, three forms -14.88 / -6.61, three forms + beta -14.24 / -6.54. Torch (trainer, 50 held-out ranks, one form): -5.80 (se 0.50) | no, by design; lean against parity: 4.7e-6 rms at the final layer on both networks = two rounding-level builds; unit checks against a float64 transcription of ncar6: plane 3-9e-5 of the fold term, weights 1e-6 | logged calls: parity **+3,568**, lean one form **+871 / +885**, spec **+929 / +913**; requests by my count about +995 for the spec | **in the S40 file** |
| **S40 file = S39 + NPf3bv4** (`stack/candidate_T_C_B_F_D_Hfull_F4Ct0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv4c0.8597296897041633,0.8597296897041633_EcutS_L3_L3o_SS0_NPf3bv4_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_cand3_five.py`, sha e2be4c75; section 23) | the lead's 13:40 UTC spec: the pair law with three forms removed, the card's beta for the carried weights, all sources and the fold at strength 1, layers 1-14 | **+0.876 against S39** [95% +0.874, +0.879], 10 fresh ranks (list a), quick | **-7.93** [95% -9.97, -5.95], lower on 10 of 10; all layers -7.32 [-7.95, -6.70]; MSE x bill -7.16 [-9.23, -5.14] | no, by design | logged calls +913; at the boundary 43,628 requests a network against 42,624, kernel instructions x1.025, projected wall 79.1 / 89.7 s, residual 0.309 / 0.355 s | **fired as #333559** (14:52 UTC) |
| a38_pairlaw NP v5 + a15_k4l K4L v5 (S40 with both: `stack/candidate_..._K4Lv5c0.8597296897041633,0.8597296897041633_EcutS_L3_L3o_SS0_NPf3bv5_...__runM_cand3_five.py`, sha cf1f735c; NP v5 alone 5bd2ec6f; K4L v5 alone c5f0a289) | NOT physics: the pair law's exchange of the pair columns by split + concatenate, its diagonal by a product and a matrix-vector product, the f-scaling in place; the K4L birth coefficients from two polynomial tables | 0 for NP v5; +0.011 for K4L v5 (two table gathers a layer) | NP v5: rounding level expected (the diagonal's summation order); K4L v5: byte-identical expected | NP v5: no (rounding); K4L v5: coefficients bit-identical in the unit check | expected: kernel instructions -5 G of S40's +8.4 G; requests -13 (NP v5) and -238 (K4L v5) | unit-checked; native checks queued; a candidate for the designated variant |
| **a38_pairlaw NP v6 / v7, the switch k (K31L4; section 24)** (`stack/candidate_..._K4Lv5c..._NPf3bk1v7_...__runM_cand3_five.py`, sha 8d110449) | NOT a bill lever: the lead's fourth-chaos plane of the young pairs' local quartic in the same read (`O/k31l.py`, chaos4, level 'read'), from K4L's own birth coefficients | **+0.292 / +0.294** against the same file without it (v6: +0.39; means over the response rows: +0.36) | **-2.33 / -2.36** (651 / 378); v6 -2.15 / -2.58; torch (lead) -2.09 / -2.65; trainer's 25 ranks -1.87 (se 0.39) | no, by design; v7 against v6 4.3-4.6e-6 rms; unit check against the reference's formulas 1-3e-5 | +479 / +475 logged calls | **ready for S41**; waiting for the dial values |
| **S41 = the stack with the trainer's first dials, the table and HV** (`stack/law_file_from_trainer.sh law_dials_d6 2 s41`; sha 33bac9db, its arithmetic b99d7562; fired by the lead as #333587 with the release lane's two fragments; section 25) | pair law v8 with every literal from a dials file, K31L4 at ages 1-2, the K4L birth by tables, the strips' leaf views by one index expression each | **+0.304** against the fired S40 (ten fresh networks) | **-3.26** [95% -4.97, -1.66], 9 of 10 lower; all layers -2.80 | no, by design | +250 logged calls; HV -580 / -362 index requests (first / second network) | **fired 16:51 UTC, grade pending** |
| a40_setup, CS (`a40_setup/fragment_cs_v1.py`; section 26) | the once-per-process constants and buffers built at the head of the binding phase | first network -0.057 (their reshapes are billed in the setup) | 0 | **yes** (651 / 378) | first network **-452 requests**, others unchanged; setup +0.03 s | **ready for the next file**; reset path checked (`a40_setup/reset_check.py`); boundary not run by me |
| byte-identical diet: a27_f4c v3, a15_k4l v7, a38_pairlaw v9, a19_edge_cut v2 (`law_file_from_trainer.sh ... same`; sha d610157e on the first dials; section 26) | views by one split per array, block gathers by one index, a row factor by one einsum, scatters that widen | **-0.023** (the edge cut's gathers) | 0 | **yes** (651 / 378) | **-323 requests on every network** | **ready for the next file** |
| hzs ks v8, HV2 (`hzs/fragment_ks_v8.py`) | the strips' leaf views cached per tree instead of per product | 0 | 0 | **yes** (651 / 378; same views by unit check `hzs/unit_views_v8.py`) | **-146 / -122 index requests** (first / second network); replay of a ten-network key census: 285 -> 135 a network | **ready for the next file** (in `law_file_from_trainer.sh ... same`) |
| deep diet: a27_f4c v2, a15_k4l v6 (`... deep`; sha 299e858c; section 26) | all cohorts in one block of operations (own entry and difference to the partner; plus blocks side by side) | +0.073 (ten fresh networks) | +0.26 [95% -0.16, +0.60], 8 of 10 higher (a second rounding draw: section 27) | no: 4.1-5.6e-6 rms | -670 requests on every network | not a default: 0.07% of bill for 350 requests more than the byte-identical diet |
| second dial set, `law_dials_d7_ages12` (table sha faa4bdd7, constant regressor sha e0598c4e; section 27) | the trainer's second solve of the pair law's dials (torch: -0.84%, se 0.19, on the constant regressor against the first dials on it) | -0.03 (the diet) | table **-0.18** (se 0.20), constant regressor **+0.14** (se 0.30), ten fresh networks against two rounding draws of S41; second against first on the constant regressor -1.15 (se 0.27) = the torch comparison, but first dials + constant regressor is +1.31 against S41 | no | as the diet | **no measurable gain over S41 natively**: the solve's base was 1.3 points behind the fired file |
| S42 + skew-corrected birth `BSK1` (sha 6c4840f2; section 31) | birth weight x (1 - (t3 / 6) He3(alpha)), layers 0-14 | +0.002 | **-0.34 pooled** against two draws of S42 (mean of per-network changes -0.36, se 0.32, 8 of 10 lower; torch -0.34) | no | +105 requests | real and small; in S43 if the caps allow |
| S42's arithmetic in the deeper diet (sha 90ab448a; section 31) | F4c v2 + K4L v6 in place of v3 / v7 | +0.078 | +0.10 (se 0.14) against the fired arithmetic's run | no (rounding level) | **-410 requests on every network** | the base of S43 by the lead's choice |
| **S43 skeleton** (`stack/law_file_from_trainer.sh law_dials_d6 2 deep bsk1 s2`; sha 0df2ba07; section 32) | S42's step-2 file on the deeper diet + `BSK1` | +0.080 | **-0.28 pooled** against two S42 draws (mean of per-network changes -0.29, se 0.26; 8 of 10 lower) | no | **-305 requests on every network** | the base of the trainer's hand-overs |
| a44_cov, DCV (lever `DCV<r>`; `a44_cov/fragment_dcv_v1.py`; sha ae5c296e on the S43 skeleton; section 32) | the covariance sandwich of layers 1-15 on the live roster of the layer below (rounded up to 64): C_pre = W_a' C_aa W_a | -0.65 / -0.75 (651 / 378) | **+16.32 / +6.72** (651 / 378, against the skeleton's run) | no | -1,985 / -2,056 | **CLOSED**: the cut units' covariances with the live ones carry 0.6-2e-3 of C_pre; a rank-32 repair costs what the cut saves |
| **a44_cov, DCV at gate floor 1e-3** (lever `DCV64f0.001`; `a44_cov/fragment_dcv_v2.py`; sha b7d33806; section 32) | the same cut on the units with gate >= 1e-3 (alpha > -3.09): on 651 layers 9-14 at m = 960 / 896 | **-0.250** (ten networks) | **+0.03 pooled** (se 0.06, 5 of 10 lower) against the skeleton's run | no | **-1,397 requests** | **in S43** (lead 03:24Z) |
| DCV at 1e-3, the cut product's kernel (`DCV64f0.001p` sha 9e45e268 / `t` sha 18266922; `a44_cov/fragment_dcv_v3.py`) | the padded specialised 1,024 product / the specialised product's tiles at the inner size | -0.316 / -0.433 (651) | -0.18 / -0.15 (651, rounding) | no | -18 / **+6,537 on the first network** (lazy binds) | padded: the request-neutral form; tiles need their keys bound at setup |
| **S42 candidate, step-2 weights**: the same file with `HC@s2` (sha 0fccf396; section 30) | the trainer's second steepest-descent step of the 128 weights | -0.027 | **-2.15 pooled** (mean of per-network changes -2.09, se 0.28; **10 of 10 lower**); against the step-1 file -0.61 (se 0.35) | no | as the step-1 file | **the file I recommend for S42; with the lead** |
| **S42 candidate**: S41's arithmetic + free diet + head column `HC@s1` (sha 33582663; section 30) | one new input column of the learned mean correction, 128 weights from the trainer | -0.027 (ten fresh networks, against the S41 source) | **-1.55 pooled** against the mean of two S41 draws (mean of per-network changes -1.52, se 0.38; 8 of 10 lower) | no | +294 requests against the same file without the column; fewer than the fired S41 on every network | **meets the lead's fire rule by 0.25 points; with the lead** |
| base-term switches `SO2m..t..f..`, `BSK`, `GSK`, `BK3r` (sections 28-29) | the refit's physics base, each term a lever | +0.30 for the target's full form, +0.15 without rho terms, 0 for the rest | not measured on ten networks (single-network parity for three of them, section 28) | no | 658 / 434 / 196 requests for the target's three forms, 444 for the other four together | written and unit-checked; waiting for the base |
| a09_stack / rc v1 | the three row cuts rounded to 32 instead of 64, with strips | about -0.1 (estimate from the cut counts of network 100: 6 of 23 cut sites lose 32 rows, three more are held by the 7-level push, a 32-row strip saves 2.6% of a low-rank reconstruction product) | would change which rows beyond the thresholds are kept | no | - | dropped, not run |
| hzs gen v6 | strips under the low-rank base's B-tree cache | 0: the bill rule takes none (5 levels 1214 Mflop, strip 1205 + requests) | - | - | - | dropped; it also redefines `_predict_binary`, which builders patch |
| a11_bare_views, V | setup binds the bare exact tree view of every key the kernels can ask for, so lazy binds bill nothing | up to -0.37 (RRPI, 12-network session), byte-identical | 0 | yes by construction (not run) | setup flops 35 -> 141 Gflop per worker for the full key families | withdrawn, not recommended (section 5) |
| d01_plain_matmul (diagnostic, not a candidate) | every fast-multiplication kernel off (`_WINOGRAD = False`) | +90.24 (screen), +90.26 (check), harness | +0.082, +0.080 | no | 13.2 s per network against 31.1 (quiet box) | the noise-free reference of section 3; the transforms are more than half of our local CPU |

CPU: the harness's `cpu_s` on this shared machine swings 2-3x with load (the same RRPI run read 26-37 s per network on the screen
set and 30-63 s on the check set), so CPU statements use the release lane's instruction-count instrument
(`O/release/cpu_anatomy.py` + `variant_eval.py`, parent reference `release/runs/anatomy/rrpi_clean`; outputs in `engineer/anatomy/`).

## 2. The levers

**Half-zero strips (H).** The selection rounds its counts to multiples of 32, so many products have a dimension that is an odd
multiple of 32. The parent's tree kernel (alternative-basis Winograd, in place) runs those either at 5 levels of recursion, or
zero-padded by 32 to 6 levels, where the 32 pad rows cost as much as 64 real rows. The strip kernel interleaves the 32 odd rows
with 32 zero rows on the deepest index bit (the strip sits at the even 64-blocks, last leaf row). Then, exactly (`hzs/probe1.py`):
the strip leaf ROW needs only 4 of the 7 deepest sub-products (0, 2, 3, 5: the A tree is exactly zero there in 4 and 6, and 1 feeds
only discarded outputs); the strip leaf COLUMN needs 0, 2, 5, 6; an odd INNER dimension leaves the strip term exactly zero in
sub-products 2, 3, 6. So an odd 32 is billed at the 5-level rate on 32 rows instead of the 6-level rate on 64, and the rest of the
product runs at 6 levels. Trees, keys, forward transforms, accumulation and unwind are the parent's. Certified against float64 on
all 17 strip layouts (`hzs/certify_forced.py`: relative error 2.5e-5 to 5.3e-5, the level of the parent's kernels) and in situ on
every strip product of network 100 (`quick.py --check`: at most 1.9e-4, the parent's own level on those reads).
Files: `hzs/fragment_ks_v6.py` (plane reads), `hzs/fragment_gen_v7.py` (packet pushes, single products; v7 = v5 with the 7-level
exclusion tied to `_D7_ON`, same behaviour while that switch is on). Switches: `_HZ_ON` / `_HZG_ON`; `_HZ_D5` / `_HZG_D5`
(False = pad-only); `_HZ_PRICE` / `_HZG_PRICE` = flops a product must save per extra request.
Cost (network 100, shadow timing): products the parent already pads are time-neutral under strips; the full variant's extra
kernel time comes from products moved from 5 to 6 levels: about 0.65 points of bill for about +3% kernel time, i.e. about 4.5%
kernel time per 1% bill (the release lane measured 8-16% per 1% for the 7-level layers).

**Float32 where float64 bought nothing (T, B, F), and one algebraic identity (C).** flopscope bills float64 at twice the float32
rate. Network 100 pays 1.53% of its bill at the float64 rate; these four levers remove the parts whose results are stored in
float32 or feed an estimate that needs no more: tail traces 0.31%, join core 0.32%, birth update 0.32%, feature reductions 0.12%
(shares before the change; each roughly halves, the core loses its product as well).

Rounding parity in situ (`stack/parity_probe.py`, network 100: RRPI's own prediction, and at every call the float32 version on
the same inputs, `stack/parity_100.json`): tail traces, 18 calls, relative difference at most 2.2e-6 (median 7e-7); birth update,
15 layers, new base at most 4.9e-8, plane delta at most 1.4e-6; feature matrix, 14 layers, at most 7.9e-8 (worst column 5e-7).
Join core: 4-5e-8 (`a08_f32_core/probe_core_100.json`).

**Live rows by slice (D).** Byte-identical; see the table.

`stack/build.py <levers joined by +>` builds any subset from RRPI (and can append release-lane fragments, e.g.
`T+C+B+F+D+Hfull+Rlvfuse_sized32`); every candidate has a `.build.json` with the parent sha, every text edit and the appended text sha.

## 3. MSE: the stack has RRPI's rounding-noise power; the +0.2% is RRPI's selected draw

`noise_decomp.py` (stored predictions only). The plain-matmul diagnostic d01 is the estimator without fast-multiplication rounding
noise. Write a variant's final-layer prediction as y = y_d01 + n and the noise-free error as c = y_d01 - truth; then
MSE = |c|^2 + 2<c, n> + |n|^2. Pooled over the 12 screen and check networks, in % of the noise-free MSE:

| variant | noise power |n|^2 | alignment 2<c, n> (se of a 12-network mean) | MSE vs noise-free | MSE vs RRPI |
|---|---|---|---|---|
| RRPI | +0.068 | **-0.149** (0.069) | -0.081 | 0 |
| a04 strips, first version | +0.074 | +0.056 (0.068) | +0.130 | +0.211 |
| a07 strips, pad-only | +0.070 | +0.019 (0.116) | +0.089 | +0.170 |
| a07 strips, full | +0.064 | -0.038 (0.096) | +0.026 | +0.106 |
| stack T+C+B+F+D+Hfull | +0.069 | +0.056 (0.068) | +0.125 | +0.205 |

- The noise power, the only part of the difference that is a real cost, is the same for every variant: 0.064-0.074% of the MSE
  against RRPI's 0.068%. Strips and float32 levers add no measurable rounding noise to the final prediction.
- Every difference to RRPI is the alignment term. For a draw nobody selected it is zero on average (per-network sd 0.24%); RRPI's
  is -0.149% over these 12 networks, 2.2 standard errors from zero, and largest on 35 and 100 (-0.44, -0.45%), the two networks every
  hill-climb step was screened on. The variants' own alignment terms scatter around zero (-0.04 to +0.06%).
- Any rounding-level change, however small (the float32 traces alone, or removing all kernel rounding), moves the final prediction
  by the same 3.3-5.1e-6 rms, 3-4% of the error rms: single networks then move by +-0.2..0.5% whatever is changed.
- Consequence: on networks RRPI's kernels were not selected on, the expected MSE change of the stack is 0.00 +- (0.24% / sqrt(n)).
- **Checked on fresh networks.** Ranks 660-669 (never used for kernel selection, parameter training or validation), RRPI and the
  stack each as one 10-network session (`logs/quick_rrpi_fresh_f1.json`, `logs/quick_tcbfd_hfull_fresh_f1.json`): bill -1.863%,
  pooled final-layer MSE **-0.107%**, 8 of 10 networks lower; per-network change mean -0.111%, sd 0.220%, standard error 0.069%.
- Validation batch 1 (37, 39, 42, 48, 59, 63, 66, 67, 71, 73), the stack as one 10-network session against the bank's own native
  rows (`vcompare.py logs/quick_tcbfd_hfull_valid_b1.json`): bill -2.113% against S35, pooled MSE -0.201%; per-network change
  mean -0.214%, standard error 0.144%, 5 of 10 up. Fresh and validation networks together: 20 networks, mean -0.16%.

## 4. CPU, requests, and composition with the release lane's fragments

Release instrument (instruction counts, networks 35 and 100, against `release/runs/anatomy/rrpi_clean`; `anatomy/variant_eval_stack*.json`):

| build (parent's prebind list) | bill | kernel calls | local CPU | memory-bound | CPU-bound | projected platform kernel |
|---|---|---|---|---|---|---|
| a07 strips, pad-only | -0.626% | +232 | +1.1% | +1.7% | +0.4% | +1.35% |
| a07 strips, full | -1.275% | +1046 | +4.2% | +4.5% | +3.8% | +4.3% |
| stack `T+C+B+F+D+Hpad` | -1.295% | +276 | +0.6% | +1.2% | -0.2% | **+0.8%** (0.6..0.9) |
| stack `T+C+B+F+D+Hfull` | -1.944% | +1090 | +3.5% | +3.8% | +3.1% | **+3.6%** (3.5..3.7) |
| stack `T+C+B+F+D` | -0.669% | +44 | -0.5% | -0.4% | -0.5% | **-0.5%** (repeat run; the first one paged and was rejected by the instrument) |

The float32 levers give back 0.5-0.7 points of kernel time. Peak RSS (guard samples on this loaded machine, 4- to 12-network
runs): RRPI 5.9-6.1 GB in the harness, strip builds 5.0-5.8 GB, the stack 5.2 GB over the 12-network session: no increase.
Requests: the strips add 0.7% (pad-only) to 2.9% (full) charged ops; the regenerated prebind list takes 2.2-3.2% off (section 5;
with it the full-strip build makes as many charged ops as RRPI on networks 35 and 100, the no-kernel build 3% fewer).

The release fragments re-assign `_lvf_ops` and flags; none of their names overlap with mine, and the strips' 7-level exclusion
follows `_D7_ON`. Built with `stack/build.py` (text only, setup-tested, not run): each deliverable with `Rlvfuse_sized32`, and with
`Rlvfuse_sized32+Rtsr_off` (their fastC), with and without the regenerated list, e.g.
`stack/candidate_T_C_B_F_D_Hfull_Phfull_v1_Rlvfuse_sized32.py`. Expected from the parts: fastC + `T+C+B+F+D` is billed about 0.3%
BELOW S35 instead of 0.37% above it, at fastC's kernel time; with full strips about 1.6% below S35 at kernel -14.7% + 3.6%.
With `Rtsr_off` the thin star products leave the tree kernels, so the nine depth-3 star keys at the head of a prebind list are
then dead: regenerate the list for that variant.
By the release lane's exchange rates the full strips are worth their CPU: 1.3 points of bill for 3.6-4.3% kernel time, against
0.43 points for 6.7% from the 7-level 1024-cube products.

## 5. Lazy tree binds and the prebind list

A tree key that is not prebound is bound inside predict; its reshape is billed at the tree's element count. Measured with
`quick.py --binds` over worker-like sessions (several networks in one process):

| session | lazy binds, share of the session's bill | per network % |
|---|---|---|
| RRPI, 12 screen/check networks | 0.374% (11.9 Gflop) | 0.49 0.57 0.27 0.32 0.17 0.09 0.12 0.11 0.87 0.74 0.38 0.36 |
| RRPI, 10 fresh networks 660-669 | 0.459% (12.2 Gflop) | 1.05 0.72 0.35 0.12 0.57 0.23 0.34 0.27 0.72 0.24 |
| stack with full strips, 12 screen/check | 0.341% | |
| stack with full strips, 10 fresh | 0.399% | |

**The parent's prebind list is fitted to the old census networks.** On the 10 fresh networks (key census `quick.py --keys`,
`logs/quick_rrpi_fresh_f1.json`): 377 of the 649 listed keys are ever asked for; of the 914 tree uses per network 60.5% fall on the
first 128 listed keys (the ones whose level ops setup fuses on the platform; 32 of those 128 are never used) and 80.4% on the
whole list. 70 keys are asked for by all ten networks and carry 66% of the uses; 336 keys are asked for by one network only (5%).

`a13_prebind/regen_prebind.py <candidate> <out fragment> <length> <fit censuses> [-- <holdout censuses>]` writes a new list: the 128
most-used keys first, then by expected saved bind flops in a 12-network session. It also simulates a session's lazy binds for any
list; the simulation of the parent's own list reproduces the measurement (0.451% against 0.459%). Lever `P<name>` of
`stack/build.py` appends the list (only `_PB_SINGLE`, `_KS_ACC_MN` and the named head size `_PB_EAGER` change: predictions
byte-identical by construction). The list needs one census per kernel variant.

Out of sample (stack with full strips; censuses of the fresh batch 660-669 and of validation batch 1):

| | parent's list | regenerated list, fitted on the OTHER batch |
|---|---|---|
| tree uses on the first 128 listed keys | 56.7% | 74.7% / 75.1% |
| tree uses on the first 480 listed keys | 74.3% | 89.6% / 89.6% |
| simulated lazy binds, all listed keys bound | 0.426% / 0.392% of the bill | 0.429% / 0.411% |
| simulated lazy binds, first 480 keys bound (the platform's guard cuts the tail) | 0.510% / 0.465% | 0.438% / 0.418% |
| setup, in process | 649 keys, 34.9 Gflop | 566-649 keys, 23.6-29.6 Gflop |

Direct holdout run (`logs/quick_tcbfd_hfull_pf1_valid_b1.json` against `logs/quick_tcbfd_hfull_valid_b1.json`: the stack with the
list fitted on the fresh batch only, run on validation batch 1): predictions byte-identical on all 10 networks, bill -0.006%,
charged ops per network 39,179 -> 38,328 (-2.2%). So the regenerated list is a REQUEST lever: more of the tree uses run fused.
The bind flops themselves are dominated by keys only one network uses, which no list can hold; on the platform, where the guard
cuts the list, the simulation gives 0.05-0.07% of bill as well.
Lists made: `a13_prebind/prebind_rrpi_v1.py` (RRPI's kernels, from the 10 fresh networks; for stacks without strips or with
pad-only strips, which use the parent's keys) and `a13_prebind/prebind_hfull_v1.py` (full strips, from 20 networks).

Binding the bare exact view of every possible key in setup instead (`a11_bare_views/`, drafted, setup-only tested) would remove
all lazy binds, but the key families needed (3,287 keys at 6 levels alone) raise the setup bill from 35 to 141 Gflop per worker to
save about 12: nine tenths of those views are never used. Not recommended, not built further.

A list fitted to S37's own kernels (`a13_prebind/prebind_s37_v1.py`, from the key census of the S37 session on fresh list a, in
sample): the list S37 carries is already close (502 of its 649 keys are used; 67.2% of the 718 tree uses per network fall on its
first 128 keys, 85.4% on the first 480); the refit gives 69.8% / 91.0% and simulated lazy binds 0.147% instead of 0.190% of the
bill. A small item (about -0.04% bill), not worth a new file by itself.

## 6. Closed with reasons, and what is left

Closed (no candidate): transform additions (12 per level is the minimum for 2x2 schemes; the 48-multiplication 4x4 schemes lose
on additions and requests); deeper recursion (the extra additions on the wide output exceed the leaf saving on every large family:
low-rank reconstruction at 6 / 7 / 8 levels = 1114 / 1124 / 1245 Mflop); covariance W'CW via Cholesky and triangular products
(1.74 vs 1.72 Gflop); join QR (Cholesky-QR is not cheaper under this meter; the columns have condition number 3-8e4); a tree
kernel for the learned nets' plain matmuls (8 Mflop saved for about 45 requests per product); finer selection granularity (the
floors sit at their optimum); tanh in the learned covariance edge (0.71% of the bill at 16 flops per element; not tried: exp
costs the same 16, a table through `interp` is priced 1 flop per element but by my estimate costs seconds of CPU per network on
118M elements, and a norm.cdf activation would need a refit and leans on a 1-flop price the organisers may call a mispricing).

Also closed: W row and column gathers by scatter (-0.05% of bill for about +280 requests per network: a loss at the request price).

Also closed (torch simulation, `a24_edgecut/torch_msgcut.py`, network 35): skipping the mean-correction net's messages for some
target units, as the edge cut skips rows and columns. Dead units only (alpha < -2.33, 14% of the units): +1.17% of MSE for at
most 0.14% of bill; dead and saturated units (27%): +6.4%. The net needs its messages on every unit; the edge is different
because its output is weighted by pdf(row) x gate(column).

Also closed (torch simulation, `a25_readcut/torch_readcut.py`, network 35): a column floor on the third-order plane read (27% of
the bill; its column roster is sorted by gate, so a leading slice would be free to build). Leaving out the columns with gate
below 0.02 (5% of the alive columns): +2.5% of MSE; below 0.03: +4.3%; below 0.05: +7.9%.

Also closed, priced on the fired S38 file's own census (`a30_hz7/price_hz7.py`, `census/s38_4net.json`: networks 35, 100, 36, 45;
the parent's fitted bill model plus my strip terms, which reproduces the measured bill of today's 7-level pushes and reads to
0.1% and of the 6-level ones to 0.2-2.8%): **7 levels of recursion for more products.** 22% of the bill is packet pushes and 30%
plane reads that run at 6 levels because a dimension is not a multiple of 128 (or not one of the parent's listed shapes). At 7
levels, with half-zero strips for dimensions that are odd multiples of 64 and zero pads for the rest:

| products taken | bill | share of the bill moved to 7 levels | wall, by the release lane's platform rates for today's 7-level pushes and reads |
|---|---|---|---|
| every product the model says saves anything | -0.70% (pushes -0.43%, reads -0.27%) | 26.3% | about +12.8 s (18 s per 1% of bill) |
| products that save more than 2% | -0.60% | 16.6% | about +8.0 s |
| products that save more than 3% | -0.43% (pushes -0.31%, reads -0.12%) | 9.1% | about +4.3 s (10 s per 1%) |
| the best single class: the layer-6 push with inner dimension 960 | -0.195% | 3.7% | about +1.6 s |

A product saves 2-5%, not the 7-9% of a clean multiple of 128: the strip runs at the 6-level rate and the operands have to
be packed. On top of the wall come about 35 requests per 2,048-column packet, about +1,000 per network for the 3% set. With
the lead's designation rule (35 s of wall room, 0.095 s of residual room; S38 is projected at 84.5-86.8 s on its worst
network and 0.293 s) there is no room to spend either. Not built. (My earlier note of -0.25% was for one class.)

Also closed (`a31_star/`, lever `SX<first layer>`; probe on network 651, then one native session on 651 and 378): **the low-rank
star with one thin product less.** The parent factors the star plane A as Q B with Q0 = orth(32 columns of A), Q1 = orth(A' Q0),
Q = orth(A Q1), B = Q' A; Z = A Q1 is already a rank-32 factorization (A ~ Z Q1'), which would drop the product Q' A and one
QR: -0.32% / -0.33% of bill measured (from layer 4 on), no request. But the star is not low rank (a third of its norm lies
beyond rank 32 at late layers, nine tenths at layer 1); what makes rank 32 work is the transport through the weights, where the
parent's form leaves 2.2-3.5% error at layers 4-14 (the best possible rank-32 form: 2.0-3.1%), and the shorter form 6-19% more
(36-61% more at layers 2-3). Natively the final prediction moves by 9.6 / 11.2e-6 rms (two rounding-level builds differ by 4-5e-6):
an expected cost of 0.6 / 0.9% of MSE for 0.33% of bill. Not a trade. It also says the parent's point is steep on this side:
the star's factorization is not where bill can be found.

Left, each small, none built: bp formed directly in float32 (-0.02%), one shared float32 square of the covariance (-0.02%),
scattering the cut rows straight into the final plane instead of concatenating (-0.06%, byte-identical, needs `put_along_axis`
into a view), the whole push-to-read pipeline on the kept rows in their sorted order (response rows become a slice, no row
expansion, 10% fewer pointwise passes: about -0.15%, rounding level, a deep edit of the live loop). Sum about -0.25%.

## 8. Native ports of the physics levers F4 and K4L, and the S37 file

Both are levers of `stack/build.py` and stack with every other lever (`T+C+B+F+D+Hfull+F4v2+K4L+P...+R...`). Neither touches a
tree kernel or `setup()`, so prebind lists and the release lane's setup findings are unaffected.

**F4** (`a14_f4/fragment_f4_v2.py` + three edits of the live loop; rule and evidence: `O/physicist/RESULT.md` section T4).
Every bank column keeps one scalar from its birth, A = -alpha / (sigma (1 + radial/3)) of its hub. At the end of step l the gated
columns born 1 or 2 layers earlier are shifted in place, u -> u + theta A w with w = (d/2) pp^2, and the plane this adds to the bank
is taken out of the star: the star's trace term is built with b + beta instead of b, its diagonal with the exact diagonal of the
added plane. Native form: live rows only (u and w vanish elsewhere), every sum factored through r = pp on the live rows
(v.(u u) = (v g^2).r^2, v.(u w) = (v g d/2).r^3, w.x = (d/2)(r^2.x), u.x = g (r.x), diagonal = 3 g^2 (d/2)(r^4.c)), all float32:
16 flops per element of live rows x shifted columns (r^2, r^2 A, r^4 A, r^3 at 1 flop, five matrix-vector passes at 2, the row
scaling and the addition of the shift at 1). The cohort born at layer 0 is skipped: its hubs have pre-activation mean exactly 0, so
its A is exactly 0 (v1 shifted it by exact zeros: byte-identical predictions on 35 and 100, 7% of the step's bill).

| check | result |
|---|---|
| strength 0 with the step running (`F4t0`), networks 35 / 100 | predictions byte-identical to the build without F4; bill +0.251% / +0.250% (v1) |
| bill of F4 v2, networks 35 / 100 | +0.2255% / +0.2239% of the build's bill (589 / 586 Mflop; 36.7M / 36.6M elements, 40,256 / 40,064 shifted columns over 13 layers); charged ops +693 per network |
| final-layer MSE change from F4, native (build `T+C+B+F+D`) against torch (`physicist/t3_pred_rank*_cp4c.npz`, on_0 -> on_0.4) | network 35: -4.48% against -4.25%; network 100: +0.73% against +0.95% |
| prediction change native against torch, per layer 3..15 (`a14_f4/parity_torch.py`) | cosine 0.955-0.968, size ratio 0.98-1.04; layers 0-2 unchanged in both |
| what is left of the change after subtracting the torch change, final layer | 4.4e-6 / 5.2e-6 rms (35 / 100) = the difference between two rounding-level native variants (4.3e-6 / 5.1e-6) |
| **20 fresh ranks** (first 20 of `O/trainer/cert/runB_val.txt`; build `T+C+B+F+D+Hfull+F4v2` against RRPI's native rows stored by the release lane, same two lists and order; `a14_f4/fresh_compare.py`) | **final-layer MSE -3.83% [95% -4.46, -3.15], lower on 19 of 20 (per-network mean -3.85%, se 0.32%, worst +0.12%, best -6.53%); bill -1.695% [-1.77, -1.61]; adjusted -5.47% [-6.08, -4.81]; all-layer MSE -3.08%** |
| torch F4 on the same 20 networks | mean -3.77%; native minus torch per network: mean -0.09%, sd 0.43%, correlation 0.956 |

**Per-age strengths for run M's export** (the lead's 03:17Z plan: F4 strength per age and K4L strength per age as trainable
dials): `F4v4t<theta age 1>,<theta age 2>` (`a14_f4/fragment_f4_v4.py`) and `K4Lv3c<c age 1>,<c age 2>` (`a15_k4l/fragment_k4l_v3.py`);
age = current layer minus the pair's birth layer; a single value gives the arithmetic of the fragments in S37. Light unit checks:
`a14_f4/unit_f4.py <fragment>` against the physicist's `f4_native_ref.f4_step` in float64: shift 5e-7, beta 4e-7, plane diagonal
1.5e-7, identical for v2 / v3 / v4, and 3e-7 / 4e-7 / 1.4e-7 with strengths (0.7, 0.45); `a15_k4l/unit_k4l_v3.py [c1 c2]` against the
lead's formulas: 1.6e-7 with one strength and with (1.2, 0.6). `stack/build.py <levers> <parent.py> <tag>` applies the levers to a
candidate the trainer's exporter wrote. None of v3 / v4 has run natively yet.

`F4v3` is v2's arithmetic with every temporary held to the end of the call and the physicist's two optional dials
(`F4v3t<theta>a<age>o<offset>k<kappa>`: birth scalar A + offset / sigma, source term of the b carrier times kappa); built, its
identity with v2 at default settings is queued, the dials are untested until values are given.

Side number from the same stored predictions: native RRPI / S35 is 1.20% (se 0.26, 40 validation networks) and 1.22% (se 0.43, the
20 fresh ranks) above the torch card in final-layer MSE; the two predictions separate from layer 7 on (difference 1.9e-6 rms at
layer 6, 4.4e-6 at layer 7, 16.5e-6 at layer 15; two rounding-level native variants differ by 4-5e-6). This is the analyst's known
gap (the native row cuts, tail traces, Nystrom core and common-factor reconstruction), now with a standard error.

**S37 file** (the lead's 02:05Z decision: F4 strength 0.6, K4L c 1, both age 2):
`stack/candidate_T_C_B_F_D_Hfull_F4v2t0.6_K4Lv2_Phfull_v1_Rlvfuse_sized32_Rtsr_off.py`, sha256 2a25a07469ac848805d3f70403d615a5220efffac2a31965f6cdd275987dcc97.
Against the lead's S36 file (`..._Hfull_Phfull_v1_Rlvfuse_sized32_Rtsr_off.py`, sha 87b3d6ef) it differs by 2 replaced and 14 inserted
lines in the live loop and the final-layer row cut plus the F4 v2 and K4L v2 fragments (214 lines, before the prebind list). Its `setup()`
runs (1.5 s here, 29.6 Gflop, 703 bound), as S36's.

| S37 file, native session on fresh list a (651, 378, 424, 328, 287, 641, 548, 557, 316, 390), `logs/quick_s37_t06_fresh_a.json` | value |
|---|---|
| final-layer MSE against RRPI's stored native rows | **-6.00%** [95% -7.51, -4.35], lower on 10 of 10; per-network mean -6.07% (se 0.84), best -9.25%, worst -0.90% |
| bill | **-1.320%** [-1.45, -1.20] against RRPI (-1.42% against S35) |
| adjusted (MSE x bill per network, summed) | **-7.21%** [-8.65, -5.67] against RRPI |
| all-layer MSE | -5.75% |
| the same list, MSE against RRPI, for the three builds | F4 0.4: -4.26%; F4 0.4 + K4L: -5.09%; S37 (F4 0.6 + K4L): -6.00% |
| S37 against the F4 0.4 + K4L build (which has no prebind list and no release fragments) | MSE -0.96% [-1.59, -0.36], lower on 9 of 10; bill +0.305% |
| health | all predictions finite; charged ops 37,206 per network (RRPI about 38,400); lazy tree binds 0.195% of the bill; peak RSS 4.31 GB |
| torch reference for the strength | 0.6 against 0.4: -0.52% (se 0.22) on 17 fresh ranks (lead, `O/e16e_fresh24_theta`); there is no torch run of 0.6 + K4L on these ranks |

The session also recorded the tree-key census of S37's kernels (`key_uses` in the same file) for a prebind list fitted to them.

**K4L** (`a15_k4l/fragment_k4l_v2.py` (v1: `fragment_k4l.py`) + five edits; rule: `O/K4L_SPEC.md`, torch reference `O/e16b_k4_closed_loop.py`). The per-unit
fourth cumulant the pair map receives (layers 1-14) and the final layer's get c (loc - beta v^2), where loc sums over the pairs
born at most 2 layers earlier the hub's four fourth-cumulant terms and beta keeps the level the card was trained with. Native form:
the quartic is written on the pair's + column p and D = p - q as D (e1 p^3 + D (e2 p^2 + D (e3 p + e4 D))), with four float32
coefficients per pair fixed at its birth (the shift from Y to p and the powers of the gap folded in), and p^2, p^3 are the squares
and cubes the live loop already forms: 12 flops per element of rows x pairs, about +0.11% bill. Rows the native row cuts do not
push have loc = 0 exactly; they get no correction and are left out of the beta sums. At the final layer the terms come from the
row-cut push (|mu/sigma| <= 3.5). Unit check `a15_k4l/unit_k4l.py` against the lead's formulas in float64: local terms equal to
1.1-1.6e-7, correction to 1.2-2.1e-7, 12.00 flops per pair element billed.

| K4L check | result |
|---|---|
| final-layer MSE change, build `T+C+B+F+D` with the lever(s) against the same build without, list 35, 100, 37, 39; native / torch (`O/e16b_rank*_s*.log`) | K4L alone: -2.34 / -2.69, -3.31 / -3.65, -2.53 / -2.94, -2.23 / -1.96; pooled -2.61% / -2.83%. F4 + K4L: -7.64 / -7.43, -3.04 / -3.16, -3.47 / -3.65, -8.03 / -8.23; pooled -5.43% / -5.50% |
| per-layer MSE change, network 100, K4L alone, layers 1..15, native | +2.87 -3.82 -4.70 -2.29 -0.48 -1.30 -1.81 -3.36 -1.89 -2.42 -2.27 -2.43 -3.39 -3.68 -3.31 |
| the same, torch (`O/e16b_rank0100_s4.log`) | +2.87 -3.62 -4.44 -2.33 -0.53 -1.40 -2.18 -3.62 -2.15 -2.84 -3.12 -3.11 -4.08 -4.17 -3.65 |
| bill | K4L +0.1155 / +0.1146 / +0.1152 / +0.1127% (12 flops per element of rows x pairs on 23.5M elements, plus about 19 Mflop of small per-layer vector work); F4 + K4L +0.341 / +0.339 / +0.338 / +0.334% |
| charged ops per network | K4L +1,082, F4 + K4L +1,775 (+4.6% of the build's 38,400) |
| **10 fresh ranks** (list a: 651, 378, 424, 328, 287, 641, 548, 557, 316, 390; build `T+C+B+F+D+Hfull+F4v2+K4Lv2`, one session, same list and order as the F4-only session and RRPI's) | against RRPI: final-layer MSE **-5.09%** [95% -6.33, -3.67], 10 of 10 lower; bill -1.619%; adjusted **-6.59%** [-7.78, -5.28]; all-layer -5.12%. K4L on top of F4 0.4: final-layer **-0.86%** pooled [-2.23, +0.72], per-network mean -0.83% (se 0.80), lower on 7 of 10; all-layer -1.97%; bill +0.117% |
| torch K4L increment on the same 10 networks (`O/e16c_fresh24`) | mean -0.69% (se 0.73); native minus torch per network: mean -0.14, sd 0.42, **correlation 0.990** (native / torch: +0.82/+0.81, -0.30/+0.66, +2.67/+2.33, -1.59/-1.72, -1.18/-1.03, +3.36/+2.87, -4.72/-4.26, -1.47/-1.32, -3.14/-2.66, -2.72/-2.56) |
| K4L fragment v2 (temporaries held, lower-case einsum subscripts; the version in the S37 file) against v1 (the parity table above), networks 35 / 100 | predictions and bill identical (hashes c4704290..., 2fef74e9...; 262,094,411,639 and 262,752,490,678 flops) |
| einsum gate `fable_porter_v1/einsum_site_diff.py` of the S37 file against the S35 file | PASS: no new repeated-operand einsum, no symmetric tag (K4L adds two 3-operand einsums and one 2-operand, all on distinct objects; F4 none) |

Native equals torch at layers 1-6 and is 10-25% weaker from layer 7 on. Layers 7-14 are where the native young negative-tail row
cut is active: about 110 dead units per layer (mu/sigma < -2.5, gate below 1%) have no pushed young columns, so their hub terms
cannot be formed and they keep the isotropic value. Their own means are tiny, but the kurtosis term moves a tail unit's mean by as
much in absolute terms as a typical unit's error. Forming the terms there would need the push the cut saves (about 0.5-2% of the
bill), so the port accepts the loss; the test with the cut switched off is `logs/quick_k4lv2_Zynt99_35_100.json` when it has run.
The base run of this table reproduced the earlier run's predictions and bill exactly on 35 and 100.

## 9. F4b in native form (draft, unit-checked, not run natively)

`O/physicist/f4b.py` fits, per pair and per layer, the hub's fourth-cumulant feed by two column shifts with a 3 x 3 Gram matrix, a
3 x 2 design and a batched 2 x 2 solve. Written on the pair's own two pushed columns P = Y + t+ Wt and Q = Y + t- Wt the same thing
is short (`a18_f4b/check_projection.py`, equal to the module's arithmetic to 1e-15 in float64):
- five coefficients per pair, fixed at birth: Psi_m = (t+ - t-)^-4 sum_{i<=4-m, j<=m} binom(4-m,i) binom(m,j) (-1)^j (-t-)^(4-m-i) t+^(m-j) S_(i+j);
- x+ = (d/2) [(Psi0 + k+ Psi1) P^2 + 2 (Psi1 + k+ Psi2) P Q + (Psi2 + k+ Psi3) Q^2], x- = (d/2) [(Psi2 + k- Psi1) P^2 + 2 (Psi3 + k- Psi2) P Q + (Psi4 + k- Psi3) Q^2];
- k+ = 2 spq sqq / (spp sqq + spq^2), k- = 2 spq spp / (spp sqq + spq^2), with spp, spq, sqq the inner products of the pair's two gated columns.
F4's rule A is the case Psi = (A lam+, 0, 0, 0, A lam-); K4L's local term is sum_m binom(4,m) Psi_m P^(4-m) Q^m.

Fragment `a18_f4b/fragment_f4b_v3.py` (v3 = v2 plus the reference's guard for a column weight of exactly 0; lever `F4Bv3c<strength>a<age>t<terms>l<lo>-<hi>[b]`; replaces F4; the loop edits are F4's plus
one birth call, and for the b anchor one line that lowers the carried b): live rows only, float32; one square of the whole shifted
block, per cohort the product P Q, three column sums and two quadratic forms written into views of one shift block, then the
anchor's five matrix-vector passes on the whole block. Unit check `a18_f4b/unit_f4b_v2.py` against f4b.py's arithmetic in float64
on synthetic data, for all four terms at strength 0.5 and for S1 + S2 at strength 1: column shifts equal to 9e-8, beta to 2e-7,
plane diagonal to 1e-7; billed 22.9 flops per element of live rows x shifted columns (F4: 16), 117 charged ops per layer.
Not run natively yet.

## 10. The run M export path (built, rehearsed, and checked natively against the torch card: the parity table at the end)

The trainer's run M card (`O/trainer/otr_m.py`) trains, next to the learned arrays and the 12 dials, global dial families. The
exporter (`O/trainer/export.py`) has native forms for rc, edge and bsrc and refuses the rest. The five it refuses are levers here:

| family (otr_m.py) | meaning | native lever | bill |
|---|---|---|---|
| f4a1, f4a2 | F4 strength for column pairs 1 and 2 layers old (base 0.6) | `F4v4t<age 1>,<age 2>` (`a14_f4/fragment_f4_v4.py`) | as F4 (+0.225%) |
| k4la1, k4la2 | K4L strength per pair age (base 1), applied to each age's terms before the level is projected out | `K4Lv3c<age 1>,<age 2>` (`a15_k4l/fragment_k4l_v3.py`) | as K4L (+0.115%) |
| k22 | scale of the off-diagonal part of the isotropic (2,2) plane the pair map receives, layers 1-14 | `K22s<scale>` (one edited loop line: the plane's scalar is multiplied, its diagonal is filled separately as before) | one scalar multiplication per layer |

Read line by line against `otr_m.py`: age = current layer minus the pair's birth layer in both; the cohort born at layer 0 has
birth scalar 0; K4L sums the cohorts born at layers l-2 and l-1 (the first cohort included) and projects once; the final layer
(15) uses the ages of cohorts 13 and 14; k22 acts at layers 1-14 only (the card's final layer does not go through the pair-input
hook).

One command from a parameter file to the candidate: `stack/from_params.sh <params.pt> <tag>` = `stack/export_for_build.py` (the
trainer's `export()` unchanged, called without the five families) + `stack/levers_from_params.py` (the lever string: base values
read from `otr_m.py` plus the stored shifts; it refuses per-layer tables and unknown families) + `stack/build.py` with the S37
lever stack + the einsum gate against the S37 file. Text only; the light setup run and the native session come after, on request.

Rehearsal (`a20_runm/make_rehearsal.py`): a parameter file of my own, RRPI's parameters plus shifts on all seven dials (F4 0.7 /
0.1, K4L 1.5 / 0.4, k22 1.004, bsrc 0.85, rc -0.28), exported (sha 9005c38c, the same bytes through `make_rehearsal.py` and through
`export_for_build.py`), built without strips (`stack/candidate_T_C_B_F_D_F4v4t0.7,0.09999999999999998_K4Lv3c1.5,0.4_K22s1.004__rehearsal_v1.py`,
sha 09fbe869) and with the full S37 stack (sha 80d0ec89); einsum gate PASS for both. The stack also builds on the trainer's
exported file with the b-source edit (`...__ema25_bsrc085.py`, sha c9f0d933).

Torch side of the parity test (`a20_runm/torch_arms.py`: the run M card through the trainer's engine, read-only, predictions
kept; one process, 3 threads, about 35 s per forward; `a20_runm/logs/torch_r0035_v1.*`, `torch_r0100_v1.*`). Reference = F4 0.4
on both ages + K4L 1 (final-layer MSE 13.0593e-9 on network 35 and 14.8795e-9 on network 100, the trainer's certificate values);
each dial alone at the rehearsal's value, and all seven together:

| arm (change against the reference) | network 35: final MSE, prediction change rms at layer 15 | network 100 |
|---|---|---|
| F4 age 1: 0.4 -> 0.7 | -0.82%, 9.3e-6 | +0.77%, 9.6e-6 |
| F4 age 2: 0.4 -> 0.1 | +1.01%, 7.1e-6 | -0.79%, 8.8e-6 |
| K4L age 1: 1 -> 1.5 | +0.29%, 8.7e-6 | +0.02%, 8.1e-6 |
| K4L age 2: 1 -> 0.4 | +0.31%, 7.5e-6 | +0.57%, 6.8e-6 |
| k22: 1 -> 1.004 | +0.11%, 8.4e-6 | +3.42%, 10.8e-6 |
| bsrc: 1 -> 0.85 | +1.66%, 13.7e-6 | +0.37%, 16.5e-6 |
| rc: -0.08 -> -0.28 | -0.41%, 5.8e-6 | +0.43%, 5.5e-6 |
| all seven | +2.38%, 25.0e-6 | +6.42%, 23.7e-6 |

- **k22 is very sharp**: a first exploration with k22 1.15 took network 35's final-layer MSE from 13.06e-9 to 282.9e-9 (x21.7; the
  prediction moved 520e-6 rms against an error of 114e-6 rms). A step of 0.004 already moves the prediction as much as a step of
  0.3 in an F4 strength. A wrong native k22 would be the most expensive porting error of the seven, hence its place in the test.
- The dials do not add: the all-seven change fitted on the seven single changes gives coefficients 0.76-1.11 on network 35
  (27% of the change left over) and 0.33-1.07 on network 100 (the F4 age-2 strength interacts most). So the test of the native
  build (`a20_runm/parity_regress.py`) is on what is LEFT after subtracting the torch card's own all-seven change: fitted on the
  single changes, a dial ported as in the card leaves 0, a missing dial -1, a dial applied twice +1; expected standard error about
  0.03 per dial at the final layer (two rounding-level native variants differ by 4-5e-6 rms, the single changes are 5.5-16.5e-6).
- **Rehearsal v2** adds the three covariance-output knobs (lever `GN<knob><value>`: the parent's `_GN_KNOBS` literal overridden;
  iso -0.094 -> -0.064, k21n -0.028 -> -0.058, k31 -0.212 -> -0.062), since the trainer's best single-dial lead is k21n. Torch
  arms `a20_runm/logs/torch_r0035_v2.*`, `torch_r0100_v2.*` (each of the ten dials alone and all together; 12 s per forward at 2
  threads on a quiet box): the three new dials move the final-layer prediction by 5.8 / 6.1, 7.9 / 9.1 and 7.7 / 8.1e-6 rms on
  networks 35 / 100; all ten together 24.4 / 22.9e-6 (final-layer MSE +1.92% / +3.92%). Candidate
  `stack/candidate_T_C_B_F_D_F4v4t0.7,0.09999999999999998_K4Lv3c1.5,0.4_K22s1.004_GNiso-0.064_GNk21n-0.057999999999999996_GNk31-0.062__rehearsal_v2.py`
  (sha 67d1b145), einsum gate PASS.
- **Rehearsal v3 = the real handoff.** Run M (launched 05:59 UTC) trains nine physics dials, and the trainer's announcer
  (`O/trainer/announce.py`) exports rc, bsrc, the `_GN_KNOBS` values (iso, k21n, k31) and the scale of the learned mean
  correction (ds, folded into the head literals) INTO the candidate. So `stack/levers_from_params.py` prints levers only for F4
  per age, K4L per age and k22 (a lever on top of an exported dial would apply it twice; `--standalone` prints DS / GN levers for
  a parent that was not exported that way), `stack/export_for_build.py` exports by the announcer's rule, and
  `stack/from_params.sh <params.pt> <tag> '<extra levers>' [<the trainer's exported candidate.py>]` is the one command (env
  `PRE=T+C+B+F+D POST=` gives the screen stack). v3: RRPI's parameters with ELEVEN dials moved (the ten of v2 plus ds 0.9):
  export sha 422415ae, levers `F4v4t0.7,0.1 + K4Lv3c1.5,0.4 + K22s1.004`, build
  `stack/candidate_T_C_B_F_D_F4v4t0.7,0.09999999999999998_K4Lv3c1.5,0.4_K22s1.004__rehearsal_v3.py` (sha 83d9cfdf), einsum gate PASS.
**The native parity session (06:35 UTC; `logs/quick_rehearsal_v2_35_100`).** The rehearsal v2 file (sha 67d1b145: ten dials
moved at once, the three knobs through my GN levers; v3 holds the same values in the exporter's form plus ds) on networks 35
and 100, against my stored native run of the same stack at F4 0.4 + K4L 1 (`logs/quick_tcbfd_f4v2_k4lv2_35_100`), next to the
torch card's stored arms: `python a20_runm/parity_regress.py a20_runm/logs/torch_r0035_v2.npz
logs/quick_tcbfd_f4v2_k4lv2_35_100_pred_35.npy logs/quick_rehearsal_v2_35_100_pred_35.npy` (and the same for 100).

| native change against the torch card's change for the same ten dials | network 35 | network 100 |
|---|---|---|
| cosine per layer, layers 2-15 (layer 1: 1.000) | 0.970 to 0.992 | 0.969 to 0.990 |
| size ratio native / torch per layer | 1.009 to 1.038 | 1.008 to 1.031 |
| torch change at the final layer, rms | 24.4e-6 | 22.9e-6 |
| left after subtracting the torch change, final layer, rms (two rounding-level native variants differ by 4-5e-6) | 4.96e-6 | 5.77e-6 |
| the same at layers 2 / 7 / 12 | 0.71 / 2.78 / 4.09e-6 | 0.73 / 2.57 / 4.62e-6 |

| what is left (native minus torch), fitted on the card's single-dial changes at the final layer: 0 = ported as in the card, -1 = dial missing, +1 = applied twice | network 35 (se) | network 100 (se) |
|---|---|---|
| F4 strength, age 1 (0.4 -> 0.7) | -0.055 (0.020) | -0.103 (0.020) |
| F4 strength, age 2 (0.4 -> 0.1) | +0.010 (0.024) | +0.006 (0.022) |
| K4L strength, age 1 (1 -> 1.5) | -0.052 (0.018) | -0.104 (0.024) |
| K4L strength, age 2 (1 -> 0.4) | -0.041 (0.021) | -0.083 (0.028) |
| k22 (1 -> 1.004) | +0.053 (0.022) | -0.029 (0.022) |
| bsrc (1 -> 0.85) | +0.001 (0.014) | +0.028 (0.013) |
| rc (-0.08 -> -0.28) | -0.040 (0.027) | -0.054 (0.032) |
| iso (-0.094 -> -0.064) | +0.001 (0.037) | +0.155 (0.044) |
| k21n (-0.028 -> -0.058) | -0.014 (0.026) | +0.006 (0.027) |
| k31 (-0.212 -> -0.062) | +0.040 (0.030) | +0.107 (0.033) |

- No dial is missing (that would read -1), none is applied twice (+1), and the two ages are not swapped (a swap would show as
  -1 on one age and a wrong-signed residue on the other; the two ages were moved in opposite directions for this reason).
- The departures from 0 are small but not all noise: on network 100 the age-1 strengths read 10% weak and iso 15% strong
  (3-5 standard errors). The native file is not the card (its row cuts start at layer 7, section 13), so a dial's response is
  not exactly the card's; the leftover (5.0 / 5.8e-6) is the size of a rounding-level difference between two native builds,
  4-5% of the error's rms (0.2% of its power), so nothing here is worth a correction.
- Over layers 1-15 together the fits are tighter (se 0.003-0.010) and all within +-0.09.
- Not in this session: ds and k21n in the exporter's own form. They are covered by the S38 session (section 14): the native
  file gains -1.06% where the card gains -0.98%.
- The handoff for run M's file, as rehearsed: `bash stack/from_params.sh <ema params.pt> <tag> 'EcutS+L3+L3o' <the trainer's
  exported candidate.py>`, then `a11_bare_views/setup_only.py <file>`, then one native session on a fresh list against S38's rows.

## 11. Where the native file loses to the torch card (stored predictions; the audit session is not run)

With the same physics (F4 0.4 + K4L 1, RRPI's parameters) the native build `T+C+B+F+D+F4v2+K4Lv2` and the torch run M card agree
through layer 6 and separate at layer 7, the first layer with the native old-tier and young-column row cuts (rms of the
prediction difference by layer 1..15, in 1e-6):

| network | 1-6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | at layer 15: difference power / torch MSE | native MSE against torch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 35 | 0.4 to 2.0 | 10.1 | 14.0 | 17.7 | 17.2 | 17.2 | 18.4 | 19.3 | 21.2 | 19.6 | 2.95% | +3.79% |
| 100 | 0.4 to 1.9 | 3.5 | 6.1 | 6.6 | 8.2 | 10.6 | 14.6 | 15.1 | 17.5 | 17.1 | 1.96% | +0.52% |

Over the 40 validation networks the native MSE is 1.20% (se 0.26) above the torch card's (section 8). The difference is not a
scale error of the fourth-order plane: projected on the torch card's response to k22 it explains under 5% of the difference on
network 35 (equivalent k22 shift -0.001), and no single dial's response has a cosine above 0.27 with it.
The join is the same deterministic algorithm on both sides (energy-ordered Nystrom rows, `s6bcard_loop5js.py`), so what starts at
layer 7 on the native side only is the row cuts (old tier positive tail and dead negative tail at 2.5, young dead tail at -2.5)
with their tail traces. Audit build with all five cuts off on the same physics:
`stack/candidate_T_C_B_F_D_F4v2_K4Lv2_Zott99_Zneg99_Zynt99_Zdead99_Zl15_99.py` (sha a8ac0e63, einsum gate PASS); one 2-network
session against `logs/quick_tcbfd_f4v2_k4lv2_35_100` gives the distance to the torch card with and without the cuts, the bill
all cuts save together and the MSE they cost (`a16_audit/audit_table.py`); four more sessions price the cuts one by one. Asked
of the lead 03:51 UTC.

## 12. Requests: the three-request operand load (L3) and the pair write of the output (L3o); both byte-identical natively

**Native result first (06:31 and 06:41 UTC).** Three files that differ only in these two levers, each on networks 651 and 378
as the first and second network of its own process (`logs/quick_s38_noL3_651_378`, the first two rows of
`logs/quick_s38_ds08_ecuts_l3_fresh_a`, `logs/quick_s38_L3o_651_378`):

| the S38 physics, networks 651 / 378 | without L3 (sha e544d59a) | with L3 (sha 0d51ccb9) | with L3 + L3o (sha 882dd7d1) |
|---|---|---|---|
| predictions | reference | byte-identical on both | byte-identical on both |
| bill, flops | 265,547,869,526 / 256,289,856,091 | 265,884,291,670 / 256,481,897,563 (**+0.127% / +0.075%**) | the same to the flop (**+0**) |
| logged calls | 37,676 / 37,587 | 36,674 / 36,360 (**-1,002 / -1,227**) | 36,420 / 35,996 (**-254 / -364** more) |

- L3's bill has two parts: every load copies one quarter more of the operand (the steady +0.05%), and every tree key pays one
  reshape of the operand's size at its first load in a process (160 keys on the first network, 53 on the second, fewer later:
  the 10-network session built 473 in all). Over a worker's 20 networks about +0.065% of bill, a third more than my +0.049%
  estimate, which had the first-load reshapes at 0.01%.
- L3o's saving grows after the first networks (the pair views of a key are bound at its first use: 85 keys on the first
  network, 34 more on the second); from these two numbers about 2.2 calls per newly bound key and about 220 products per
  network, so about -440 calls per network in steady state.
- These are calls logged by flopscope in process. The platform's request count also includes the free view calls and the
  frees; the release lane's boundary session of the chosen file gives that number and the residual time.

The rest of this section is the design and the unit checks as written before the native runs.

The platform's residual limit (0.400 s per network of the participant's own Python, about 7 microseconds per request) is what the
leaders run against; S37 sits at 0.303 s with 42,864 requests per network. The release lane's census of RRPI (network 35, 43,984
requests): tree-level arithmetic 17,959, estimator body about 17,000, operand loads 3,570 (`_rs_load_a` 284 x 6, `_rs_load_b`
311 x 6), output quadrant views 1,200, lazy binds 2,209.

A load today: reshape the operand into its quadrant layout, split it twice, copy one half into the tree's pair view, copy one
quarter, and form the alt-basis sum (A: S1 = a21 + a22 into a21's slot; B: T3 = b22 - b12 into b12's slot). L3
(`a21_load3/fragment_load3.py`, builder lever `L3`) binds the key's four level-1 data slots as ONE view in the operand's own axis
order (the slots are chunks 4 i1 + 2 j1 of A's tree and 2 i1 + j1 of B's, so the four form a 2 x 2 axis pair by a reshape), and a
load becomes: reshape, one copy of the whole operand, the same sum or difference taken in place (X4 <- X4 + X6, Y1 <- Y3 - Y1).
Three requests instead of six; every tree element ends with the value it had before. The view is built at the key's first load,
not at bind time, so the binding phase of setup gets lighter, not heavier (the old pair view is no longer made).

| unit check (`a21_load3/unit_load3.py`: the S37 file against S37 + L3, each in its own 9-second process: `setup()` and 14 tree products on random float32 matrices) | result |
|---|---|
| outputs, 7 cases x 2 calls (single products at 5 / 6 / 7 levels, a lazily bound odd key, batched packets of 2 and 3) | byte-identical in every case |
| calls per load | -1 copyto, -2 unstack |
| bill per load | + a quarter of the operand's elements (1024-cube product, one load: +262,144 flops = +0.025% of that product) |
| first load of a key | one reshape of the operand's size (billed once per key and process), two or three view calls |
| setup | same 29,571,934,767 flops; 7,074 logged calls against 7,269 |

Expected per network (RRPI census, `census/rrpi2_35_100.json`): on RRPI -1,785 requests and bill +0.0563% (the extra quarter
copies, 148 Mflop). On S37 the thin-star products (87 per network, one A and one B load each) are plain matmuls, so 421 loads
remain: about **-1,260 requests, bill +0.049%**, plus about 0.01% averaged over a worker's 20 networks for the first-load
reshapes. At the release lane's exchange rate for the thin-star diet (9,000 requests per 1% of bill) L3 sells requests at about
25,000 per 1%: the freed requests either widen the residual room (0.303 s -> about 0.294 s) or pay for putting part of the
thin-star products back on the tree kernel at a lower bill.

**L3o, the output side at no bill** (`a21_load3/fragment_load3o.py`, lever `L3o`, needs L3). The last unwind writes four output
quadrants with four additions into four views split off the output array on every call. The upper two are one pair addition,
(tl, tr) = (Z0, Z5) + (Z2, Z3), written straight into the output's upper half (which carries the two quadrants along its
column-half axis): one addition and one split fewer per product, in the plain unwind and in the k-split's unwind. A pair op bills
its two quadrants, so the bill is exactly unchanged; the tree's pair views are bound at the key's first use (views only).

| unit check, second round (11 cases x 2 calls: the seven above plus, through the dispatcher `_mm`, the k-split reads 1024x4096x1024 and 896x6144x960, the strip k-split 736x4096x1024, and 1024x2048x1024; three 15-second processes) | S37 -> S37 + L3 | S37 + L3 -> S37 + L3 + L3o |
|---|---|---|
| outputs | byte-identical, all 22 calls | byte-identical, all 22 calls |
| calls | -3 per operand load (-12 on a four-load k-split read) | -2 per product (batched packets unchanged: they have their own unwind) |
| flops | +0.02 to +0.06% of the product | +0 |

Expected on S37 per network: L3 about -1,260 requests, L3o about -400 (about 200 unwinds): together about **-1,660 requests for
+0.049% bill and no wall time**, under the release lane's 41,500 target without un-fusing less. Files (einsum gate PASS): S37 + L3
sha ab626d8c; S37 + L3 + L3o `stack/candidate_T_C_B_F_D_Hfull_F4v2t0.6_K4Lv2_L3_L3o_Phfull_v1_Rlvfuse_sized32_Rtsr_off.py` sha
042bf49a; S37 + L3 + the release lane's `lvfs_limit64` sha 3ad400d8. A first draft of L3o that raised on a cache mismatch is in
`_stale/` (never run); the kept version falls back to the four single writes.

A dearer variant (not built): the whole level-1 unwind taken in the tree, then one copy of the four result slots into the output:
-2 more requests per product for one more copy of every product's output (+0.089% bill).

Side fact from the census: the tree kernel pays 0.31% of the bill for reshapes that move no number (operand layouts 0.225%, output
layouts 0.089%; flopscope bills a reshape at its element count), and lazy tree binds are reshapes too. I see no honest way around
them in this arena layout (the tree stores quadrants blocked, so a matching view of a 2-D operand needs the reshape); free view
calls that imitate a reshape would be exactly the under-priced-op trick the organisers may disqualify.

## 13. The native file's row cuts, priced on the torch card (simulation validated against stored native runs; native confirmation not run)

Section 11's question (why the native file is 1.2% worse than the card) is answered without a native session.
`a22_cutsim/torch_cutsim_v2.py` applies the native row-cut rules to the torch run M card's pushed bank: the dead-side cut
(alpha < -3.5, layers 1-14, no repair), the old-tier cut of saturated rows (alpha > 2.5) and of dead rows (alpha < -2.5, not
alive) with its first-chaos trace repair 3 rho U_i . F (lam o |F_r|^2), the young-column cut of dead rows (alpha < -2.5) with its
spherical trace repair 3 |w_i|^2 / (n + 2) w_i . tau, the repaired diagonal where the native file uses it (pair map, births,
learned parts), and the native tail part of the fourth-order scalar's source for the old-tier rows. The final layer's row cut is
already in the torch card. Counts are rounded up to 64 as in the native file.

| validation: simulated native cuts (S) against native minus torch (G), same physics (F4 0.4 + K4L, RRPI parameters) | network 35 | network 100 |
|---|---|---|
| cosine of G and S at layers 7 / 8 / 9 | 0.965 / 0.976 / 0.983 | 0.788 / 0.906 / 0.894 |
| cosine at layers 10-15 | 0.92-0.96 | 0.83-0.89 |
| rms at layer 15, G / S (1e-6) | 19.6 / 18.6 | 17.1 / 15.2 |
| what is left of G after subtracting S, layer 7 / layer 15 (1e-6; before layer 7 the two programs differ by 2.0) | 2.7 / 7.8 | 2.2 / 9.6 |
| final-layer MSE against the card, native / simulated | +3.79% / +3.55% | +0.52% / +0.12% |

Cost of each cut. "Implied" = power of the final-layer prediction change over the card's MSE: the expected cost of a perturbation
unrelated to the error the estimator already has (a single network's measured MSE change scatters by about +-1% around it at
these sizes: on network 100 three cuts "improve" the MSE by 0.7-1.1%).

| arm (cut alone unless said) | network 35: implied, measured, rows cut over layers 7-14 | network 100 |
|---|---|---|
| all four cuts at the native thresholds | +2.66%, +3.55% | +1.55%, +0.12% |
| young columns of dead rows, alpha < -2.5 (the parent's YNT cut) | **+2.10%**, +2.39%, 1,344 | **+0.77%**, +1.29%, 1,152 |
| the same at -3.0 | **+0.08%**, +0.09%, 832 | **+0.07%**, +0.36%, 768 |
| the same at -3.5 | +0.01%, +0.14%, 448 | +0.03%, 0.00%, 576 |
| the same at -2.5 without its repair | +4.75%, +2.81% | +2.18%, -0.73% |
| old-tier columns of dead rows, alpha < -2.5 | +0.38%, +1.38%, 1,344 | +0.14%, +0.08%, 1,152 |
| the same at -3.0 | +0.02%, +0.04%, 832 | +0.01%, +0.03%, 768 |
| old-tier columns of saturated rows, alpha > 2.5 | +0.27%, -0.39%, 1,152 | +0.20%, -1.07%, 1,216 |
| the same at 3.0 | +0.07%, -0.16%, 768 | +0.09%, -0.23%, 832 |
| dead-side cut at -3.5 | +0.06%, -0.07% | +0.11%, -0.67% |

Networks 35 and 100 are not typical. The same scan at the S37 physics on fresh ranks (list a; all four cuts on, only the
young dead-tail threshold moved; `a22_cutsim/scan_report.py s37 s37_t06_fresh_a`; three of the ten ranks done when the lead
cleared the machine for run M at 05:11 UTC):

| young dead-tail threshold (old-tier and dead-side cuts as in the native file) | implied cost of all cuts, pooled (378, 424, 651) | young row-layers not pushed | bill they are worth (6.0 Mflop per row-layer, network 35's census) |
|---|---|---|---|
| 2.5 (the parent) | +1.34% (1.62, 1.05, 1.37) | 661 | -1.52% |
| 2.75 | +0.82% | 491 | -1.13% |
| 3.0 | +0.71% | 320 | -0.74% |
| 3.5 (= the dead-side cut: nothing left for this cut) | +0.69% | 0 | 0 |
| 3.0, and the old-tier dead rows at 3.0 as well | +0.45% | 320 | -0.74%, plus the old-tier rows given back |

| the stored S37 native predictions against the card (G) and the simulated cuts (S), final layer | 378 | 424 | 651 |
|---|---|---|---|
| rms G / rms S (1e-6), cosine | 18.1 / 15.1, 0.855 | 16.3 / 13.0, 0.796 | 16.3 / 14.9, 0.842 |
| native MSE against the card / simulated | -0.06% / -0.61% | +0.90% / +0.63% | -2.03% / -2.68% |

- **Reading.** The cuts cost about 1.3-1.5% of MSE in expectation and are worth about 5% of bill (old-tier reconstruction on
  576-832 of 1024 rows: about 3.3%; young push without the dead rows: about 1.5-2.1%). Of that cost the young dead-tail band
  between 2.5 and 3.0 is 0.63% on the fresh ranks (2.0% on network 35, 0.7% on network 100) and keeping it would cost about 0.78%
  of bill: break-even. The band between 2.5 and 2.75 alone: 0.52% of MSE for 0.39% of bill, a gain of about 0.1% that two-digit
  network counts would be needed to see. **The parent's thresholds sit close to the optimum; no threshold change is proposed.**
- Why the dead rows matter at all: a unit at alpha = -2.5 still has a mean about 20 times the error level, a third-cumulant
  error moves that mean in proportion to |alpha| pdf(alpha) (3.3 times more at -2.5 than at -3.0, 11 times in power), and the
  trace estimate that replaces the cumulant correlates only 0.74-0.94 with the exact value on the cut rows (old tier: 0.92-0.96,
  best scale 0.95-1.06). Without its repair the young cut would cost +4.75% / +2.18% on networks 35 / 100.
- What this is good for: (1) the 1.2% between the native file and the card is explained and is a deliberate trade, not a
  porting defect; (2) the refit trains a card WITHOUT these cuts and is scored WITH them: the cut rule now exists as a torch
  module (forward only), and a refit on the card with the cuts might let the learned parts absorb some of the 1.3-1.5%;
  (3) a per-network single MSE number cannot judge a cut (on network 100 three cuts "improve" the MSE by 0.7-1.1%); the power
  of the prediction change can.
- Lever `YNT<z>` of `stack/build.py` exists (two literals of the parent become `_YNT_Z`), with two builds at 3.0 (S37 stack sha
  08e77f00, screen stack sha df178635, einsum gate PASS); they are NOT recommended and were not run.
- Not simulated: the native K4L leaves unpushed rows out of its level; the old-tier repair on two blocks at layers 13-14.

## 14. The next board file (S38): two builds, both measured natively on the same ten fresh ranks

**14a. The build in qualification (dials 2; the lead's 07:03 UTC recipe).** Parent: the trainer's second dial-only export
`O/trainer/candidates/runM_dials2/candidate.py` (sha 716f66dd: RRPI's learned arrays with the learned mean correction x 0.8
folded into its output weights, k21n -0.048, b-source scale 0.965, bscale 1.03414, X1 0.59158; no trained parameter). Built by
`bash stack/from_params.sh O/trainer/cert/runM_dials2.pt runM_dials2 'EcutS+L3+L3o' <the trainer's file>`: the lever string the
tool derives from the parameter file (`F4v4t0.656,0.6+K4Lv3c0.75,0.75`) equals the one in the trainer's `physics_dials.json`.
File `stack/candidate_T_C_B_F_D_Hfull_F4v4t0.656,0.6_K4Lv3c0.75,0.75_EcutS_L3_L3o_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_dials2.py`,
sha256 7c1a67f4e7dd4201895b9461c400173cec607489f267c87d7e0db6241146ea0a. Einsum gate PASS against the S37 file; setup alone
0.92 s, 29,571,934,767 flops (S37's), 703 keys bound. Read back from the built file: k21n -0.048, X1 0.5915819405586057,
bscale 1.0341394097164294, b-source divisor 6.217616580310881 (= 6 / 0.965), F4 (0.656, 0.6), K4L (0.75, 0.75), stand-in on.

Native session on fresh list a (one process, one thread, under the guard, 07:06-07:24 UTC, 1056 s on a busier machine;
`logs/quick_s38_dials2_fresh_a.json`), against the stored S37 session, same list and order
(`python a14_f4/fresh_compare.py s38_dials2_fresh_a --base s37_t06_fresh_a`):

| network | bill | final-layer MSE (x 1e-9) | MSE change | all-layer MSE change | logged calls |
|---|---|---|---|---|---|
| 651 | -0.533% | 15.2404 | -3.90% | -1.74% | 36,458 |
| 378 | -0.629% | 13.8902 | -1.55% | -1.02% | 36,034 |
| 424 | -0.538% | 15.5021 | -3.56% | -2.78% | 36,873 |
| 328 | -0.630% | 16.2968 | +0.78% | -0.34% | 35,534 |
| 287 | -0.651% | 17.2522 | -1.04% | -0.78% | 35,799 |
| 641 | -0.624% | 17.6282 | -0.82% | -0.06% | 35,630 |
| 548 | -0.669% | 14.8186 | -1.38% | -0.23% | 35,442 |
| 557 | -0.636% | 16.2269 | -1.11% | -0.60% | 35,497 |
| 316 | -0.673% | 14.7784 | -2.45% | -1.92% | 35,483 |
| 390 | -0.669% | 19.2936 | -4.78% | -2.26% | 35,582 |

| pooled over the 10 networks | against the S37 file | against the dials-1 build (14b) |
|---|---|---|
| final-layer MSE | **-2.034%** [95% -3.122, -0.986]; per-network mean -1.982% (se 0.531), lower on 9 of 10 | -0.982% [95% -1.556, -0.345]; per-network mean -0.985% (se 0.314), lower on 8 of 10 |
| bill | **-0.624%** [95% -0.652, -0.591] | +0.00003% (about 80 kflop per network: the per-age F4 strength vector) |
| adjusted (MSE x bill per network, summed) | **-2.661%** [95% -3.732, -1.629] | -0.982% |
| all-layer MSE | -1.188% [95% -1.769, -0.627] | -0.818% |
| logged calls per network | 35,833 against 37,206 (-1,373) | against 36,173 (L3 only): -340 |
| health | all predictions finite; exit 0; guard not tripped; peak RSS 4.91 GB; no swap growth; the edge cut kept 0.51 of its grid | |

- **Expectation for the board.** The trainer's torch number for these dials on 50 held-out ranks is -1.44% (se 0.26, 40 of 50
  lower). My 10 ranks read 1.1 standard errors better, as the S37 list did (which overstated the board gain by 1.7 points). So:
  MSE about -1.4%, budget share about -0.6%, adjusted about **-2.1% (1.951e-9 -> about 1.91e-9)**.
- **Op forms in this file that S37 does not use** (they have run natively only; the release stages run them across the
  boundary): the stand-in's `minimum`, `sqrt` and `divide` with `out=`; the edge cut's `put` on a flat index grid; L3's `reshape`
  of a strided basic slice of the tree; L3o's `transpose` of a two-chunk strided slice; F4 v4's per-column strength vector (a
  `concatenate` of two `full` blocks) and K4L v3.
- For this session I stopped my S37 run on list b three minutes in (partial files in `_stale/`). The dials-1 file with L3o had
  finished on list b (919 s, peak RSS 5.51 GB, 35,770 calls per network, all finite), but without its S37 partner there is no
  list-b comparison.

**14b. The first build (dials 1; the lead's 06:05 UTC recipe; superseded, not fired).**

Parent: the trainer's dial-only export `O/trainer/candidates/runM_dials_ds08/candidate.py` (sha 4de2b950: RRPI's learned arrays
with the learned mean correction x 0.8 folded into its output weights, and the covariance-output knob k21n at -0.048; no
trained parameter). On it: `python stack/build.py T+C+B+F+D+Hfull+EcutS+F4v2t0.6+K4Lv3c0.75,0.75+L3+Phfull_v1+Rlvfuse_sized32+Rtsr_off
<parent> runM_dials_ds08` = the S37 lever stack with K4L strength 0.75 on both pair ages, the edge cut with the tanh stand-in,
and the three-request operand loads.

| file | sha256 | against the next one up |
|---|---|---|
| `stack/candidate_T_C_B_F_D_Hfull_EcutS_F4v2t0.6_K4Lv3c0.75,0.75_L3_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_dials_ds08.py` (**S38 as specified**) | 0d51ccb90f9cc9db853de2a4801f0af2908a87f0baac863ed3fd9876a465818c | - |
| the same with `_L3_L3o_` (**S38 + L3o, the one I would send**) | 882dd7d135ec79d2d96d1d6b37c3554f48292d16d5075ecf72f44966241dec6f | predictions byte-identical, bill equal to the flop, 254 / 364 fewer calls (651 / 378) |
| the same without `_L3` (reference for L3 only) | e544d59a176b17b9c6f33d99892182498489cda60f956cdbaf8d7e63fec4111c | predictions byte-identical, bill -0.127% / -0.075%, 1,002 / 1,227 more calls |

Einsum gate against the S37 file: PASS for all three. Setup alone: 0.92 s, 29,571,934,767 flops (S37's), 703 keys bound.

**Native session on fresh list a** (651, 378, 424, 328, 287, 641, 548, 557, 316, 390; one process, one thread, under the guard,
840 s; `logs/quick_s38_ds08_ecuts_l3_fresh_a.json`), against my stored session of the S37 file on the same list in the same
order (`python a14_f4/fresh_compare.py s38_ds08_ecuts_l3_fresh_a --base s37_t06_fresh_a`):

| network | bill | final-layer MSE (x 1e-9) | MSE change | all-layer MSE change | logged calls |
|---|---|---|---|---|---|
| 651 | -0.533% | 15.2937 | -3.56% | -1.96% | 36,674 |
| 378 | -0.629% | 14.0806 | -0.20% | -0.02% | 36,360 |
| 424 | -0.538% | 15.8181 | -1.60% | -0.99% | 37,229 |
| 328 | -0.630% | 16.2206 | +0.30% | -0.37% | 35,860 |
| 287 | -0.651% | 17.3913 | -0.24% | +0.45% | 36,137 |
| 641 | -0.624% | 17.4936 | -1.57% | -0.42% | 35,990 |
| 548 | -0.669% | 15.0724 | +0.31% | +0.15% | 35,800 |
| 557 | -0.636% | 16.4498 | +0.25% | +1.04% | 35,861 |
| 316 | -0.673% | 14.9967 | -1.01% | -0.43% | 35,861 |
| 390 | -0.669% | 19.7068 | -2.74% | -1.05% | 35,958 |

| pooled over the 10 networks, S38 against S37 | value |
|---|---|
| final-layer MSE | **-1.062%** [95% -1.878, -0.267]; per-network mean -1.007% (se 0.428), lower on 7 of 10 |
| bill | **-0.624%** [95% -0.652, -0.591] |
| adjusted (MSE x bill per network, summed) | **-1.696%** [95% -2.495, -0.910] |
| all-layer MSE | -0.372% [95% -0.857, +0.102] |
| logged calls per network | 36,173 against 37,206 (-1,033) |
| local CPU per network (one thread, loaded machine) | 72.6 s against 74.0 s |
| lazy tree binds | 0.196% of the bill (S37 0.194%) |
| edge cut: share of the 1024 x 1024 grid evaluated | 0.51-0.53 |
| health | all predictions finite; guard not tripped |

- **Where the -0.62% of bill comes from** (networks 651 / 378): the edge cut with the stand-in -0.659% / -0.703%, the L3 loads
  +0.127% / +0.075%, the dials 0.
- **Where the -1.06% of MSE comes from**: the three dials (correction x 0.8, k21n -0.048, K4L 0.75). The trainer's torch number
  for the same dials on 50 held-out ranks is -0.98% (se 0.23, 38 of 50 lower); the edge cut's own cost is +0.02%. The native
  file delivers the card's gain.
- **How much to trust 10 networks**: my 10-network session of S37 overstated its board gain by about 1.7 points (state block).
  The bill change is solid (a per-network range of -0.53 to -0.67%); for the MSE the trainer's 50 ranks are the better number,
  and the board result should be expected near -1.0% of MSE and -0.6% of budget share, about **-1.6% adjusted (1.951e-9 ->
  about 1.92e-9)**.
- **What the release lane must run on the exact file before an upload**: the boundary stages (client and server) and the
  request count. New op forms in this file that S37 does not use: in the stand-in `minimum`, `sqrt` and `divide` with `out=`;
  in the edge cut a `put` on a flat index grid; in L3 a `reshape` of a strided basic slice of the tree (the old loads reshaped
  a contiguous slice), and in L3o a `transpose` of a two-element strided slice.

## 15. F4c, the physicist's exact-term mixture, in native form: the port is his module (four networks); bill +0.13%, about 850 requests; no gain over S38 at his literal strengths

What it is (`O/physicist/RESULT.md` T13 / T14, reference `O/physicist/f4c_native_ref.py`): F4's rule stays (each young gated
column shifted along its own square, strength theta), and the hub's exact fourth-cumulant terms S1 = -alpha pdf / sigma^2 and
S2 = 2 (g - g^2 - m pdf / sigma) are added at strengths c1, c2. For a pair's pushed columns p (+) and q (-), D = p - q:
shift_p = (d/2) [theta A p^2 + (n_p0 p^2 + n_p1 p D + n_p2 D^2) / lam], and the same for q with its own three numbers and the
weight -lam. The six n are a 2 x 2 least-squares solution whose metric is fixed at the pair's birth, so they are constants of
the pair. Torch, 48 held-out ranks: -0.48% [95% -0.83, -0.12] against F4 + K4L, lower on 34 of 48.

Native form (`a27_f4c/fragment_f4c_v1.py`, lever `F4Ct<theta>c<c1>,<c2>`, replaces F4; per-age strengths as
`t<age 1>,<age 2>` and `c<c1 a1>,<c2 a1>,<c1 a2>,<c2 a2>`):
- **At the birth** everything that does not depend on the later layer is folded into six float32 coefficients per pair and
  age: the six n, the rule theta A (with q^2 = p^2 - 2 p D + D^2), and the weight the columns carry at that age (newborn weight
  x s_new x s_carry^age). The step then only evaluates two quadratic forms: shift = (d/2) (a p^2 + b p D + c D^2).
- **The birth constants in two small einsums.** In units of the pair's common leg (x = t+ / |s|, y = t- / |s|) every n is
  [S1 / |s| P1(x, y) + S2 / |s|^2 P2(x, y)] / ((x - y)^(j+1) Delta), Delta = 2 x^2 y^2 + (x + y)^2 + 2, with integer polynomials
  P of degree at most 5 in x and 3 in y. The fragment derives them once per process from the reference's own matrices by exact
  polynomial arithmetic in plain Python (the adjugate of the 2 x 2 system times its right-hand side, the change from
  (Y^2, Y Wt, Wt^2) to (p^2, p D, D^2), and one exact division by x - y that the closed form allows; `a27_f4c/closed_forms_sympy.py`
  is the symbolic derivation this was read from). The network-dependent part is then two power tables and two einsums on
  K-vectors. The reference's two guards are kept (gap below 1e-12; determinant below 1e-8 of the diagonal product): such a pair
  gets no exact part.
- **The cohort born at layer 0 is shifted too** (the physicist's warning): its rule part is exactly 0, its S2 is 0.18.
- **The step** is F4b's draft step (section 9) without its per-layer least squares: live rows only, float32, per cohort D, p D,
  D^2 and ten combine passes written into one shift block, then F4's anchor on the whole block.

| cost | F4 (in S37 / S38) | F4c |
|---|---|---|
| flops per element of live rows x shifted columns (unit check) | 16 | 20.7 measured (my count 21.5: r^2 1; D, p D, D^2 and the two forms 6.5; five matrix-vector passes 10; two products 2; scaling and adding the shift 2) |
| a birth (unit check) | about 8 charged ops (the rule's scalar) | 52 charged ops; about 4,600 flops per pair at n = 1024 = 3.6 Mflop per layer (2,048 per pair for the leg norms) |
| charged ops per layer, step, two cohorts (unit check) | 46 | 67 |
| **bill, measured natively** (mixture build against the F4 build, networks 474 / 648) | - | **+0.130% / +0.127%** |
| **logged calls per network, measured natively** | - | **+856 / +846** |

The paper count before the runs was +0.095% of bill; the difference is the cohort born at layer 0, which F4 skips and the
mixture shifts at layers 1 and 2. A plain line-by-line port of the reference's birth algebra would have been about 70
requests per layer (about +1,300 per network); the polynomial form buys the difference for about 0.01% of bill.

**Unit check** (`a27_f4c/unit_f4c.py`, synthetic data, two cohorts; with one strength set and with one set per age): the six
coefficient vectors per pair and age against `exact_constants` + the rule: relative difference 2e-8 to 4e-7 (float32
storage); three degenerate pairs (gap 0, a leg of norm 0, a nearly singular system) get no exact part in both; shift 3-4e-7,
beta 3e-7, plane diagonal 1.5-1.7e-7. PASS on the first run.

**Native parity** (`bash a27_f4c/run_parity.sh`: two builds on RRPI, screen stack, each one process, one thread, under the
guard, on the physicist's held-out ranks 474 and 648; `python a27_f4c/parity_fr48.py f4c_fk_474_648 f4c_mbk_474_648`).
FK = F4 0.58 + K4L 0.73 (`stack/candidate_T_C_B_F_D_F4v4t0.58,0.58_K4Lv3c0.73,0.73.py`, sha 6cefab2e); MBK = the mixture in
his held-out form + K4L 0.71 (`stack/candidate_T_C_B_F_D_F4Ct0.378c0.48134398907692455,0.48,0.4801278724647027,0.48_K4Lv4c0.71,0.71.py`,
sha d692d8cf, einsum gate PASS). His stored arm MBK is the three-entry form (the rule without its own S1, exact S1 at 0.858);
it is the lever's four-entry form at c1 = 0.858 - 0.378 x s_new x s_carry^age exactly, because the fit is linear in the S
terms and the rule's own S1 is the exact S1 times the weight dials.

| MBK - FK, native against torch (`O/physicist/k4l_pred_rank*_fr48.npz`) | network 474 | network 648 |
|---|---|---|
| rms of the prediction change at the final layer, native / torch | 13.63e-6 / 12.64e-6 | 12.34e-6 / 11.21e-6 |
| cosine of the two changes, layers 2-15 | 0.886 to 0.939 (final layer 0.912) | 0.857 to 0.935 (final layer 0.922) |
| projection of the native change on the torch change, layers 5 / 10 / 15 (1 = same size) | 1.04 / 1.00 / 0.98 | 0.99 / 0.97 / 1.02 |
| left after subtracting the torch change, layers 2 / 7 / 15 | 0.76 / 2.77 / 5.59e-6 | 0.74 / 2.53 / 4.78e-6 |
| final-layer MSE change, native / torch | -0.44% / +0.13% | -0.36% / -0.80% |

- The leftover is the distance between two rounding-level native builds (4-5e-6 at the final layer, 0.7 at layer 2: the same
  numbers as in the F4 and the ten-dial parity runs). It also explains the 8-10% larger rms of the native change:
  sqrt(12.64^2 + 5.59^2) = 13.8 against 13.63 measured. No missing or doubled term (that would read 0.5 or 2 in the projection).
- Pooled over the two networks the MSE change is -0.40% natively and -0.35% in torch; one network moves by half a percent
  with any rounding-level change.
- **The trade**: the physicist's -0.48% of MSE (se 0.18, torch, 48 held-out ranks) for +0.13% of bill, net about -0.35%,
  and about 850 requests (0.006 s of residual).
- **Rehearsal of the one command for candidate 3** (`a20_runm/rehearsal_v4`: the trainer's dials-2 values plus the new
  leaves x1a1 = x2a1 = x1a2 = x2a2 = 0.48, f4a -0.222, k4la -0.29, e2 0.3): levers
  `F4Ct0.378,0.378c0.48,0.48,0.48,0.48+K4Lv4c0.71,0.71+E2s0.3` on the full S38 stack, sha 6f85934d, einsum gate PASS, setup
  alone 0.91 s with S38's flops; natively on 651 and 378 (`logs/quick_rehearsal_v4_651_378`): finite, bill +0.131% / +0.134%
  against the fired S38 file, logged calls +983 / +973 (the mixture about +850, the skew term about +130), peak RSS 4.08 GB.

**Parity on the lead's two networks** (`bash a27_f4c/run_parity.sh 35,651 35_651 port`, against the physicist's port references
`O/physicist/k4l_pred_rank00{35,651}_port.npz`, same two builds):

| MBK - FK | network 35 | network 651 |
|---|---|---|
| final-layer MSE change, native / torch | -2.46% / -2.66% | -1.58% / -1.89% |
| cosine of the two changes, layers 2-15 | 0.893 to 0.941 | 0.852 to 0.918 |
| left after subtracting the torch change, final layer | 4.74e-6 | 4.72e-6 |

Four networks in all (474, 648, 35, 651): the native change is the torch change.

**Preview on the fired S38 stack: no gain at the physicist's literal strengths.** The S38 file with F4C in place of F4 (rule
0.378 both ages, exact S1 and S2 at 0.48, K4L left at 0.75; `stack/candidate_T_C_B_F_D_Hfull_F4Ct0.378c0.48,0.48_K4Lv4c0.75,0.75_EcutS_L3_L3o_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_dials2.py`,
sha f52d28b1) on fresh list a against the stored S38 rows (`python a14_f4/fresh_compare.py s38_f4c_preview_fresh_a --base
s38_dials2_fresh_a`; one process, one thread, under the guard, 869 s):

| against the fired S38 file, 10 fresh ranks | value |
|---|---|
| final-layer MSE | +0.18% [95% -0.40, +0.91]; per-network mean +0.21% (se 0.37); lower on 5 of 10; best -0.85%, worst +2.87% (network 424) |
| per network (651 378 424 328 287 641 548 557 316 390) | -0.49 +0.58 +2.87 -0.66 +0.21 +0.03 -0.85 -0.58 +1.48 -0.47 |
| per layer, pooled | 0.1-0.3% better at layers 2-9, 0.2-0.6% worse at layers 12-15 |
| bill | +0.133% on every network |
| adjusted | +0.33% [95% -0.27, +1.08] |
| logged calls | about +810 per network |
| health | all predictions finite; peak RSS 5.44 GB |

- On the untrained parent the mixture beats F4 0.58 + K4L 0.73 by 1.6% on network 651 natively; on the dials-2 file it beats
  F4 0.656 / 0.6 + K4L 0.75 by 0.5% on the same network and by nothing over the ten ranks. The mixture's strengths were
  fitted on the untrained parent; candidate 2's dials seem to have taken part of what the exact terms gave.
- Ten ranks cannot exclude a gain of 0.4%, but they do not show one. The lever costs +0.13% of bill and about 850 requests,
  so the trainer's line search at candidate 2 has to find at least 0.2-0.3% before F4C belongs in a file.

## 16. A CPU fact about the tree kernel (measured on small numpy probes, not on the estimator)

numpy runs an element-wise operation on strided views through its buffered iterator when the contiguous run is shorter than
4,096 floats: about 1.3-2.6 ns per float instead of 0.3 (`a28_numpy_runs/run_length_probe.py`, numpy 2.4.6; the same on 1.26
and 2.2). In the in-place tree the level-j components are views with runs of 7^(j-1) x leaf floats, so at 16 x 16 leaves the
level-1 and level-2 operations (runs of 256 and 1,792 floats) take that path:

| one level operation of a 1024 x 1024 operand, 6 levels (`a28_numpy_runs/level_ops_probe.py`) | level 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| floats per component | 262,144 | 458,752 | 802,816 | 1,404,928 | 2,458,624 | 4,302,592 |
| contiguous run, floats | 256 | 1,792 | 12,544 | 87,808 | 614,656 | 4,302,592 |
| ns per float in the tree / on contiguous copies | 2.58 / 0.32 | 1.35 / 0.28 | 0.33 / 0.29 | 0.25 / 0.31 | 0.34 / 0.31 | 0.27 / 0.27 |

Levels 1 and 2 hold 7% of a transform's elements and about a third of its time (3.9 ms against 2.8 ms for one operation per
level). The shortest runs always fall on the two cheapest levels whatever the order of the level axes, so reordering does not
help, and I found no billed-neutral way around it. It is a fact for the release lane's CPU model (leaf size and depth change
how many levels sit below the 4,096 threshold), not a lever.

## 17. The star carried by its row sums (lever SS): bill -2.37%, 150 calls fewer, error +0.5% on ten networks; in the S39 file

**What the star is.** After a layer's pair map the third-order content that the bank columns and the trace terms do not hold
is kept as one n x n plane per layer, the star A (the live loop's `star = 3 * rest.T`). The next layer takes
`s2 = (W o W)' A' W` from it (a third of it off the diagonal, its diagonal as it is). The fired S38 file holds A in rank-32
factors Q (Q' A): three thin products with A, three more to move the factors, two QR factorizations a layer: 2.4% of the
bill in all (the six thin products alone 2.2%).

**The finding** (network 651, planes captured in native runs and compared in float64: `a31_star/probe_star3.py`,
`a33_starsum/dump_planes.py` + `forms_offline.py`). Squared weights are positive, so `(W o W)' = m 1' + R` with m_i the mean
of unit i's squared weights and R a zero-mean remainder, and

    s2 = m (W' A 1)' + R A' W.

The first part needs only the ROW SUMS of A: how strongly each unit is tied, at third order, to the squared norm of its layer.

| relative size or error, network 651 | layers 1 / 2 / 3 | layers 4-14 |
|---|---|---|
| share of s2's power in the first part | 97.2 / 98.6 / 99.6% | 99.4-99.8% |
| row sums the rank-32 factors miss, relative to the row sums | 3.2 / 8.5 / 4.0% | 0.8-1.8% |
| error of the transported plane: the fired form (rank 32) | 16.3 / 12.2 / 6.0% | 2.2-3.5% |
| the same with the missed row sums put back (rank 32 + row sums) | 15.9 / 8.8 / 4.5% | 1.9-3.2% |
| rank 16 + row sums | 16.3 / 9.2 / 4.8% | 2.3-3.5% |
| rank 8 + row sums | 16.5 / 9.5 / 5.1% | 2.6-4.1% |
| one direction along the row sums (a = A 1, s2 = ((W o W)' A' a)(W' a)' / a'a) | 16.7 / 10.7 / 5.7% | 3.0-6.5% |
| the first part alone (rank 0) | 16.7 / 11.9 / 6.6% | 4.0-7.6% |

So 14-45% of the power of the fired form's error is in the row sums themselves, and the star plane is 60-75% ONE rank-one
piece at layers 4-12 (which is why one direction recovers 45% of what rank 32 holds beyond the row sums).

**The lever** `SS<form>` (`a33_starsum/fragment_ss_v3.py`, no loop edit; it redefines the two functions that move the star,
`_flms_cov_star` for layers 1-14 and `_flms_star_diagonal` for the final layer): the missed row sums `u = A 1 - Q (B 1)` go
through W as one vector and enter as one more column of the two factors of s2 (`T2` gets m, `P1` gets `W' u`). `<form>` is a
rank (the parent's factors at that rank plus the column), `0` (the first part alone: no factorization, no QR, no thin
product) or `d` (the one-direction form); `SS<f>x<g>x<l>` uses form f below layer l and g from l on. Unit check against
float64 numpy on a random plane (`a33_starsum/unit_ss.py`, the covariance product replaced by a plain matmul): rank 16 equal
to 7e-7, rank 0 to 1e-6; one layer bills -204 Mflop (rank 16) and -413 Mflop (rank 0) against the fired form.

**Native sessions** (S38 stack + SS, the stored S38 rows as reference, fresh list a = 651 378 424 328 287 641 548 557 316 390,
one process per form, one thread; `a33_starsum/compare.py <name> [<base>]` and `a14_f4/fresh_compare.py <name> --base s38_dials2_fresh_a`):

| form | networks | bill | logged calls | final-layer MSE | adjusted | the final prediction moves by (rms, 1e-6; share of the MSE's power) |
|---|---|---|---|---|---|---|
| rank 32 + row sums (sha 4a9d3ee4 / 0ea1c94e) | 2 (651, 378); the ten-network session is queued | +0.036 / +0.037% | +135 | -0.24 / -0.52% | - | 8.6-9.0; 0.48-0.59% |
| rank 16 + row sums (sha f764f3c0) | 2 (651, 378) | -1.14 / -1.18% | +135 | +0.20 / -0.19% | - | 9.4-9.9; 0.58-0.70% |
| rank 8 + row sums (sha 3f125260) | 10 | -1.765% [-1.79, -1.74] | +135 | +0.31% [95% -0.28, +0.84] | -1.47% [-2.04, -0.96] | 10.2-12.8; 0.66-0.95% |
| one direction (sha e631d2b3) | 10 | -2.353% [-2.39, -2.31] | -105 | +0.89% [+0.36, +1.47] | -1.49% [-2.00, -0.95] | 11.6-14.6; 0.83-1.27% |
| **the row sums alone, rank 0** (sha 9c6d3692) | 10 | **-2.372%** [-2.41, -2.33] | **-150** | **+0.50%** [-0.06, +0.97] | **-1.88%** [-2.44, -1.43] | 12.2-14.8; 0.99-1.34% |

Between forms, on the same ten networks: one direction against rank 0: MSE +0.39% (se 0.28) for +0.018% of bill; rank 8 against
rank 0: MSE -0.19% (se 0.25) for +0.62% of bill, the prediction moving by 0.33-0.53% of the MSE's power; rank 16 and rank 8
against rank 32 on two networks: the prediction moves by 5.4-5.8e-6 and 6.7-7.2e-6 rms (two rounding-level builds: 4-5e-6),
rank 0 by 9.3-10.2e-6.

**Reading.** (1) A change cannot gain more MSE than the power it moves. Everything the rank-32 factors add to the row sums
moves the final prediction by about 0.6% of the MSE's power and costs 2.4% of bill; the rank-8 part moves 0.4% and costs
0.6%. So no rank earns its bill back, whatever the ten-network noise (about 0.3% per form) hides. (2) What cost MSE in the
fired form was the 1-2% of row sums it missed: putting them back moves the prediction by 0.5% of the MSE's power and lowered
the MSE on both networks it was run on (-0.24%, -0.52%; the ten-network session is queued). (3) Closer to the exactly transported plane is not better MSE for the part beyond the row sums: the
one-direction form is closer at every layer (error power -36% at layers 4-14 against rank 0) and measures worse. The same
was seen when the star's s1 term was dropped (09-24). (4) Rank 0 is 4.7 times more bill than MSE on ten networks, a little
under the lead's 5x rule; the lead took it for S39 (10:40Z). 48-50 torch ranks would fix the MSE side to 0.1-0.15%.

**For the torch cards**: `a33_starsum/star_ss_torch.py`, `install(cd, torch, rank)` replaces the card's two star functions
with this form. Written, NOT run (no torch in this lane); the cards still train on the plain rank-32 star.

Not pursued, with reasons: the own-cumulant term of the star's diagonal made exact (`sum_k w_ki^3 A_kk`: the diagonal holds
0.5-2% of the plane's energy and the exact term lowers the diagonal's error from 4.2% to 4.15% at layer 9); forming only the
row sums instead of the plane (about 7 n^2 flops a layer, 0.04% of the bill); a second direction at late layers (the same
bound as (1)).

## 18. The pair map's extra (3,1) plane (remainder31): the native input exists and has parity with the torch hook

The carrier of the fourth-order norm coupling (lead, physicist; `O/NCAR_SPEC.md`) hands the pair map an extra K31 plane,
`remainder31_ij = k31 (3/n) R_ij / (sigma_i^3 sigma_j)`, diagonal zero (the torch card's `pair_in` value[5]; the physicist's
`ntor.py`). The native pair map already has that input: argument `extra31` of `_nonlinear_pair_batch` (a plane in physical
units, standardised inside as `plane_ij / (sigma_i^3 sigma_j)`, diagonal zeroed, added to `T31`, which both series paths
read), switched on per layer by `_T31_LAYERS` with the plane supplied by `_t31_extra(l, ctx)`.

Parity (`a34_r31/oracle_native.py`, `parity.py`; network 651; the S38 file unchanged, the two names set on the loaded
module): the physicist's ORACLE matrix (`O/physicist/t19_oracle_rank0651.npz`, symmetrised, full, strength 1, layers 1-14)
against the physicist's torch arm K31f (`k4l_pred_rank0651_ora.npz`):

| layer | 2 | 3 | 4 | 6 | 8 | 10 | 12 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|
| native change, rms 1e-6 | 3.01 | 5.80 | 8.60 | 14.32 | 21.74 | 30.58 | 39.13 | 43.89 | 50.05 |
| torch change, rms 1e-6 | 2.82 | 5.64 | 8.34 | 14.03 | 21.56 | 30.66 | 38.99 | 43.71 | 49.97 |
| cosine | 0.879 | 0.943 | 0.967 | 0.983 | 0.990 | 0.993 | 0.995 | 0.994 | 0.995 |
| left over after the torch change, rms 1e-6 | 1.44 | 1.93 | 2.18 | 2.63 | 3.08 | 3.58 | 4.01 | 4.71 | 5.11 |
| MSE change, native / torch | -0.9 / -1.5% | -3.3 / -3.2% | -3.9 / -3.6% | -5.3 / -4.5% | -6.2 / -6.3% | -9.3 / -9.3% | -8.0 / -8.2% | -11.4 / -12.0% | -15.31 / -15.60% |

The left-over is at the level of two rounding-level native builds (4-5e-6 at the final layer). Final-layer MSE: native
15.240 -> 12.907e-9, torch 15.622 -> 13.185e-9. So a carrier only has to produce its plane; the pair-map side is done. In
this test the plane is a full float64 matrix built outside the budget, so the run prices nothing; a carrier's rank-2K plane
`U Q' + Q U'` formed in float32 costs about (2K + 3) n^2 flops a layer (K = 16: 37 Mflop, 0.19% of the bill over 13 layers).
Nothing is ported yet: the lead's 10:40Z decision is that the carrier is S40 and its design (state, law, which terms) is
still being priced by the physicist.

## 19. The S39 file: candidate 3 (five changes) + the star by its row sums; native on ten fresh ranks: adjusted -2.27% against S38

The lead's 10:40 UTC decision. File:
`stack/candidate_T_C_B_F_D_Hfull_F4Ct0.656,0.6c0.0,0.44439084790069294,0.0,0.17112793763960094_K4Lv4c0.8597296897041633,0.8597296897041633_EcutS_L3_L3o_SS0_Phfull_v1_Rlvfuse_sized32_Rtsr_off__runM_cand3_five.py`,
sha256 a158b9abd2c260c31bf8025141fcf6e3a3c1a07a96361bcbb17260862ae52c29. It is the trainer's candidate 3 file
(`O/trainer/candidates/runM_cand3_five/candidate.py`, sha a4515f39: learned mean correction x 0.784, cstar leg j2 x 0.791,
learned covariance edge x 0.75, and S38's k21n, b source, bscale, X1) built by the one command
(`stack/from_params.sh O/trainer/cert/mix3/mix3b_optimum_min5.pt runM_cand3_five 'EcutS+L3+L3o' <the trainer's file>`, which
reads the trainer's leaves: rule strength 0.656 / 0.6 per pair age, exact S2 strength 0.44439 / 0.17113, exact S1 0 / 0, K4L
0.85973 on both ages) with the lever SS0 added. Einsum gate PASS; setup 1.27 s, 29.57 Gflop, 7,074 calls (as S38). The edge
strength (a per-layer table read at call time) multiplies the cut edge of EcutS as it multiplied the full edge (checked in the
source).

Native session (fresh list a, 10:51-11:08 UTC, 993 s at a load of 23-28; `a14_f4/fresh_compare.py s39_cand3_five_ss0_fresh_a --base s38_dials2_fresh_a`),
against the fired S38 file on the same ten networks in the same order:

| | S39 against S38 | of which the star by its row sums (S38 + SS0) | of which candidate 3 on top of it |
|---|---|---|---|
| final-layer MSE, pooled | **-0.04%** [95% -0.75, +0.67]; per-network mean +0.01% (se 0.39), lower on 4 of 10 | +0.50% [-0.06, +0.97] | -0.54% |
| bill | **-2.239%** [-2.27, -2.20] | -2.372% | +0.136% |
| adjusted | **-2.27%** [-2.98, -1.57] | -1.88% | |
| logged calls per network | 36,492 against 35,833 (+659) | -150 | +809 |
| all-layer MSE | +0.11% | +0.32% | |

All predictions finite; peak RSS 4.62 GB. Candidate 3 per network, native (S39 against S38 + SS0) beside the trainer's torch
card (candidate 3 against candidate 2, `O/trainer/cert/mix3/mix3b_optimum_min5_val50.json` and `runM_dials2_val50.json`):

| network | 651 | 378 | 424 | 328 | 287 | 641 | 548 | 557 | 316 | 390 | pooled |
|---|---|---|---|---|---|---|---|---|---|---|---|
| native, % | +0.57 | -0.05 | +0.12 | -0.65 | -0.01 | -1.69 | -1.04 | -2.37 | +0.74 | -0.65 | -0.54 |
| torch, % | +0.00 | +0.16 | +1.44 | -0.61 | +0.56 | -1.11 | -1.57 | -1.47 | +0.68 | -0.93 | -0.31 |

Correlation 0.81: the dial part is ported as trained, and these ten ranks are a weak draw for it (the torch card has -0.70%,
se 0.15, on all 50 held-out ranks). With the 50-rank number the board expectation is MSE about -0.2%, bill -2.24%, adjusted
about -2.4% against S38's 1.907e-9. The release lane's pre-check of the file (11:01 UTC): four new einsum signatures, all in
the module's once-per-layer coefficient step, none with a repeated operand; about 42,560 requests per network at the boundary
(S38 41,862), residual about 0.299 s mean, inside the designation rule with about 740 requests to spare.

Not run before the window: candidate 3 without the star change, and rank 32 with row sums on ten networks (both stopped to
give the S39 session and the lead's chain the machine; they resume after the fire and change nothing in the file).

## 20. What a fourth-order carrier would cost natively (price list; nothing ported)

Asked by the lead (11:39 UTC). Read from flopscope's own counter on real calls (`a36_carrier/price_probe.py`, under a second, no
estimator), float32 operands (float64 doubles every line), in % of the S39 file's bill (2.513e11 flops per network on list a) when
the piece runs at the 13 layers 2-14:

| piece | flops a layer | % of the bill |
|---|---|---|
| one product of W or of the covariance with a block of k vectors (plain matmul, 2 n^2 k) | k = 8 / 16 / 24 / 32 / 56: 16.8 / 33.5 / 50.3 / 67.1 / 117.4 Mflop | 0.087 / 0.174 / 0.260 / 0.347 / 0.607 (0.0109 per vector) |
| one pass over the young bank columns with a block (about 6,000 columns x 900 live rows) | k = 16: 173 Mflop; k = 24: 259 Mflop | 0.89; 1.34 (0.056 per vector); the conversion term is two passes |
| the old tier through its factors (384 x about 9,000 coefficient columns) | k = 16: 123 Mflop | 0.64 a pass: no cheaper than the columns |
| QR of an n x k block (`fnp.linalg.qr`, Q and R) | k = 16 / 32 / 56: 1.04 / 4.15 / 12.6 Mflop | 0.005 / 0.02 / 0.065 |
| symmetric eigen-decomposition of a k x k matrix (`fnp.linalg.eigh`, 9 k^3) | k = 32 / 56: 0.29 / 1.58 Mflop | 0.0015 / 0.008 |
| eigen-decomposition of the covariance itself (9 n^3) | 9.66 Gflop | 3.85 PER CALL: not available as a method |
| the plane Z M Z' at rank 32, standardised and added to T31 | 71 Mflop | 0.37 (rank-16 two-factor plane: 0.195) |
| probes: a slice of weight columns | 0 (a view) | 0; seeded Gaussian numbers bill 32 flops each |

Every line is one request. S39 has about 700 requests of room under the release lane's designation rule.

**Eigenvectors by power sweeps** (`a36_carrier/dump_state.py` stores a network's covariances; `eig_offline.py`, numpy, network
651): warm start = the previous layer's block pushed through the gates and W, s sweeps with the covariance, a Rayleigh-Ritz
rotation. Share of the exact top-K eigen-subspace held at layers 5 / 9 / 12 / 14:

| block | sweeps | share held | vector products a layer | % of the bill |
|---|---|---|---|---|
| K = 16, no extra columns | 2 | 0.81 / 0.86 / 0.88 / 0.89 | 64 | 0.70 |
| K = 16 + 8 extra | 2 | 0.91 / 0.98 / 0.95 / 0.995 | 96 | 1.05 |
| K = 16 + 16 extra | 2 | 0.98 / 0.996 / 0.995 / 0.999 | 128 | 1.40 |
| K = 8 + 8 extra | 2 | 0.88 / 0.97 / 0.99 / 0.99 | 64 | 0.70 |
| K = 8 + 8 extra | 3 | 0.90 / 0.99 / 1.00 / 1.00 | 80 | 0.87 |

The spectrum explains it: at layer 12 the eigenvalues 1 / 2 / 8 / 16 / 17 / 32 are 0.062 / 0.032 / 0.020 / 0.014 / 0.014 / 0.0084
of the trace (the 16th and 17th are not separated), and at layers 1-2 the spectrum is flat (first over 32nd = 1.3-1.5).

By these rates the vector carrier of `O/NCAR_SPEC.md` at K = 16 costs 1.4-1.6% of the bill without its bank pass and 3.2-3.4%
with it (the trainer's 50-rank gain for the form with the pass: -2.06%); the sketched carrier `O/ncar4.py` at r = 32, k = 24:
2.0% without the conversion, 4.7% with it. Posted to the team log at 11:22, 11:43 and 11:51 UTC.

## 21. The joins lose norm coupling; putting it back is exact and does not lower the final error (lever JR, closed as a file change)

Same quantity as in section 17, for the old tier. A join projects the joined columns J (weights lam) on its rank-r frame Qn. The
third-order norm-coupling vector of those columns, t_c = sum_r lam_r |J_r|^2 J_rc, changes by
`miss = J (lam o |J_r|^2) - Qn (Fj (lam o |Fj_r|^2))`, and the old tier never gets it back. Network 651, the estimator's own
columns, the join repeated in float64 (`a35_join/probe_join.py`; `dump_join.py` + `join_offline.py`, `old_offline.py` offline):

| join | frame rank | energy of the joined columns outside the frame | miss of their norm coupling | the fired rank-32 star's miss of its row sums at that layer (absolute; the join's miss is) | share of the dropped plane's power (after the weights) that is the coherent part |
|---|---|---|---|---|---|
| layer 7 | 384 | 0.033% | 1.25% (0.122 absolute) | 0.077 (1.6x) | 97.7% |
| layer 10 | 320 | 0.030% | 2.30% (0.172) | 0.066 (2.6x) | 96.1% |
| layer 13 | 384 | 0.002% | 1.02% (0.050) | 0.021 (2.4x) | 95.9% |

By rank (layer 10): 384: 1.2%, 320: 2.3%, 256: 5.2%, 192: 8.8%, 128: 16%. The old columns that the layer-10 join re-projects add
0.30 of the new columns' miss (cosine 0.23): 10% more in all; the layer-13 join is lazy and re-projects nothing.

Lever `JR[<x>]` (`a35_join/fragment_jr_v1.py`; one line in the join, one in the live loop): the miss is recorded at the join (two
passes over the joined columns) and its coherent part m_i (W' miss)_j is added to the transported star of the join layer, three
times that on the diagonal; the residual star carries it on. Bill +0.044%, +84 logged calls.

| S39 file + JR against the stored S39 rows, fresh list a | |
|---|---|
| final-layer MSE, pooled | **-0.11%** [95% -0.74, +0.53]; per-network mean -0.09% (se 0.34), lower on 5 of 10 |
| all-layer MSE | -0.28% [-0.70, +0.10] |
| bill, calls | +0.044%, +84 |
| the final prediction moves by | 9.2-13.9e-6 rms = 0.59-1.14% of the MSE's power (mean 0.79%) |

A pure removal of error of that size would have given -0.79%, an unrelated change +0.79%; the ten networks say neither. Not for
a file by itself. It is an exact piece of physics with a strength literal (`_JR_SCALE`), so it is a candidate dial for the
trainer; torch drop-in `a35_join/jr_torch.py` (written, not run).

Other old-tier arms, two networks each (651 / 378) against the S39 rows (levers `JK<r7>,<r10>,<r13>` = frame ranks, `LU<r>` =
rank of the old pairs' common part at layers 10-14):

| arm | bill | final-layer MSE | verdict |
|---|---|---|---|
| layer-10 join at rank 384 instead of 320 (`JK384,384,384`) | +1.70% | -2.79% / -1.56% | break-even: the fired rank cut saves what it costs |
| the same with the join repair | +1.74% | -2.65% / -2.32% | break-even |
| ranks 384 / 256 / 384 with the join repair | -1.22% | +2.55% / +5.66% | no |
| the old pairs' common part at full rank (`LU384`) | +1.33% | -0.60% / -0.52% | no (the fired 256 is the better trade) |

## 22. The bank channel of the fourth-order norm coupling, folded into the bank read (lever NB): -2.5% on ten networks; superseded by the pair law

**What it is.** The estimator carries, per layer, a bank of columns pp_r with weights lam_r; the third-order plane it reads is
`sum_r lam_r pp_ir^2 pp_jr`. The physicist's bank channel (`O/physicist/ncar_fold.py`, level 'read') puts a fourth-order
quantity on the same columns: how every pair of units varies together with the squared norm of its layer, as the matrix
`B = sum_r w_r pp_r pp_r'` with one more weight w per column. Per layer the weights are carried as
`w_r <- 2 (gbar w_r + lam_r V' pp_r)` (V = 2 m (1 - g), gbar the mean of g - m pdf / sigma), and the read's left factor
`pp_ir^2 lam_r` becomes `pp_ir (pp_ir lam_r - f_i w_r)` with `f_i = strength mu_i / (n v_i)`: the read multiplies by the
right factor as before, so the channel costs passes over the columns but no product. What the fold must not contain, a
multiple of the covariance, comes from a regression of B's diagonal on (v_i, mu_i^2, mu_i (C mu)_i) and is put back on the
read's block.

**Native results** (against the stored S39 rows; `a14_f4/fresh_compare.py <name> --base s39_cand3_five_ss0_fresh_a`):

| build | networks | final-layer MSE | bill | logged calls |
|---|---|---|---|---|
| NB, first version (one form, the multiple of C at 1.0) | 10 fresh (list a) | **-2.49%** [95% -3.87, -1.34], lower on 10 of 10; all layers -3.25% | +0.539% | +727 |
| the same | 651 / 378 | -3.78% / -1.18% | +0.537% / +0.540% | +727 |
| one form, the multiple of C x 1.41 (the physicist's factor) | 651 / 378 | -4.74% / -1.92% | +0.522% / +0.526% | +670 |
| three forms, x 1.41 | 651 / 378 | -4.46% / -1.04% | +0.553% / +0.559% | +904 |
| one form, x 1.41, the card's beta in place of V | 651 / 378 | -4.99% / -1.90% | +0.372% / +0.373% | +628 |
| one form, x 1.41, the regression on v_i alone | 651 / 378 | -4.72% / -1.42% | +0.501% / +0.504% | +605 |

MSE x bill of the ten-network session: -1.95% [-3.33, -0.81].

**Parity with the torch module** (network 651, the S38 stack + NB one form x 1.41 against the physicist's stored arm; the
tool of section 18): per layer cosine 0.940-0.971, size ratio 0.968-1.011, left over 0.8-4.9e-6 rms; final-layer MSE
-4.46% natively against -4.51% in torch.

**What I got wrong and reported:** my bill estimate (0.5%) held; my request estimate (15-20 a layer) did not: the first
version made 55-58 logged calls a layer. That is what the lean form of section 23 fixes for the full law.

The lever stays in the builder, but nothing uses it: the pair law contains this channel (its carried weights are these
weights) and adds the births' own sources.

## 23. The pair law of the norm coupling (lever NP): the S40 file

**What it is** (the lead's `O/ncar6.py`, spec `O/NCAR6_SPEC.md`; the physicist's audit by reading and by numbers, 13:20 and
13:46 UTC). The matrix of section 22 with one more number per PAIR of columns:

    B = sum_r w_r pp_r pp_r' + sum_pairs x (pp+ pp-' + pp- pp+').

A pair is the two columns born at one unit (its hub) in one layer, `p+- = s_h + t+- e_h`. Three numbers per pair (w+, w-, x)
can hold any symmetric form on the plane the pair spans, so a per-unit law at the hub can be written into them exactly:

    w+ = 2 (S2 t-^2 - 2 rv t- + D) / gap^2,  w- = 2 (S2 t+^2 - 2 rv t+ + D) / gap^2,  x = 2 (- S2 t+ t- + rv (t+ + t-) - D) / gap^2

with S2, rv, D the law's three sources at the hub (the hub's own fourth-order response, its coupling to the other units through
the covariance and the pair map's K21 plane, and the diagonal). Carried columns keep section 22's rule; the cross weight is
carried as `x <- 2 gbar x`. The read's left factor becomes `pp_ir^2 lam_r - f_i (w_r pp_ir + x_r pp_i,partner(r))`.

**The parity version (fragment v1) on ten fresh networks** (list a, 13:14-13:29 UTC, one process, one thread; file sha
321c59b6; `a14_f4/fresh_compare.py s39_np_fresh_a --base s39_cand3_five_ss0_fresh_a`):

| | the pair law against S39 |
|---|---|
| final-layer MSE, pooled | **-7.50%** [95% -9.45, -5.49]; lower on 10 of 10, best -13.04% (651), worst -1.85% (641) |
| all-layer MSE, pooled | -6.95% [-7.73, -6.13] |
| bill | +1.168% on every network |
| MSE x bill | **-6.46%** [-8.45, -4.40] |
| logged calls | +3,568 per network |

Native against the trainer's torch card (one form removed, on candidate 3) on the six ranks both ran, native / torch:
287 -3.23 / -3.2, 328 -8.18 / -7.6, 378 -6.51 / -6.3, 424 -10.63 / -10.4, 641 -1.85 / -1.5, 651 -13.04 / -13.8. On the trainer's
50 held-out ranks the torch card reads -5.80% (se 0.50), so list a is a strong draw of ten and the board should be expected
nearer the torch number than my interval's middle.

Two native facts the reference does not show: (1) the birth hubs are not exactly the read's response rows, so natively B's
diagonal exists on the response rows and is 0 at the few hubs outside them; (2) the loop masks t+ and t- of a selected hub
that is not alive to 0, so the law must take the unmasked ones, otherwise gap = 0 (my first native run died of that at layer 11).

**The lean version (fragments v2, v4): the same arithmetic in a quarter of the calls.**

| what changed | effect |
|---|---|
| nothing is computed that no read uses: the fold is off at layer 15, so the update after layer 14 is dropped | one layer of column passes (the largest) and its calls |
| the fold takes the read's response rows itself: the rows of the pushed columns once, and their squares are the parent's left factor | the second row gather (a pass over all n rows of the columns, 7 requests) is gone |
| the per-unit vectors of the law (V, gam, S2, S3, S4, the reciprocal of gap^2 a v) are ONE polynomial evaluation: a stack of the loop's vectors, a gather by a constant index table, a product over the degree axis, one product with a constant coefficient matrix | 6 requests instead of about 60 |
| the newborn weights at the hubs are a second polynomial table on a stack of those vectors, the matrix-vector products and three scalars, gathered at the hubs once (7 x 47 monomials) | 6 requests instead of about 140 |
| the tables are the law's formulas expanded by exact polynomial arithmetic in plain Python when the file is imported: no array, 4 ms, outside every prediction | the first prediction of a process only turns them into arrays (17 requests) |
| every per-layer literal (fold strength, the table of multiples of C, the carry factors) sits inside the constant tables; the state is stored already scaled for the layer that reads it | a per-layer dial costs no call |
| the diagonal regression, its normalisation and the forms' vectors are one Gram matrix, one solve, one gather of the solution by a constant index table and one small product | three forms cost 4 more requests a layer, not 7; a constant regressor costs none |

Requests per layer as the platform counts them (logged calls + views; counted in the unit check `a38_pairlaw/unit_np3.py`,
where every view is counted): the fold 30 (28 with one form), the forms 4 (2), the update 38 with beta and the diagonal source
(37 with V, 32 without the diagonal source, 26 at layer 0); per network 14 folds and 14 updates: **about 995** (one form, V:
about 925). The parity version made 247 logged calls a layer. The release lane's census at the boundary is the number
that counts; mine is a count of calls in a unit check.

Unit checks (float64 transcription of `ncar6.py`, level 'read', with the native choices; hubs that are no response rows; two
fold strengths in one run): plane 3-9e-5 of the fold term, the stored diagonal 2-3e-7, the weights 3e-7 to 1.5e-6, partner
positions exact, in eight combinations of the switches.

**Native, networks 651 / 378, against the stored S39 rows** (`a38_pairlaw/arms_table.py`):

| file | final-layer MSE | bill | logged calls | final prediction against the lean one-form file, rms |
|---|---|---|---|---|
| parity version (v1), one form, V (sha 321c59b6) | -13.04% / -6.51% | +1.168% / +1.170% | +3,568 | 4.7e-6 / 4.7e-6 |
| lean (v2), one form, V (sha 05994d2e) | -13.60% / -6.40% | +0.976% / +0.984% | +885 / +871 | - |
| lean (v3), three forms, V (sha 4f7efa74) | -14.88% / -6.61% | +1.004% / +1.013% | +929 / +913 | 10.8e-6 / 7.3e-6 |
| lean (v3), three forms, beta (sha 7018f4bd) | -14.24% / -6.54% | +0.873% / +0.878% | +942 / +926 | 10.6e-6 / 7.6e-6 |
| **lean (v4), three forms, beta = the S40 file** (sha e2be4c75) | **-14.24% / -6.54%** | **+0.873% / +0.878%** | **+929 / +913** | byte-identical to the row above |

**The fired file on the ten fresh networks of list a** (14:53-15:03 UTC, one process, one thread, against the stored S39 rows;
`a14_f4/fresh_compare.py s40_npf3bv4_fresh_a --base s39_cand3_five_ss0_fresh_a`):

| | S40 against S39 | against the parity version (one form, V) on the same networks |
|---|---|---|
| final-layer MSE, pooled | **-7.93%** [95% -9.97, -5.95]; lower on 10 of 10, best -14.24% (651), worst -2.17% (641) | -0.47% |
| all-layer MSE, pooled | -7.32% [-7.95, -6.70] | |
| bill | **+0.876%** [+0.874, +0.879] | -0.29% |
| MSE x bill | **-7.16%** [-9.23, -5.14] | |
| logged calls | +913 (+929 on the first network of the process) | -2,655 |

Per network (651 378 424 328 287 641 548 557 316 390): -14.24 -6.54 -9.71 -8.82 -4.44 -2.17 -7.91 -10.37 -9.15 -7.13%.
The torch card reads -7.1% on six of these ranks where it reads -5.8% on the trainer's 50, so list a flatters the law by
about 1.3 points; with three forms the board should be expected near -6.3% in error and -5.5% adjusted.

Reading: (1) lean against parity is two rounding-level builds (they differ by 4-5e-6 rms; the MSE then moves by up to
0.6% on one network). (2) Three forms against one: the prediction moves by 0.4-0.9% of the MSE's power and the MSE falls by
1.49% / 0.22% (torch: -0.54%, se 0.21, on 25 ranks). (3) Beta against V: 5.1e-6 / 4.8e-6 rms, the rounding level, for 0.13%
less bill: free, as the physicist's torch run said. Two sessions were stopped by the lead's quiet window and have no rows:
beta alone, and the constant regressor in place of the table (builds sha 222afddc, d8ee172c).

**Where the +0.875% of bill is** (census of the fired file on network 651, every charge attributed to its call chain:
`census/census2.py`, `census/s40_651.json`, 15:07 UTC; the network's bill is 2.6239e11 flops):

| part | Mflop | share of the bill | what it is |
|---|---|---|---|
| the fold (`_np_left`), without the row gather | 1,453 | 0.554% | 10 passes over (response rows x columns), 137.6 Mflop each (the rows are 0.707 of the units): B's diagonal 3, the f-scaling 1 (both einsum, 550), the two scalings and the squares (multiply, 413), the exchange (put_along_axis, 138), the sum (138), the subtraction (138); the regression's products 70; the table gathers 6 |
| its row gather | 195 | 0.074% | replaces the parent's own row gather of the squares one for one: not an addition |
| the forms (`_np_forms`) | 103 | 0.039% | the multiple of C and the two mean forms on the n x n block, and their two additions |
| the update (`_np_update`) | 715 | 0.273% | matrix-vector products 541 (the convolution term over all columns 342, the law's n x n products 199), the two three-operand products 82, the table gathers 61 and products 12 |
| **the pair law, net of the replaced row gather** | **2,271** | **0.866%** | measured against S39 on this network: +0.873% |

**Where the pair law's kernel time is, and the faster form (fragment v5).** From the release lane's two worker runs (S39
and S40, 19 warm networks each, the server's counters by operation; `O/release/runs/d_s39`, `d_s40`): S40 costs +8.4 G
kernel instructions a network (x1.025).

| operation | calls, S39 -> S40 | kernel instructions, S40 - S39 | what it is in the pair law |
|---|---|---|---|
| put_along_axis | 50 -> 64 | **+5.18 G** | the exchange of every pair's two columns (one scatter along the columns per fold) |
| einsum | 490 -> 572 | +1.16 G | B's diagonal (three operands), the f-scaling, the forms, two n x n products |
| multiply | 4,221 -> 4,330 | +0.81 G | the two column scalings, the squares |
| matmul | 863 -> 1,000 | +0.31 G | the law's matrix-vector products, the convolution term |
| add | 11,840 -> 11,895 | +0.27 G | |

numpy's put_along_axis takes 6-7 ns per element here (an index pair per element); a block copy takes 1. Fragment
`a38_pairlaw/fragment_np_v5.py` = v4's arithmetic with (1) the exchange as `split` of the rows into the roster's blocks (free
views; the parent uses split at 12 sites) and one `concatenate` with every cohort's two blocks swapped (billed 1 per element,
as the scatter was); (2) B's diagonal as a product written into the exchange's buffer and a matrix-vector product with lam
(0.66 against 2.26 ns per element; 3 passes of bill either way); (3) the f-scaling in place. The roster's block sizes are plain
numbers, so the partner index array and its three requests per update are gone. Unit check PASS in the eight switch
combinations, with two cohorts of different sizes; requests per layer 71 (v4: 72). Expected: about 5 of the 8.4 G
instructions back and 13 requests fewer per network; rounding level against v4 (the diagonal's summation order). Files:
S40 + NP v5 sha 5bd2ec6f; S40 + NP v5 + K4L v5 sha cf1f735c.

**The K4L birth coefficients from two polynomial tables** (`a15_k4l/fragment_k4l_v5.py`, lever `K4Lv5c...`; same technique):
14 requests a layer instead of 31 (about -238 a network), +2 Mflop a layer (+0.011% of bill). Unit check
`a15_k4l/unit_k4l_v5.py`: all 4,816 float32 coefficients of four test layers bit-identical to fragment v4's. The same
rewrite does not pay for the F4c birth: its two guards and its power tables leave about 41 of its 46 requests.

**What could still come off, and what decides it:**
- the diagonal source off: 5 requests a layer and 0.206% of bill. MEASURED on 651 / 378 (sha baa71e02, three forms, beta): final-layer
  MSE -14.36% / -4.23% against S39 where the fired file has -14.24% / -6.54% (pooled -9.51% against -10.55%; the prediction
  moves by 24-30e-6 rms, a real change): on these two networks the source is worth more than its bill. The trainer's dial
  solve has the arm on 16 ranks;
- the constant regressor in place of the table (sha 8805546b): MEASURED on 651 / 378: -14.41% / -6.11% (pooled -10.44% against
  -10.55%), same bill, same calls: equal within the noise, as on the trainer's 25 ranks (+0.16%, se 0.16);
- the fold on the young cohorts only: its passes scale with the columns kept; a physics change nobody has measured;
- the regression on v_i alone (identical in the bank-only fold, physicist 13:20 UTC): 5 requests a layer;
- the release lane's un-fusing of the trees above 64M floats: 535 requests for 1.8 s of wall, no physics.

**At the client-server boundary** (the lead's qualification chain, stage C, networks 35 / 100, 14:24-14:34 UTC; not run by
me): PASS. **43,628 requests per network against S39's 42,624: +1,004** (my count from the unit check: +995); kernel
instructions x1.026, bill x1.00876; projected platform wall 79.2 s mean / 89.8 s on the worst network (S39 measured 77.3 /
87.6), residual 0.309 s mean / 0.355 s worst (S39 0.302 / 0.347; the limit is 0.4). What was new there and passed: the
operation `prod` (three sites), a float64 vector added into a float32 view with `out=`, and scalars (a mean, a sum) going
into `stack` and `multiply` as numbers (the client holds them as plain values).

## 24. The fourth-chaos plane of the young pairs' local quartic (K31L4), in the same read: native parity; -2.3% on two networks for 0.29% of bill

**What it is** (the lead's `O/k31l.py` with `chaos4`, level 'read'; addendum of `O/NCAR6_SPEC.md`; 15:09 UTC). K4L uses the local
quartic of a young pair, `Q = 4 S1 Y^3 E + 6 S2 Y^2 E^2 + 4 S3 Y E^3 + S4 E^4` on its two pushed vectors, on the diagonal only
(the per-unit fourth cumulant). The same form gives a (3,1) plane, `K31L_ij = sum_h [AY_ih Y_jh + AE_ih E_jh]` with
`(AY, AE) = (1/4) grad Q`; its second-chaos part is already in the pair law, so only the rest is added: two of the three
i-slots replaced by the pair's means, subtracted.

**The native form needs nothing new at the births.** On the pair's pushed columns p (+), q (-) and D = p - q, K4L's stored
coefficients are exactly `Q = e1 p^3 D + e2 p^2 D^2 + e3 p D^3 + e4 D^4`, and (checked to 1e-15 in plain numpy,
`a39_k31l/identity_check.py`, `roster_form_check.py`)

    K31L_ij           = (1/4) sum_h [dQ/dp_i p_j + dQ/dD_i D_j]
    its second chaos  = (1/8) sum_h [dH/dp_i p_j + dH/dD_i D_j],   H = m_pp Q_pp + 2 m_pD Q_pD + m_DD Q_DD

with m the pair's means of p^2, p D, D^2 over the units. So every column r of the two youngest cohorts gets, in the left factor
of the read, `(k31 / 3)(1 / lam_r)` times a cubic + linear polynomial in its own entry o and the difference Delta to its
partner, with six coefficients per column from one more polynomial table (inputs: whether the column is the plus one, 1 / lam,
e1..e4, the three means). It rides on the pair law's own row factor: the addition goes into the same array the fold scales by
mu_i / v_i and subtracts, after B's diagonal has been taken.

**Fragment** `a38_pairlaw/fragment_np_v6.py` (the six monomials as slabs of one array, summed by an einsum) and `_v7.py` (the
same polynomial in Horner form, the table in float32); lever `NP..k<strength>[m]v7`, `m` = the means over the response rows
instead of all units. Unit check `a38_pairlaw/unit_np6.py [fragment]`: the plane against `k31l.py`'s formulas to 1-3e-5 of
itself, with one and with two young cohorts, both kinds of means; the pair law's own checks unchanged.

**Native, networks 651 / 378** (one process, one thread; base = S40 + pair law v5 + K4L v5, sha cf1f735c):

| file | final-layer MSE against the base | bill | logged calls | torch (lead, e62) |
|---|---|---|---|---|
| + K31L4, means over all units, v6 (sha 1017aab8) | -2.15% / -2.58% | +0.388% / +0.391% | +479 / +475 | -2.09% / -2.65% |
| + K31L4, means over the response rows, v6 (sha 2bb3763e) | -1.75% / -2.13% | +0.359% / +0.360% | +395 / +391 | |
| **+ K31L4, means over all units, v7** (sha 8d110449) | **-2.33% / -2.36%** | **+0.292% / +0.294%** | +479 / +475 | |

v7 against v6: 4.3e-6 / 4.6e-6 rms at the final layer (rounding level). The means over all units are worth about 0.4 points
for 0.03% of bill and 84 calls: kept. The trainer's torch on the first 25 judging ranks: -1.87% (se 0.39) on top of the law.

**Why v6 cost a third more than v7.** flopscope bills an einsum (operands - 1) per point of its index space plus one addition
per point summed away: the sum of six monomial slabs with their coefficients is 11 passes, and the slabs took 9 more to build.
Horner form, `o (o (c3 o + c2 Delta) + c1 Delta^2 + c_o) + Delta (c0 Delta^2 + c_d)`, is 14 passes in all. The rest of the term's
bill: the pair means over all units 0.07% (a column sum of the squares the loop has, one product of the young columns with
their partners), the coefficient table 0.03%.

**The S41 stack so far, against the fired S40 file** (651 / 378): MSE -2.12% / -2.36%, bill +0.304% / +0.305%, logged calls
+253 / +250 (the K4L birth by tables takes 225 back).

## 25. The S41 file: the trainer's dials as literals, one command from his hand-over, ten fresh networks; and the strips' leaf views

**S41 as the lead fired it (16:27 UTC decision)** = the stack of section 24 on the trainer's card `law_dials_d6` with his first dial
values and the table (my file sha b99d7562) + lever HV (sha 33bac9db) + the release lane's two fragments Rlvfs_limit64 and
Rl3_bind128 (their file sha 169fd371). The arithmetic is b99d7562's.

**Dials.** Fragment `a38_pairlaw/fragment_np_v8.py` (= v7 with the young pairs' plane's strength by cohort age and the factor on
the multiple of C per layer; unit check `unit_np8.py` with strengths (0, 1.3, 0.6) and (0.8, 0, 0.5, 1) by age). Every literal
of the fragment is set by name from `a38_pairlaw/dials/<name>.json` (lever `NPv8@<name>`): STRENGTH (16 values), RHO, DSCALE,
BIRTHS, ROWS, DIAG, CA, FORMS, DFIT, BETA, K3L. `a38_pairlaw/dials_from_trainer.py <physics_dials.json> <name> [2 | 3]` writes
that file from the trainer's hand-over (entry `pair_law_dials`), and `stack/law_file_from_trainer.sh <directory under
O/trainer/candidates> <2 | 3> [plain]` does the whole path: copy of the card and the json, the dials file, the build on the S40
lever stack with the K4L birth by tables and HV (`plain` = without HV), the einsum gate. Rehearsed on `law_dials_d6`: the measured
file byte for byte.

**The first dial values** (trainer 15:32 UTC, `O/trainer/candidates/law_dials_d6/`: card sha 89191265 with c0 0.6222 and the k31
knob -0.1659; fold strength 1.829 at layers 1-4, 1.395 at 5-9, 1.181 at 10-14; factor on the multiple of C 1.2716; births
1.142, rows 0.697, diag 0.791; K31L4 at ages 1-2, strength 1; his torch verdict for the dials alone: -1.48%, se 0.28, on 25
ranks). Native, networks 651 / 378, final-layer MSE:

| file | against the same stack at default dials (table, K31L4) | against the fired S40 file | bill against S40 |
|---|---|---|---|
| dials, the table (sha b99d7562) | -2.51% / +0.20% (pooled -1.16%) | -4.58% / -2.17% | +0.303% / +0.305% |
| dials, the constant regressor (sha c1cf077d) | -0.06% / +0.42% (pooled +0.18%) | -2.18% / -1.95% | +0.304% / +0.306% |

The factor 1.2716 was solved on the table, whose multiples fall from 1.04 at layer 1 to 0.555 at layer 14; on the constant
regressor's coefficient (about 1 at every layer) it removes 1.27 times the multiple of C everywhere. At default dials the two
forms were equal on these networks (section 23). So these dials go with the table.

**Ten fresh networks** (list a, 15:40-15:50 UTC, the table file against the stored S40 rows of the same networks;
`a14_f4/fresh_compare.py s41p_d6_r2_fresh_a --base s40_npf3bv4_fresh_a`):

| | the S41 arithmetic (sha b99d7562) against the fired S40 file |
|---|---|
| final-layer MSE, pooled | **-3.26%** [95% -4.97, -1.66]; lower on 9 of 10, best -7.94% (390), worst +1.31% (328) |
| all-layer MSE, pooled | -2.80% [-3.79, -1.87] |
| bill | **+0.304%** [+0.303, +0.305] |
| MSE x bill | **-2.97%** [-4.66, -1.36] |
| logged calls | +250 |

Per network (651 378 424 328 287 641 548 557 316 390): -4.58 -2.17 -5.07 +1.31 -5.81 -1.79 -2.65 -1.15 -1.54 -7.94%. The two
torch parts add to the same: K31L4 -1.87% and the dials -1.48%, each on 25 ranks.

**The strips' leaf views (lever HV, `hzs/fragment_ks_v7.py`).** The half-zero strip kernels cache, per product shape, views of
the operand and output trees for each group of leaf chunks. The fired S40 form builds each view with up to three chained index
calls (chunks, then rows, then inner or column range) and builds the operand views twice, once for the output tree and once
for the leaf-sum buffer. One index expression per view and shared operand views give the same views: `hzs/unit_views_v7.py`
checks shape, contents, memory offset and strides in all eight strip combinations, 346 index calls against 156. No
arithmetic, no bill. Native, 651 then 378 in one process (`quick.py --views` counts every index call by function; sha 33bac9db
against b99d7562): predictions byte-identical, bill identical, logged operations identical, index calls of that function
1,216 -> 636 on the first network (-580) and 784 -> 422 on the second (-362). The release lane's boundary count of the same
two files: -570 on the first network after a setup, -380 on a warm one.

## 26. Requests after S41: the constants at setup, a diet that keeps every bit, and a deeper one that does not

**Why.** The platform stops a network whose own Python time exceeds 0.4 s, and that time is the number of requests (about 7.2
microseconds each). The release lane's reading of S40's graded run (16:02 UTC): a fresh 100-network run loses a network in
about 3 runs of 100, always the FIRST network a worker runs after its setup, which makes about 1,700 requests more than the
others: 579 first-load views of the kernel's operand loads, 432 leaf views of the strips, 482 constants and buffers that exist
once per process, 220 for tree keys the prebind list does not hold. Their caps for the next file: first network at most 43,800
requests, a warm one at most 42,700 (S40: 44,578 / 42,678; the S41 stack before any diet: 44,881 / 42,982).

**The constants at setup (lever CS, `a40_setup/fragment_cs_v1.py`; the release lane's request).** A wrapper of the binding
phase makes the kernel arena with the binding phase's own three lines and then calls, with the arguments a prediction uses,
the functions that build what exists once per process: the decoded literals of the learned parts, the learned correction's
buffers and its parameter views for layers 1-15, the feature columns, the birth head's rows for layers 1-14, the edge's
planes, the pair map's persistent buffers and views, the birth update's stack, the weight-power stack (filled once from a zero
matrix, then marked empty), and the tables of the pair law, of the exact-term mixture and of K4L. Then the parent's binding
phase runs and finds the arena. Any failure clears those caches and the binding goes on (they then bind inside predict as
before); the reset path clears them too.

| check | result |
|---|---|
| setup alone (`a11_bare_views/setup_only.py`) | the log gains ('constants', 0.031 s); +269 logged operations and +153 Mflop in the setup |
| 651 then 378 in one process, against the same file without CS (`quick.py --views`) | prediction sha equal on both; first network 43,221 requests by my count (logged operations + index calls) against 43,673: **-452**; its bill -0.057% (the buffers' reshapes are billed in the setup); second network identical |
| reset path (`a40_setup/reset_check.py <file> <parent>`) | after setup and `_pb_reset((None, None))` no global holds an array that the parent file does not hold too (both keep one in `_VIEW_SPECS`: the parent's own) |
| not checked here | the boundary run, a setup with a real loss of handles |

File sha c0ae245f = the S41 source (33bac9db) + CS.

**A diet that keeps every bit (in `law_file_from_trainer.sh ... same`).** Four fragments rewritten so that the arithmetic and
its order are untouched and only views, gathers and index arithmetic change:

| fragment | what changes | requests | bill | checked |
|---|---|---|---|---|
| F4c v3 (`a27_f4c/fragment_f4c_v3.py`) | the cohorts' column blocks of r, r^2 and the shift block by three splits instead of five views per cohort; the row factor by a two-operand einsum; the two scattered vectors written in float32 and widened by the scatter | 74 -> 64 a step | same | `unit_f4c_v2.py fragment_f4c_v3.py`: shift, beta, diagonal bit-identical to v1 |
| K4L v7 (`a15_k4l/fragment_k4l_v7.py`) | the blocks of the pushed columns, their squares and cubes by one split per array | 33 -> 28 a layer | same | `unit_k4l_v6.py fragment_k4l_v7.py`: row sums bit-identical to v5 |
| pair law v9 (`a38_pairlaw/fragment_np_v9.py`) | the young plane's table rows (one, 1 / lam) from one power with two exponents; the forms' vectors by one split; K4L's coefficients as one array | -4 a layer | +0.0002% | alone on the S41 source: predictions byte-identical, -42 requests |
| edge cut v2 (`a19_edge_cut/fragment_ecut_v2.py`) | the three kept blocks of R, S and S' gathered by ONE index each (rows as a column, columns as a row), the transposed block straight from its gather, channel products as two-operand einsums | 41 -> 34 a layer | -9 Mflop a layer | `unit_ecut_v2.py`: output bit-identical to v1 |

Native, 651 / 378, file sha d610157e against the S41 source with CS: prediction sha equal on both networks, **requests -323 /
-323** (logged +22, index calls -345: a split is a logged call), **bill -0.0233% / -0.0235%**.

**The deeper diet moves the rounding and costs bill (`... deep`).** F4c v2 writes every shifted column with its own entry and
the difference to its partner (one block of eight operations for all cohorts, 74 -> 43 requests a step, 24 flops per element
against 21.5); K4L v6 puts the plus columns of both cohorts side by side (33 -> 18, 16 flops per element against 12). Both
change the order of float32 operations.

| | F4c v2 + K4L v6 + pair law v9 + CS (sha 299e858c) against the same arithmetic before (sha b99d7562), ten fresh networks |
|---|---|
| final-layer MSE, pooled | +0.26% [95% -0.16, +0.60]; higher on 8 of 10 (-0.61 -1.26 +0.49 +0.89 +0.87 +0.18 +0.74 +0.40 +0.57 +0.14%) |
| bill | +0.073% |
| MSE x bill | +0.33% [-0.09, +0.67] |
| requests | -670 a network (my count on 651 / 378); logged calls -381 |
| final prediction moved | 4.1-5.6e-6 rms on every network |

Each piece alone on 651 / 378 moves the final prediction by 4.1-4.6e-6 rms: whatever changes the rounding gives the same
4-5e-6, the estimator's rounding floor. No evidence of harm and none of gain; 0.07% of bill for certain. Not a default.

**The files in one table** (requests = logged operations + index calls of 651 as first network of a process / 378 as second;
the boundary counts more, since it also sees transposes and reshapes):

| file (first dials) | sha | requests 651 first / 378 second | bill against the S41 source | predictions |
|---|---|---|---|---|
| the S41 source | 33bac9db | 43,673 / 42,431 | | |
| + CS | c0ae245f | 43,221 / 42,431 | first network -0.057% | byte-identical |
| + CS + F4c v3 + K4L v7 + pair law v9 | 246fda2c | 42,961 / 42,171 | +0.0002% | byte-identical |
| + edge cut v2 as well | d610157e | 42,898 / 42,108 | -0.023% | byte-identical |
| + CS + F4c v2 + K4L v6 + pair law v9 | 299e858c | 42,551 / 41,761 | +0.078% | rounding level |
| **everything that is free: HV2 in place of HV, CS, F4c v3, K4L v7, pair law v9, edge cut v2** (`law_file_from_trainer.sh law_dials_d6 2 same`) | d464c12c | **42,752 / 41,986** | -0.023% | byte-identical |

**The strips' leaf views per tree (lever HV2, `hzs/fragment_ks_v8.py`).** The strip kernels kept one set of leaf views per
product (M, q, N). A product now takes its views from four per-tree entries: the left tree (M, q), the right tree (q, N), the
output tree and the leaf-sum buffer (M, N); every other product with the same tree shares them. `hzs/unit_views_v8.py`: the
same views in all eight strip combinations; a product that shares two of its three trees makes 4-5 index requests instead
of 20-28. A ten-network session's strip keys (`quick.py --hzkeys`; 26-50 strip products a network, 119 distinct product keys,
107 left trees, 95 right trees, 38 output trees) replayed by `hzs/sim_side_views.py`: 285 -> 135 index requests a network on
average, first network 508 -> 370. Native (651 then 378): predictions byte-identical, bill equal, 146 / 122 index requests
fewer. Binding the 60 most used trees at setup would take another 140 off a first network for 326 requests in the setup:
not built (the binding phase has about 700 requests of room and the release lane's fragment uses 430 of them).

## 27. The rounding draw of a native run, and the second dial set read with it

**Two runs of the same estimator.** The deep-diet file (sha 299e858c) is S41's arithmetic with another order of float32
operations, so its ten fresh networks against the stored S41 rows (sha b99d7562) are two rounding draws of one estimator:

| network | 651 | 378 | 424 | 328 | 287 | 641 | 548 | 557 | 316 | 390 |
|---|---|---|---|---|---|---|---|---|---|---|
| final-layer MSE, draw 2 against draw 1 | -0.61% | -1.26% | +0.49% | +0.89% | +0.87% | +0.18% | +0.74% | +0.40% | +0.57% | +0.14% |

Standard deviation 0.69% between two runs: **0.49% of a network's final-layer MSE per run** (the final prediction moves 4-5.6e-6
rms; the MSE moves through the cross term with the error). Consequences:

- Ten networks against ONE stored run resolve 0.3% at best (standard error 0.22% from rounding alone).
- Two candidates compared against the same stored run share its draw: the deep diet's and the second dials' per-network
  changes against the stored S41 rows correlate at 0.79 although the changes have nothing in common.
- 651 and 378, my pair for quick checks, drew high in the stored run: against it every candidate reads about 0.9 points too
  good on them. The two-network numbers of 2 October against the S41 rows are to be read with that.
- Rule from here on: an effect below 1% is judged on ten networks against at least two draws of the base. The board is not
  affected (100 networks average the draw to 0.05%).

**The second dial set** (trainer, `law_dials_d7_ages12`: fold strength 1.850 / 1.207 / 1.167, factor 1.307, births 1.305, rows
0.854, diag 0.905, the young plane at ages 1-2 as in S41; solved and judged in torch on the law with the constant regressor:
-0.84%, se 0.19, on 25 ranks against the first dials on the same form). Native, ten fresh networks of list a, final-layer MSE:

| file | against draw 1 of S41 | against draw 2 | against the mean of both | lower on | standard error |
|---|---|---|---|---|---|
| second dials, table form (sha faa4bdd7) | -0.05% | -0.31% | **-0.18%** | 6 of 10 | 0.20 |
| second dials, constant regressor (sha e0598c4e) | +0.27% | +0.01% | **+0.14%** | 5 of 10 | 0.30 |
| first dials, constant regressor (sha c1cf077d) | +1.44% | +1.17% | **+1.31%** | 0 of 10 | 0.24 |

Second against first dials, both on the constant regressor: **-1.15% (se 0.27), lower on 9 of 10**: the trainer's comparison,
natively; torch said -0.84% (se 0.19). Native and torch agree. But the first dials on the constant regressor are 1.31% worse
than the first dials on the table, which is S41 (worse on all ten: +2.82 +0.85 +0.57 +0.94 +2.09 +0.54 +0.95 +1.49 +0.77
+2.05%). So the second solve ends where S41 already is: the table is the better form for the first dials (they were solved
on it), the two forms are level for the second dials (constant regressor against table +0.32%, se 0.32), and a file on the
second dials would not move the board in either form. A verdict "against the S41 stack" has to use first dials + table as
its base. Both second-dial files carry CS and the byte-identical diet: bill -0.03% against the S41 source. The young plane at ages 1-3
(`law_dials_d7`, torch -0.19% more) would add about 0.15% of bill natively: not built into a session.

## 28. Native switches for the refit's physics base (written ahead of the base's choice; four of them, each off unless its lever is named)

The lead's plan of 18:01 UTC: the next file is the refit file, on a base of terms that are physically right and that the
present learned parts reject; the physicist's table (`O/physicist/t4_ora52.log`: each term with the learned correction off
and on, six held-out ranks) names the base. While it ran I wrote the switches for its cheap candidates. All of them are
elementwise on per-unit vectors or on planes the loop holds; with the lever absent the file is byte for byte the one before.

| lever | term (reference) | native form | requests a layer | bill | unit check | native against the torch arm, network 651, learned parts on |
|---|---|---|---|---|---|---|
| `SO2m<c>[r<r>]` (`a41_base/fragment_so2_v1.py`) | second-order mean term inside the pair map and the consistent per-unit variance (`physicist/so2.py`, mean / raw2) | a wrapper of the pair map: one polynomial table on nine per-unit rows, the covariance's diagonal and `post_v` rewritten, the cached denominator invalidated | 10 | 0.005% | `unit_so2.py`: 1e-15 against a transcription | cosine 1.000 at layer 1, 0.96-0.98 at layers 2-14, 0.94 at 15; size ratio 0.96-1.03; final-layer MSE +1.01% native, +1.19% torch |
| `BSK<x>` (`fragment_bsk_v1.py`) | birth weight with the skew-corrected kink density | one loop line after the card's own | 6 | 0 | `unit_bsk.py`: 2e-16 | cosine 0.97, size ratio 0.99 at layer 2 (the first it reaches); later layers 0.4-0.8 (the change is the size of a rounding draw); final-layer MSE -0.72% native, -1.01% torch |
| `GSK<s>[k<k>][b<b>]` (`fragment_gsk_v1.py`) | transport gate with the mean gate's skew / kurtosis corrections and the b-mode feed | six loop edits: the carried vector b, the inherited plane and its diagonal, the gated young columns, the old tier's basis, and the exact-term mixture's anchor take the corrected gate | 5 | 0 | `unit_gsk.py`: 2e-16 | final-layer MSE native / torch: skew +0.02% / -0.38%, skew + kurtosis -0.51% / -0.40%, b-mode feed -0.74% / -0.67% (a build before the mixture's anchor took the gate; the corrected build has NOT run) |
| `BK3r<x1>,<x2>` (`fragment_bk3_v1.py`) | collective mode times the specific third cumulant in the (3,1) plane (`physicist/bk3.py`), level 'r31' (the form the physicist's table measured) | the pair map's extra plane through the loop's own hook for it (`_T31_LAYERS` / `_t31_extra`) | 11 | 0.03% | `unit_bk3.py`: 1e-7 of the plane (float32) | NOT run. Level 'pair' (`BK3p`) is written and unit-checked as a fold of the K21 plane, but the builder refuses it: the fold would change the plane for the features, the edge, the covariance knob and the birth update too, and the reference's hook changes it for the pair map's call only (`covstep.py`: the step passes its own k21 to those consumers). It needs a second plane for the pair map, about the same 0.03% |
| `BT@<name>` | the levers listed in `a41_base/bases/<name>.json` (keeps file names short) | | | | probe build with all four switches compiles (`probe4r`, sha e14ff18e) | NOT run |

The parity tool: `a41_base/parity_ora52.py <native session> <native base session> <torch arm> [rank]` (per layer the cosine and
the size ratio of the change the term makes to the 16 x 1,024 means, native against the stored torch predictions).

**(Superseded by section 29 for the target's part: it is written, in a form that needs half of what this paragraph says.)**
**Not written:** the second-order covariance list and the target's second-order part (`so2.py` cov / tgt). As the reference
writes them they are ten plane-times-outer-product terms and six plane products a layer for the covariance (about 420
requests a network, 0.2% of bill) and the same again twice for the target, which also needs the covariance list: about 1,100
requests and 0.6% of bill together. That is more requests than the whole diet of section 26 freed; if the target part is in
the base it needs a leaner form (no rho term, or response rows only) and the refit has to train on that form.

**A breach, for the record.** The five single-network sessions behind the last column, and a sixth stopped after 30 s, ran at
18:01-18:06 UTC, after the lead's order of 18:01 that nothing native runs until the base is named: I had launched them without
reading the team log first. One process, one thread, about 3 GB, 45 s each. Nothing native has run since.

## 29. The target's second-order part and the final layer's mean term (written and unit-checked; not run natively)

The lead's reading of the physicist's table (18:30 UTC) kept the target's second-order part "only if it survives in a lean
native form". It does, at half of my first price. The physicist's reference (`so2.py`, now the keywords `so2t so2k so2p so2td`
of `physicist/pbase_frozen_s4a.py`) makes two lists of ten plane terms each, one for the covariance and one for E f_i^2 f_j,
and combines them into the change of the target cum(f_i, f_i, f_j). The target needs only the combination, and the two lists
differ only in their left rows. So it is ONE list:

    U_r = v B_r - 2 m1 sigma A_r   (left rows),      W_q = sigma A_q   (right rows),      A_r = the pair map's derivative rows
    dt = T o [(M_g0 + rho o M_g1) + T o (M_gq0 + rho o M_gq1)]  +  T' o [(M_h0 + rho o M_h1) + T' o (M_hq0 + rho o M_hq1)]
         + (T o T') o (M_c0 + rho o M_c1)  +  rho o M_de  -  2 m2_i cov_ij

T = K21 / (v_i sigma_j), its transposed view and the correlation rho are float32 planes the pair map already holds; every M is
an outer-product sum of rank 1 to 3 of per-unit rows. The rows come from two small float32 tables of monomials on the pair
map's own derivative rows (`st['FT']`) and its t3 (`st['t3']`): a gather, a product over the degree axis, a sum over two slots,
with the coefficients as constant base rows. Only the float32 copy of the target and the diagonal of the float64 one are
written: with `_FULL_LOWER_MOMENTS` on, the live loop reads nothing else (the fragment asserts it).

Measured on one layer's call at the estimator's width (`a41_base/unit_so2_v2.py 1024`, random inputs of the sizes network 651
has at layer 13; requests = logged operations + index calls; bill against 2.573e11 flops a network; 14 layers):

| form (physicist's keywords) | lever | plane passes a layer | bill | requests a network |
|---|---|---|---|---|
| full: parts a b q c d e, first power of rho (`so2t=1`) | `SO2m0t1` | 52.1 | 0.297% | 658 |
| without the rho terms (`so2k=0`) | `SO2m0t1k0` | 25.4 | 0.145% | 434 |
| without the products of two K21 entries (`so2p=1256`) | `SO2m0t1p1256` | 32.7 | 0.186% | 434 |
| the K21 terms only, no rho (`so2k=0,so2p=12`) | `SO2m0t1k0p12` | 15.1 | 0.086% | 308 |
| the diagonal alone (`so2td=1`) | `SO2m0t1d` | 0.8 | 0.005% | 196 |
| with the mean term in the same call (`so2m=1`) | `SO2m1t1...` | +0.2 | | +70 (the mean term alone: 140) |
| final layer's mean term (`so2f=1`) | `...f1` | 0 | 0 | 7 |

Unit check (`unit_so2_v2.py`, PASS): against a numpy transcription of the reference's `_A`, `single`, `pair_list` and its
wrapper, for twelve settings of strength, highest power of rho and parts, the off-diagonal change of the target agrees to
1e-6 of the change (the planes are float32), its diagonal to 3e-7, the mean term in the same call to 1e-10; the final term
to 5e-16.

What it would have to earn: on the six ranks of the first table the target's part alone is -1.03% (se 0.49) with the learned
parts off and -0.02% with them on; on the S41 card's first network -1.88% off. At 0.30% of bill the full form pays only if a
refit keeps most of the learned-off gain; the physicist's running table (`t4_ora60.log`) has the three cheaper forms as arms.

`a41_base/base_from_request.py <name> '<keywords>'` turns the physicist's keyword string into the builder's lever set (and
refuses what has no native form: the covariance list, the (2,2)-plane cross form, the b-form size and K31 row level, which
are dials of the card). Built as text for the arms of the running table (einsum gate PASS, none run): the lead's 18:30
proposal m + xh + gk + gb + bsk `b1830` sha d0310bae; m + t + f + bsk `p3` 635fba08, with the diagonal alone `p3d` f6ada1e4,
without rho terms `p3k` 4d58b0d6; all seven terms `p6` 50ffd10b (1,039 requests a network, +0.33% of bill).

## 30. The head column: the S42 candidate (unit check, parity with the trainer's torch column, ten fresh networks)

The trainer's zero-weight test (18:41 UTC) found that the learned mean correction wants one new per-unit input: the part of
the pair law's matrix diagonal that its own regression does not explain. With its 128 first-layer weights solved by one
line-search step (hand-over `O/trainer/candidates/head_col_s1/`), torch gives -1.79% (se 0.29) on 25 judging ranks against
the same card without the column. The lead defined S42 = S41 + this column (19:13 UTC).

**Definition** (the trainer's, 18:47 UTC), layers 1-14: on the response rows of the layer's bank read,
x_i = (B_ii - fit_i) / (tr C v_i), where B_ii is the law matrix's diagonal and fit is its least-squares fit over those rows on
(tr C v_i, 1, mu_i^2, 2 mu_i (C mu)_i); z_i = clip((x_i - mean) / sd, -8, 8) over the rows; 0 for every other unit. The first
layer of the correction gets enc0_b + z_i u' in place of enc0_b.

**Native form** (`a42_head/fragment_hc_v1.py`, lever `HC@<weights>`): the fold already holds kap B_ii on the response rows
and the masked regressor rows of its own diagonal regression. The residual is one 5 x 5 solve: with G the Gram matrix of
(regressors, target) on the rows, G^-1 e_target is (- coefficients, 1) divided by the residual sum of squares, so its product
with the rows is a positive multiple of the residual, and every positive per-layer factor drops out in the standardisation.
S41's arithmetic uses the table regression (three regressors, no constant), so the constant is added as a fifth row there.
Two lines in the fold, one in `gnn_step`, no setup state. **21 requests a layer, 5.8 Mflop a network (+0.002% of bill).**

| check | result |
|---|---|
| unit (`a42_head/unit_hc.py`): column against a numpy transcription of the trainer's column 6, both layouts of the regression rows, with units at the clip | equal to the last float32 bit; the bias term too; a column is used once and never by another layer |
| parity on S41's arithmetic, column on minus off against torch on minus off, networks 651 / 378 (`a42_head/parity_hc.py`) | correlation per layer 0.995 / 0.995 at layer 1 (rms 1.25e-6 / 1.37e-6 on both sides), 0.76 / 0.82 at layer 2 (the change itself is 1e-6 there), 0.86-0.96 at layers 3-14, 0.914 / 0.955 at the final layer; size ratio 1.02-1.06 from layer 4 on |
| parity on the weights' own card (second dials, constant regressor) | 1.000 at layer 1 on both; 0.94-0.97 at layers 3-15. The shortfall is the native rounding draw: final layer native 15.5e-6 rms against torch 14.8e-6, difference of squares (4.7e-6)^2 |
| setup and reset path (`a40_setup/reset_check.py` against the same file without the column) | same 41,370 arrays after setup, nothing left after the reset that the parent does not leave: PASS |
| **ten fresh networks** (session `hc_s1_d6_r2_fresh_a`, list a in the order of the stored rows; `a42_head/judge_ten.py`) | **final-layer MSE 14.1233e-9 against 14.3459e-9 (mean of the two stored S41 draws) = -1.55% pooled; against each draw -1.42% / -1.68%; mean of the per-network changes -1.52%, se 0.38; lower on 8 of 10 (the other two +0.06%); bill -0.027%** |
| requests | +297 / +294 on a first / later network against the same file without the column; against the fired S41 arithmetic about -627 / -151 (the free diet of section 26 is in the file) |
| **ten fresh networks, the trainer's step-2 weights** (`a42_head/weights/s2.json` from `O/trainer/cert/head/v2r_weights.json` row 6, length 0.3394 against 0.2484; file sha 0fccf396; session `hc_s2_d6_r2_fresh_a`) | **final-layer MSE 14.0376e-9 against 14.3459e-9 = -2.15% pooled; against each draw -2.02% / -2.28%; mean of the per-network changes -2.09%, se 0.28; lower on 10 of 10; bill -0.027%; requests as the step-1 file** |
| step 2 against step 1 on the same ten networks | -0.61% pooled (mean of the per-network changes -0.56%, se 0.35, lower on 7 of 10); torch -0.36% (se 0.15) |

Per network against the mean of the two draws, in run order 651 378 424 328 287 641 548 557 316 390:
step-1 weights -1.45, +0.06, -1.52, -0.46, -2.57, +0.06, -1.67, -2.19, -1.75, -3.76 %;
step-2 weights -1.80, -0.98, -1.55, -1.12, -2.02, -3.30, -2.52, -2.01, -1.84, -3.75 %.

The lead's fire rule (pooled at least -1.3%, at least 8 of 10 lower, bill within +0.1%, requests inside the release caps) is
met by the step-1 file with a margin of a quarter of a point (less than one standard error) and by the step-2 file with
0.85 points and every network. This list read 0.3 points better than the board for S41 and 2.3 points better for S40.

New weights from the trainer's next steps become a file with two commands:
`python a42_head/weights_from_trainer.py <hand-over directory> <name>` and
`bash stack/law_file_from_trainer.sh law_dials_d6 2 same - <name>`.

**Other head inputs the lead listed (prices, no session;** `a42_head/HEAD_INPUT_C_DEFINITION.md`**):** the two parts of the
diagonal separately cost one more matrix-vector product over (rows x columns), 0.11% of bill; the young pairs' plane diagonal
0.02%; the law's per-unit sources pushed through the weights 0.02%; the "row sizes of the left factors" that I had called free
at 18:13 UTC cost 0.16% + 0.02% and measure the row of the whole matrix, not of its specific part: withdrawn as a suggestion.

## 31. After S42: the skew-corrected birth on ten networks, a second draw of S42, and several head columns in one fragment

**The skew-corrected birth (lever BSK1) on the fired S42 arithmetic** (sha 6c4840f2; session `s42_bsk1_fresh_a`). The lead's
condition for S43 was "if the engineer's ten networks confirm it". Judged against the mean of two rounding draws of S42 (the
stored run of the fired arithmetic and `s42_deepdraw_fresh_a`, the same arithmetic in the deeper diet's float32 order):

| | value |
|---|---|
| pooled final-layer MSE | 13.9967e-9 against 14.0443e-9 = **-0.34%** (against each draw -0.29% / -0.39%) |
| mean of the ten per-network changes | -0.36%, se 0.32; lower on 8 of 10 |
| per network, run order 651 378 424 328 287 641 548 557 316 390 | -0.66 -0.04 -0.42 **+2.05** -1.71 -0.23 -0.60 -0.95 -1.14 +0.11 % |
| the physicist's torch arm (learned parts on, old base), networks 651 378 424 328 287 | -1.01 +0.05 -0.83 +1.10 -0.51 %; six-rank mean -0.34% |
| price | +105 requests a network (7 a layer, layers 0-14), bill +0.002% |

Native and torch agree network by network, network 328 included. The term is real and small.

**The second draw of S42** (sha 90ab448a = S42's arithmetic with F4c v2 and K4L v6 instead of v3 / v7): against the fired
arithmetic's run **+0.10% pooled (mean of the per-network changes +0.09%, se 0.14, lower on 4 of 10), bill +0.078%, 410
requests fewer on every network** (42,639 on the first network of the session, 40,830-42,943 on the others by my count).
Together with section 26's reading of the same diet on S41 (+0.26% [-0.16, +0.60]) this says the deeper diet costs its bill
and nothing measurable in error. The lead chose it as the base of S43 (20:07 UTC) to make room under the request caps.

**Several head columns in one fragment** (`a42_head/fragment_hc_v2.py`, lever `HC@<weights file with "columns">`). Every
column leaves its per-unit vector where it is made; one standardisation runs on the stacked rows (each column over its own
units), one product Z' U goes on the first layer's bias; a column a layer does not make counts as 0 there. Written for the
trainer's columns 6 (`diag`), 9 (`k31l_diag_rows`: the young pairs' plane diagonal, one product and one matrix-vector
product over the young columns in the fold), 12-15 (parts of the per-unit third cumulant), 16 (`k4_local`, and
`k4_local_rows` for the response-rows convention).
- Unit check `a42_head/unit_hc_v2.py`: equal to a numpy transcription of the trainer's rule to the last float32 bit; with
  one column equal to v1 bit for bit.
- Native: the S42 column through v2 (`hc_s2v2_651`) gives the stored S42 run's prediction hash, bill and request count. A
  build with seven columns at random weights (a smoke test) runs clean: +332 requests and +0.085% of bill over one column,
  most of it the split of the bank's third cumulant, which did not pass the trainer's table.
- The native columns of network 651 are stored (`a42_head/dumps/columns_dummy7_651.npz`, by `a42_head/dump_columns.py`).
  The response rows per layer are the trainer's to the unit. One finding from it: **natively K4L's per-unit value is exactly
  zero on the rows the native push drops** (64-192 units at layers 7-14, never a response row), so column 16 has to be
  defined on the response rows to be the same number on both sides; the trainer has been told.
- The trainer's joint step over columns 6, 9, 16 on the S42 arithmetic gave -0.2% at best (se 0.23): not handed over.

**Price of the same three columns at the final layer** (asked by the lead, counted from the code): about 0.05% of bill and
105 requests: the law's update at layer 14 (skipped today because no read follows) 0.02% / 35; the law diagonal on the kept
rows at layer 15 0.02% / 33; the young pairs' plane diagonal there 0.006% / 25; K4L's value exists; 12 for the standardisation.

## 32. After the restart (10-10): the S43 skeleton, the final layer's columns, the readouts, and the covariance sandwich's dead cut (closed)

**S43 skeleton.** `law_file_from_trainer.sh law_dials_d6 2 deep bsk1 s2` (sha 0df2ba07): S42's step-2 file with F4c v2 / K4L v6
(the deeper request diet) and the skew-corrected birth. Ten fresh networks of list a in one process (session
`s43skel_fresh_a`), judged against the two stored S42 draws (`a42_head/judge_ten.py s43skel_fresh_a hc_s2_d6_r2_fresh_a
s42_deepdraw_fresh_a`): pooled 14.0057e-9 against 14.0443e-9, -0.28% (each draw -0.23% / -0.32%; mean of the per-network
changes -0.29%, se 0.26; per network in run order -0.61 -0.06 -0.55 +1.80 -1.26 -0.53 -0.75 -0.52 -0.63 +0.19%). Bill
+0.080%; 305 requests fewer on every network (by my count first network 42,747; at the release lane's boundary about
43,165 / 42,268 against the caps 43,800 / 42,700).

**Final-layer columns** (`a42_head/fragment_hcf_v2.py`, unit check `unit_hcf.py` for the K4L column): the final layer's
learned correction is gnn_step at layer 15 (`scalar_terminal_hook`), so the columns enter through a wrapper of `_hc_bias` at
l = 15 (my first build edited `final_delta`, which the live file never calls; an instrumented run showed it). Rows = the kept
rows of the final layer's row cut (|mu / sigma| <= 3.5), the only rows the bank is pushed on into layer 15. 17 = the law
diagonal (needs the fold's update at layer 14, skipped before), residual on (v, 1, mu^2) over the kept rows, over v; 18 =
the young pairs' plane diagonal with its chaos4 means over the kept rows; 19 = K4L's final value over v^2. Smoke test with
random weights on 651 (`hf3_smoke_651`): +132 requests, bill +0.10% (19 alone `hcf_smoke2_651`: +25 requests, 0.3 Mflop).

**Readouts** (`a43_readout/fragment_ro_v1.py`, unit check `unit_ro.py` exact against a transcription of the trainer's lines;
smoke `ro_smoke_651`): +254 requests, +7.5 Mflop.

**The covariance sandwich with the dead units cut: closed.** Lead 02:29Z: "the covariance sandwich is not dead-cut today,
about -1.5..-2.5% if clean". The file computes C_pre = W' C W over all 1,024 units of the layer below: the product C W
(1.009 Gflop, 106 requests, depth 7) and the upper blocks of W' (C W) (0.708 Gflop, 398 requests), layers 1-14, and C W
once more for the final layer's diagonal.
- How many units the live roster cuts (`a44_cov/probe_deadcov.py` on 651): none at layers 1-4, then live 992 960 960 928 896
  864 832 832 800 800 at layers 5-14 (mean 926 of 1,024). At most about -0.7% of bill, not -1.5..-2.5%.
- Prices of the cut products through the file's own kernels (`a44_cov/bench_shapes.py`, warm calls): C_aa W_a + generic
  symmetric W_a' (C_aa W_a) = 1.715 / 1.548 / 1.389 / 1.237 Gflop at m = 960 / 896 / 832 / 768 against 1.717 at 1,024; the
  generic symmetric path takes about 190 requests against the specialised 398 (the specialised one is for 1,024 only).
- What the cut drops: 0.6-2.0e-3 of C_pre's Frobenius norm and 0.7-3.6e-4 of its diagonal (rms) at layers 5-14, almost all of
  it the cut units' covariances with the live ones; their own variances are 1e-4 of the trace (putting them back, on the
  diagonal or through a Gram product, changes the error in the third digit). The file's float32 product is 1e-4 off.
- Native, lever `DCV64` on the skeleton (sha ae5c296e), session `s43dcv_fresh_a` against `s43skel_fresh_a` (same order, one
  process each): **651: +16.32% final-layer error, bill -0.651%, -1,985 requests; 378: +6.72%, -0.748%, -2,056.** Stopped
  after two networks.
- A repair through a few directions does not pay (`a44_cov/probe_deadcov_v2.py`, exact SVD of the cut units' rows of C):
  rank 8 / 16 / 32 hold 63-82% / 85-96% / 97-99% of their energy; the C_pre error at layer 12 goes 2.0e-3 (cut) -> 1.2e-3 /
  8.0e-4 / 3.5e-4, at layer 14 7.7e-4 -> 3.9e-4 / 2.3e-4 / 8.9e-5. The rank-32 term costs about 175 Mflop a layer (the cross
  block through rank 32 plus a range finder) against 170-330 Mflop the cut saves at m = 896-832.
- What it teaches: the estimator feels C_pre errors of 1e-3 at the 10% level of the final-layer error; units under the gate
  floor still carry covariance the next layers use.
- **At a lower floor the cut is clean** (lead 03:0xZ asked for the push's dead side, alpha < -3.5). The error is the cut units'
  gates' doing, so the floor decides it (`a44_cov/probe_deadcov_v3.py` / `_v4.py`, 651; m = the count at or above the floor rounded
  up to 64): floor 1e-3 cuts layers 9-14 only (m 960 / 960 / 896 / 960 / 896 / 896) at C_pre errors 0.6-1.5e-4; alpha -3.5 cuts
  layers 11-14 at m 960; 2e-3 adds layer 12 at 896 (1.9e-4); 5e-3 cuts layers 7-14 at 3-8e-4. Lever `DCV64f0.001` (sha b7d33806),
  ten fresh networks of list a against the skeleton's session (`a44_cov/compare_sessions.py s43dcv3_fresh_a s43skel_fresh_a`):
  **final-layer error +0.03% pooled (se 0.06, 5 of 10 lower; per network -0.01 +0.00 -0.25 -0.22 +0.18 +0.27 -0.07 +0.23 +0.23
  -0.04%), bill -0.250% (-0.06 .. -0.34%), charged ops -1,397 (-1,223 .. -1,766)**. The lead put it into S43 (03:24Z).
- The cut product's kernel at 1e-3 (651, first network): generic symmetric path -0.186% / -1,223 requests; the specialised
  1,024 product on the cut operand padded with zero rows (W's rows in roster order) -0.316% / -18 requests; the specialised
  product's tiles at the inner size -0.433% but +6,537 requests on the first network (lazy binds of the new tree keys).

**Batch-6 head columns** (`a46_b6/fragment_b6_v1.py`; the HC lever's columns 'm2', 'qb', 'eold', 'qold'; unit check
`a46_b6/unit_b6.py` PASS against numpy transcriptions of O/physicist/b6cols.py: m2 to 4e-16, the others to 2e-7 (float32)):
m2 = t3^2 He4(a) phi(a) / 72 from the K21 diagonal before the pair map (14 requests a layer); qb = the read's plane without its
diagonal against c = (g / s)^2 mu, over (c.mu - c_i mu_i) v_i (15 requests, one float32 matrix-vector product); eold = the old
cohorts' (born at j <= l - 3) |lam| pp^2 sums over v (4 requests, one matrix-vector product over their columns); qold = their
local quartic with K4L's birth coefficients (strength 1) over v^2 (21 requests, 16 flops per element of units x old pairs).

**Layer-15 copy of the learned correction** (`a45_l15/fragment_l15_v1.py`, lever `L15@<name>` with arrays from
`a45_l15/weights/<name>.npz`, or `L15@off`): gnn_step at layer 15 takes its own enc0 / enc2 / enc4, msg weights and layer 15's
rows where given; the same operations (no bill), about 30 requests once for the unpack; +3.5 KB off, +223 KB with all 44,271
numbers. Built (off / same values / perturbed: shas b88e86ff / 3c30e979 / 880819d7), not run: the dump was cancelled.


## 33. S43, S44 and the parts they needed (10-10)

**S43** = the trainer's `s43_mix` hand-over (learned-array step, k31l plane by age 1 / 1 / 0.640) + BSK at 0.34 + S42's head
column + the deeper diet + the padded dead cut: `stack/s43_build.sh s43_mix` (sha 4011f35f). Ten fresh networks of list a against
the two S42 draws: -0.16% pooled (each draw -0.11 / -0.21%; se 0.26), 6 of 10 lower, bill -0.16%. With column 17 at layer 15 (the
physicist's u15, `a42_head/weights/c17p.json`, base `s43c17`; sha f03a044f): -0.43% (8 of 10), bill -0.08%; column 17 against
S43 itself -0.28% (se 0.07, 8 of 10), +76 charged ops, +0.077% bill.

**S44** = jmC (trainer) without qold: `stack/s43_build.sh jmC jmCnq s43c17` (jmCnq = `head_weights_31x128_without_qold.json`
converted by `a42_head/weights_from_trainer.py --json <table> all jmCnq`: rows 6 / 20 / 21 / 28 / 29 -> diag / k4_local_rows / m2
/ eold / qb at layers 1-14, row 17 -> `jmCnq_f15.json` at layer 15; kappa_C 0.697 read from physics_dials.json as kc = 0.303;
generated base kb4d2d = s43c17 with KC@jmC and HCF@jmCnq_f15). Sha 67dfcf71. Ten fresh networks against the two S42 draws:
-0.78% pooled (se 0.61), 8 of 10 lower, bill +0.051%. Network by network against torch (each against its own S42; native / torch,
%): 651 -1.14 / -1.07, 378 -2.95 / -2.77, 424 -1.25 / -1.27, 328 -0.95 / -1.29, 287 -1.09 / -0.71, 641 +0.27 / -0.69, 548 +3.95 /
+3.60, 557 -0.38 / -1.33, 316 -2.71 / -2.28, 390 -1.62 / -2.17: means -0.79 / -1.00. Parity (`a48_parity/`): the change from each
side's own S42 (torch a_0s42par) has cosine 0.92-0.98 and size ratio 1.01-1.07 at layers 8-15 on 651 / 378.

**kappa_C** (`a47_kc/fragment_kc_v1.py`, lever `KC@<name>`, values from `a47_kc/kc_from_trainer.py`): the card's rank-2 update
after the learned mean correction, S = host d' + d host' + d d', is already scaled by 1 + rc (rc = _GN_RC = -0.08); the trainer's
card adds kc S back after every other state hook, so natively rc_eff = rc - kc in the same two lines. kc = 0: byte-identical
(651); kc = 0.05 everywhere: bill +0, ops +0.

**Batch-6 columns** (`a46_b6/fragment_b6_v1.py`): price on 651 for all four (m2, qb, eold, qold, random weights): +0.58% bill, +775
charged ops, nearly all qold's quartic over the old pairs; qold left S44 (the trainer: +0.08% with it).

**Cholesky route for the sandwich: closed by its price** (flopscope bills cholesky n^3/3 = 0.358 Gflop in float32; no triangular
solve; the triangular product through the tree kernels about 0.77-0.83 of a full product: 1.84 Gflop a layer against today's 1.72).

## 34. Approximation audit of the native file and the native-torch gap (10-10 evening; S45 b36b7154, S46 jmIv)

Each native-only approximation of S45 switched to exact alone (`a53_audit/make_variant.py <parent> <name>`: text edits + literal
overrides appended last; files `a53_audit/files/b36b7154_<name>.py`; the edge cut by a rebuild without EcutSv2, `EC_OVERRIDE=none`,
sha 3ba25ecc), sessions `ax45_<name>` on 651 / 378 in their own fresh processes against `s45h_fresh_a`; table `a53_audit/audit_table.py
s45h_fresh_a a_0mHpar <sessions>` ('implied' = power of the final-layer change over MSE = the expected cost of a change unrelated to
the error; 'vs torch' = rms of native - torch S45 at the final layer, 688-truth).

| switched to exact | in torch? | final-layer MSE 651 / 378 (pooled) | implied | bill | vs torch rms (1e-6) |
|---|---|---|---|---|---|
| (parent S45) | | | | | 16.44 / 17.12 |
| old-tier tails (a > 2.5; dead rows a < -2.5), first-chaos trace | no | +2.58 / -0.61 (+0.98) | 0.82 / 1.02 | +2.915% | 12.66 / 12.99 |
| young dead tail (a < -2.5, L7-14), spherical trace | no | +0.16 / -0.50 (-0.17) | 0.69 / 0.68 | +1.650% (-495 ops) | 12.71 / 14.63 |
| pair-mean truncation r_u 256 (L10-15) | no | -0.50 / -0.92 (-0.71) | 0.49 / 0.52 | +1.476% | 14.94 / 15.08 |
| edge cut + tanh stand-in | no | +0.29 / -0.09 (+0.10) | 0.03 / 0.03 | +0.716% | 16.28 / 16.79 |
| tanh stand-in alone | no | -0.03 / +0.06 (+0.02) | 0.02 | +0.140% | 16.41 / 17.20 |
| covariance dead cut (gate 1e-3) | no | +0.55 / -0.00 (+0.27) | 0.09 / 0.12 | +0.352% | 16.85 / 17.79 |
| push dead-side cut (a < -3.5, L1-14) | no | +0.12 / -0.11 (+0.01) | 0.04 / 0.07 | +0.080% | 16.36 / 16.76 |
| L15 row cut (k3 on all rows, head rows kept) | YES (gn7_s6b k3cut) | +0.07 / +0.28 (+0.17) | 0.04 / 0.09 | +0.984% | 16.68 / 17.45 |
| the first three together | | +1.99 / -1.62 (+0.18) | 2.12 / 2.24 | +6.297% | **4.76 / 5.00** |
| every native-only one | | +2.49 / -1.80 (+0.33) | 2.26 / 2.43 | +8.817% | **4.24 / 4.26** |
| partials: truncation exact at L10-14 / at L15 / r_u 320 | | -0.78 / +0.06 / -0.73 (pooled) | 0.45 / 0.05 / 0.47 | +1.313 / +0.240 / +1.311% | |
| old-tier positive tail exact / dead tail exact | | +1.48 / -0.55, +2.59 / +0.20 | 0.6-0.9 / 0.7-0.8 | +2.069 / +1.395% | |
| old-tier positive tail at 2.0 (both tails at 2.0) | | +14.4 / +15.2 (+2.4 / +9.5) | 22 / 13 | -0.398% | |
| edge rosters at floor 0.05 (rows and columns) | | -0.27 / +0.52 (+0.13) | 0.11 | -0.149% (+180 ops) | |

- **No approximation is flagged**: each saves more bill than its error cost; the thresholds sit at their optimum (2.0 for the
  positive tail is a cliff; exactness costs 1.3-2.9% of bill per cut for at most -0.8% of error). A rounding-level change (kernel
  prices) moves the final layer by rms 2.3e-6 (implied 0.04%). Shared with torch and closed by the physicist on 10-10: the 0.01
  gate floors (the knee), the 384 / 320 / 384 old-tier ranks (1:1 with bill), Rayleigh-Ritz at the L10 join.
- **The native-torch difference is the three cuts**: old-tier tails, young dead tail, pair-mean truncation (91% of its power at
  the final layer); without any native-only approximation what is left (rms 4.2e-6, 0.15% of MSE) is rounding grown from layer 0.
- **The change gap, step by step** (native/torch - 1 on list a, pooled, 688-truth, `a53_audit/level_table.py` with the matching
  torch judgements): S42 +0.63% (draws +0.58 / +0.68), S43 +0.92% (a_0s43c_j), S44 +0.86% (a_0mCJq), S45 +0.84% (a_0mHJ), S46
  +1.20% (a_0mIJv; native S46 on list a -0.32% against native S45, torch -0.68%). Steps +0.29, -0.06, -0.02, +0.36% (mean +0.14%,
  se about 0.09). Per network the level is a persistent constant (651 -2.2%, 316 -1.5%, 287 +3.5%, 557 +2.4%). The S46 step is
  NOT its push cut at gate 1e-3: S46 with the cut back at 3.5 against the fired S46, same ten, paired: -0.04% (se 0.13) for
  +0.089% bill (the cut stays). At layer 2 the native and torch changes differ by the same rms 0.74-0.78e-6 in the S44, S45 and
  S46 steps whatever moved: the native-torch difference at layer 2 (0.7e-6, also with every native-only cut off) re-drawn.
- What could still be had here belongs to the trainer: his card has none of the three cuts, so the learned parts are fitted
  without them and scored with them (net about 0.8% of MSE on list a); `a22_cutsim/torch_cutsim_v2.py` applies them in torch.

**S46 = jmIv** (lead 21:02Z): the checkpoint `trainer/cert/refit/jmI_p2_noAS.pt`, read without torch (`a54_jmiv/read_pt.py`),
differs from jmIf's `jmI_p2_free.pt` only by the iso shift +0.030240600493245826 at layers 10-14, so jmIv = the jmIf hand-over +
iso -0.06375939950675417 at 10-14 through `DLY@jmIv` (`a52_pld/values/jmIv.json`; `DLY_NAME=jmIv bash stack/s43_build.sh jmIf jmIfh
s46s|s46p`): safe 63f89e8d, primary f89b8951. Parity of the iso step against torch (a_0mIparv - a_0mIparf) on 651 / 378: cosine
0.96-0.99 at layers 11-15, size ratio 1.02-1.06.

## 35. After S46 (10-10 night): the per-layer dial fragment for K4L v7, and the layer-15 unmask port

- **S46 fired** (lead 21:55Z): the safe file 63f89e8d + release's Rl3_bind128 / Rlvfs_limit_off / Rprices_zero (#335259). The
  release twin with K4L v7 died with NameError _K4L_E: `a52_pld/fragment_pld_v1.py` scaled K4L v6's birth-time age tables only.
  **v2** (`a52_pld/fragment_pld_v2.py`, appended by `stack/build.py` from 22:11Z): v6 as before; in a v7 file (no _K4L_E; the
  strength by age is read in `_k4l_local`) the age-1 strength of layer l is set around that call. A v6 build with v2 (sha 79635d35)
  has the fired file's predictions on 651 (pred sha 83cfbcb2); the v7 twin (sha 7bb599c2) runs clean, bill -0.078% and +419 ops
  against v6, rounding-level predictions (rms 3.2 / 2.8e-6 at the final layer).
- **Checkpoint values without torch**: `a54_jmiv/read_pt.py <a.pt> [<b.pt>]` (the pickle's tensor rebuilds mapped to numpy over the zip's
  storages; prints dials, per-layer shifts and the differences between two files) and `a52_pld/dly_from_pt.py <pt> <name> [<dials>]`
  (per-layer iso / bscale / X1 for DLY@, every layer explicit). `s43_build.sh` takes `DLY_NAME=<values>`.
- **Layer-15 unmask, group (a)** (lead 20:49Z; lever `U15a` or `U15<p>.<p>`, `a55_u15/fragment_u15_v1.py`, needs DCV): inputs 13-15
  and 26 of the final layer's correction from the full C_pre15 (the diagonal kept bit for bit; the symmetric half added to the
  C_aa W_a product the final layer already pays). On the S46 safe (base `u15t`, sha 2851aebd) against s46v_651_378b: bill +0.27 /
  +0.28%, +411 charged ops; with the existing head the final layer moves by rms 14 / 17.5e-6 (implied 1.6 / 2.5%), MSE +2.86 /
  +0.15%: only a layer-15 head refit can make these inputs pay. Prices of the other masked inputs: 24 / 25 through the plane's
  factors (free), 22 / 23 the plane read at 15 (+2.6%), 27 / 28 the layer-15 post covariance.

## 36. S47 twin pieces (10-11 early): request census, request diet, 7-level strip pushes (reads closed)

**Request census of the fired S46 safe file** (70f7857f; `a56_req/run_sites.sh` = one boundary worker session with release's
--sites recorder in this lane; `a56_req/sites_table.py`, `buckets.py`). Warm network (100), 44,508 requests in that session (the
recorder slows the setup: 2,035 of them are tree views bound at first use that a normal setup binds): tree-kernel arithmetic 20,137,
physics + learned arithmetic 13,491, physics + learned without arithmetic 5,519, tree kernels without arithmetic 3,307.
`a56_req/probe_repeat.py` (in process, weak references): 489 requests per network repeat the SAME view or conversion of the SAME live
object (pair series about 220, _ps32 75, _ppi_copy_p16 78, ...), plus fixed-shape allocations.

**Request diet** (lever `XRQ` = `a56_req/fragment_rqd_v1.py`; `XRQ2` = v2 + a per-layer cache of the float32 conversions):
the pair series' views of its persistent arena / state arrays made once per owner, persistent scratch for the packed-inverse puts.
Byte-identical predictions on 651 / 378. Boundary (`a56_req/run_boundary.sh` with --setup-slowdown 0, networks 35 / 100):
S46 safe 43,518 / 42,216; XRQ 43,059 / 41,726 (-459 / -490). XRQ2: charged ops -191 / -203 (XRQ -156 / -168), bill -0.011%.

**7-level strip pushes** (lever `HZ7` / `HZ7g<x>` = `a57_hz7/fragment_hz7_v1.py`): a packet push with m or k having a rest of 32 / 64
beyond its multiples of 128 runs at 7 levels with that rest as a half-zero strip (the 6-level strip layout one level deeper),
2048-column packets on the 7-level trees, taken where the (calibrated, within 1%) bill model gains > 3%. Unit check
(`a57_hz7/unit_hz7.py`): exact to rel 6e-5 (the 7th level doubles the rounding); (1024, 960, 6464) -5.9% / +214 ops, (896, 832, 5248)
-4.2% / +141 ops. Native S46 + HZ7 on 651 / 378: bill -0.178 / -0.173%, charged ops +683 / +257, final layer rounding-level (rms
2.3 / 2.1e-6). Taken: the layer-6 push (1024, 960, ~10k) on every network, (896, 832, ~6k) and (960, 960, ~5.6k) on some. Twin
S46 + XRQ + HZ7 (sha 992ee31d): boundary 43,901 / 41,980 (+383 / -236 against S46; about +600 of the first network are HZ7's new
7-level keys bound at first use), bill -0.180 / -0.194%.

**Safe-eligible form: HZ7R** (`a57_hz7/fragment_hz7_v4.py`, lead 05:52Z). The first network's extra requests of HZ7 were the
7-level B / C trees of each product's last, partial packet (a new width per product and network, bound at first use), not the strip
views (binding those in setup, HZ7P = v3, saved 16 requests). HZ7R runs only FULL 2048-column packets at 7 levels and gives the
remainder columns to the parent's own push (the 6-level keys S46 binds for that remainder), written into one output; the rule prices the
mix. S46 safe + XRQ2 + HZ7R (base s46h7r, sha 7e812cea): boundary 43,566 / 41,935 (+48 / -281 against S46), native 651 / 378 bill
-0.249 / -0.146%, charged ops +162 / +38, final rms 2.6 / 2.0e-6. One command for S47: `stack/s47_build.sh <hand-over> <head> [<pt>]`
builds S47s (base s46h7r) and the fallback S47n (base s46xq2).

**7-level strip reads: closed.** Built and exact (`fragment_hz7_v2.py`), but against the parent's 6-level reads +0.01% (896, 9984,
960) to +13% (672, 6208, 832): the price model's 7-level read bill was never calibrated and is 5% low (the per-chunk transforms at 7
levels cost 1.8x the 6-level ones over (m + n) k).

## 37. After S47 (10-11 morning): residual cuts at zero bill (lead 06:26Z) and the prebind list for the S47 physics

**Persistent scratch of the covariance sandwich** (lever `XRQS` = `a56_req/fragment_rqs_v1.py`, appended after XRQ2 and HZ7R;
bases `s46h7rs`, `s46xqs`). The outputs of `_psd_sym_tile` (three outer tiles) and of `_psdb_group` (two P4 groups) are kept per shape
(5 buffers, 1.4 MB) with every view the kernels take of them: the lower-copy transposes, the group split, the unwind's quadrant views,
and, through an `_l3o_last` override for kept outputs, the last unwind's output views. Each buffer is a tree product's output and is
fully written before it is read. S47s + XRQS (sha 284bfc71, base kffc45) against S47s (30123696) on native 651 / 378, each in its own
fresh process: predictions byte-identical 2/2, bill -0.003% (reshapes no longer made), charged ops -312 / -336; in the census
session every sandwich after the first network hits (70 gets per network, 39 output-view reuses). Expected at the boundary: -38
requests per sandwich x 14 layers = about -530 per warm network.

**What else repeats** (`a56_req/probe_repeat.py` on S47s, `a56_req/probe_s47s.out`): 150 repeated views per network spread over 8-24
per site (`_ps32` across layers, `_ubc_slice`, `_uvp_split`, `omitted_radial_source`, `_gn_dcov`); the fixed-shape zeros (`select`,
`_np_left`, `_f4c_step`, the main loop's `cp_pre`) are put / accumulate targets, so a kept buffer would need re-zeroing (no gain);
the strip packs (`_hzg_pack`, `_hz_ks_product`, `_hz7_pack_*`) change shape with the layer (8-14 shapes over 12-18 calls): no gain.

**The prebind list.** The P lever's list (`a13_prebind/prebind_hfull_v1.py`, 10-01) is appended near the end of the file and
REPLACES `_PB_SINGLE` wholesale: the parent's own setup extensions (TRAIN6's 600 keys, of which 217 are not in our list; the S32
public-shape extension, 12 of 27 missing; the PSD layout's 7 single-side keys) are dead in every file we build. In-process census
of S47s + XRQS (`quick.py --keys --binds`, all 649 listed keys bound in setup): lazy tree binds 0.21% of the bill on the first
network of a session, 0.02-0.30% on later ones. At the platform's setup pace only about 593 of the 649 keys bind (release stage C).
Tools: `a13_prebind/regen_prebind_v2.py` (= v1 + the platform-pace scenario, PACE env), `a13_prebind/bind_table.py`; the builder
takes `PLIST=<name>` (law_file_from_trainer.sh) for `P<name>`.

**Prebind regen: closed.** A list refit on census session 660-669 (`a13_prebind/regen_prebind_v2.py`) overfits: on the held-out
session 37-73 its lazy binds at the platform pace are 0.328% of the bill against 0.220% with the 10-01 list (in sample 0.162% vs 0.182%).
What is still bound lazily is network-specific (k-split reads with k = 3,800-5,100, about 110 Mflop per tree). The fired configuration
(release fragments, `stack/release_like.py` rebuilds it here byte for byte: rl/s47s_safe = 55b951a6) binds the same: 0.207% on 660.

**The residual is requests** (`a58_floor/floor_candidate.py` + `floor_table.py`, boundary session a56_req/runs/floor_v1): a stub
estimator making 10,000 requests of one kind in a tight loop costs 42.9k (view, reshape) / 43.5k (in-place addition) residual
instructions per request = flopscope's own client work outside its spans (6.0 / 6.4 us locally). Our files: 44.6-45.5k per request
(S46 + XRQ2 + HZ7R 44.6k warm, S46 safe 45.2k), i.e. our own Python is 44-114 M instructions per network = 3-5% of the residual. Only
fewer requests cut the residual; every request removed saves its full ~6-7 us.

**XRQS in the fired configuration** (`stack/release_like.py` rebuilds release variants here byte for byte; rl/s47s_safe = 55b951a6
reproduces release's stage C exactly: 43,362 / 42,218 requests, same bill to the flop): + XRQS (rl/s47rs_safe 612e78cd) 42,868 / 41,686
(-494 / -532), bill -0.003%, residual instructions -1.3 / -1.6%, boundary predictions byte-identical, server footprint +19 MB.

**Direct strip loads** (lever `HZD` = `a59_direct/fragment_hzd_v1.py`, k-split reads; `HZD2` = v2, + the depth-6 strip products, one loader
`_hzd_load` for every strip layout): a strip product copied its operand into a zero-interleaved pack and the tree load copied the pack
again (reshape + copy, each billed per element). The operand now goes straight into the tree's level-1 quadrant view (axes: row bits x L,
row block, column bits x L, column block): main blocks, the strip's 32 rows / columns on the deepest bit's 0 half of the last block, the
corner, zeros on the 1 half. Probe of the fired file (`a59_direct/probe_packs.py`): packs copy 133 / 127 M elements per network in the
k-split reads, 44 / 46 M in the strip products, 16 / 9 M in the 7-level pushes (not done: 0.01%). S47s + XRQS + HZD (042b1490) against
S47s + XRQS on native 651 / 378: byte-identical 2/2, bill -0.088% / -0.088%, charged ops +57 / +59.

HZD3 (= HZD2 + a bounded view cache; base `s46h7rsd3` = S47s's base + XRQS + HZD3, the S48 engineering base, `stack/s48_build.sh`
default): S47s physics sha 8df9325a, native 651 / 378 against S47s + XRQS byte-identical 2/2, bill -0.104% / -0.109%, charged ops +43 / +47;
fired configuration (rl/s47rsd3_safe a61ed8e7) network 35: 42,946 requests (-416 against the fired file), bill -0.077%, byte-identical.

**Inner-size covariance sandwich** (lever `XSDM` = `a60_psdm/fragment_psdm_v1.py`, needs XRQS and DCV ..p; NOT `PSDM`: a lever starting
with P is read as a prebind list). With the dead cut the symmetric product W_a' (C_aa W_a) ran at inner size 1024 with 1024 - m zero
rows (the specialised product's fixed keys); XSDM runs it at inner m (multiple of 64 below 1024) with the same three tiles and the two P4
groups at depth 5 (left / right sides built for inner m by `_psdm_side`, output sides the parent's), and gathers only the m kept rows
of W. Rounding-level. Estimate on 651 (S45 product census: m 960 / 896 at layers 9-14): about -0.37 Gflop = -0.14% of the bill.
Not worth it (estimates): direct loads of the 7-level pushes' packs (0.01%), of the output unpack (0.014%); depth-3 trees for the
GNN's thin batched products (n = 17-30 padded to 32: -0.05% at best).

XSDM measured (S47s physics + XRQS + HZD3 + XSDM, sha a22d8770, base k9b874): native 651 / 378 against s46h7rsd3 bill -0.102% / -0.175%,
final MSE +0.22% / -0.15% (rounding level), charged ops +314 / +196 (first-use binds of the inner-m keys); fired configuration
(rl/s47rsd3x_safe 01103249, platform setup pace) XSDM alone: network 35 bill -0.123% at +650 requests, network 100 -0.140% at +195; the
whole stack against the fired S47s: 43,596 / 41,898 requests (+234 / -320), bill -0.200% / -0.284%. Release gated it as S47e (08:35Z,
their recipe on a22d8770 = 01103249 byte for byte). Zero-padded packs (probe v2: 7-level kpad, depth-5 -> 6, padded pushes): 0.003-0.016%
of the bill, closed. Base for S48: `BASE=s46h7rsd3x bash stack/s48_build.sh <hand-over> <head> [<checkpoint .pt>]`.

**Per-layer s_carry** (lead 08:20Z / GO 08:5xZ; the trainer's 11th Ne arm at layers 1-6). DLY key `scarry` (a52_pld/fragment_pld_v3.py,
builder edits in stack/build.py; `a52_pld/dly_from_pt_v2.py` reads pl['s_carry'] + the dial from a checkpoint; `stack/s48_build.sh` uses
it). s_carry (_S19_LAM[0]) enters four native sites: the loop's carried lam (x s_carry(l) at the end of layer l = the card's state_out),
F4c's step lam2, the pair law's per-layer constants (aV / aG / iaV / iaG; the law's tables are built at import, before _PLD exists:
a `_pldc` shim, and pld v3 rebuilds `_NP_TABLES`), and F4c's birth mix (s_carry ** (age + 1) -> the product over layers l + 1 .. l + age
+ 1). Identity: values equal to Nd's global at layers 1-6 byte-identical to k9b874 on 651 / 378. Parity against the trainer's torch card
(v11 cut simulator, change against each one's own Nd base, `a48_parity/parity_change.py`): +0.01 at 1-6 cosine 0.9997 at layer 2 and
0.984-0.994 at 3-15, size ratio 0.999-1.031; the trainer's Ne trial arm (arm_ds_carry_1_6.pt, -0.0039 at 1-6, built from the checkpoint)
cosine 0.92-0.96, ratio 1.03-1.09; both leave the same ~3e-6 rms final-layer mismatch whatever the shift (rounding / cut-flip level).

**Bill-side research bets on the young bank: both closed** (lead 11:0xZ). The young bank (newborn-bank pushes + young-square reads) is about
55% of the bill, linear in its column count. Census of the fired S47e file (rl/s47rsd3x_safe) at every young read of 651 / 378
(`a61_prune/census_v1.py`, `_v2.py`; `census_v*_s47e.json`), pair energy E = ||a (x+ y+' - x- y-')||_F exact:
(A) pruning low-energy births: the energy is FLAT, 99.9% of sum E needs 97.4-99.9% of the pairs at every layer (99%: 91-99%; top-10% share
12-27%; Gini 0.05-0.40), and the pairs' terms cancel: dropping the weakest 10% (1-8% of sum E) moves the whole plane by 2.5-4% at layers
7-14 and 5-27% at 1-6 (25%: 8-12% / 12-45%). (B) low-rank young reads: the young plane is not low-rank: projected on the top-r range of
the read's right factor the plane keeps 26-37% (r 256) / 16-24% (r 384) error at L2-L6, the plane's own best rank-384 approximation 8-14%.
Every pair carries comparable, mutually cancelling third-order content: any truncation of this representation costs error about 1:1.

**Physicist's bet #1 (scale-mode K21 in the covariance output only): closed at its gate, no port** (11:12Z: gamma 0.5 at 10-14 +0.20%,
gamma 1 +287%; the card already carries 97-99% of the true coefficient). For a future covariance-only port: their torch arm called the
pair map a second time with K21_C and kept only its covariance output (everything inside _nonlinear_pair_batch saw K21_C; hooks after the
map, e.g. cov_pm / the GN dcov k21n term, saw the card's K21); b_pre = W' b of the state the layer receives (the b-form already in K21),
radial = that state's incoming radial. Native sketch (TEAM_LOG 11:15:03Z): T21C on the packed series' cov channel only, about 8 Mflop and
16 requests per active layer.

## 38. RESTART (10-11 afternoon): module bills, the read by weight, the star's leading piece, r_u, and S48 as fragments

All in `a62_modules/`. Gates are native: one process per arm, the same seeds in the same order, final-layer MSE against the analyst's
fresh Monte Carlo truths (`analyst/fresh/loader.py`), pooled change and the se of the mean log ratio; bills are deterministic.

**Module bill census of the fired S47e** (`module_census_v1.py` wraps `BudgetContext._charge_op` and keys every charge by the main
loop's line and the candidate frames under it; `table_v1.py` maps keys to modules; 651 + 378). The physics modules are 5.4% of the
bill: head 2.02 (only the output projection and the rank-2 update of C, 0.16, go with it alone: its GNN state and messages also feed
the edge at 8-14 and the learned birth coefficients), law 0.85, edge 0.53, K31L 0.43, F4C 0.41 (only with rule and S2 both off), K4L
0.22, head columns 0.14, K31 row factor 0.13, covariance knobs 0.12, K22 0.11, the rest about 0. Base 94.6: young push 31.3, young read
27.8, old-tier push 16.1, old-tier read 8.1, covariance sandwich + star 9.7. Half strength saves nothing anywhere (dials only).

**The read's rows feed the births.** The read's rows are the response roster and its columns the live roster; both come out of
`select` sorted most central / most active first, so a tighter read is a prefix slice. `rows_probe_v1.py` / `_v2.py`: rows with
|a| > 2.0 are 11-16% of the rows at 7-14 and hold 0.2-0.4% of the covariance-channel energy (pdf_i g_j / (2 s_i) on K21_ij); columns
with a < -1.6 hold 0.01-0.03%. The zeroing gate (`make_rdz_v1.py`: the full read, then the plane zeroed outside the prefixes = what a
restricted read leaves; certificate = the fired sha) says otherwise: rows |a| <= 2.0 cost +13.75% (0 of 8 lower), growing layer by
layer; feeding the head's features the uncut plane (`make_rdz_v2.py`) leaves +12.2%. Cause: `_bd_update` builds every newborn leg from
the hub's whole K21 row (weight g_i) and column (weight d_i), entrywise, and the response roster is the hub roster. No O(nK) shortcut:
each leg entry needs its own plane entry. Columns only: a >= -1.6 +2.13% for -2.57% bill; a >= -2.0 +0.28% (se 0.44) for -1.04%.
The real build (`make_rd_v1.py`, loop edits; `fragment_readcol_v1.py`, the same arithmetic as an appended fragment, same prediction
sha) on the 16 gate seeds 7100136-151: MSE +0.38% (se 0.40), bill -0.88%, charged ops +148. (The +13% local wall I first reported was
load: a same-work zeroing arm showed the same walls.) Dead-tail crop at 1-6: fewer than its 64-row minimum, about 0.

**The star.** Native `_SS_RANK = k` at every layer (`make_ss_v1.py`), bill on 7100136 over the fired row-sum star: rank 8 +0.60%, 16
+1.18%, 32 +2.37% (+285 ops each): the physicist's torch gains (-0.43..-0.49%) cannot pay. k = -1, one direction along the row sums
(s2 = ((W o W)' A' a)(W' a)' / a'a, 8 n^2 flops a layer): -0.26% MSE (se 0.11, 11 of 16 lower), bill +0.035%; at layers 7-15 only
-0.17%. Fragment `fragment_star_dir_v1.py`; torch drop-in `a33_starsum/star_ss_torch_v2.py` (v1, which the trainer's engine imports,
would read -1 as all-but-one columns).

**r_u** (`_LCR_RU`, the rank kept of every old pair's mean at layers 10-14; `make_lu_v1.py`), bill on 7100136 over 256: 272 +4.64%
(bad kernel width), 288 +0.96%, 320 +1.22%, 384 +1.34%; native MSE on that network -2.1 / -2.6 / -2.8% against the torch scan's
-0.60 / -0.89% pooled.

**S48 as fragments** (`make_s48_v1.py`: the fired S47e + a62 fragments appended between BEGIN / END lines with their sha; NAME=VALUE
items for certificates). (a) MS law literals `fragment_ms_v1.py` from the trainer's frozen point by `law_literals_from_pt_v1.py` (the
exporter's rules; the same conversion of jmNd_p1.pt reproduces the fired law literals exactly); the native source strengths are one
value for all layers, so layer-0 births take MS's new strengths (the trainer's point kept Nd's there). (b) `fragment_lu320_v1.py`.
(c) `fragment_star_dir_v1.py`. (d) `fragment_readcol_v1.py`. Certificate: (c) + (d) at neutral values = the fired S47e bit for bit.
r_u 320 OUT (8 of 16 gate networks: score +0.33%, +429 ops). Platform-pace counts (release `boundary_run.py --short-setup`, networks
35 / 100; `run_boundary_counts.sh`): S47e 43,596 / 41,898; MS + star (s48/m_sd) 43,668 / 41,970; MS + star + column (s48/m_sd_rc)
44,222 / 41,971 (warm +73: the first network's extra are first-use binds of the cut read's shapes). SEALED LOOK (lead's pre-registered
rule; `judge_sealed_v1.py`): m_sd_rc vs S47e on A37 7100328-359 = score -0.580% pooled (se 0.31, better on 18 of 32), MSE +0.244%, bill
-0.822% -> S48. (Five runs were stopped once by the guard's 256 MiB swap limit with six of my processes on a loaded box: at most two.)
MS native-torch parity (651 / 378): MSE change native +1.30 / +0.20% vs torch +0.77 / +0.11%, change correlation 0.77 / 0.86 (the
state gap, as the star's 0.84). PORT PATH certified: the MS hand-over through `stack/s48_build.sh` (base s46h7rsd3x) + `release_like.py`
+ `make_s49_v1.py` with the star and column fragments = m_sd_rc bit for bit (7100328). Ready for RAD-S48: `star_beta_from_pt_v1.py`
(the trainer's ssb as `fragment_star_beta_v1.py`'s `_SS_BETA`). Join re-timing (physicist T115) priced by `price_joins_v1.py` (per-layer
per-column bank costs from the census): early joins 6/9/12 -4.5% bill against +10.1% error.

**Evening (S48 fired as #335411).** Joins 7/10/- natively (`fragment_join710_v1.py`; restored-join certificate = S48 bit for bit):
bill +1.39 / +1.64%, MSE -1.09 / -1.26% on 7100136 / 137, a net loss; the census model under-prices the late young columns about 2x.
Closed with the torch arms. **S48b** (`s48/s48b`, sha 8378970c) = S48 + `fragment_prebind_s48b_v2.py` (`prebind_targeted_v1.py`: 140
keys that none of 20 census networks asks for swapped for the column cut's lazily bound read shapes): predictions cmp-equal to S48 on
35 / 100; platform-pace requests 43,661 / 41,702 (S48 44,222 / 41,971), bill -0.051% / -0.006%, setup unchanged (592 of 649 keys,
last phase 4.123 s). **RAD port in one command**: `port_rad_v1b.sh <hand-over> <pt>` (head table, `s48_build.sh`, law literal check,
`release_like.py`, star strength from ssb, + star_dir + star_beta + readcol + prebind_s48b_v2); on the MS hand-over it reproduces S48's
prediction sha 5f09aa4a on 7100328. v1 picked its source by mtime and took another base's file when the builder refused to overwrite.
**Strip twins** (S48 with only the strip lever swapped; platform pace 35 / 100 vs S48): HZ7 (every packet, 3% floor) +310 / +7
requests, bill +0.055% / -0.040%; HZ7g0.0 +1,053 / +703, +0.069% / -0.027%; HZ7Rg0.0 +731 / +243, -0.033% / -0.008%. The bills carry
the lazily bound trees of new packet widths, so HZ7Rg0.0 is worth about -0.01%, not the -0.05% measured in-process on S47.

**Late evening (all native, S48b reference, fresh 7100136-143 unless noted).**
- Early edge (`fragment_early_edge_v1.py`): torch's base source called with layer 8 inside `cov_pm`, no knob factor. `_EDX = {}` is bit for bit. At edx 0.1 on layers 1-7: bill +0.80-0.86%, +551 requests on every network. Parity on 7100136: correlation 0.85 at L2 and 0.62 at L15; native change norm 1.2-1.7x torch's.
- Per-read-layer K3L (`fragment_k3l_bylayer_v1.py`): certified. The RAD port `port_rad_v1d.sh` carries both pieces when the checkpoint has them.
- Final-layer row cut z 3.0: score -0.04% (se 0.17), MSE +0.25% against bill -0.29%. Neutral.
- K3-weighted join sketch: MSE +0.38 / +0.41%. Closed.
- One-sided read row cut (`make_rdz_v3.py`): a > +2 costs +16.8%, a < -2 costs +7.1%, for about 2.5% of bill each. Closed.
- Join capture (`make_jcap_v2.py`, 7100136): the native basis keeps the same lambda^2-energy as the torch range finder, 0.99970 / 0.99969 / 0.99997 at L7 / L10 / L13; the best basis keeps 0.99979.
- Births floor: census model bill +2-5%, as a floor.

## 7. Files

- Builders: `stack/build.py <levers joined by +>` (T, C, B, F, D, Hpad | Hfull, P<list>, R<release fragment>), `buildlib.py`,
  `edits_lib.py`. Every candidate has a `.build.json` (parent sha, every text edit, appended text sha).
- Runners and comparisons: `run_native_v2.sh` + `compare.py` (harness, rows under `H/opus_engineer_*`); `run_quick.sh` + `quick.py`
  (`--check` float64 check of every strip product, `--binds` lazy binds, `--keys` tree-key census) + `qcompare.py` + `vcompare.py`
  (against the bank's validation rows), rows and predictions under `logs/quick_*`; `run_anatomy.sh` (release CPU instrument, outputs
  under `anatomy/`); `noise_decomp.py`; `stack/parity_probe.py`; `a11_bare_views/setup_only.py` (what setup binds and bills).
- Levers: `hzs/` strip kernels, certificates, timing; `a08_f32_core/` traces and core probe; `a10_bd32/` birth update and unit check;
  `a12_live_slice/`; `a13_prebind/` list generator, the two lists and their logs; `a11_bare_views/` (withdrawn idea, bind analysis).
- Evidence: `census/` bill census per call site and per product; `rounding/` in-situ read rounding; `logs/chain3c.sh` and
  `logs/chain3c.out` (the last measurement chain as it ran); `_stale/` superseded files (nothing was deleted).
- Regenerating a prebind list for another kernel variant: `run_quick.sh <candidate> <10 ranks> <name> "--keys --binds"` (one
  worker-like session, repeat with other ranks), then `a13_prebind/regen_prebind.py <candidate> a13_prebind/prebind_<list>.py 649
  logs/quick_<name>.json ... [-- <holdout>.json]`, then `stack/build.py <levers>+P<list>`.
- Physics levers: `a14_f4/` (fragments v1-v4, `parity_torch_v2.py` native against stored torch arms, `fresh_compare.py` against RRPI's
  stored fresh-rank rows, `unit_f4.py` against the physicist's plain-array reference), `a15_k4l/` (fragments v1-v3, `unit_k4l*.py`,
  `fresh_torch_table.py`), `a17_tanh/` (stand-in probe and timing), `a18_f4b/` (closed form check, draft fragments, unit checks),
  `a19_edge_cut/` (fragment and unit check), `explainer/index.html` (one page with charts, regenerated by `explainer/make_explainer.py`).
- Run M export: `stack/from_params.sh <params.pt> <tag>` (= `stack/export_for_build.py` + `stack/levers_from_params.py` + `stack/build.py` + the
  einsum gate); rehearsal and parity tools in `a20_runm/` (`make_rehearsal.py`, `torch_arms.py` = torch predictions of the run M card
  at chosen dial shifts, `parity_regress.py`; stored torch arms `a20_runm/logs/torch_r0035_v1.*`, `torch_r0100_v1.*`).
- Requests: `a21_load3/fragment_load3.py` (lever L3), `a21_load3/unit_load3.py` (kernel unit check: `<candidate> <out.json>`, then
  `--compare <old.json> <new.json>`).
- Row-cut audit: `stack/build.py ...+Z<cut>99` builds, `a16_audit/audit_table.py <base> <variants> [--torch <npz pattern>:<key>]`.
- The next bank file from a trained parameter set: `python stack/build.py T+C+B+F+D+Hfull+F4v4t<theta age 1>,<theta age 2>+K4Lv3c<c age 1>,<c age 2>+Phfull_v1+Rlvfuse_sized32+Rtsr_off <exported candidate.py> <tag>`
  writes `stack/candidate_<levers>__<tag>.py` and its `.build.json`; the exporter's own edits (parameter literals, dials, bsrc, rc) are
  left as they are, every lever edit still has to match exactly once. Before it goes anywhere: the einsum gate
  (`P2/fable_porter_v1/einsum_site_diff.py <file> <last fired file>`), the light setup run (`a11_bare_views/setup_only.py <file>`), and one
  native session.
- Star by its row sums: `a31_star/probe_star3.py` (the split, in a native run), `a33_starsum/dump_planes.py` (stores a network's star
  planes: `a33_starsum/planes/s38_rank0651.npz`), `forms_offline.py` (every form against the exact transport, offline numpy),
  `fragment_ss_v3.py` (the lever; v1 and v2 are the fragments of the first sessions), `unit_ss.py <SS candidate> <parent candidate>`,
  `compare.py <name> [<base name>]` (per network: bill, calls, MSE, how far the final prediction moves), `star_ss_torch.py` (torch
  drop-in, not run), `run_list_a.sh` (the S38-stack arms on fresh list a).
- The pair map's extra plane: `a34_r31/oracle_native.py <candidate> <rank> <oracle.npz> <k31> <eig rank> <out.npy>`, `a34_r31/parity.py`.
- Session chains: `run_chain.sh <name>=<candidate.py> ...` (fresh list a against the stored S38 rows, one session after the other, no
  session while the lead's last QUIET WINDOW line is an OPEN, one retry when the guard stops a session); `quiet_watch.sh` (stops this
  lane's sessions when a window opens). `a35_join/probe_join.py` (written, NOT run: how much norm coupling a join loses).
- The fourth-order norm coupling: `a36_carrier/price_probe.py` (what flopscope bills for the pieces a sketched carrier needs),
  `a37_fold/fragment_nb_v{1,3,4}.py` + `unit_nb*.py` (the bank channel, section 22), `a38_pairlaw/fragment_np_v1.py` (the pair
  law, parity version) and `fragment_np_v2.py` / `_v3.py` / `_v4.py` (lean; v4 is in the S40 file), `unit_np.py` (v1),
  `unit_np2.py` (v2), `unit_np3.py [fragment]` (v3, v4: every switch, requests per function), `arms_table.py` (the 651 / 378
  arms against S39 and against the lean one-form file), `debug_fail.py <candidate> <rank>` (frame variables when a run fails).
- Refit base and head column: `a41_base/` (fragments `so2_v1`, `so2_v2`, `bsk_v1`, `gsk_v1`, `bk3_v1`; unit checks; `base_from_request.py`;
  `bases/<name>.json`; `parity_ora52.py`; `parity_chain.sh` for the parity sessions that have not run), `a42_head/` (`fragment_hc_v1.py`,
  `unit_hc.py`, `weights_from_trainer.py`, `weights/<name>.json`, `parity_hc.py`, `judge_ten.py`, `HEAD_INPUT_C_DEFINITION.md`).
  Launch discipline: `ack_lead.sh` prints the unread lead lines and records them as read; `wait_and_run.sh` refuses to start a session
  while a lead line is unread or the file `NATIVE_HOLD` exists.
- Not done anywhere in this lane: client/server (boundary) runs and platform timing of setup. Every new op form used is one the
  parent already uses (`copyto` with a scalar source, Ellipsis and strided basic slices, `matmul` with `out=`, comparisons under
  `bool(fnp.all(...))`), but the release lane's boundary stages must run the chosen candidate before any upload.
