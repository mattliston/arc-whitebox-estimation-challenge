# Lean card: what could run ~10 dense products per layer? (fable_physicist_v1/lean, 2026-09-24)

**Answer.**
- The organizers' own design is not the lean card. Their factored third cumulant is never compressed, so it costs 143 dense 1024^3 products per layer, 2.2 times the whole flop budget, and it is 2.7-3.5 times LESS accurate than our card.
- Our card is that same design (mean, covariance, full third cumulant, isotropic fourth), made affordable by tiering and corrected by learned parts.
- A card at ~10 products per layer most likely keeps the structure but carries ALL third-order content in one fixed-size compressed carrier from birth. It would drop our exact young tier (young push 25.6% + atoms read 21.4% = 47% of our bill).

## Measured: the organizers' code on our nets (kprop_run.py, kprop_runs.jsonl)

The code is mlp_cumulant_propagation's mlp_kprop (average metric, float64, CPU) on our 1024x16 weights, compared with the 1e9 truth of E[y_15]. Products per layer are matmul flops / 16 / (2 x 1024^3), counted with torch's flop counter.

| design | products per layer | final MSE r35 | final MSE r100 | wall |
|---|---|---|---|---|
| k_max 2 (mean + covariance only) | 2.0 | 3.02e-6 | 5.08e-6 | 1 s |
| k_max 3, factored (+ full third cumulant, isotropic fourth) | 143 | 4.26e-8 | 5.49e-8 | 50 s |
| our S6b card (torch) | ~20 (11.9 dense-product flops, after Winograd) | 1.56e-8 | 1.57e-8 | |
| Luna #332100 (board, 50 public nets) | ~10 (9.9 dense-product flops, plain GEMM) | raw 1.672e-8 (ours 1.774e-8) | | 43.9 s (ours ~92 s) |

- k_max 4 was not run: k_max 3 already costs 2.2x the budget, and the fourth-order factor adds a factor of n.
- Why 143: each layer's nonlinearity appends two rank-n factor blocks (factor_k3.py factored_nonlin_kprop_k3), and nothing ever compresses them. The rank reaches 2n x layer, and pushing three factor matrices through W costs about 12 n^3 x layer at each layer.

## Pricing a ~10-product card (per layer, in dense 1024^3 products)

| part | lean card | our card (S2 census, dense-product flops) |
|---|---|---|
| covariance push W^T C W | 2 | 1.0 (8.5%) |
| third-order carrier | one fixed-rank frame of rank r: push r/n + plane read ~2r/n + fold births and recompress ~2r/n, so about 5 at r = 1024 | young push + atoms read 5.6 (47%) + old tier 3.0 (25.4%) = 8.6 |
| plane-residual (star) carrier | 0-3 | 1.5 (12.3%) |
| isotropic fourth order, mean, learned parts | ~0.5 | ~0.5 |
| total | about 7.5-10.5 | 11.9 (about 20 products before Winograd) |

## Which component a lean card drops or replaces: the exact young tier

- **It is the only part whose size grows with births x age.** Up to 6-8k exact columns at deep layers are pushed and read every layer. Yet its content fits a bounded carrier: one cohort holds 99.99% of its energy in rank 768 at age 1, 640 at age 2 and 512 at age 3 (our cohort audit).
- **The other parts are not the fat:**
  - The old tier is already the bounded carrier (rank 384, 3.0 dense products). Cutting it further is a cliff, and rank 288 plus refit is dead.
  - The star is cheap (1.5) and valuable (+24-52% raw when removed).
  - The covariance push cannot be reduced below its 2 products.
- **Counter-evidence and why it does not decide it:** in OUR card, putting young cohorts into the frame at birth costs +79% to +258% raw without a refit. Our learned parts are co-adapted to exact young content. A lean card trains its learned parts on the compressed carrier from the start, and the GNN-trunk refit (-1.1% VALID40 from a refit alone) shows these parts re-adapt.
- **Luna's profile fits.** It is better at every layer, not only the final one (all-layers MSE 6.8e-9 against our 7.2e-9; the final/all-layers ratio ~2.5 matches ours). It runs half our wall with plain GEMMs. Both point to a fixed-size card with a better trunk carrier, not a final-layer trick.

**Caveats.** The lean pricing is arithmetic, not a build. The real test is a card with one fixed-rank third-order frame from birth (rank 768-1024, recompressed each layer), the star optional, and a joint refit of all learned parts. Only the organizers' average-metric option was run.

## Correction after the fold-at-birth test (2026-09-24; refit6b/foldpush.py, refit_s6b_fold.py, fold_bill.py)

The "one fixed-rank carrier, about 5 products" line above assumed the carrier holds a bounded number of columns. Folding young cohorts into the frame does not bound them.

Test setup: the torch S6b card with B2 epoch-1 literals; every young cohort folds into the frame from layer 8, and every later cohort folds at birth. The every3 control of the same push reproduces B2 epoch 1 bit-exactly.

| fold from layer 8 | no-refit VALID16 vs B2 ep1 (1.74614e-8) | estimated bill vs S2 | dense-product units, layers 8-15 (S2 13.9) |
|---|---|---|---|
| rank 768 | 2.636e-8, +51%, 0/16 better | +30% | 21.1 |
| rank 384 | 2.833e-7, 16x worse, 0/16 | -10% | 11.6 |

- **Why the fold fails.** In a column (CP) representation, the bill is columns x per-column cost. A column carried in the frame is rebuilt over all rows and read at about 2·1024·r + 2·resp·r flops. That halves the young cost at r = 384, which is a 16x lossy rank, and breaks even near r = 768, which is still +51% without refit. At deep layers there are 15-24k columns either way.
- **Revised answer to the lean-card question.** A ~10-product card must carry FEWER third-order columns (fewer births, or a non-column carrier for old content), not the same columns in a frame. Neither fold rank can reach adjusted -15%, so the 1-epoch refit was not started.

## Carriers that bound the COLUMN COUNT (paper only, 2026-09-24; nothing run until covhunter's rank-320 + refit lands)

Why this is the question: the column-driven parts (young push, atoms read, old tier) are about 8.5 of S2's 11.8 dense-product units per layer, at 15-24k columns at L8-15. The rest (covariance 1.0, star 1.5, pair map and learned parts about 0.8) is fixed at about 3.3. Each column costs about 3.3e-4 units per layer in the frame (rank 384) and 6.8e-4 young.

Gate for all three: covhunter's rank-320 cut is +3.35% no-refit (15/16 worse). If the trunk refit cannot absorb that, it cannot absorb any of these larger cuts. Close all three if so.

| carrier | dense-product units per layer | loss mechanism | cheapest falsifiable test |
|---|---|---|---|
| **1. Budgeted births** (top-K born units per layer, K = 512 or 256, ranked by CONSUMPTION importance: the born columns' read-weighted share of the plane on kink rows, not tensor norm) | K=512: about 8.9 (-25%); K=256: about 6.2 (-48%), since columns scale with K | Plane importance is flat: each quarter of births carries 13-16% of the consumed plane (norm-ranked keep .75 was +100% raw, keep .5 was 7.5x, bank, no refit). Consumption ranking was never tried. | No-refit VALID16 on torch S6b + B2 ep1 at keep .75 and .5 from L8 (5 min each). Refit one epoch only if keep .5 is at most +20%. |
| **2. Old content as a plane carrier** (a cohort leaves the CP bank at age 3; its pushed K21 plane and k3 diagonal go into a second star, transported by the star push, 3 products regardless of columns) | about 8.4 (-29%): old tier -3.0, young columns beyond age 3 about -2.2, second star +1.8. The every3-join version (ages 5-7, replaces only the old tier) is about -10%. | A plane-supported carrier keeps only plane-to-plane transport. The pushed off-plane parts are dropped every layer; the old plane is only about 70% regenerable from plane quantities (R2 at most .69). The old tier is worth 18-29x raw, so even the 30% transport loss is large. | No-refit VALID16 with "join into the star" at the every3 layers L7/10/13 (5 min), the mildest version. If that is a cliff, the age-3 version is dead too. Refit only if at most +20%. |
| **3. Merged births** (cluster the born units of a layer by their covariance column s_j in the top-64 subspace; one ± pair per cluster with summed birth weights; 128-256 clusters) | about 5.5-6.5 (-45..-53%) | Each born tensor has one leg pinned to its unit (e_j, which becomes the unit's outgoing weight row after one push). Averaging within a cluster erases unit identity on that leg, which the kink-row plane reads directly. Expected worse than pruning at equal columns. | No-refit VALID16 at 256 clusters from L8 (5 min). Run only after test 1, and only if it beats test 1 at equal columns. |

- **Ruled out on paper:** a top-k principal-subspace core (the organizers' "high-variance directions" hint). The unit-pinned leg spreads over all directions after one push, so a k = 128 subspace holds only about k/n of it.
- **Ruled out on paper:** fewer columns per birth. Two columns (±) is the minimum for a born binary cubic; one column cannot carry its s²e and se² terms.
- **Prior:** all three lose information that the co-adapted card uses. They are only live if the trunk refit absorbs structural cuts, which is exactly what covhunter's run measures. Order if the gate passes: 1, then 2, then 3.

## Test 1 result: budgeted births are DEAD (no refit; 2026-09-24; refit6b/births_prune_eval.py, births_bill.py, births_grade.py)

Setup:
- Ranking, computed only from the card's own state: each born pair's read-weighted plane in the next layer, pushed with the next layer's weights and weighted by the pair map's d on kink rows and g on live columns.
- Pruned pairs get lam = 0.
- Torch S6b + B2 epoch-1 literals, VALID16 against B2 epoch 1 (1.74614e-8).
- Whole-card bill from the S2 native census, with the newborn cohort's first push paid in full.

| births kept | from layer | raw ratio | nets better | whole-card bill ratio | projected adjusted ratio |
|---|---|---|---|---|---|
| 75% | 8 | 1.63 | 0/16 | .975 | 1.59 |
| 50% | 8 | 5.52 | 0/16 | .950 | 5.24 |
| 50%, pruned plane caught by the star | 8 | 3.05 | 0/16 | .950 | 2.89 |
| 75% | 1 | 3.91 | 0/16 | .874 | 3.42 |
| 50% | 1 | 20.8 | 0/16 | .749 | 15.6 |

- **From layer 8 the bill barely moves (-2.5 to -7.6% even at 25% kept).** The early cohorts (layers 0-7, 1.5-2k columns each) dominate the column count.
- **From layer 1 the loss is catastrophic** (4x to 21x). Consumption ranking does not rescue the flat importance: dropping a quarter of the late births alone costs +63%.
- **The star catch halves the damage but not enough.** Carrying the pruned pairs' plane for one more layer brings keep 50% from layer 8 down from 5.5x to 3.0x.
- **Verdict:** no refit can plausibly close 1.6x-21x, since prior refits recovered at most half of much smaller cuts. Carrier 1 is closed. Carriers 2 (old content as a plane carrier) and 3 (merged births) remain untested. Carrier 3 was expected to be worse than pruning at equal columns, so it is closed by this result too.

## Candidate 4 (lead): Tucker carrier with exact pinned legs is DEAD (no refit; refit6b/tucker_eval.py)

Algebra check. A born pair is λ[(s+γe)³ − (s−γe)³] = λ[6γ·Sym(s̃,s̃,e) + 12γu·Sym(s̃,e,e) + (6γu²+2γ³)·e³], with s = s̃ + u·e.
- The (s̃,s̃,e) part fits Y ×F ×F with the pinned leg exact.
- The (s̃,e,e) part is about 2|u|/|s̃| of the main term, so every birth keeps one exact pinned column.

Cost check (S11 prices at L14): today about 10.1 U. Tucker is about 7.0 U at r = 64 and about 14.8 U at r = 128 (Y's r(r+1)/2 slices plus a 4·n·r³ mode rotation per layer). It pays only near r = 64.

Emulation (torch S6b + B2 ep1):
- From L8, every young pair is split into its exact pinned leg e' = (p+−p−)/2, its exact unit bond ρe' (ρ = u/γ recorded at birth), and its covariance leg s̃'.
- All s̃' legs are projected onto ONE common top-r eigenspace of Σ(|λ||e'|)²s̃'s̃'ᵀ, refreshed every layer. The old tier is untouched.
- The full-rank control reproduces the card.

| r | VALID16 raw ratio vs B2 ep1 | nets better | weighted covariance-leg energy captured, L8 / L10 / L12 / L14 |
|---|---|---|---|
| 256 | 5.52 | 0/16 | .976 / .989 / .992 / .994 |
| 128 | 10.75 | 0/16 | .951 / .982 / .987 / .989 |
| 64 | 16.4 | 0/16 | .890 / .967 / .977 / .982 |

- **Killed.** r = 128 is far above the +15% kill line.
- **The Jacobian worry is ruled out.** One common frame DOES hold 98-99% of every cohort's pushed covariance-leg energy at deep layers.
- **The card is hypersensitive to the last 1-2%.** This is the same law as the old-tier rank cliff (rank 256 at age 4 holds 99.8-99.96% and costs +22-33%). Energy fractions do not predict tolerance here.
- **Consequence for the lean card.** Every compression of the third-order columns tested so far loses 5-20x in no-refit raw: fold, prune, Tucker. A lean card would need learned parts trained on the compressed carrier from scratch. The retrofit route through our co-adapted card is closed.

## Is the hypersensitivity information or co-adaptation? INFORMATION (refit6b/tucker_eval_v2.py, VALID16, no refit)

| test | ratio | nets better |
|---|---|---|
| Tucker r = 256, each newborn cohort projected ONCE at birth (learned on), against B2 ep1 | 4.07 | 0/16 |
| Tucker r = 256, all young pairs every layer (learned on), against B2 ep1 (from above) | 5.52 | 0/16 |
| learned-OFF control (GNN head and edge output zeroed), against learned-ON B2 ep1 | 1.24 | |
| Tucker r = 256, learned OFF, against the learned-OFF control | 4.43 | 0/16 |
| Tucker r = 128, learned OFF, against the learned-OFF control | 8.71 | 0/16 |

- **Not the re-projection.** Projecting each cohort once still costs 4x, so re-projecting every layer is not the problem.
- **Not co-adaptation.** With learned parts off the loss is nearly the same (4.4x and 8.7x against 5.5x and 10.8x). The learned parts add only about 1.25x on top.
- **LAW.** The card's carried covariance needs every young birth's covariance leg to about 99.9% of its energy, a physical requirement. A carrier that drops the last 1-2% of young content fails at any rank, with or without retraining.
- **What remains.** Bounded-column designs that keep young content EXACT, or a different estimator family altogether.

## Which consumer carries the sensitivity (refit6b/route_eval.py; birth-only Tucker r = 256 from L9; 8 VALID nets vs B2 ep1)

Method: a copy of the hook card's step carries the projected-minus-exact difference of the young columns, pushed and gated like the card's exact columns. Each consumer then reads exact or projected content. With no consumer routed, the copy reproduces the card.

| projected content fed to | raw ratio | nets better |
|---|---|---|
| every consumer | 3.32 | 0/8 |
| the plane read only (K21 on response rows, and everything downstream of it) | 1.79 | 0/8 |
| the per-unit k3 diagonal only | 2.64 | 0/8 |
| the omitted-radial source only | 1.00 | 2/8 |
| every consumer, cohorts aged 2 or more only (first read exact) | 2.39 | 0/8 |

Net 35 against the 8M truth planes (kink block, off-diagonal; k3 over alive units):

| layer | plane change from the projection | card's own plane error | k3 change from the projection | card's own k3 error |
|---|---|---|---|---|
| L10 | 10.2% | 7.5% | 7.4% | 5.4% |
| L14 | 16.2% | 8.5% | 12.8% | 6.3% |

- **The most sensitive consumer is the per-unit k3 diagonal**, then the plane read. The omitted-radial source does not matter. Most of the damage travels with the carried content.
- **The energy fraction was the wrong measure.** Holding 98-99% of the energy still moves the kink-row plane by 10-16% and the per-unit k3 by 7-13%. A global low-rank frame misplaces exactly the per-unit and kink-row values.
- **No accuracy lever found.** The card's own errors in both consumers (5-8%, upper bounds because they include truth noise) are smaller than the projection's. No consumer is shown to be one; the earlier truth-k3 oracle (-3%) agrees.
