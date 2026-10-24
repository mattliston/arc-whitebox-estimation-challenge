# Leader gap: which architecture change could close it? (fable_physicist_v1/leadergap, 2026-09-23 ~00:10, addenda to 00:37)

**Summary for a cold read (5 lines).**
1. The leaders are about 10% more accurate and 25% cheaper than our fired card. Public telemetry shows two kinds: suliman, marius and reds are better at every layer, while J2W and ben3 gain only at the final layer, which our anatomy ties to each unit's final variance and so to covariance errors carried from earlier layers.
2. No change to our card closes that gap without retraining: every cheaper setting tested costs 2 to 8 times more error than it saves (thresholds at deep layers, births straight into the compressed tier, cheaper star), and an exact star transport at the key layers makes the error worse.
3. One small free saving was found and built as a native build step: skip the last layer's third-order work for units far from the ReLU kink, worth -0.4% to -0.6% on the adjusted score; it goes into the next stack.
4. The carried covariance error cannot be predicted from anything cheap: no feature built from the weights and the card's state, including the fourth-order loop terms, predicts it across nets (held-out R2 at most 0.012 against a 0.2 bar), so a learned correction channel is dead before training.
5. What remains is a real fourth-order carrier plus a joint refit for accuracy. The bill route of structural cuts plus a joint refit failed overnight: a rank-288 old tier refit for one epoch got 5% worse on all 16 held-out nets, like the earlier drop-age-9 test. A plain refit of the uncut S2 closure gained 0.58% and went into S6. Refitting the GNN trunk itself on the fired physics (S6b) gains 1.09% on 40 held-out nets (33 of 40 better) and goes into S9.


**Answer.** No change inside the current card reaches either target in a no-refit test: not -25% bill at equal raw, and not -10% raw at equal bill. All three suggested directions are cliffs: births into the frame, a cheaper star product, and a new row/column split at deep layers. Each loses 2-8x more raw than it saves in bill. The one free change is a terminal-layer row cut, worth about -0.8% adjusted. The public telemetry and our own last-layer anatomy locate the leaders' edge in the per-unit variance at the final layer. That variance is our carried off-diagonal covariance, and no local change reaches it.

Setup: torch S2 card (model_bank_audit + K31 row .85/.5 at L1-14 + b-fix .1 + every3 push), learned parts ON, no refit, CPU. Final-layer MSE vs the truthpost file (1e9 samples), nets 35 and 100. The S2 baseline is 1.6121e-8 / 1.5770e-8. The torch bank reproduces 1.6485e-8 / 1.6834e-8 exactly.

Bill estimates use the analyst's native per-layer, per-function census of the bank (fable_analyst_v1/bill_by_layer_candidate_r35.json). Each function's per-layer cost is divided by the torch bank's counts for that layer: rows, live columns, young and old columns. Variants are priced by their count changes against S2 (bill_model.py, about ±20%). Adjusted = (1 + raw change) x (1 + bill change) - 1.

## Ranked list

### 1. Terminal-layer row cut: keep the CP and star parts of k3 only on rows with |mu/sigma| < 3.5 (ACCEPT; small)

Mechanism: at L15 nothing propagates. k3 only corrects the per-unit ReLU mean, by a term proportional to z·phi(z). Rows beyond |z| = 3.5 carry under 0.7% of the largest correction. Three L15 costs scale with rows: the old-tier reconstruct, the young push and the star diagonal. Together they are 3.4% of the bill. The b term (3 v b) stays on every row, and the variance is still needed on every row to find z.

| cut | rows kept (r35 / r100) | raw r35 | raw r100 | bill (est.) | adjusted (est.) |
|---|---|---|---|---|---|
| \|z\| < 4 | 79% / 75% | -0.02% | +0.33% | -0.7 / -0.8% | -0.75 / -0.51% |
| \|z\| < 3.5 | 72% / 69% | +0.01% | +0.41% | -1.0 / -1.05% | **-0.95 / -0.64%** |
| \|z\| < 3 | 64% / 61% | +0.45% | +0.51% | -1.2 / -1.3% | -0.79 / -0.82% |
| \|z\| < 2.33 (the response rows) | 50% / 49% | +19.9% | +11.6% | -1.7% | +17.9 / +9.6% |

Port note: compute v and mu first (both are already computed first natively). Then run the L15 push and the star diagonal only on the kept rows, and set k3 = 3 v b elsewhere. Row counts vary by net, so the Winograd tiles need padding. The rows at 2.33 < |z| < 3 matter, because their skew correction is still 5-25% of its peak.

### 2. New row/column split at deep layers, L10-14 (REJECT)

| dial at L10-14 | raw r35 | raw r100 | bill (est.) | adjusted (est.) |
|---|---|---|---|---|
| response rows: threshold .01 -> .03 | +9.9% | +10.2% | -2.6 / -2.4% | +7.0 / +7.6% |
| response rows: threshold .01 -> .05 | +21.4% | +33.1% | -3.9 / -4.3% | +16.7 / +27.3% |
| live columns: .01 -> .03 | +9.5% | +5.7% | -1.7 / -1.6% | +7.7 / +4.0% |
| births: .01 -> .03 | +5.3% | +3.0% | -0.7 / -0.8% | +4.6 / +2.2% |

The deep layers are already the cheap part of these kernels, because response rows and live columns shrink with depth (at L14 on r35: 544 response rows, 768 live). The rows and columns being cut still carry plane content the pair map uses. The .01 thresholds were already found co-adapted in both directions (audit/RESULT_AUDIT.md).

### 3. Birth deep cohorts directly into the rank-384 frame (REJECT)

| schedule (on every3) | raw r35 | raw r100 | bill (est.) |
|---|---|---|---|
| cohorts 9-13 join at L14 | +79% | - | +2.1% |
| cohorts 6-12 join at L13, 13 at L14 | +197% | +258% | +0.9% |

Young content truncated at age 1-4 is the hypersensitivity cliff again (rank 256 at age 4 = +22-33%). The bill does not even fall. Reading a column through the frame costs about as much as reading it young (two products against the rank-384 frame per response row, against one against the live columns). Joins add work on top.

### Not re-run: a cheaper star middle product

The star push is three 1024^3 products per layer (W^T A, then (W^T A ∘ W^T) W and (W^T∘W^T)(W^T A)^T), 12% of the bill. A low-rank star does not help. The Hadamard coupling makes every rank-1 term of A cost a full W^T diag(v) W product. So no cheaper exact form exists. The approximate forms are already measured: the alive-compact star and covariance push gives -3% bill for +6-10% raw, and the star off at L>=8 gives about -6% bill for +24% raw (p2-331824 dials). The organizers' kprop code carries the same objects (factored k3/k4 plus diagonal slices), so it gives no shortcut.

## [section removed for publication: competitor per-network telemetry]























## Recommendation

1. Port the terminal cut at |z| < 3.5 if it is easy, for -0.6 to -1.0% adjusted.
2. Keep the bill effort on the refit line (drop old cohorts plus a joint refit). Every no-refit bill dial here costs 2-8x more raw than it saves.
3. Aim the raw effort at the carried covariance: fourth-order planes plus a joint refit, judged by the final-layer variance error.

Files: arch_screen.py / arch_screen_v2.py (screens), screen1.jsonl, screen2.jsonl, smoke.jsonl (results with per-layer counts), bill_model.py and bill_model.out (bill estimates).

---

# Addendum (2026-09-23 ~00:30): L15 row cut as a native build step; loop-term and information tests

## A. L15 k3 row cut: native build step (build_l15_rowcut.py)

Usage: `build_l15_rowcut.py <in.py> <out.py> [zc=3.5]`. The patch is count-asserted and refuses to overwrite its output or to patch an input twice.

- **What it replaces:** four terminal lines of the live _predict_binary. These are the push, the cube sum and the two star-diagonal lines, replaced by one call to `_l15_rowcut_k3`.
- **How the rows are chosen:** rows = the card's own `select(zc - |mu/sigma|, 0)`, which rounds up to a multiple of 32 so every product keeps its Winograd path.
- **How the kept rows are computed:** the live `_shared_old384_push` is called unchanged, with W restricted to those columns. The W5 port scope is suspended for that one call, because the port requires the registered 1024-column W. All other rows keep only the 3 v b term.
- **Where it applies:** it anchors on lines that are identical in the bank, e2 v7 and e2 v8 (_e2_H) forms. It applied to S2 and to the porter's candS3_HEB_porter.py.
- **Torch reference:** arch_screen_v3.py cfg term32:3.5, which uses the same row set.

Native (score_variant.py; net 35 cold, then net 100 warm in one process; S2 baseline from fable_fire_v1/score_candS2_r35_r100.log):

| build | net | raw change | torch raw change | util | bill change | adjusted change |
|---|---|---|---|---|---|---|
| S2 + cut | 35 | +0.009% | +0.011% | .18462 (S2 .18575) | -0.61% | **-0.60%** |
| S2 + cut | 100 | +0.434% | +0.442% | .18194 (S2 .1835) | -0.85% | **-0.42%** |
| S3-HEB + cut | 35 | MSE identical to S2 + cut | - | .18233 (base .18341) | -0.59% | -0.58% |
| S3-HEB + cut | 100 | MSE identical to S2 + cut | - | .1798 (base .18133, same order) | -0.84% | -0.41% |

Native and torch agree to within 0.01 points of raw.

The bill saving is smaller than the -1.0% estimate. The bank census scaled by row count predicted it. The per-layer census of the build (fable_analyst_v1/bill_by_layer_candS2_l15cut35_r35.json) shows why. With 72% of the rows kept, the L15 old-tier reconstruct falls only to 0.887x and the young push to 0.803x. `_rule_depth` grants the depth-6 Winograd push only when m == 1024, so a row subset falls back to depth 4, which costs about 1.3x more per row. Recovering the missing ~0.3-0.4% would need a kernel-rule change (depth 6 for a row-subset push), which is the engineer's call.

Chain placement: anywhere after build_e2. It touches only the terminal block and appends one function. The join step and the kernel steps leave the anchored lines alone.

## B. Gaussian-born loop terms vs the card's fourth-order error (loop_plane_test.py, r35, L12)

Setup:
- Candidates are pushed from the L11 Gaussian part (S = Cpre_11, vertex factors g, d, d3) through W_12. Each family is fitted with free weights to the pair-map read of the truth-minus-card planes (8M truth, response x live, net_read_fit.py weighting).
- The card's isotropic K31 is 82% off the truth on that block. K22 is 24% off.
- Top-k accuracy of the 4-cycle against the exact full-S 4-cycle on 48 response rows: cosine .69 / .86 / .96 at k = 16 / 32 / 64.

| family (k = 64) | K31 error read explained | K22 error read explained | net covariance change explained |
|---|---|---|---|
| 4-cycle (K31, K22 adjacent and alternating) | 0.1% | 1.6% | 0.07% |
| triangle + pendant (three label patterns) | 5.8% | - | 0.2% |
| (2,1,1) doubled vertex, phi = 2g(1-g) - 2 m d | 4.1% | 5.3% | 0.7% |
| coincident W o W pushes of the per-unit Gaussian k4 | 0.25% | 0.0% | 0.06% |
| all together | 7.3% | 5.7% | 0.9% |

The loop terms are below the 15% bar: DROP. What the isotropic K31 misses is not Gaussian-born loop content from the previous layer. The net change is 89% K21 read at this layer (r21 .0374 of net .0419).

## C. Information test on the one-sided top-k correction R (R_loop_data_v2.py, R_loop_fit_v2.py)

Protocol: the analyst's R_data.py, unchanged.
- Torch bank card, learned ON. After step l, R = P (C - C_truth) P U_k, with the 32M truth; R is 89-95% signal at L10-14.
- Ridge fit on one net, R2 on the other (35 to 100 and 100 to 35).
- New features are the pair-map reads (the analyst's forms) of the section B planes, plus a (W o W) TRANSPORT family: the card's own L-1 K22 and K31 planes, linear-response gated and pushed to L. That is fourth-order memory the card never carries.

| feature set (k = 30) | held-out R2, pooled L10-14 (35 to 100 / 100 to 35) | in-sample | best single layer, held-out |
|---|---|---|---|
| analyst's 11 | .001 / .009 | .012 / .005 | .023 |
| + 4-cycle | .001 / .009 | .013 / .005 | .019 |
| + triangle + pendant | .002 / .010 | .013 / .005 | .024 |
| + (2,1,1) doubled vertex | .002 / .011 | .015 / .007 | .022 |
| + W o W coincident | .001 / .009 | .013 / .005 | .023 |
| + W o W transport | .001 / .009 | .012 / .005 | .024 |
| + all new (32 features) | .003 / .012 | .016 / .007 | .019 |
| new only (21) | .003 / .011 | .013 / .006 | .016 |

At k = 100 every set is between .002 and .009. Nothing comes near the 0.2 bar. The in-sample fits are as small, so the features are orthogonal to R, not overfit. The learned covariance-channel route is DEAD ON INFORMATION before training, for every cheap feature built from W and the card's state that we could name. That includes the Gaussian-born fourth-order loops and trees, the doubled vertex, and fourth-order plane transport.

## D. A final-layer-specific estimator of v15 / k3_15 / k4_15?

The telemetry (J2W and ben3 gain only at the final layer) and our L15 anatomy point at v15. But the v15 error is the carried L14 covariance error (cos .96), which lives in the top-k one-sided block (the analyst's closed-loop swaps put 81-99% of the lever there). Section C shows no local feature predicts that block. So a final-only head faces the same information wall as the trunk. The per-unit v/k3/k4 heads were already dead for the same reason. A final-layer edge needs new information: a better plane or fourth-order carrier upstream, or more samples. I see no mechanism that avoids a better trunk.

## E. Costed exact star transport at L10-12 (covhunter's target layers; arch_screen_v4.py cfg starcols, screen4.jsonl)

The audit card's star-as-columns switch turns the plane-supported star residual into exact CP columns, so the star content keeps its off-plane parts and its memory. That is the "more exact star transport". It is applied only at L10-12, where covhunter finds two thirds of the truth-plane lever. Base: the torch S2 card, learned ON, no refit. The bill is estimated with bill_model.py.

| strength (share of the star moved into columns) | raw r35 | raw r100 | bill (est.) |
|---|---|---|---|
| 1.0 | +1.49% | +0.83% | +11.8 / +12.0% |
| 0.5 | +1.13% | +0.53% | +11.8 / +12.0% |
| 0.25 | +0.67% | +0.24% | +11.8 / +12.0% |
| 0.5 at every layer L2-14 (reference) | -0.91% | +1.37% | about +112% |

This is REJECTED: raw gets worse at every strength, and the bill rises by 12%. The columns (about 3.3k per layer) ride the young tier from L10 through L15. The exact transport fixes only the K21 side, which covhunter shows overshoots in closed loop unless K31 and K22 move with it. The all-layer -1.42% measured on the bank audit card (audit/RESULT_AUDIT.md) does not carry over to S2.

## F. Structural cut + joint refit (overnight, 09-23; lead-owned jobs judged here)

Setup: torch card, final-layer MSE on the 16 held-out nets (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130), batches of 4 of the 160 training nets, lr 2e-4 cosine.

| job | before refit | after epoch 0 | change | nets improved |
|---|---|---|---|---|
| rank-288 old tier, bank physics, GNN + birth + edge parameters (48,574) | 2.2484e-8 | 2.3609e-8 | +5.0% | 0 / 16 |
| uncut S2 closure, heads + birth + edge parameters (3,437) | 1.8088e-8 | 1.8021e-8 | -0.37% | 12 / 16 |
| same, after epoch 1 / epoch 2 (final) | 1.8088e-8 | 1.7985e-8 / 1.7984e-8 | -0.57% / -0.58% | 12 / 16 each |

Rank 288 on the bank would need to reach 2.030e-8 (+7.7% over the uncut bank's 1.8852e-8) to pay for its -7.7% bill.
- Its training loss spiked after the first update (2.26 to 5.4e-8), recovered, and then plateaued at 2.33-2.57e-8, above where it started. The run was stopped after epoch 0; the parameters are kept.
- Together with drop-age 9 (7.71e-8 to 4.71e-8 after one epoch, still 2.4 times the bank), the joint refit does not win back what a cut of the old tier removes. The route is closed for tier cuts.
- On S2 a cut would buy even less, because the old tier is only 23.9% of S2's bill.

End state of the refit line (09-23 04:12):
- **(b), tier cuts:** DEAD, stopped after epoch 0.
- **(a), uncut S2 closure refit:** saturated at -0.58% (epochs -0.37 / -0.57 / -0.58%). It changes learned literals only, so it is bill- and CPU-neutral. The lead is putting fable_analyst_v1/refit_s2_full_lr2e4_v2/params_epoch2.pt into S6 (S5e + --refit + engineer2's mem, yright, arena), and VALID40 decides whether the S2-trained literals keep their gain on S5e.
- **Cost:** the three epochs took 118 / 64 / 48 min of wall time at 5 threads. The first epoch ran under load of about 15.

## G. Design for Matt's review (NOT started): refit the closure on S5e's own physics

Why it might gain more: S5e differs from the S2 physics the literals were trained on in three ways.
- covhunter's GN knobs v3: K21 kink-row, non-kink K21, T31 and recentering scales, plus a 1.164 scale on the learned covariance edge.
- The L15 row cut.
- float32 third-order sums (rounding level).
The refit trains the edge and birth literals, so it overlaps the knobs. Transplanted literals may double-count the edge adjustment, or be tuned to physics S5e no longer has.

Expected gain: small. The S2 refit saturated at -0.58% after two epochs. An S5e-native refit can at best recover whatever part of that the transplant loses, so it is worth at most about 0.5% and probably less.

Gate (cheap, before any training):
1. **Read S6's VALID40.** If S6 is at least 0.4% better than S5e, the literals transferred, and an S5e-native refit would add at most about 0.2%: skip it.
2. **Otherwise, one torch evaluation on VALID16** (about 15 min, no training) of the S5e-physics torch card with the original literals and then with params_epoch2.pt.
   - If the S2 literals are WORSE than the original ones, that is a mismatch: run the refit.
   - If they are equal, the knobs have already taken the gain: skip it.

The run, if the gate says so:
- **Card:** fable_analyst_v1/s2card.py + covhunter's v3 knob hook (fable_covhunter_v1 torch harness) + the L15 terminal k3 row mask (leadergap/arch_screen_v3.py cfg term32:3.5). Check native parity on r35/r100 against the fired S5e first (about 1 h of building and checking).
- **Groups:** 'full' of refit_s2.py (heads + A + cstar + edges, 3,437 parameters). NOT the GNN trunk P: training it at 1e-4 to 2e-4 made the uncut bank 2.1% worse after one epoch, and it did not rescue rank 288.
- **Start** from params_epoch2.pt (warm start), lr 1e-4 (half the S2 run's, because of the warm start), 3-update warmup, clip 1, batches of 4.
- **Length:** 2 epochs on the same 160 TRAIN nets; VALID16 after each epoch, VALID40 at the end.
- **Cost:** about 1.5-2 h of wall time at 5 threads when load is under 10, so about 8-10 thread-hours plus the VALID40 evaluation (about 15 min). Peak memory about 7 GB (refit_s2.py's fine checkpoints).
- **Pass bar:** VALID40 at least 0.3% better than S6 (or than S5e if S6 fails), with 60% or more of nets improving. Then the porter's build_stack.sh --refit, native parity, panel and census as usual.

## H. Refit on the fired S6b physics: the GNN trunk is the lever (09-23 morning; fable_physicist_v1/refit6b/)

Setup: the analyst's s6bcard.py (S2 physics + GN knobs v3 + L15 row cut + epoch-2 literals; torch against native S6b pooled over VALID40 -0.009%). Driver refit_s6b.py (refit_s2.py's loss, cosine schedule and fine checkpoints, plus a second optimizer group for the trunk). 160 TRAIN nets, batches of 4, 5 threads, warm start from the S6b literals.

| run | trained | lr | VALID16 after epoch 0 | nets better |
|---|---|---|---|---|
| (A) | head, birth coefficients, edge (3,437) | 1e-4 | -0.18% | 11 / 16 |
| (B2) | the same + GNN trunk: msg and enc0/2/4 weights and biases (44,144) | 1e-4 / trunk 5e-5, trunk warmup 5 | **-1.31%** | 14 / 16 |

B2's epoch-0 parameters on all 40 held-out nets, against the analyst's torch S6b rows at the same thread count:
- 1.82273e-8 to 1.80286e-8, pooled **-1.09%** (the lead's bootstrap CI is -1.43 to -0.75%).
- 33 of 40 nets better, per-net range -3.54% to +1.90%.
- This PASSES the bar (at most -0.3%, at least 24/40). The parameters go into S9 through build_stack --refit. The change is to learned literals only, so there is no bill or CPU change.

The trunk is the refit lever. It is the largest learned part (removing it costs +26.8% raw), and it had never been refit on the new physics.

The first (B) run spiked +70% on its first update for a different reason. It had also trained the GNN's normalization constants (feature centers and scales, rms of about 5.6e-5, gain, and the flag "first"). Adam's first step moves every parameter by lr x sign(g), so a global gradient clip cannot shrink it, and a 1e-5 step moved rms by 18%. Never train those keys: the trunk is the network weights only. Run (A) was stopped after epoch 0 at the lead's request. B2 continues; its epoch 1 gets a VALID40 check if it beats epoch 0 on VALID16 by at least 0.2 points.

## I. Edgeworth-corrected births (09-24, lead's question; refit6b/edgeworth/)

**What the birth code does today** (covstep step + the parent's _bd_update):
- The vertex E[f''] is the Gaussian φ(α)/σ times (1 + radial/3). That factor is the elliptical κ4 cross term κ4(a,a,i,j), not a per-unit correction.
- The legs use the Gaussian gate Φ(α) times Cpre. Four learned directions are then added to each leg, each orthogonalized against the Gaussian leg:
  - the bivariate skew term (d/2)·K21, learned at about 0.8 on every layer;
  - the born unit's side, η·K21ᵀ, at 0.02-0.35;
  - the Gaussian C² term, at about 0.8;
  - the marginal skew of the gate, at about 0 on L9-14.
- beta = 2(mg − host·g) mixes the Gaussian E[f f'] with the corrected mean.
- The born diagonal is fitted to the pair map's corrected post k3.

**Test.** The eval script (ew_eval.py, certified by cert.py) replaces E[f'] on the legs, E[f''] at the vertex, and E[f f'] and E[f'] in beta with their Edgeworth values. It uses the card's own per-unit pre k3 and k4, with no refit. The results are VALID16 ratios against B2 ep1. The strength-0 control is bit-exact.

| arm | ratio | nets better | per-net range |
|---|---|---|---|
| skew (order 1), full | 1.0129 | 1/16 | 0.997-1.028 |
| skew, half | 1.0054 | 1/16 | 0.996-1.012 |
| order 2 (t4 + t3²), full | 1.0102 | 2/16 | 0.997-1.030 |
| order 2, half | 1.0039 | 4/16 | 0.996-1.013 |
| learned OFF, skew, full (vs the learned-off control) | 1.0100 | 0/16 | 1.003-1.020 |
| learned OFF, order 2, full | 1.0080 | 0/16 | 1.000-1.021 |

- **Negative at every strength, and with the learned parts off.** It is not co-adaptation with the heads or edges, so it is not a refit candidate.
- **The damage is close to linear in strength.** A quadratic fit puts the optimum on the non-physical side: skew at strength −1.1 for −0.47%, order 2 at −0.6 for −0.16%.
- **Births are not where the plane error comes from.** Pre-activations are nearly Gaussian per unit. On net 37 at L9-14, the median |skew| is 0.03-0.04 (90th percentile 0.07-0.09), kurtosis is about 0.05 and skew² is 0.001-0.002. The corrections move the vertex by a median of about 1% and the gate by about 0.25%.
- **The card's own plane errors are real.** The 8M truth noise floor is about 2% at L10 and 1.3% at L14, so noise-corrected errors are about 7.3% and 8.4%.

## J. Sizing the dropped distinct-index carry term (09-24, lead's question; refit6b/carry/, MPS float32)

**Certificate (cert_transport.py).** The test is a 3-variable Monte Carlo whose only third cumulant is the distinct κ(1,2,3), with correlations of 0.3-0.5. Against its transport factor of 0.2046 at ε 0.2:

| candidate | value | vs Monte Carlo |
|---|---|---|
| linear gate (the card) | 0.1789 | 12.6% low |
| the v4 instrument's formula | 0.1772 | 13.4% low |
| linear plus first-order dressing d_i d_j C_ij g_k | 0.2120 | 3.6% high |
| exact E[f'f'f'] | 0.2105 | 2.9% high |

The residual is O(ε²). The v4 instrument's pattern list skips every bare-vertex pattern, so it never had this term.

**Dense recursion (dense_carry.py).** A 1024³ third-cumulant tensor is carried open-loop, driven by one unmodified card run on r35. The capture reproduces the card's MSE exactly. The metric is the kink block against the 8M truth, noise-corrected with floors of 1.8% (L10) and 1.3% (L14).
- **C0, the lead's A.** This is the card's own bookkeeping, done densely with no old-tier truncation: bank transport by the linear gate, the card's newborn columns, and the star recomputed from this arm's own bank with the card's formula. It matches the card's star to 2e-5 and its star push to 1e-5 before the first join (L7).
- **D1, the lead's B.** This is C0 plus the first-order dressing on distinct entries only. Plane and diagonal entries keep the card's linear transport and star.

| layer | card | C0 (A) | D1 (B) |
|---|---|---|---|
| L10 plane error, raw / noise-corrected | 7.51% / 7.29% | 7.46% / 7.24% | 7.42% / 7.19% |
| L14 plane error, raw / noise-corrected | 8.49% / 8.39% | 8.46% / 8.36% | 8.45% / 8.35% |

- **B does not close the error.** The dressing moves the kink-block plane by about 1.4% of the truth norm, but only about 0.05 points (L10) and 0.01 points (L14) of that closes error. That is a cosine of about 0.1 with the needed correction. **The carry hypothesis is closed.**
- **Old-tier truncation is not the source either.** The untruncated C0 is within 0.05 points of the card.
- **Two discarded arms, both my own errors:**
  - A/B with a full-row plane reset reached 25-31%. The reset erased the carried bank plane on the non-kink rows, where the card keeps it on top of a target computed without it.
  - D0 dressed the plane entries too and reached 17-27%. Repeated indices have a different transport factor, and that plane mass leaks past the star.

## K. Where the card's 7-8% deep plane error is born (09-24, lead's localization; refit6b/carry/, r35)

Each arm is the validated dense replica C0 plus one change (dense_carry2.py). All runs are on MPS at about 40 s per arm, and swap stayed flat. Errors are on the kink block against the 8M truth, noise-corrected. C0 has 7.24% at L10 and 8.36% at L14.

| arm | L10 | L14 | gain (points) | cos to C0 error |
|---|---|---|---|---|
| S: carry the star's pushed distinct content | 7.09% | 8.14% | +0.15 / +0.22 | 0.21 / 0.23 |
| T3: born triangle (cert_triangle.py) | 7.24% | 8.36% | 0 / 0 | 0.03 / 0.02 |
| TL: all born loops to order 6 | 7.24% | 8.36% | 0 / 0 | 0.02 / 0.01 |
| P: truth K22, K31 and k4 in the pair map | 7.17% | 8.32% | +0.07 / +0.04 | 0.14 / 0.10 |
| K: truth pre plane K21 in the pair map, all-row reset | 7.30% | 8.73% | −0.06 / −0.37 | 0.06 / −0.02 |
| M: truth pre mean and covariance at every layer (card run, second_order_oracle.py) | 7.19% | 8.36% | +0.02 / +0.01 | n/a |

- **The truth noise is small.** A k-statistic noise model reproduces the measured floors and the 8M-vs-32M k3 difference. It puts the kink-block floor at 2.08% (L10) and 1.45% (L14), so the card's error is real.
- **Exact split (decompose.py).** The next layer's pre plane is the pushed post plane plus the pushed post distinct content. The truth post planes come from the same seed and draw. The error energy is 3-8% post plane and 98-107% distinct content, with a small negative cross term. The card's own post-plane error is only 2.1-4.6%. So the error is born in the post distinct content.
- **Which part of the distinct content (distinct_regress.py).** The parts are the carried bank (0.43 of the truth norm at L10), the newborn columns (0.25) and the b channel's distinct Sym(C⊗b) (0.05). The b part lines up with the error at a cosine of −0.51 (L10) and −0.57 (L14). A fit wants b at 0.24-0.42x, and carried and newborn content 3-5% smaller. That fit cuts the distinct error from 7.4% to 6.2% at L10 and from 8.8% to 6.9% at L14.
- **Torch card, b scaled through b_post, r35, no refit (b_scale_test.py).**

| arm | plane L10 / L14 | per-unit k3 L10 / L14 | final MSE |
|---|---|---|---|
| card | 7.21 / 8.37% | 5.37 / 6.32% | 1.551e-8 |
| b × 0.9 | 6.32 / 7.17% | 4.64 / 4.91% | +5.3% |
| b × 0.8 | 6.09 / 7.05% | 4.47 / 4.81% | +11% |
| b × 0.6 | 6.11 / 7.14% | 4.51 / 4.99% | +18% |
| b-fix γ = 0 | 6.82 / 7.86% | 5.03 / 5.70% | +2.1% |
| learned OFF, b × 0.8 (vs learned-off card 1.936e-8) | 6.09 / 7.03% | 4.46 / 4.81% | −1.8% |
| learned OFF, b × 0.6 | 6.11 / 7.13% | 4.50 / 4.99% | +1.8% |
| learned OFF, b-fix γ = 0 | 6.83 / 7.85% | 5.02 / 5.67% | −0.8% |

- **Reading.** The carried b channel is the over-represented third-order content. The scale compounds through the carry, and about 40% of the effect is the S2 b-fix. The third-order state gets 1.0-1.5 points better with less b, but the final output does not follow at no refit, because the learned parts compensate. Even with the learned parts off, the gain is only about 2% on one net.
- **VALID16, learned OFF, b scaled (vs the learned-off control tk2_control_off).** b × 0.9 gives 0.9729 (−2.7%, 13/16 better), and b × 0.8 gives 0.9970 (−0.3%, 8/16). The lighter carried b is right physics that the learned parts cancel at no refit (learned on r35, b × 0.9: +5.3%). That makes it a refit candidate, sized as a 1-3% bet: the B2 recipe with b × 0.9, or a trainable b scale.

## L. The aligned distinct-content term: the born-unit-side plane feed j1 (09-24; refit6b/carry/leg_screen.py, variant_eval.py)

**One-step alignment screens.** Layer l's births are pushed to L = l+1 and compared with the distinct error on the kink block. My rebuild of the card's four learned leg directions (bp + Σ c_j orth_j, with c from _bdnh_coefficients) matches the captured newborn columns to 2.6e-8.

| candidate | size (share of truth norm) | cosine to distinct error, L6 / L8 / L10 / L12 / L14 | distinct error L10 / L14 after one layer |
|---|---|---|---|
| card | – | – | 7.43% / 8.75% |
| j1 physical (−(g_i/2) η_a K21[a,i], unorthogonalized, coefficient 1) | 1.8-2.0% | +0.49 / +0.48 / +0.46 / +0.40 / +0.42 | 6.76% / 8.13% |
| j1 coefficient raised to 1 (orthogonalized) | 1.3-1.8% | +0.44 / +0.39 / +0.35 / +0.25 / +0.28 | 7.02% / 8.48% |
| j0 (bivariate, learned ~0.76) | ~0.2% | −0.02 to −0.10 | none |
| j2, j3 | ≤0.15% | ≤0.14 | none |

The card learns j1 at a mean coefficient of 0.13-0.33 over alive units, on the orthogonalized direction.

**No refit, torch card (the switched-off copy of _bd_update is bit-exact).** Plane and k3 errors at L10 / L14; the card has plane 7.21 / 8.37% and k3 5.37 / 6.32%.

| variant | plane | k3 | MSE, learned on (r35) | MSE, learned off (r35) |
|---|---|---|---|---|
| j1 physical, X = 1 | 5.23 / 5.92% | 3.71 / 3.92% | +40% | +3.5% |
| j1 physical, X = 0.5 | 5.74 / 6.68% | 4.19 / 4.79% | +4.6% | −5.6% |
| j1 physical, X = 0.3 | 6.53 / 7.91% | 4.81 / 5.92% | +1.0% | – |
| j1 coefficient = 1 | 5.56 / 6.47% | 3.89 / 4.01% | +40% | +12% |

**VALID16, learned off, vs the learned-off control.**

| variant | pooled ratio | nets better |
|---|---|---|
| j1 X = 0.5 | 0.9367 (−6.3%) | 15/16 |
| b × 0.9 | 0.9729 (−2.7%) | 13/16 |
| j1 X = 0.5 + b × 0.9 | 0.9650 (−3.5%) | 13/16 |

- **j1 is the physics the card under-learns.** At physical strength it takes 2.0 / 2.5 points off the plane error, which clears the lead's 2-point bar.
- **The learned parts cancel it at no refit.** The best output strength is about half physical, the overshoot law again.
- **The excess b partly stood in for the missing j1.** The two effects are not additive.
- **Refit module ready.** refit6b/s6bcard_j05.py is validated: it reproduces the evaluator's number exactly and runs a fine-checkpoint backward pass.
- **Orthogonalized vs parallel (lead: keep this in).** The card's parameterization projects each learned leg direction onto the complement of the Gaussian leg bp, per born column.
  - For j1, that projection removes a component that carries much of the physics.
  - Raising the learned coefficient to 1 on the orthogonalized direction aligns less with the distinct error (cosine 0.25-0.44 vs 0.40-0.49) and hurts even with the learned parts off (+12% on r35).
  - The unorthogonalized physical direction helps at half strength (−6.3% on VALID16, learned off).
  - So the parallel component matters. The card could not learn it through c1, which may be why c1 stalled at 0.13-0.33.
- **Refit slot (lead's decision).** The combination (−3.5%) did not beat j1 alone (−6.3%), so j1 X = 0.5 ALONE is refit, replacing b × 0.9. b × 0.9 is parked and will be revisited only on top of a j1-refit card. Pre-registered bar: VALID16 ≤ 1.7374e-8, then VALID40.

## M. Mean path, per-unit kurtosis, and the learned-part audits (09-24; refit6b/carry/)

**The final error is a per-unit mean-path error, and the host is exact.** Net 35, B2 ep1, learned on, no refit.
- The card's L15 host mean (m = mg + k3·e/6 + k4·f/24), fed the independent 32M cumulants, scores 1.96e-9: that is the 32M noise floor (1.84e-9).
- L15 input attribution (host only, one input set to 32M truth): μ −65%, v −5%, k3 −8%, k4 −7%.
- NOISE TRAP: 8M-truth mean oracles are floored, because the 8M sample post mean alone is 0.51 of the card MSE.
- With the 32M mean: exact pre mean at L0-14 gives −49%, at L0-12 −29%. Adding the 8M covariance gives −57%, and adding 32M per-unit k3/k4 gives −65% (mean_path_oracle.py plus inline oracles).

**Lever 1: per-unit kurtosis.**
- Oracle: 32M per-unit k4 at L1-15 gives −8.0% (k3 +8%, both −10.4%). The card's k4 = radial·v² has zero per-unit spread; the truth spread in t4 grows from 0.004 (L5) to 0.016 (L15).
- Cheap physical forms (kurt_screen.py) capture about none of it:
  - the diagonal deviation Σ W⁴(post_k4 − radial·C_ii²) raises the correlation with truth by only +0.001;
  - the one-GEMM form (plus the K22-plane deviation) by +0.001 to +0.003, with a t4 spread of 0.0010-0.0016 against the truth's 0.004-0.016;
  - a raw-moment scale mixture (v+μ²)² fails (relative error 0.80-0.93);
  - t4 correlates with α² only +0.19 to +0.36.
- So the per-unit kurtosis lives in the distinct fourth-order content, the closed family. The one-GEMM plug-in on r35 gives −0.28% learned on and −2.5% learned off, at about +9% bill.

**Lever 3: audit of the learned mean head (head_audit.py, nets 35/37/39/42, pooled).** At every layer L1-15:
- cos(delta, truth − host) is +0.21 to +0.35;
- the best-fit scale is 0.84-1.36 (about 1);
- the correction is 0.23-0.46 of the needed size;
- it removes 4-12% of the residual energy, which equals its maximum at best scale.
So the head is correctly scaled and information-limited. No j1-like fix.

**Lever 3b: audit of the covariance edges (edge_audit.py, r35 and r100, L8-14, against the 8M truth post covariance).**
- Off-diagonal: cosine +0.06 to +0.16, best-fit scale 0.4-0.94, size 15-18% of the needed correction, removing ≤2.5% of the error energy per layer.
- Diagonal: cosine at most 0.08.
The edges are weakly informative and over-scaled against physics, yet the v3 knob fit raised them 16% for the final MSE. That is co-adaptation, and there is no clean fix.
- **j05 REFIT RESULT (1 epoch B2 recipe, warm from B2 ep1, card s6bcard_j05).**
  - VALID16: 1.71227e-8, −1.94% vs B2 ep1, 12/16 better. It passed the pre-registered bar (≤ 1.7374e-8), starting from a no-refit +4.97%.
  - VALID40: 1.76186e-8 vs 1.79867e-8, −2.05%, 30/40 better. The 24 nets outside the bar give −2.11% (18/24). Per-net median −2.19%, best −9.02%, worst +4.91%.
  - Extra-epoch controls: covhunter's function-preserving wide2x, 1 epoch, ended at +0.14%, and R320 absorbed ~0. So the gain is the j1 physics.
  - Native port spec went to the porter (S17-j05). Params: s6b_B2ep1_j05_1ep/params_epoch0.pt, sha256 2000046cdcf25a9c926c4916fc47ee0524aacc717ce15ec2152f18aadd9d95ac.
- **Lever 1 closed for no-refit use (VALID16, learned ON, vs B2 ep1).**
  - Per-band k4 scalar × 1.2 / 1.5 / 2.0:
    - L1-7: +4.3 / +30 / +140%;
    - L8-11: +7.3 / +47 / +230%;
    - L12-15: +9.7 / +59 / +330%;
    - 0-1/16 nets better in every arm.
  - Today's card k4 norm matches the truth within 1-10%; the missing piece is per-unit variation.
  - One-GEMM form: +0.83% learned on (4/16), −3.0% learned off (16/16). Parked as a refit-bundle candidate.
- **Native parity (porter, S17p vs S16, VALID40, .venv).**
  - Raw −2.159% [−3.116, −1.211], 31/40; adjusted −2.138%, 30/40.
  - Against the torch rows: native −2.159% vs torch −2.046%, per-net correlation 0.996, per-net difference sd 0.289%.
  - Passed the lead's bar. fire_candS17 staged after the residual gate.

## N. A deterministic per-unit k4 source at L13-14? (lead's question after analyst §35; 09-24 23:10 to 09-25 00:15; refit6b/carry/)

**Answer: no.** No deterministic construction gets per-unit k4 within ~7% at L13-14 for 10-20% of the bill. Neither proposed form
can do it even in principle. The unit-specific k4 at L13-14 is made mostly in L0-L6 and carried for 8-14 layers, so a source must be
nearly exact from L1-2 onward. That is a full-depth fourth-order bank. It would cost about what our third-order bank costs (80% of the
bill), and it would lose even if exact.

**1. The per-unit k3/k4 at L13-14 is old** (`gauss_restart_mc.py`, `gauss_restart_eval.py`; nets 35 and 100; 16M samples per row, MPS).
- Method: at layer l0 the true state is replaced by a Gaussian with the true 32M mean and covariance (covhunter's mcall files). The
  true network then runs forward, and its per-unit k3/k4 is compared with the independent 32M truth. Two independent halves remove the
  Monte Carlo noise.
- This is the ceiling of any construction that starts at l0 and is exact afterwards. Every error below also fits the card's
  isotropic form (k4 ∝ v², k3 ∝ mean × variance) for everything made before l0.
- Control: l0 = 0 is the exact network and reproduces the truth to within 1.2%.

| start at | share of the truth's k4 at L14 made after it | k4 error at L14, net 35 / 100 | k3 error at L14, net 35 / 100 |
|---|---|---|---|
| L13 (last layer only) | 0% | 25.5 / 23.0% | 26.1 / 21.6% |
| L12 (last two layers) | 0-2% | 25.1 / 22.3% | 23.2 / 19.4% |
| L10 | 5-8% | 22.5 / 20.2% | 16.4 / 14.6% |
| L8 | 14-17% | 18.9 / 18.2% | 11.4 / 10.1% |
| L6 | 25-30% | 14.6 / 16.2% | 7.7 / 7.0% |
| L4 | 40-45% | 11.6 / 12.7% | 4.7 / 4.4% |
| L2 | 61% | 7.7 / 7.7% | 2.3 / 2.4% |
| L1 | 73% | 5.3 / 5.2% | 1.5 / 1.6% |
| isotropic form alone | | 26.3 / 24.1% | 28.8 / 25.6% |
| the card (§34, B2 ep1) | | 27.1 / 24.8% | 6.1 / 7.1% |
| spec (break-even) | | 7% (11%) | 1.5% (2.6%) |

L13 reads within 2 points of L14 for k4 and within 3 points for k3.
- At depth, the last layers make almost no per-unit kurtosis. One ReLU on a Gaussian state at L13 gives a mean standardized k4 of 0.001
  at L14, against 0.06 in the truth; two layers give 0.002-0.003. Most of each deep unit's variance is shared, correlated signal, and
  that dilutes what one layer can add. The first ReLU alone makes 27% of the L14 kurtosis, and its unit pattern survives 13 layers.
- So both proposals sit at the isotropic floor. The exact last-two-layer contraction gives 22-25%, and a carrier started at L10 gives
  20-22%. The 7% spec needs exact physics from L1-2; break-even needs L3-4.
- k3 behaves the same way and is stricter: 1.5% needs exact physics from L1. Our third-order bank (6-7%) is as good as exact physics
  from about L5-6.

**2. Nothing cheap carries the unit-specific part, even fitted to the truth** (k4 error at L12-15, nets 35/100; the isotropic form
gives 22-28%).

| per-unit form | k4 error | script |
|---|---|---|
| top-mode variance share (1-64 covariance modes), standardized mean, diagonal share, all together (in-sample fits) | 19-27% | `multiradial_screen.py` |
| scale mixture over the upstream layer norms (ReLU homogeneity; parameter-free; r35) | level 1.2-1.3x the truth, unit correlation .29-.34; refit 22-26% | `norm_coupling_mc.py` |
| the truth's own per-unit k3 (t3, t3², standardized mean; in-sample) | 17-21% | inline |
| one free fourth-order weight per column of our bank (~21k columns; held-out units) | 20-24% | `col_k4_unit_fit_v2.py` |
| born trees at L-1; top-16 covariance directions; diagonal, one-GEMM and raw-moment forms | R² ≤ .04; ≤ .25; ≈ 0 | §M, `k4_tree_screen.py`, `lowrank_kurt.py` |

- The column fit is underpowered: its positive control recovers about half of a planted column signal. Scaled by that, 20-40% of the
  unit-specific k4 is column-supported, which matches the plane-level fit (36-44%, fourth/RESULT_FOURTH.md §1). It is not enough.
- Rank census (`k22_rank_census.py`, r35, two independent truth draws, noise cancelled by crossing them). The non-isotropic part is
  5.5% (L12) and 7.8% (L15) of the per-unit k4 energy. The 7% spec needs 91% of it, which takes the top 256-512 components of the
  residual (2,2) plane: rank 16 holds 36-43%, rank 64 59-70%, rank 256 88-93%. That is the best possible basis at one layer. Our
  third-order bank uses ~20k physically born columns to hold k3 at 6%.

**3. Cost and value.** The adjusted score is raw MSE × bill (util .14, above the .1 floor). So a source costing a share x of the bill
must cut raw MSE by x/(1+x).

| source cost | break-even raw cut | net at spec (-11.7%) | net if exact (-18.4%) |
|---|---|---|---|
| +10% | 9.1% | -2.9% | -10.2% |
| +20% | 16.7% | +6.0% | -2.1% |
| a second bank (~+80%) | 44% | loses | loses |

**Cost of a full-depth carrier, per layer** (lead's request; unit price from our own bill: a symmetric plane push bills like our
covariance push, 13.3 U over L1-14 = 0.95 U per layer; a non-symmetric one, K31, about 1.3 U; the bill is ~150 U):

| carrier, L1-14 | U per layer | total | share of bill | break-even raw cut | can it reach the 7% spec? |
|---|---|---|---|---|---|
| fourth-order weights on our bank's columns (read only; transport free) | ~0.03 | ~0.4 U | ~0.3% | ~0.3% | no: 20-24% even with free weights (section 2) |
| dressed planes-only recursion: K22 and K31 pushes, ReLU dressing, k3-to-k4 conversion | 2.3-3.5 | 32-49 U | +21-33% | 18-25% | no: planes lose about half their content per layer to the distinct classes |
| the same plus the distinct content: a fourth-order bank with its own columns | ~9 (our third-order bank: 124 U per-birth) | ~120 U | ~+80% | ~44% | unknown; it would need exact-from-L2 fidelity |

- The analyst's independent estimate for the planes is 2-4 U per layer (30-55 U), which agrees.
- Two analyst measurements say why the cheap rows fail (fable_analyst_v1/RESULT.md §37, nets 35 and 100, built on these restart runs).
  - A column carrier with average-gate linear transport reaches only 5-8% of the L0-1 births' true k4 at L14 and 6-16% of L2-9's, and
    its unit pattern does not align with the true spread for births at L0-7 (cosine -0.20 to +0.15). No number of columns fixes that.
  - An odd/even split shows the early spread arrives through the true ReLUs in two channels. Born fourth-order content, carried with
    coincident-index dressing, gives 50-72% of it. Born third-order content converting to k4 at later ReLUs gives 28-50%.
  - So a working carrier is a dressed fourth-order recursion at every layer plus a k3-to-k4 conversion fed by the full third-order tensor.
  - The birth split also shows the spread (the part the card misses) is born evenly: every two-layer band holds 10-18% of it.
- Even the planes-only row, which cannot reach the spec, needs an 18-25% raw cut to break even, against a lever of -12% at the spec and
  -18..-20% exact.
- The values are the analyst's (§35: 7 held-out nets, L12-14, correction 0.5). Covhunter's §29 measures -19.6% exact on the current card.
- k4 alone is a small lever. Measured tonight on the analyst's frame and 7 held-out nets (`k34_spec_ch.py`, a versioned copy with a
  channel switch; the deployed and joint rows reproduce the analyst's bit-exactly), truth at L12-14, correction 0.5, geo vs deployed:

| truth fed at L12-14 | vs deployed | nets better |
|---|---|---|
| k3 and k4 (parity with §35) | -18.4% | 7/7 |
| k3 only (card k4) | -8.7% | 7/7 |
| k4 only (card k3) | -4.9% | 6/7 |
| k4 only at the spec (truth + 0.25 x the card's error as noise) | -3.2% | 5/7 |
| k4 only, correction left at 1 | -2.7% | 5/7 |
| control: correction 0.5, no truth | +3.0% | 1/7 |

- The joint value is larger than the two parts together (-18.4% against -13.6%), so most of the lever needs both channels. A k4-only
  source at the spec would pay only if it cost under ~3% of the bill (5 U); an exact one under ~5% (8 U).

**4. The cheapest experiment that could change the answer.** Every cheap construction I could design was run tonight (section 2),
and none moved the answer.
- Value: the cheapest check (truth k4 alone) was run tonight; see the table in section 3. k4 alone is worth -4.9% exact and -3.2% at
  the spec. That closes the k4-only question: no construction reaches 7%, and the value is too small for a bank.
- The only experiment left that could change the answer is on construction, and it is not cheap: a dense MPS replica of a full-depth
  fourth-order bank, born and transported like our third-order bank, on r35. It must beat "exact from L4" in the table above (12%) to
  reach break-even, it would cost ~80% of the bill, and it only pays jointly with a k3 source four times better than our bank.
- A zero-cost aside for a future refit bundle: a per-layer law of k4 on the card's own k3 (t3, t3²) takes the k4 error from ~25% to
  ~20% when fed the true k3. It is not a source.

## O. Why three rivals beat our raw by 6-9% at similar cost: three structural hypotheses (lead's question, 09-25; paper only, no compute)

**What the cost signatures say** (analyst §34, measured against S20):
- suliman: our per-birth slope, half our fixed cost. The fixed half is most simply billing: deeper fast-matmul on the square n³
  products (covariance, star), which our 120 s wall caps. That says nothing about their accuracy, which must come from elsewhere.
- J2W: final layer 6.2% better but all layers 6.5% worse, half our per-birth cost, twice our fixed cost. That is the signature of a
  final-layer-weighted component riding on a cheaper propagation.
- marius: 151 U on every net, so no data-dependent births or pruning. A fixed-size card, 9.3% better, flat across net hardness.

**What our closures already exclude**, so the hypotheses below avoid it:
- per-unit k3/k4 sources (the restart law, section N);
- fourth-order carriers bolted onto the bank: per-column weights, the plane-only star, third-order x nonlinearity terms (fourth/, covhunter §18-19);
- REPLACING bank content by a low-rank subspace: Tucker common frame r256 gives 5.5x the MSE (tucker_eval.py, TEAM_LOG 09-24 11:15),
  old tier r128 +90% raw;
- spending more at the last layers (§34), or an exact old tier at all joins L7/10/13 (~0);
- the learned side: T640 data -0.13%, wide2x capacity +0.14%, the R320 tier cut absorbed ~0 by a trunk refit, per-band knobs,
  per-net routing (<= 0.19%), and the converged dials.
- Three of the lead's examples are already covered:
  - "a fixed-size lossy carrier the learned parts absorb": R320 and Tucker show warm-started learned parts do not absorb carrier loss.
  - "post-activation moments as the closure variable": moments and cumulants carry the same information order by order, and the
    next layer needs the joint tensors either way.
  - "our card sits in a compensated basin": covhunter's learned-OFF 12-dial fit (-10%) moves the same way as the learned-ON optimum
    (c0 down, k4 down). The off-physics values are the optimal response to missing content, not a basin made by the learned parts.

**H1 (highest ceiling): the collective, high-variance modes are treated non-perturbatively, on top of a full bank.**
- Mechanism: at each layer, take the top-R principal directions of the covariance (R ~ 16-64; the effective rank falls with depth).
  Carry the non-Gaussian structure of those R coordinates exactly, including the distinct-index content our CP bank truncates. That can
  be dense third/fourth cumulant tensors in R dimensions, or a small quadrature/mixture over the R coordinates with a Gaussian closure
  per node. Keep the bank for everything else. This is the organizers' own hint: "spend more compute on higher-variance directions".
- Signature fit: the cost depends on R, not on births. That gives a fixed intercept (J2W's +34 U over ours) or a fully fixed card
  (marius). The gain is flat across hardness, because the collective modes exist on every net.
- Why not excluded: every subspace test we ran either REPLACED bank content (Tucker, old-tier cuts) or fitted special forms one layer back
  (independent factors on the top-16 covariance directions: k4 R² <= .25; per-subspace scale mixture: R² <= .16). None kept the bank
  and ADDED exact collective-mode content. One pointer in favour: the card's own per-unit k3 error at L12-14 is partly low-rank along
  the top covariance directions and the mean direction (in-sample R² .11-.51, against .02-.04 for random directions; lowrank_kurt.json).
- Falsifier (<= 1 h, MPS, the restart harness): a subspace-Gaussianized restart. At l0 = 2 and 6, keep the TRUE state's component in
  the top-R principal subspace (R = 16, 64, 256) and replace only the complement by a Gaussian with the matching covariance. Run the true
  network, then score per-unit k3/k4 at L13-14 and the final post mean against the 32M truth on nets 35 and 100.
  - PASS: R = 64 at l0 = 6 keeps k4 <= 11% and k3 <= 3% (the full restart at L6 gives 15-16% and 7-8%). The collective modes hold what
    matters, and a dressed subspace carrier is worth costing.
  - FAIL: R = 256 stays near the full-restart errors. What matters lives in the complement, consistent with Tucker, and H1 is dead.

**H2 (most likely, fastest to exploit): more j1-like terms, i.e. second-order born diagrams our card lacks.**
- Mechanism: each layer the card re-anchors the (2,1) planes to the exact pair-map target (births sized by the diagonal innovation;
  the star carries the off-diagonal residual). Distinct-index third-order content (a ≠ b ≠ c) has no target. It comes only from CP
  transport and the four birth legs j0-j3, and the plane error is born in exactly that content (§K: 98-107%). j1 was one physical term of
  it (cosine .40-.49 with the distinct error, -2% after refit), but it removes only ~17% of that error's energy (7.43 -> 6.76% at L10).
  The second-order diagrams for a ReLU acting on a Gaussian-plus-K3 input include terms outside the j-leg family:
  - f''_a f''_b g_c C_ab K3_abc: a full-K3 vertex dressed by a covariance bond;
  - f'''_a C_ab K21_ac: a plane dressed through the third derivative;
  - f''_a K4_aabc: the (2,1,1) fourth-order feed, which the card has only in isotropic form, through b.
- Signature fit: suliman. The extra terms live inside the birth legs (our per-birth slope), and they make every layer better.
- Why not excluded: my alignment screen (§L, leg_screen.py) tested only the physical versions of the card's own four legs. The analyst's
  §36 ablation audit removes existing terms, so it cannot see missing ones.
- Falsifier (<= 1 h, MPS, dense replica): add these terms at physical strength to leg_screen.py. At L6-14 on r35, measure the cosine
  with the card's distinct-content error and the one-layer cut in that error.
  - PASS: any term with cosine >= .3 and a cut >= 5%, i.e. a j1-sized gain of ~1-2% after refit.
  - FAIL: all cosines < .15.

**H3 (thin): a large final-optimized learned model trained from scratch.**
- Mechanism: a much larger learned model, trained from scratch on many nets against the final-layer loss, correcting the final means
  from rich features on top of a cheaper propagation. J2W's extra fixed cost is room for ~16x our GNN's 2.1 U. It would help at L15 and
  be indifferent or worse earlier: J2W's split.
- Why not excluded: every learned-side test was warm-started from our co-adapted card for about one epoch. §33's "final error
  unpredictable" used our card's own state features. A from-scratch nonlinear model on neighbour and subspace features is untested.
- Evidence against: T640 and wide2x are both ~0, and §33 is ~0 at four granularities.
- Falsifier (<= 1 h): cache per-unit features at L12-15 for ~300 TRAIN640 nets (MPS eval mode, ~3 s per net). Use the card's per-unit
  state, W14/W15 row statistics, weighted sums over the 8 most correlated neighbours, and the top-8 subspace loadings. Fit a from-scratch
  2-layer MLP and gradient-boosted trees to the card's per-unit final error, held out by net.
  - PASS: held-out R² >= .10, so a final-layer model can buy >= 10% raw.
  - FAIL: R² <= .02.

**Ranking.** Expected value is probability x raw gain if true:
- H1 ≈ .2 x 6-8% ≈ 1.4%;
- H2 ≈ .45 x 2-4% ≈ 1.3%;
- H3 ≈ .1 x 5% ≈ .5%.
H1 has the highest ceiling and is the only hypothesis that explains marius's whole profile. H2 is the most likely, and a pass converts
in a day. Run both falsifiers first, one MPS job at a time. If both fail, the honest read is that the rivals' edge is a sum of smaller
things our component closures cannot see one at a time, and our path is the joint-refit line (loopR4) plus the bill.

**O addendum: H2 FAILS (09-25 02:02; refit6b/carry/t3_screen.py, capture_bank.py, t3_screen.json; B2 ep1 card, r35, MPS).**
- Mapping the second-order diagrams onto the card showed two of the three candidates are already its birth legs. The third-derivative
  plane term f'''_a g g C K21 is j1, and the curvature plane term f'' f'' g C K21 is j0. The (2,1,1) fourth-order feed exists only in
  isotropic form, through b and the birth weight radial/3.
- That left one genuinely missing diagram: T3 = f''_a f''_b g_c C_ab K3_abc, the carried FULL third cumulant (the card's CP bank at
  pre-activation, 10-22k columns) dressed by one covariance bond. A brute-force certificate (n = 6) matched the per-column formula, the
  plane formula, the plane-push convention and the distinct-entry split to 1e-15.
- Screened like j1: the term's distinct push to the next layer's (2,1) plane on the kink block, against the card's distinct-content error.

| layer | T3 size (vs the truth plane) | cosine with the distinct error | one-layer cut at physical strength | j1 for scale: size / cosine / cut |
|---|---|---|---|---|
| L6 | .0080 | .099 | 0.5% | .020 / .49 / 11% |
| L8 | .0075 | .079 | 0.3% | .018 / .48 / 10% |
| L10 | .0076 | .066 | 0.2% | .020 / .46 / 9% |
| L12 | .0071 | .063 | 0.2% | .019 / .40 / 7% |
| L14 | .0071 | .036 | 0.0% | .020 / .42 / 7% |

- T3 is real and about 40% of j1's size, but it is almost orthogonal to what the card misses. It FAILS the pre-registered bar
  (cosine >= .3 with a cut >= 5%) at every layer, and is under the .15 fail line everywhere.
- It would also have been undeployable: exact T3 costs two n³ products per carried column per layer (~30,000 U per layer).
- Scope: the CP part of K3 only. The star and b parts are plane-pattern or Gaussian-like and are covered by j0-j3. With H2 closed, the
  remaining distinct error is not reachable by local second-order born terms.

**O addendum 2: H1 FAILS on k3, both nets (the analyst's subspace-Gaussianized restart, 09-25 01:39-02:36; fable_analyst_v1/
subspace_restart_mc.py, subrestart_r{35,100}_R*_h8M_eval.json, scored with gauss_restart_eval.py).** The true top-R part of the
pre-activation state at l0 was kept (eigenvectors of the 32M covariance), and only the complement was Gaussianized. Errors at L13 / L14,
restart plus the isotropic term:

| l0 = 6 | k4, net 35 | k4, net 100 | k3, net 35 | k3, net 100 |
|---|---|---|---|---|
| R = 0 (the full restart) | .153 / .146 | .156 / .162 | .085 / .077 | .080 / .070 |
| R = 16 | .117 / .104 | .114 / .113 | .091 / .086 | .092 / .086 |
| R = 64 | .092 / .087 | .086 / .089 | .082 / .073 | .078 / .072 |
| R = 256 | .047 / .037 | .038 / .040 | .038 / .031 | .035 / .032 |
| pre-registered bar at R = 64 | <= .11 | <= .11 | <= .03 | <= .03 |

- The top 64 modes' own non-Gaussianity carries about half of the kurtosis-relevant content (k4 falls from .15 to .09) and k3's level,
  but not k3's per-unit spread: k3 stays at .07-.08.
- The analyst's complement test (r35, l0 = 6) Gaussianizes ONLY the top 64 or 256 modes and keeps the rest true. It is as bad as the
  full restart: k4 .149-.154, k3 .072-.086. So the bulk's own higher cumulants give nothing.
- The per-unit k3 spread therefore sits in MIXED cumulants that couple the top modes to the bulk, (top, top, bulk) and (top, bulk,
  bulk). No subspace alone holds it; keeping R = 256 true (k3 .03-.04) works because it keeps more of those mixed terms. This fits
  Tucker's catastrophe for subspace compression of the third order, and explains why the card's k3 error lines up partly with the top
  directions (R² .11-.51) while no subspace carrier can fix it (fable_analyst_v1/RESULT.md section 40).
- So a collective-mode carrier could buy part of the small k4-alone lever (-4.9% exact), but not the k3 part (-8.7%) that the joint
  lever needs.
- With H1 and H2 both failed, the read in section O stands: the rivals' edge is a sum of smaller things our component closures cannot
  isolate, and our path is the bill plus refits after real structural changes. loopR4 (-0.18%) and the c0 line search (flat minimum at
  S24's value) show the dial and refit surface around S24 is exhausted.
