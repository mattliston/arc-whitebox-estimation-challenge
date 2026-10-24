# fable_physicist_v1: fourth-order partner of the (2,1)-plane gain term, and a joint consistent supply

Lane: physicist (2026-09-22, ~18:40-20:00 machine time). Torch research card only (no native runs, no refit, no submission).
Card copy with supply/capture hooks: `model_bank_phys.py` (parent `fable_refit_v1/model_bank_bfix.py`, gamma 0; unmodified path
reproduces r35 1.6484657558908164e-08 bit-exact, re-checked after every hook edit). Truth: 8M MC planes
`public_g19_attribution_v1/mps_planes_rank{0035,0100}.npz` (independent draw `..._noise2.npz` at L2/5/9/12), 8M mean/cov/k3/k4
`mps_cov2_rank*.npz` (same draw), independent 32M per-unit truth `mps_truth_rank*_indep32M.npz`, final targets `truthpost`.
Baselines (torch, final-layer MSE): r35 1.64847e-8, r100 1.68341e-8, r37 1.84683e-8, r39 1.53819e-8.

## Answer in one paragraph

The compensation the lead looked for is NOT in K22/K31/k4. With every input at MC truth, the card's pair map reproduces the true post
covariance at the MC noise floor. On the true state the Sym(C,mu) gain term of the truth K21 is essential to the covariance: removing it
makes the next-layer variance 7x worse. The card's own K21 already carries a read-equivalent stand-in, so adding kappa*Sym double-counts.
The fourth-order lever is real, and it is K31. Truth K31 planes fed to the pair map (L2-14, no refit) cut the final error by 12.1% on r35.
Truth K22+K31 cuts 12.2% on r35 and 14.7% on r100. The gain is certified against an independent draw, and it enters only through the pair
map's covariance output propagated to the next layer. It is broad-spectrum per-pair structure. Every term I could derive is certified
against truth: the gain-on-skewed-base terms, the tree-level Gaussian-born diagrams, a second-order product, slice scales. Each is real
and stable across layers and nets, but together they carry at most ~30% of the needed covariance read. The only affordable piece is a zero-cost
per-row K31 model: -1.4% raw MSE, 40/40 held-out VALID40 wins (torch, no refit).

## Q1. The card's K22 and K31 against MC truth (r35; r100 agrees)

Off-diagonal relative error of the card's plane vs 8M truth, next to the relative noise of one 8M draw (from the independent draw):

| layer | radial | K22 err | K22 noise | K31 err | K31 noise | K31 truth/card scale | K22 scale | k4 scale | rms mu/sd |
|---|---|---|---|---|---|---|---|---|---|
| 2 | .0158 | .176 | .136 | .861 | .656 | .90 | .99 | 1.00 | 1.0 |
| 5 | .0285 | .164 | .076 | .765 | .345 | .85 | 1.01 | 1.01 | 1.7 |
| 9 | .0421 | .209 | .053 | .722 | .174 | .92 | 1.03 | 1.02 | 2.4 |
| 12 | .0521 | .240 | .045 | .703 | .126 | .94 | 1.04 | 1.02 | 2.8 |
| 14 | .0566 | .272 | - | .674 | - | 1.07 | 1.05 | 1.06 | - |

- **Scale:** truth projected on the card's own structure. The (3,1) slice runs 0.84-0.86 of the card at L3-7 and rises to 1.0-1.1 at
  L12-14. The (2,2) and diagonal slices run 0.99-1.06. r100 reproduces these ratios within ~0.03 at every layer.
- **Mean-structured terms:** mu^2 v, mu mu C, mu^2 mu^2 and friends explain at most 12% of the K22 error and 22% of the K31 error.
- **K31 error spectrum:** the top singular pair is left ~ per-unit k3 (cos .87-.91) and right ~ mu (cos .92-.94). It holds 7% (L5),
  18% (L9), 19% (L12) and 28% (L14) of the error energy. The remaining 70-80% is spread over many components and is not noise: one-draw
  noise is 3-6% of the error energy at L9-14.
- **Per-unit k4:** isotropic R2 falls from .99 (L2) to .93 (L14). Adding mu*k3 with coefficient 0.36-0.46 x radial lifts it to .95-.97.

## Pair-map fidelity on the TRUE state (pair_truthstate.py)

Next-layer variance error (rms, vs independent 32M truth) from one pair-map call with 8M truth inputs. The floor is the 8M truth post
covariance itself, pushed: 0.56-0.59e-3.

| variant (r35) | L2 | L5 | L9 | L12 |
|---|---|---|---|---|
| all truth | 0.56e-3 | 0.56 | 0.58 | 0.56 |
| K22, K31, k4 isotropic (truth radial) | 0.56 | 0.58 | 0.63 | 0.67 |
| truth K21 minus kappa*Sym(C,mu) | 0.56 | 1.47 | 3.09 | 3.96 |
| K21 off-diagonal zeroed | 3.20 | 5.45 | 7.24 | 8.97 |
| card's C, truth planes | 0.35 | 0.43 | 0.80 | 1.24 |

On the card's own state at L12 (pair_ablate.py), the card's next-layer variance error is 1.48e-3. It moves to 1.47 with truth K22, 1.45
with truth K31, 1.42 with truth K21 off-diagonal, 4.23 with kappa*Sym, and 4.20 with Sym+K22+K31. The independent draw gives the same
numbers.

## Multi-layer oracle (run_supply.py, oracle_channels.py; truth fed to the pair map only, L2-14)

| supply | r35 | r100 |
|---|---|---|
| truth K31 | **-12.1%** | - |
| truth K22+K31 | **-12.2%** | **-14.7%** |
| truth K22 | -1.4% (raises radial toward truth) | - |
| truth K22+K31+per-unit k4 | -21.4% | -8.7% |
| truth K21 off-diagonal | +430% | +827% |
| everything truth (K21 off, k3, k4, K22, K31) | +129% | +192% |

Attribution of the truth-K31 gain on r35:

- **Channel:** covariance output only, -13.0%. That covariance fed only to the next layer gives -13.0%; fed only to the GNN features,
  +0.1%. The post (2,1) plane gives -0.1%, radial +0.1%, marginal 0.0%.
- **Layers:** L2-4 0.0%, L2-6 -6.5%, L5-8 -6.9%, L7-10 -2.5%, L11-14 -8.2%.
- **Noise certificate (supply at L2/5/9/12 only):** main draw -4.7%, independent draw -5.8%, two-draw average -5.5%, extra noise -3.2%.
  The gain is signal.
- **Rank:** top-k SVD of the truth error gives +0.6% (k=1), -0.8% (3), -4.2% (10) and -9.3% (50). The full plane gives -12.1%.
- **Covariance change:** 0.17-0.24% of the off-diagonal post covariance, broad-spectrum (top singular value 3-10% of its energy). It is
  not explained by a scale, mean-outer or gate-outer structure.

## Q2. Joint consistent model: derivations and certificates

1. **Common gain on a skewed base.** Take h = u z with u independent of z, E u = 1, Var u = s2. To first order in s2:
   Cov = (1+s2) C + s2 mu mu^T; K3 = (1+3 s2) K3_z + 2 s2 Sym3(mu, C);
   K4 = (1+6 s2) K4_z + 4 s2 pairings(C (x) C) + 3 s2 Sym4(mu, K3_z).
   The new K4 term gives K31_ac += (radial/4)(3 mu_a K21_ac + mu_c k3_a), K22_ac += (radial/2)(mu_a K21_ca + mu_c K21_ac) and
   k4_a += radial mu_a k3_a. Exact toy (`toy_gainskew.py`, discrete skewed z, two-point gain): Cov and K3 exact to 1e-14; K4 residual
   scales as s2^2; dropping the Sym4 term leaves 1.64x the whole first-order change. In MC truth these terms are present, but at
   0.2-0.5x the full-radial prediction at L9-14 (the mu_a K21_ac term is 0.8-1.3x at L2-5), so only part of the radial behaves as an
   independent common gain. It is also the source of the
   rank-1 k3 (x) mu error component. As a K31 supply, least-squares fitted per layer, it gives +0.5% (truth inputs) and +1.3% (card
   inputs).
2. **Tree-level Gaussian-born K4 of the ReLU, pushed to K31** (`tree_k31.py`). Star and path diagrams with vertex factors gate, pdf/s and
   -alpha pdf/v, 6 dense products per layer. Against truth the coefficient is 0.98-1.19 at L3-14 with no free parameter. It is 14-19% of
   the truth norm and explains 5-7% of the error energy. Earlier-layer trees transported by the ReLU's linear response
   (`tree_transport.py`) enter with coefficient 0.8-1.0 and raise the explained share to 12-13%. As a card supply (`tree_supply.py`,
   from the card's own state) it gives -1.30 / -4.13 / -0.62 / -0.90% on r35 / r100 / r37 / r39, mean -1.74%.
3. **Second-order product** P1 = 3 (k3_a/v_a)(K21_ac - C_ac k3_a/v_a). Coefficient 0.43-0.53 at every layer, R2 0.25-0.36 of the K31
   error with the truth K21. As a supply: +0.6% (truth K21), +0.3% (card K21).
4. **Per-row K31 factor** (`k31_rowfactor.py`). Fit truth K31 row by row as f_a * radial * v_a * C_ac. Then f_a = a_l + b_l * mu_a *
   k3_a / v_a^2 explains 21-50% of the row-factor variance at L5-14, identically on r35 and r100. Across L3-14, a_l runs 0.71-0.86;
   b_l is 0.92-0.98 at L2-5 and falls to about 0.51 by L14. Alpha, skewness and gate add nothing. This is the regression form
   K31 = (k4_a/v_a) C_ac of the per-unit k4 from item 1, k4_a = radial (v_a^2 + mu_a k3_a); its full-radial prediction is b = 1. Even
   a perfect per-row factor only lowers the K31 error from 0.70 to 0.65 at L12. The per-pair structure dominates.
5. **Slice-resolved scales** (table above). Scaling the (2,2) and diagonal slices up to their truth ratio is strongly harmful:
   +11.5% / +19.4% / +13.6% / +16.5%. It changes the marginal kurtosis the closure is co-adapted to.
6. **Covariance-read space** (`read_align.py`, `read_combo.py`). Each piece above aligns with the truth-K31 covariance read, with
   coefficients near 1. Jointly, the zero-cost pieces explain 13% / 21% / 28% of the read at L5 / L9 / L12, and 18 / 27 / 32% with the
   tree. The top-10 SVD components of the truth error explain 24-48% of the read, yet give only -4.2% as a supply. A usable model needs
   about 0.7+ read cosine. The read-weighted error is not low-rank either (`read_lowrank.py`): once pushed, 10 / 30 / 100 components
   carry 27-50 / 51-73 / 82-93% of the read. It is also spread across units, with the top quarter by read weight holding 48-63%.

## Q3. Supplies in the torch card (no refit)

| variant (L2-14, pair map only) | r35 | r100 | r37 | r39 | mean | extra cost |
|---|---|---|---|---|---|---|
| K31 x 0.9 | -1.08 | -1.34 | -0.18 | -2.55 | -1.29% | 0 |
| K31 per-row (0.85 + 0.5 mu_a k3_a/v_a^2) | -1.27 | -1.61 | -0.19 | -3.03 | -1.52% | 0 |
| measured beta_l schedule | -1.15 | -1.65 | -0.01 | -1.97 | -1.19% | 0 |
| tree term (coefficient 1) | -1.30 | -4.13 | -0.62 | -0.90 | -1.74% | ~6 GEMM/layer |
| K31 x 0.85 + 0.45*P1 (card K21) | -1.58 | -2.05 | +0.89 | -3.53 | -1.57% | 0 |
| per-row (0.75 + 1.0 mu k3/v^2) + 0.5*P1 | -1.59 | -2.06 | +1.60 | -3.54 | -1.40% | 0 |

Scaling the K22 plane by its measured truth ratio (k4 and T31 untouched) is harmful on every net: +7.3 / +15.1 / +9.4 / +11.3%, and
+6.8 / +14.2 / +9.4 / +8.1% on top of the per-row K31. It compounds through the radial recursion.

**The two-part success criterion is met only for the final error.** Per-unit pre-activation variance against the independent 32M
truth (`vcheck_rowmod.py`) does not improve under any K31 variant:

| r35 | L5 | L9 | L12 | L15 |
|---|---|---|---|---|
| base: variance rel-rms | 0.00038 | 0.00081 | 0.00123 | 0.00200 |
| per-row K31: variance rel-rms | 0.00038 | 0.00082 | 0.00124 | 0.00200 |
| base: mean-weighted, rms(pdf/(2 sigma) dv) | 3.25e-5 | 3.79e-5 | 4.65e-5 | 4.71e-5 |
| per-row K31: mean-weighted | 3.26e-5 | 3.82e-5 | 4.70e-5 | 4.72e-5 |

r100 behaves the same way (L15 0.00197 in both). The truth-K31 oracle itself leaves variance slightly worse (r35 L12 0.00124 ->
0.00130, graded against the same-seed 32M truth) while cutting the final error 12%. The K31 gain therefore does not travel through per-unit variance. It travels through the
propagated off-diagonal covariance structure, which the next layers' pair maps and the closure's pre-covariance features read. This is
consistent with the covariance-only split: fed to the next layer, -13.0%.

VALID40 (torch, no refit; rows in `valid40_rowmod.jsonl`, `valid40_rowfit.jsonl`). Held out: every coefficient came from r35/r100
truth, and neither net is in VALID40. Ratio of mean final MSE to the base card's, with a 95% paired bootstrap interval:

| variant (L2-14) | ratio (95% CI) | wins | median | worst |
|---|---|---|---|---|
| K31 x 0.9 | 0.9880 (0.9857-0.9902) | 40/40 | -1.24% | -0.01% |
| **K31 per-row (0.85 + 0.5 mu k3/v^2)** | **0.9857 (0.9828-0.9884)** | **40/40** | **-1.38%** | **-0.10%** |
| per-layer (a_l, b_l) schedule | 0.9838 (0.9794-0.9881) | 35/40 | -1.44% | +0.50% |

The fixed two-number per-row model is the recommended form. The fitted schedule is 0.18% better on average but loses on 5 nets.

## Physics picture

- **Common gain.** A deep ReLU pre-activation vector is approximately a common gain times a skewed, correlated base. The gain is the
  fluctuating norm of the whole activation vector: the input norm, plus each layer's own norm fluctuation. Because ReLU is positively
  homogeneous, a common gain passes through every layer exactly. That is why one scalar radial with pairings(C x C) is the right
  leading fourth-order model, and why the pair map is exact for a gain mixture.
- **Gain times skewed base.** The gain multiplies the base's mean and skewness. K3 picks up 2 s2 Sym(mu, C), the lead's (2,1) term.
  K4 picks up 3 s2 Sym4(mu, K3), which contains the rank-1 k3 (x) mu part of the K31 error and the mu*k3 part of the per-unit k4. Truth
  carries these at a fraction of the full-radial prediction. Only the input-norm gain is exactly independent of the direction; gain
  born inside the network is correlated with it.
- **New structure each layer.** Each ReLU creates non-Gaussian structure from the Gaussian correlations: tree diagrams (stars and
  paths, three covariance edges). They are then carried forward by the next ReLU's linear response (the gates) and the weights, keeping
  roughly 65-90% of their size per layer. They are not isotropic, so the scalar radial cannot hold them. They are certified here with
  coefficient ~1 and no free parameter, and they are a real but modest part (12-13%) of the K31 error.
- **Why K31 matters and K22 hardly does.** The pair map reads K31 only through alpha*pdf of the triple unit times the gate of its
  partner. That read corrects the post covariance for pairs that involve kink-shoulder units, and the correction accumulates through
  covariance -> next-layer variance -> means. The dominant per-pair K31 structure (about half its energy) is beyond lowest-order trees,
  second-order products and gain terms, which points to higher-order (loop) diagrams and their transport.

## Recommendation

- **Stop the Sym(C,mu) K21 supplies.** The card's K21 already has the correct covariance read through a stand-in. Truth K21 planes in the
  co-adapted card are catastrophic (+430% / +827%).
- **K31 is where the no-refit gain lives** (oracle -12% to -15%, signal-certified), but it is per-pair and broad-spectrum. The
  physically derived per-pair pieces (fresh and transported tree diagrams) are certified, yet cost about 6 dense products per layer and
  carry less than a third of the read. One extra 1024^3 product per layer adds about 7.4% to the bill (14 x 2.15e9 FLOPs on a
  4.06e11 bill). A model capturing the full -12% therefore breaks even at about 1.8 products per layer. A fourth-order carrier does
  not pay at this ceiling.
- **Bankable now:** the zero-cost per-row K31 model, -1.4% raw with 40/40 VALID40 wins. Native port: the bank's `_nonlinear_pair_batch`
  (candidate.py line 675; the live loop calls it as `nonlinear` at L1-14) has no `remainder31`. Change line 699 to
  `T31 = (t4 * (0.85 + 0.5 * mu * k3 / (v * v)))[:, None] * rho`. Only T31 changes; t4 in the marginal terms stays. Applying it at
  L1 too is harmless (torch L1-14: -1.26 / -1.69 / -0.14 / -3.05%). Cost is ~6 vector ops per layer, ~84 requests per net (~0.1 s of
  platform wall) and no n x n FLOPs. It needs a native parity check and a paired VALID40 before any fire; it is below the fire bar alone.
- **For a refit line:** the per-pair truth K31 is the supply worth learning toward. Its value comes only through covariance propagation,
  so a refit target would be the post covariance, not the K31 plane.

## Follow-ups (2026-09-22, ~20:00-20:20 machine time)

- **Row-factor constants, TRAIN only** (`train_c0c1_sweep.py`, ranks 40/41/43/44, VALID untouched). Best c0 = 0.8, c1 = 0.5:
  -2.03% mean, 4/4 wins. Neighbours: 0.85 gives -1.79%, 0.75 -1.96%, 0.9 -1.31%, 0.7 -1.66%, 0.6 -0.37% (loses on 3 nets). c1 is
  flat between 0.5 and 1.0. Formula and constants sent to the porter.
- **The analyst's covariance-only K22 row factor adds on top** (fable_analyst_v1/rf_train_v3..v4.log). K31 rows (0.85, 0.5) plus K22
  rows (0.9, 0.35) applied only to the pair map's covariance output: -2.54% mean, 4/4 wins on the same TRAIN nets. Applied to all
  outputs it is +14% to +43%. The combination with c0 = 0.8 is left to the analyst's harness.
- **Old third-order tier at 1/3 cost: negative.** Arithmetic in `oldtier/RESULT_OLDTIER.md`. The tier is 107 dense units over
  L5-14: rebuild 38%, inner product 25%, join 26%, rotation 11%. No row restriction of the rebuild is free (+3% to +39% raw).
  Truncating the whole tier's frame fails the 0.3% bar already at rank 320 (1.0% plane error), and the lost content is not
  gain-type. The best combination is about
  0.9x the tier, a wash on adjusted score.
