# covhunter lane: what injects the carried-covariance error, and can it be fixed cheaply? (2026-09-22)

## Summary for a cold read (2026-09-23)
1. Our estimator carries, layer by layer, the covariance between every pair of units. That covariance is about 1% wrong by layer 13,
   and the error matters: replacing it with the true covariance at every layer cuts the final error by 30% (net 35) and 35% (net 100).
2. The error is created fresh at each layer and is not damped: each layer keeps about 96% of the error it inherits. The new error
   comes from two inputs to the pair formula: the (2,1) third-order plane on units near their kink (about 2/3 of it) and the
   fourth-order K31 plane (about 1/4). The mean, the learned corrections and the other settings add almost none.
3. Supplying the true planes to the covariance output is worth -24% of the final error (both test nets), and -16% when supplied at
   layers 10-12 only. Any expensive plane upgrade should be spent there.
4. The (2,1) plane alone makes things worse at full strength (+4-6%) but better at half strength (-9%). The estimator's (2,1) error
   partly offsets its K31 error, so the two must be fixed together. This was the paradox in the brief.
5. There is no cheap fix. The plane errors have no simple structure, cannot be predicted from the estimator's own features (held-out
   R2 at most .05-.09), and exact planes cost more than the whole budget. Monte Carlo would need about half the budget, at three
   layers only.
6. SHIPPED: five zero-cost settings, found with a fitting method I built for this (one trial run per setting, then a quadratic
   model of the final error). They shrink part of the plane and K31 contributions to the covariance, and turn one learned correction up
   and one down. They cut the final error by 1.7% on 40 held-out nets and by 1.3% on the fired S2 physics. They are in submission #331961.
7. CLOSED: other knob choices (all at their optimum already), reweighting the plane's internal parts (x2-16 worse), per-layer-band
   versions of the settings (-0.37%, 5/12 nets), settings acting only on the last two layers (-0.12%), and a learned final-layer
   correction. The final error is not predictable from final-layer features: held-out R2 is 0, and a control signal at R2 .08 is
   recovered.
8. So the rivals' final-layer advantage is not a final-layer trick. They reach the last layer with a more accurate covariance.
9. Tools left for reuse: all-layer 32M Monte Carlo truth for nets 35/100/37/39, gradients of the final error with respect to every
   layer's covariance (3 min per net), and the fitting method (sections 3, 10, 15).
10. What would move the score from here: a more accurate carrier of the fourth-order (K31/K22) side together with the (2,1) plane,
    spent at layers 10-12.

All numbers come from the torch research card (`fable_refit_v1/model_bank_research.py`, the bank's port), learned parts ON, bank
parameters, no refit. My step copy `covstep.py` (HookCard) reproduces `model.Card` bit-exactly (max abs diff 0.0 over all 16 layers,
re-checked after every edit). "Final MSE" is the layer-15 MSE against the 1e9 truthpost. Ratios are against the unchanged card on the
same net. Nets: r35 and r100 (training nets with 8M truth planes), r37 and r39 (VALID), TRAIN 40/41/43/44 for fitted constants.

## Answer in one paragraph

The carried covariance C is injected by the pair map's third- and fourth-order INPUTS, not by the mean, the learned parts, the
recentering or the response floor. Given an exact C at l-1, one step re-creates an off-diagonal error of .003-.005 (1.2-3.8x the MC
floor), and the propagation keeps about 96% of the relative error per layer, so injections pile up to .005-.0115. By adjoint price
at L7-13, all pair inputs together explain ~100% of the injection; the (2,1) plane read on KINK rows (|mu/sd|<1.5) is the largest
single part, K31 the second, and K22 and per-unit k4 are small. The paradox (true K21 into the covariance does not help) is
OVERSHOOT. The K21-driven part of the final error has a norm^2 of ~45% of the MSE but is anti-correlated with the rest, so at full
strength the fix loses (+4/+6%), while at half strength it gains (-9.6/-8.7%). The exact C is worth -30/-35% (r35/r100) at every
layer and -31/-37% at half strength. Its value flows 100% through the pair map: 20-50% through the per-unit variance into the post
mean, and the rest through the covariance recursion into later variances. The post plane, star, births, radial and features carry
none of it. No cheap structured fix captures the injection: the K21 and K31 corrections are broad per-pair structures, mutually
orthogonal, and not row scalings; the carrier reweighting that fits the plane best is catastrophic. What does pay is a zero-bill
set of five covariance-channel knobs fitted by Gauss-Newton on 18 TRAIN nets: -1.66% on VALID40 on top of the K31 row factor
(32/40 wins), -3.06% vs the bank (38/40), native form certified (section 10).

Where to spend (added 09-23, from section 12): three layers, L10-12, hold two thirds of the plane-driven covariance value (-16% of
-24%), and L10 alone gives -7..-9%. Any costed plane or fourth-order upgrade should therefore be spent at L10-12 (plus L5), not
everywhere. For example, a carrier that costs 30-57% of the bill across L2-14 costs about 7-13% at three layers. Two follow-ups have
already tested this band. The physicist's exact star transport at L10-12 is worse on raw MSE and costs +12% bill: K21 alone overshoots.
The analyst's predictability of the band's plane errors from card features is at most .05-.09 cross-net R2.

## 1. All-layer MC truth factory (`mc_allayers_mps.py`, MPS)
Mean and full covariance of the pre- and post-activations at all 16 layers, with A/B halves, 32M samples, same input stream as
`fable_mc_v1/mc_second_order_mps.py`. GPU float32 sums are flushed to float64 every 16 batches, and normals come from a background
thread. On the same samples, L4/7/10/13 agree with `fable_mc_v1/mc2_rank*_P32M.npz` to 6e-9 relative. Files:
`mcall_rank{0035,0100,0037,0039}_P32M.npz`, 0.54 GB each, 14-18 min per net under load. Noise floor of the off-diagonal post
covariance, |A-B|/2 / |C_off|: .0077 at L0, .0043 at L2, .0024 at L6, .0018 at L10, .0014 at L14. The off-diagonal part grows from
0.73x to 2.9x the diagonal's Frobenius norm with depth.

## 2. Per-layer injection in the card's own closed loop (`inject.py`)
Step l is re-run from the card's captured state at l-1 with the post C (arm C), the post mean (arm m) or both (arm mC) set to truth.
Its carried C is graded at three stages: straight out of the pair map, after the message edge (L8-14), and after the GNN recentering.
r35 and r100 give the same picture. Selected layers, r35:

| layer | noise floor, off-diag | card, accumulated | truth C at l-1: pair map / +edge / +recentering | v_{l+1} rel-rms: floor / card / truth-C arm | mean-weighted dv: floor / card / truth-C arm |
|---|---|---|---|---|---|
| 1 | .0056 | .0057 | .0034 / .0034 / .0035 | .026% / .025% / .015% | 3.9e-5 / 3.8e-5 / 2.3e-5 |
| 4 | .0031 | .0046 | .0029 / .0029 / .0029 | .025 / .038 / .024 | 2.4e-5 / 3.4e-5 / 2.1e-5 |
| 7 | .0023 | .0062 | .0037 / .0037 / .0037 | .025 / .066 / .039 | 1.6e-5 / 3.8e-5 / 2.2e-5 |
| 10 | .0018 | .0080 | .0044 / .0042 / .0042 | .024 / .109 / .060 | 1.1e-5 / 4.2e-5 / 2.4e-5 |
| 13 | .0015 | .0106 | .0047 / .0045 / .0050 | .025 / .169 / .085 | 8.7e-6 / 4.6e-5 / 2.3e-5 |
| 14 | .0014 | .0115 | .0047 / .0046 / .0054 | .024 / .202 / .091 | 8.4e-6 / 4.7e-5 / 2.4e-5 |

- The truth MEAN at l-1 changes the step's C by nothing, at every layer and in every metric. Mean errors do not inject covariance
  error.
- The mean-weighted next-layer variance error, rms(pdf/(2 sd) * dv_{l+1}), is the first-order post-mean error it causes. For the
  one-step injection it is flat at 2.2-2.5e-5 from L1 to L14. The accumulated value is 3.4-4.8e-5, and the post-mean error itself is
  1.0-1.4e-4.
- Error location: the injection lands 28-36% on kink rows (gate .2-.8) and 39-43% on always-on rows (g>.98). The accumulated error
  is 61% on always-on rows at L13-14, because on-units pass covariance errors through with gain 1.
- The edge lowers the step's error slightly at L8-14. The recentering adds Frobenius error at L11-14 (+10-30%) but leaves the
  mean-weighted metric unchanged, and its adjoint price is ~0 (section 4).

## 3. Adjoint of the final MSE (`adjoint.py`, `price_acc.py`)
One checkpointed backward (3.4 GB, ~3 min) gives Gamma_l = dMSE/dCpre_l for l = 1-14 and dMSE/dmu_l. The first-order price of
changing the carried C_{l-1} by dC is <Gamma_l, W_l^T dC W_l>. Finite-difference certificate (C_l -> C_l + eps (T_l - C_l)):

| r35 / r100 | eps .1 exact | eps .1 linear | eps 1 exact | eps 1 linear |
|---|---|---|---|---|
| L4 | -0.87 / -0.63% | -0.89 / -0.75% | -3.98 / -0.65% | -8.89 / -7.46% |
| L10 | -2.91 / -4.50 | -3.15 / -4.77 | -8.24 / -20.82 | -31.45 / -47.68 |
| L13 | -2.72 / -3.31 | -2.92 / -3.52 | -9.60 / -13.77 | -29.18 / -35.22 |

The adjoint is right, but the response is strongly QUADRATIC: MSE(eps) = MSE0 (1 + a eps + b eps^2) with b = .19-.26 at L10-13. The
C_13 error contributes a final-error vector with norm^2 ~20% of MSE, at cos -0.33 (r35) to the total error. First-order prices
overstate full-strength values by 2-4x. Every "value" below is therefore a closed-loop run, and prices are used only for attribution.

## 4. Attribution of the one-step injection (`attrib.py`, `knob_price.py`)
With truth C at l-1, ONE input of step l is swapped. The table gives the first-order price of the change of the carried C, in % of the
final MSE. The "injection" row prices removing the whole one-step error.

| arm (r35: L4 / L7 / L10 / L12 / L13) | r35 | r100 |
|---|---|---|
| remove the whole injection | -6.7 / -7.1 / -14.2 / -10.7 / -8.6 | -3.7 / -9.8 / -13.2 / -11.1 / -10.0 |
| all pair inputs to truth (K21 off, k3, K22, K31, k4, mu) | -2.4 / -8.5 / -17.6 / -12.5 / -8.7 | -1.6 / -9.0 / -15.2 / -11.5 / -10.3 |
| K21 off-diagonal, kink rows | -0.4 / -5.7 / -8.8 / -8.6 / -2.4 | -2.0 / -5.0 / -8.0 / -4.6 / -7.1 |
| K21 off-diagonal, other rows | +0.4 / -1.7 / -0.5 / +0.4 / +0.1 | -0.2 / +1.3 / -1.5 / -1.2 / 0.0 |
| K31 plane | -0.6 / -1.6 / -6.0 / -3.3 / -5.5 | +0.7 / -3.6 / -4.6 / -5.1 / -2.6 |
| K22 plane | -1.8 / -0.3 / -1.2 / -0.2 / -0.4 | +0.1 / -1.2 / -0.2 / -0.6 / -0.2 |
| per-unit k4 | 0.0 / +0.4 / -1.4 / -1.3 / -0.4 | +0.2 / -0.5 / -0.8 / +0.2 / -0.2 |
| pre mean mu | -0.1 / +0.1 / -0.1 / -0.2 / 0.0 | +0.2 / -0.2 / -0.2 / 0.0 / +0.2 |
| message edge removed | 0 / 0 / +1.3 / +0.6 / +0.3 | 0 / 0 / +1.2 / +1.5 / +0.1 |
| recentering removed | -0.5 / +0.8 / -0.4 / -0.2 / -0.6 | +0.1 / +0.1 / -0.4 / -1.1 / -0.1 |
| response floor .005 | -0.1 / 0.0 / -0.3 / 0.0 / -0.2 | -0.1 / +0.6 / +0.1 / -1.0 / -0.1 |

On the card's own state, summed over L2-14 (`knob_price`), the truth fixes cost: kink-row K21 -50/-52%, K31 -40/-41%, K22 -5/-7%. The
signs are negative at every layer on both nets. Generic directions do not behave like this. Doubling the kink-row K21 read costs
-49..+120% per layer, and doubling the atoms read -36..+117%, with signs that alternate by layer and differ between nets. These are
"stiff" directions of the co-adapted card.

## 5. Closed-loop value of truth supplies, covariance output only (`cl_pairswap.py`)
Input = card + eps (truth - card), 8M planes, fed only to the pair map's covariance call.

| supply | r35 | r100 |
|---|---|---|
| K21 kink rows, L10 only, eps .1 / 1 | -0.8 / -2.9% | -0.7 / -2.8% |
| K21 kink rows, L6-14, eps .25 / .5 / 1 | -7.7 / -9.6 / +4.1% | -7.3 / -8.7 / +6.2% |
| K31, L2-14, eps .5 / 1 | -13.2 / -13.0% | -14.2 / -15.5% |
| all pair inputs, L6-14, eps .5 / 1 | -25.6 / -23.8% | -27.0 / -20.6% |

The quadratic model reproduces every K21 row to 0.2 points: price a = -.42 (L6-14 sum), curvature b = .45/.48, optimum eps ~.45.
Resolution of the paradox: the kink-row K21 read IS the largest injector, but its contribution to the final error is anti-correlated
with the rest of the card's error, so the full fix overshoots.

The overshoot is PHYSICS-level, not co-adaptation of the learned closure. With the GNN mean head and the edge output zeroed
(`LEARNED_OFF=1`, base MSE 2.04e-8 on r35 and 2.16e-8 on r100), the same supplies give:
- kink-row K21, eps .5 / 1: -4.1 / +10.2% (r35), -6.4 / +5.7% (r100). It still overshoots.
- K31, eps 1: -20.6% / -20.4%. The learned closure absorbs 5-8 points of this when it is on.
- all pair inputs, eps 1: -22.2% / -25.7%.
Fixed together with K31, K22 and k4 (the 'pair' row), K21 no longer overshoots. So the card's K21-read error offsets the effect of
its other pair-input errors on the final error, even though the one-step covariance corrections are orthogonal (section 8).

## 6. Value of the exact carried C at every layer (`swap_closedloop.py`)
Carried C after step l -> 32M truth, one layer at a time. Final-MSE change, r35 / r100:

| L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| +3.3/+0.5 | +0.6/+0.6 | 0.0/+0.3 | -4.0/-0.7 | -6.6/-4.4 | -10.2/-5.8 | -13.1/-5.3 | -12.7/-12.2 | -9.1/-16.0 | -8.2/-20.8 | -10.5/-16.6 | -12.7/-17.5 | -9.6/-13.8 | -5.6/-11.1 |

All layers: -29.6 / -35.3%. All layers at half strength (C + .5 (T - C)): -31.1 / -36.9%, and .75 gives the same. All layers plus one
extra independent 32M noise draw: -22.0 / -26.2%, so the truth's own noise costs ~8 points and L1-3 sit at the noise floor.
C@4+7+10+13 reproduces the lead's .7645 / .7857.

## 7. Where the value of C flows (`adjoint_consumers.py`, `adjoint_pairout.py`)
Separate zero leaves on each consumer of Cpre_l split the first-order price of the truth swap of C_{l-1}. The total matches
`price_acc` to 0.1 point.
- The pair map's use of Cpre carries 100%. Per-unit v, the 2 C b term of K21, K22 = radial(vv' + 2C^2)/3, the born atoms
  bp = g Cpre, _bd_update and the GNN features/edge pre_cov carry |price| <= 0.2 points each.
- Inside the pair map, the post covariance, which feeds the next layer's Cpre, carries 50-90%: r35 -7.0/-19.5/-17.2/-25.8/-14.5 and
  r100 -7.5/-24.8/-34.6/-32.6/-18.4 at L5/8/10/12/14. The post mean host and aux (per-unit variance) carry the rest: r35
  -1.9/-5.3/-9.4/-9.4/-14.6, r100 +0.1/-1.3/-4.9/-12.0/-16.9. The post plane (target -> star, births) and radial carry ~0.
So the carried covariance matters only as the source of later per-unit variances. The third-order machinery does not consume its
error.

## 8. Structure of the injection (`pair_combo.py`, `k21_decomp.py`)
- The K21-kink and K31 truth corrections of the one-step covariance are .0031-.0042 and .0017-.0024 of |C_off|. They are nearly
  ORTHOGONAL: cos -.02 to -.10, with the same values in the next-layer pre-covariance metric. There is no cancellation between
  them at the covariance level.
- A per-unit row scaling (x_a + x_c) C_ac explains 1-8% of the K21 correction's energy, 2-18% of K31's and 1-9% of the joint one.
  Its price is ~0. The corrections are broad per-pair structure, as the physicist found for K31.
- The kink-row plane against the 8M truth has error .26 at L4 and .43-.51 at L7-14. A least-squares fit K21t ~ a*atoms + b*star +
  c*bterms has UNIVERSAL coefficients (r35 = r100 to ~.03): a falls .9 -> .55, b rises 2.3 -> 5, c rises 1.5 -> 2.5 from L4 to L14,
  and the residual drops to .31-.36. In the covariance call only, closed loop, this reweighting is CATASTROPHIC: eps .5 gives x2.2
  (r35) and x2.4 (r100), b-terms x1.5 give x2.1/x2.8, star x3 x1.6/x1.7, atoms x.8 x3.8/x4.4. Frobenius accuracy of the plane is
  not the value metric.

## 9. Zero-cost knobs (`knob_closedloop.py`, `edge_extend.py`, `gn_knobs.py`)
- Single knobs, 4 nets (35/100/37/39), mean change: recentering alpha 0 +3.5% (4/4 worse), alpha .5 +0.1%, off-diagonal recentering
  removed +3.7%, diagonal removed +0.4%, edge off +5.1%, edge x.5 +1.7%, response floor .005 +0.4%, .0025 +1.3%, omitted-radial
  source off +27%. All sit at a co-adapted optimum.
- The frozen message edge also applied at L4-7: -3.1/-2.2/-0.3/+6.3% (35/100/37/39); at L1-14 +7.2% mean. Not robust.
- Gauss-Newton screen: one finite-difference run per knob gives the final-error response, and the quadratic model is minimized.
  Ten knobs were fitted on TRAIN 35/100/40/41/43/44. The model is accurate: predicted vs actual on the fitted nets .952/.952,
  .974/.980, .932/.936, .961/.954, .983/.985, .919/.917. But the pooled 9-knob setting gives only -1.5/-0.5% on held-out 37/39
  (in-sample mean -4.6%; leave-one-net-out mean predicted -3.4%).

## 10. Validated zero-bill lever: covariance-channel knobs on top of the K31 row factor (`gn2_knobs.py`, `gn_select.py`)
Base: the physicist's K31 row factor as in the fire candidate, T31 = radial rho (.85 + .5 mu_a k3_a/v_a^2) at L1-14 in all pair
outputs. Knobs were fitted by Gauss-Newton on 18 TRAIN nets (35, 100, 36, 38, 40, 41, 43-47, 49-55); every knob subset was scored
by leave-one-net-out held-out ratio. The best zero-bill subset (LOO .9858, 16/18):

| knob (layers) | value | meaning |
|---|---|---|
| iso (L2-14, covariance only) | -0.094 | kink rows (abs(mu/sd) < 1.5): K21_ac += -0.094 k3_a C_ac / v_a |
| k21n (L2-14, covariance only) | -0.028 | non-kink rows: K21_ac x 0.972 |
| k31 (L2-14, covariance only) | -0.212 | T31 (with the row factor) x 0.788 |
| edge (L8-14) | +0.164 | message edge increment x 1.164 |
| rc (L1-14) | -0.080 | recentering x 0.920 |

VALID40 (torch, no refit, held out, n = 40):

| comparison | mean-MSE ratio | 95% bootstrap CI | wins | median | worst |
|---|---|---|---|---|---|
| row factor vs bank | 0.9858 | .9829-.9886 | 40/40 | -1.32% | - |
| knobs vs row factor | 0.9834 | .9789-.9883 | 32/40 | -1.52% | +2.26% |
| knobs + row factor vs bank | 0.9694 | .9644-.9746 | 38/40 | - | - |

Earlier variants: a 6-net fit that also scaled K22 (x0.938) gave 0.9869, 30/40, worst +3.9%. The frozen message edge also applied at
L4-7 (x0.316) gave 0.9777, 36/40, but that edge costs ~0.55% bill (0.14%/layer), so in adjusted terms it is the same as the zero-bill
set.

Native form, certified (`cert_delta.py`, r35 L2/5/9/12/14). The pair series is linear in T21 and T31, so the covariance-only knobs
are one closed-form delta on the pair map's covariance output, matching the second-call covariance change to 0.8-1.5e-4 relative
(float32 series rounding) with the diagonal untouched:
  dcov = s s^T * [(dT21*t21a + dT21^T*t21b)/2 + (dT31*t31a + dT31^T*t31b)/6], zero diagonal, where
  dT21 = dK21/(v_a s_c), dK21 = -0.094 kink_a k3_a C_ac/v_a - 0.028 (1-kink_a) K21_ac, dT31 = -0.212 T31_row,
  t21a = A2 A1^T + rho*A3 A2^T + rho^2/2*A4 A3^T,  t21b = A1 A2^T + rho*A2 A3^T + rho^2/2*A3 A4^T,
  t31a = A3 A1^T + rho*A4 A2^T + rho^2/2*A5 A3^T,  t31b = A1 A3^T + rho*A2 A4^T + rho^2/2*A3 A5^T,
  with A_r = derivatives(alpha, pdf, gate, gaussian_mean/sigma, 6) (A1 = gate, A2 = pdf, A3 = -alpha pdf, ...), rho = Cpre/(s s^T) with
  zero diagonal, v = diag(Cpre), s = sqrt(v), k3 = diag(K21).
The cost is about 30 elementwise n^2 ops per layer (<= 0.1% of the bill), with no extra pair-map call.

## Verdict
- Mechanism: the carried-covariance error is injected by the pair map's plane inputs (kink-row K21 read > K31 >> K22, k4) and
  compounds at ~96% per layer. Its value (-30/-35%) flows only through later per-unit variances. The K21 paradox is physics-level
  overshoot, resolved by partial strength.
- Cheap structure: none. The corrections are broad per-pair, mutually orthogonal, and not row scalings, and fitting the plane
  (Frobenius) is catastrophic.
- Lever: a zero-bill set of five covariance-channel knobs, fitted by Gauss-Newton on TRAIN, gives -1.66% on VALID40 on top of the
  K31 row factor (32/40), -3.06% vs the bank (38/40). This is just under the 2% bar in mean and above it in consistency.
- Big lever, not cheap: truth pair inputs at half strength are worth -26/-27%, and the exact C -30/-37%. A learned per-pair
  covariance corrector trained end-to-end (the message-edge architecture at all layers with K31-type inputs) is the route I would
  test next; it is a refit, not a zero-cost change.


## Files (all in fable_covhunter_v1/)
mc_allayers_mps.py, mc_certify.py, covstep.py (HookCard), inject.py, adjoint.py, price_acc.py, attrib.py, knob_price.py,
cl_pairswap.py, swap_closedloop.py, k21_decomp.py, cl_carrier.py, pair_combo.py, adjoint_consumers.py, adjoint_pairout.py,
knob_closedloop.py, edge_extend.py, gn_knobs.py, gn2_knobs.py. Results: inject_r*.json, adjoint_rank*.npz, price_acc_rank*.json,
attrib_v1.jsonl, knob_price_rank*.json, clpair_v1.jsonl, swap_v1/v2.jsonl, k21_decomp_rank*.json, clcarrier_v1.jsonl,
pair_combo_rank*.json, adjoint_consumers_rank*.json, adjoint_pairout_rank*.json, knobs_v1.jsonl, edgeext_v1.jsonl, gn_v1.npz,
gn2_v1.npz.

Resource note: the peak RSS was 7.0 GB, in `swap_closedloop.py`, which reads the 0.54 GB truth npz on every access. Cache the arrays
before reusing it. Every other job stayed at or below 3.4 GB. The MPS chain ran 20:12-21:12; MPS has not been used since.

## 11. Knobs on the S2 card and the native patch (2026-09-23)
- S2 torch card: the analyst's s2card physics (K31 row .85/.5 L1-14, gain b-fix gamma .1, every3 joins) on my hook card (`covstep_s2.py`),
  bit-exact vs `fable_analyst_v1/s2card.py` (max diff 0.0; r35 1.6120e-8). v3 knobs vs S2 on 35/100/37/39 plus 8 VALID nets: pooled
  0.9870 (-1.30%), 9/12 wins (35 -2.5, 100 +0.1, 37 -1.8, 39 0.0, 42 -1.4, 48 +0.7, 59 -0.8, 63 -2.5, 66 -2.5, 67 -0.7, 71 -4.0,
  73 +0.6; `s2_v3knobs_eval.jsonl`).
- Gauss-Newton refit on the S2 card (18 TRAIN nets, `gn3_knobs.py`, `gn3_s2_v1.npz`). With the five knobs only, it gives iso -.085,
  k21n -.024, k31 -.176, edge +.187, rc -.059, and a leave-one-out ratio of .9885. The v3 constants score .9863 on S2's quadratic
  model over the same nets, so I keep v3. Freeing the b-fix gamma and the K31 row c0/c1 adds ~0.3 pt leave-one-out, but only through
  a correlated valley (covariance-only K31 x.38 against c0 +.38). That is fragile, and I do not recommend it.
- Native build `build_covknobs.py` (count-asserted edits of the live `_predict_binary` plus an appended `_covknob_delta`). Scored with
  `score_variant.py`:

| build | r35 MSE | r35 util | r100 MSE | r100 util |
|---|---|---|---|---|
| S2 (fired) | 1.612368e-8 | .18575 | 1.576382e-8 | .1835 |
| null (0,0,0,1,1) | 1.612368033799512e-08, bit-exact | .18575 | - | - |
| v3 knobs | 1.574340e-8 (-2.36%; torch -2.48%) | .18618 (+0.23%) | 1.577986e-8 (+0.10%; torch +0.09%) | .18392 |

## 12. Per-layer injection map, mean-orthogonal (lead's request; `attrib_perp.py`, `cl_planeband.py`)
One-step attribution. Step l starts from the true m and C at l-1. P = I - hh^T/|h|^2, h = the pair map's post mean. The share is the
fraction of the orthogonal injection ||P dC_total P||_off removed by setting input X to truth (signed). Values are for r35; r100 agrees
to about .03.

| layer | injection, off-diag rel | orthogonal fraction | 32M noise / orthogonal injection | K21 all rows | K21 kink rows | K31 | K22 | k3, k4 | K21+K22+K31 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | .0028 | .99 | 1.51 | .62 | .61 | .19 | .11 | 0 | .92 |
| 5 | .0033 | .87 | .94 | .68 | .64 | .24 | .07 | 0 | .99 |
| 8 | .0038 | .78 | .67 | .69 | .64 | .27 | .05 | 0 | 1.00 |
| 10 | .0044 | .77 | .52 | .70 | .65 | .27 | .05 | 0 | 1.01 |
| 12 | .0046 | .76 | .43 | .70 | .66 | .27 | .04 | 0 | 1.01 |
| 14 | .0047 | .75 | .38 | .67 | .64 | .30 | .03 | 0 | 1.01 |

The K21 truth fix changes the covariance mostly in the ORTHOGONAL subspace: 69-99% of its Frobenius norm. It removes about 2/3 of the
orthogonal one-step injection, K31 about 1/4 and K22 under 1/10. The three planes together remove all of it. The mean-aligned structure
of the K21 plane error does not reach the covariance, because the read weights d_a g_c filter it out.

Closed-loop value by layer set. The truth planes K21+K22+K31 change only the covariance output, projected onto P, at eps 1:

| layers | r35 | r100 |
|---|---|---|
| 2-14 (reproduces the lead's .762/.746) | -23.8% | -25.4% |
| 2-5 | -1.9 | +2.1 |
| 6-9 | -11.4 | -2.9 |
| **10-12** | **-16.0** | **-16.1** |
| 13-14 | -5.2 | -9.4 |

Single layers, r35 / r100 (L2-4 are at the 8M plane noise):

| L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | L13 | L14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| +3.4/+4.0 | -1.8/0.0 | +1.1/+1.7 | -5.4/-6.0 | -6.2/-0.8 | -1.6/-0.9 | -0.8/-1.5 | +0.3/-5.3 | -8.6/-7.2 | 0.0/-4.7 | -5.6/-5.5 | -2.5/-5.7 | -1.3/-3.7 |

The value is concentrated. Three layers, L10-12, hold 2/3 of the all-layer plane value on both nets (-16%). L10 alone is worth -7..-9%,
and L5 alone -5..-6%. Band values are sub-additive: the four bands sum to -35%/-26% against -24%/-25% for all layers together.
Practical reading: an exact plane supply at L10-12 only would pay about -16%. But the only exact sources are Monte Carlo (100k samples
through 12 layers is ~1e12 flops, half the budget, with ~9x the 8M plane noise) or the full third- and fourth-order tensors. Neither
fits the bill, so the band structure shows WHERE a costed carrier upgrade would pay, not that one exists.

## 13. Kink-row carrier scales per layer band, on top of S2 + v3 knobs (lead's request; `gn4_carriers.py`, `gn4_car_v1.npz`)
Knobs added to the covariance call's kink-row K21, each as theta * carrier in bands E = L2-7 and L = L8-14: the atoms read cp, the star
sp, and the b terms bt. Gauss-Newton fit on 18 TRAIN nets, scored by leave-one-net-out (steps: cp .01, stiff; sp .2; bt .05).
- Best subset (cpE -.012, cpL -.005, spE -.117, spL -.108, btE +.036): LOO .9969 (-0.3%), 12/18 wins, worst +2.2%.
- No single carrier knob beats LOO .9989.
Closed: the carrier split adds nothing beyond the v3 knobs.

## 14. Final-layer lane (lead's step 0 and the L14-15 knobs; 2026-09-23)
Card: S2 + v3 covariance knobs, i.e. the fired S4 physics, in torch.

**Step 0: is the per-unit final error predictable?** (`final_dump.py`, `final_fit.py`, `final_fit_posctl.py`; data in `final_dump/`)
Nets: 27 TRAIN and 16 VALID. Per unit: e_i = clamped pred15 - truth15, target e_i/sigma_i. There are 29 card-only features: alpha,
gate, pdf, min(g,1-g), kink; skew in total and in its bank/star/b-term parts; radial; log sigma; host, gaussian-mean and host-deviation
over sigma; the final GNN delta/sigma; post v/k3/k4; the diag vs off-diagonal share of v_i; the energy and variance loadings of w_i on
the top-16/64 eigenvectors of C_14; sum w^1..4; the m14 mean; the top eigen share of C14.
The metric is R2_0 = 1 - sum(e-ehat)^2 / sum e^2, i.e. the final-MSE cut.

| model | TRAIN in-sample | TRAIN leave-one-net-out | VALID held-out | VALID nets improved |
|---|---|---|---|---|
| ridge (alpha 1e4) | +0.005 | +0.003 | -0.0001 | 9/16 |
| GBM depth 3, 200 it | +0.006 | - | -0.0002 | 9/16 |
| GBM depth 4, 400 it | +0.009 | - | -0.0008 | 9/16 |
| GBM on the raw-e target | - | - | -0.0005 | - |

The per-net bias share mean(e)^2/mean(e^2) is 0.4% on average (max 2.3%). Positive control: a known signal injected at R2 .08-.09
is recovered on VALID (linear by ridge .084, nonlinear by GBM .050). The null is therefore real: under 1% of the final error is
per-unit predictable from final-layer features. The final hook already took what the c2 card had (~10%).

**L14-15-only zero-bill knobs** (`gn5_final.py`; the state entering L14 is cached, so the knobs cannot touch L0-13; Gauss-Newton on
18 TRAIN nets). Knobs: the k3_15 parts (bank, star, b term), k4_15, v15 overall and kink-only, the final delta, and L14-own covariance
constants (iso, k31, k21n).
- Best subset LOO: .9988 (-0.12%), 11-14/18 wins. Final delta x0.87 alone: -0.11%.
- The v15 and kink-v15 scales fit to 0 (no systematic bias), while a 0.2% scale moves the MSE +23-38%.
- CLOSED, far below the -1.5% bar.
Reading: the rivals' final-layer edge is neither per-unit-local nor a scale of the final-layer quantities. With the final map exact
(physicist), it has to come from a more accurate state entering L15: v15 is the pushed off-diagonal C14 error, i.e. the covariance
recursion of sections 2-8.

## 15. Band knobs: the covariance-knob family is exhausted (lead's request; `gn6_bands.py`, `gn6_fit.py`, 2026-09-23)
iso, k21n and k31 were split into four bands (L2-5, L6-9, L10-12, L13-14), with edge and recentering left global: 14 deviations from v3.
They were fitted by Gauss-Newton on 18 TRAIN nets, with a ridge pull toward v3 whose strength was chosen by leave-one-net-out. Each
finite-difference run restarts from the cached state entering the knob's first layer (the base reproduces S2+v3, r35 1.572086e-8).

| knob set | leave-one-net-out (18 TRAIN) | wins | in-sample |
|---|---|---|---|
| all 14 | .9973 | 10/18 | .9931 |
| 12 band knobs | .9970 | 11/18 | .9932 |
| L10-12 + edge/rc | .9992 | 12/18 | .9984 |
| L10-14 | .9990 | 12/18 | .9983 |

Validation of the all-14 set (deviations: iso b1 -.081, b3 +.041, b4 +.023; k21n b2 +.022, b3 -.017; k31 b1 +.034, b4 +.013; edge
+.022; rc +.011; the rest small) on 35/100/37/39 plus 8 VALID nets against S2+v3:
- pooled 0.9963 (-0.37%), 5/12 wins;
- per net: 35 +0.71, 100 +0.18, 37 +0.10, 39 +1.36, 42 -1.08, 48 +0.16, 59 -0.77, 63 +0.02, 66 -1.21, 67 +0.90, 71 -2.81, 73 -0.78%.
This FAILS the bar (-0.8% with 9/12 wins). The zero-cost covariance-knob family is exhausted at the global v3 set: per-band freedom does
not transfer, even though the plane value is concentrated in L10-12.

## 16. Knob re-fit on the S6b physics (refit literals; lead's request; `gn7_s6b.py`, `gn7_fit.py`, 2026-09-23)
My torch S6b is the S2 hook card plus the v3 knobs, the L15 k3 row cut (zc 3.5, the physicist's term32) and the refit epoch-2 literals
(params_epoch2.pt: GNN head/head_b, A, cstar and all edge parameters changed). Parity vs the fired native S6b (#332005): r35
1.56428e-8 vs 1.56537e-8 (-0.07%), r100 1.57334e-8 vs 1.57023e-8 (+0.20%).
The five v3 knobs were re-fitted as deviations by Gauss-Newton on 18 TRAIN nets, with a ridge pull toward v3 and every subset
scored by leave-one-net-out:
- Best subset: edge deviation -0.104, i.e. edge x1.060 instead of x1.164 (the refit's edge literals took over part of the knob).
  Leave-one-out .9996 (-0.04%), 7/18.
- Every other deviation fits to ~0: iso +.0002, k21n -.001, k31 0, rc -.0003.
- 12-net check of edge -0.104 against S6b: pooled 0.9991 (-0.09%), 10/12 wins (35 -0.02, 100 -0.05, 37 +0.59, 39 -0.51, 42 -0.04,
  48 -0.49, 59 -0.12, 63 -0.40, 66 -0.01, 67 +0.26, 71 -0.13, 73 -0.21%).
This FAILS the bar (-0.4% with 9/12): the wins are there, the size is not. v3 stays optimal on S6b, and no v4 JSON is needed.

## 17. Price list on the CURRENT card, S6b (lead's request; `price_s6b.py`, `price_table.py`, rows in `price_s6b_v1.jsonl`; 2026-09-23)
Card: the analyst's torch S6b (`fable_analyst_v1/s6bcard.py`: S2 physics + v3 knobs + L15 row cut + refit epoch-2 literals), learned
parts ON. My copy of its covariance hook reproduces it bit-exactly on all 4 nets. In closed loop, ONE channel at ONE layer band is moved
toward truth: value = card + eps (truth - card). Each cell gives the final-MSE change vs S6b at **eps 0.5 / eps 1.0**, as the mean of the
per-net ratios. Truths: 32M all-layer MC for a/b/c (nets 35/100/37/39); 8M planes and 32M per-unit cumulants for d/e/f/g (nets 35/100).

| channel moved to truth (nets) | L2-5 | L6-9 | L10-12 | L13-14 | L15 |
|---|---|---|---|---|---|
| a post mean m_l (4) | +3.2 / +5.2 | -7.5 / -6.8 | -29.5 / -27.2 | -58.5 / -58.0 | output itself |
| b post covariance C_l, off-diagonal (4) | -5.4 / -6.1 | **-16.4 / -14.4** | **-21.4 / -20.1** | -14.0 / -13.2 | not used |
| c per-unit pre variance v_l (4) | -1.1 / -0.2 | -5.6 / -5.5 | -9.3 / -9.5 | -10.5 / -10.2 | -7.6 / -7.9 |
| d K31 read, covariance output only (2) | -1.9 / -2.2 | -4.7 / -5.6 | -3.6 / -3.8 | -2.6 / -3.6 | no pair map |
| e K21 kink-row read, covariance output only (2) | -2.8 / -0.1 | -4.5 / +1.6 | -5.5 / -2.5 | -2.8 / -2.3 | no pair map |
| f per-unit k3 and k4, all pair outputs (2) | -0.0 / +1.1 | +0.6 / +5.6 | -4.8 / -1.7 | -7.6 / -7.7 | -6.0 / -6.5 |
| g all three planes (K21+K22+K31), covariance only, projected off the mean (2; = sec 12's channel) | -4.4 / -0.8 | -9.8 / -6.1 | -12.3 / -13.8 | -5.8 / -6.8 | no pair map |

Channel g compared with S2 (sec 12, eps 1, r35 / r100): L10-12 goes from -16.0/-16.1 to -14.4/-13.1 on S6b, and L6-9 from -11.4/-2.9 to
-10.8/-1.4. The knobs and the refit absorbed only ~2-3 points of the plane value. Caveat for a at L2-5: the 32M truth mean carries
~9e-5 of noise per unit, which is comparable to the card's early mean error, so that cell injects noise (+3-5%).

Where the lever sits now:
1. S6b's final error is carried mean error (the exact mean is worth -58% at L13-14 and -28% at L10-12). The mean is the estimand, so
   the actionable question is which input still corrupts it, and that input is the carried off-diagonal covariance at L6-14. Exact
   covariance is worth -21% at L10-12 alone, -15% at L6-9 and -14% at L13-14. The per-unit variance it feeds is worth -9..-10% at
   L10-14 and -8% at L15.
2. No single plane channel holds that value any more: K31 is -2..-6% per band, the K21 kink read up to -5.5% at half strength (it still
   overshoots at full strength in L6-9), and per-unit k3+k4 -6..-8% only at L13-15. The three planes together still give -12..-14% at
   L10-12, so the covariance lever needs the (2,1) and fourth-order planes fixed jointly, as on S2.
3. So the refit, every3 and the knobs did not move the lever; it still sits at L6-14. The price list points to a second-order carrier
   redesign aimed at the L10-12 covariance (with L6-9 next). An exact carried covariance there is worth ~20%, against ~13% for the
   joint truth planes at the same band. Final-layer or single-plane work can reach at most -6..-10%.

## 18. Homogeneity check: is the (2,1) plane's Sym content the exact input-radial term? (lead's request; `cert_homog.py`, `homog.py`, `homog_oracle.py`)
Physics. For a bias-free ReLU net and x ~ N(0, I_n), every layer's pre-activation is z_l = r g_l(xhat), with r = |x| independent of xhat.
Hence kappa3(z_l) = E r^3 kappa3(g_l) + (1/E r^2) Sym(C_l (x) mu_l) + O(mu^3/n^2), with c3 = 1/E[r]^2 = 9.7704e-4 at n = 1024
(1/n = 9.7656e-4), so 6c3 = 5.862e-3. The identity is certified by a tiny Monte Carlo (n = 16, two layers, 4M samples): residual 0.7% of
|kappa3| against 0.23% sampling noise. The card's b recursion carries b = c3 m exactly iff radial = 6c3.
Measured on the torch S6b card, r35 / r100, with 32M truths and 8M planes:

| layer | card radial / 6c3 (r35 / r100) | card b_pre vs c3 mu: cos, scale | TRUE K21 = Sym(C (x) b_fit): share of plane energy | b_fit / (c3 mu_true): cos, scale | b_fit / card b | pure c3 Sym(C (x) mu) share |
|---|---|---|---|---|---|---|
| 2 | 2.7 / 2.8 | .85, 0.86 | .55 / .53 | .95, 4.5 | 4.0 / 3.8 | .19 |
| 4 | 4.2 / 4.6 | .91, 1.8-2.0 | .70 / .70 | .97, 6.2-6.7 | 3.1 / 3.0 | .18-.20 |
| 6 | 5.6 / 6.0 | .93, 2.3-2.6 | .79 / .77 | .97, 7.5-7.8 | 3.0 / 2.8 | .18 |
| 8 | 6.8 / 7.1 | .95, 2.7-3.1 | .82 / .83 | .98, 8.5-8.8 | 2.9 / 2.7 | .17 |
| 10 | 7.7 / 8.3 | .94-.96, 3.1-3.5 | .82 / .85 | .98, 9.3-9.6 | 2.8 / 2.6 | .16 |
| 12 | 8.9 / 9.3 | .95-.96, 3.4-3.7 | .85 / .87 | .98, 9.5-10.1 | 2.7 / 2.6 | .16 |
| 14 | 9.7 / 10.2 | .96, 3.7-4.0 | .87 / .89 | .98, 10.2-10.3 | 2.7 / 2.5 | .16 |

- (1) The card's radial is 1.6-10x the exact input-radial value 6c3, and the ratio grows with depth.
- (2) The card's b is aligned with mu but sits at 0.9-3.9x c3.
- (3) The TRUE plane's Sym(C (x) b) content is much larger than the input-radial term: b_fit is 4.5-10x c3 mu (about 1.0-1.7x
  radial/6), while the pure input-radial term explains only 16-20% of the true plane energy. So most of the Sym content comes from
  angular, ReLU-generated common-gain structure, not from the input radius. The card's b is too SMALL, at 25-40% of b_fit, not too big.
- (4) Closed loop, learned ON, b changed at L1-14 (b -> b + eps (target - b)); final-MSE change vs S6b over 35/100/37/39:

| b oracle | eps .25 | eps .5 | eps 1 |
|---|---|---|---|
| b -> c3 m (exact input-radial) | - | +13.3% (0/4) | +20.4% (0/4) |
| b source radial -> 6c3 (split) | - | +4.5% (1/4) | +19.1% (0/4) |
| b -> (radial/6) m (toward the true Sym scale) | +33.2% (0/4) | +48.8% (0/4) | +57.6% (0/4) |

Verdict: the input-radial term is exact but small, and the card's b sits at a sharp co-adapted optimum in both directions. Nothing
zero-cost ships from here without a refit, and the earlier heads-only refit of the b-fix recovered only 4 of 32 points. The b-fix
gamma .1 already in S2/S6b is the only b move that paid.

Addendum to §18 (lead's follow-up):
- **beta_fit against radial/6.** beta_fit (the true-plane Sym coefficient) divided by the card's radial/6 is smooth and the same on both
  nets: 1.69 / 1.65 (r35 / r100) at L2, 1.45 / 1.45 at L4, 1.34 / 1.31 at L6, 1.26 / 1.24 at L8, 1.20 / 1.15 at L10, 1.07 / 1.09 at L12
  and 1.06 / 1.01 at L14.
- **Against the true kurtosis.** The lead's r35 MC gives a median excess kurtosis of .024 / .036 / .050 / .056 at L4 / L8 / L12 / L14,
  close to the card's radial (.025 / .040 / .052 / .057). So beta_fit / (radial_true / 6) follows the same factors. The Sym content is
  therefore ~radial/6, i.e. an angular common-gain fluctuation (the lead's s-mixture: beta = 2 sigma^2 = radial/6). The input-radial part
  is only ~1/10 of it.
- **Why no b move can pay (analyst RESULT.md §1-3).** Moving Sym(C x mu) into b is WORSE in every mode (partition .5 / .85 / 1.0,
  additive, offb). The covariance read weights the plane by d_a g_c, and under that weighting the Sym error has ~zero share: it lives in
  the always-off and always-on rows (38% + 36% of the error energy, but only 3% + 4% of the read), while per-unit k3 and v get worse.
- **The homogeneity split adds nothing new.** It contributes only the exact input-radial 16% piece, which is invisible to the covariance
  read for the same reason. My b oracles above (all worse) repeat this, and a second scalar recursion for beta would feed the same
  read-invisible channel. The b line is closed.

## 19. Design memo: is there a carrier for the read-visible fourth-order content (K31/K22 at kink rows, L6-14) that pays? (2026-09-23)
**Answer: no candidate clears break-even on paper.** Even a PERFECT carrier of K31+K22 on kink rows at L6-14 is worth only -10..-12%
of raw MSE, and the cheapest physical carriers cost 13-27% of the bill while capturing at most half of that content. The larger
joint-plane value, -18..-24%, also needs the (2,1) kink-row read fixed at the same time, and no third-order carrier upgrade has yet
lowered raw MSE.

**Rule for break-even.** With a bill B = 180 units per net (1 unit = one dense 1024^3 product; ~0.55 unit per Winograd product), adding
D units pays only if raw MSE falls by more than D/(B + D): 10 units needs 5.3%, 24 units 11.8%, 27 units 13%, 47 units 21%, 87 units 33%.

**New oracle measurements** (torch S6b, learned ON, no refit, r35/r100 mean; truth fed to the covariance output only at L6-14; eps .5 / 1;
`price_s6b_v2.py`, `price_s6b_v2.jsonl`):

| truth fed at L6-14 | eps .5 | eps 1 | per band at eps 1 (L6-9 / L10-12 / L13-14) |
|---|---|---|---|
| K31 on kink rows only | -7.8% | -8.7% | -5.3 / -2.1 / -2.1 |
| **K31 + K22 on kink rows (read-visible fourth order)** | **-10.0%** | **-12.2%** | -6.6 / -3.6 / -2.2 |
| K21 on kink rows only | -12.7% | -2.5% (overshoot) | see §17 row e |
| K21 + K31 + K22 on kink rows (joint) | -23.6% | -18.4% | -6.2 / -8.8 / -5.1 |
| all three planes on all rows, projected (§17 row g) | -26.8% | -22.6% | |

Kink rows hold ~88% of the plane value (-23.6% vs -26.8%). The fourth-order part alone is about half of the joint value.

**Candidates**

| candidate | (a) what it carries / how it moves | (b) cost, L6-14 | (c) capture bound | (e) break-even | verdict |
|---|---|---|---|---|---|
| 1. per-column lam4 on the bank's columns (independent sources) | K4 = sum lam4_j p_j^4 on the SAME columns as the third-order bank; transport free, but births need a lam4 rule | K31+K22 reads on kink rows: 24 units (young columns) / 47 (all) | free-weight ORACLE fit carries 40% / 51% of the K31 error and 36% / 44% of K22 (physicist RESULT_FOURTH §1), and the weights are mutually inconsistent: <= ~-5..-6% | needs -11.8% / -21% | fails by 2-4x even with oracle weights |
| 2. plane-only (3,1)+(2,2) star on kink rows | the post K31, K22 planes and k4, pushed as if kappa4 lived only on (a,a,a,a), (a,a,a,c), (a,a,c,c) patterns (5-6 products/layer; formula certified to 5e-14 by `cert_star31.py`) | ~3 units/layer = 27 units | **measured now** (`star31_transport.py`, r35, TRUE 8M post planes -> TRUE next-layer K31, read-weighted): relative error .90 / .93 / .95 / .96 at L5->6 / 8->9 / 12->13 / 13->14 (cos .43-.29, i.e. 8-19% of the read captured), WORSE than the card's isotropic form k4 C/v (.76-.82) | needs -13% | dead: the next layer's K31 lives mostly in distinct-index kappa4 patterns, which no plane carrier holds |
| 3. joint third + fourth order: exact (2,1) star transport + a fourth-order carrier | the (2,1) star residual as exact columns, plus 1 or 2 | exact star at L10-12 alone: ~+12% of the bill (physicist), so ~+36% at L6-14, plus 13-27% | joint perfect ceiling -18..-24%. The exact star alone at L10-12 made raw WORSE (+0.2..+1.5%) | needs -33..-39% | fails even if perfect |
| 4. band-only version of 1 or 2 at L10-12 | as above, 3 layers | ~8 units | perfect ceiling -3.3..-3.6% | needs -4.3% | loses even if perfect |
| 5. learned per-pair covariance corrector | a direct model of the net covariance change | n^2-class, cheap | cross-net R2 <= .02-.09 from any local feature family (analyst §13-15; physicist leadergap) | - | dead on information |

**(d) First falsifiable tests.** For candidate 2 the test was run above and it fails. For candidate 1 the cheapest test is a closed loop
on S6b with the physicist's free-weight lam4 fit (r35, 20k columns) feeding K31+K22 on kink rows at L6-14. Pass bar: at least -12%
(break-even at +24 units). Expected: at most -6%, a fail, since the weights carry at most half of an oracle that tops out at -12%.

**What would have to be true to pay.** A carrier that costs at most ~10 units (5.6% of the bill) and still captures at least half of
the JOINT kink-row value (-9..-12%). That means carrying the (2,1) and fourth-order kink-row reads at near-oracle accuracy for about one
product per layer. Nothing measured tonight comes close: every physical family captures under half of its own target, and a
plane-only transport loses 80-90% of the fourth-order content in one layer. The price list (§17) still says the error is worth ~20% at
L10-12. It is simply not reachable by a carrier bolted onto this card. A new card design, in which the per-pair fourth-order content is
not carried at all but computed from something cheaper, would be the multi-day bet; nothing on paper supports it yet.

## 20. Auxiliary intermediate-mean supervision for the refit: code and smoke test (lead's request; `refit_s6b_aux.py`, `aux_smoke/smoke.json`)
**Code.** `refit_s6b_aux.py` is a versioned copy of the analyst's `fable_analyst_v1/refit_s6b.py`: same card (s6bcard / s6bknobs),
groups, schedule, clip, split and .pt format. The only change is one block behind flags: `--aux_w` (default 0.3), `--aux_layers`
(default 10,11,12,13,14) and `--aux_calib` (default 8). The loss becomes

  loss = MSE_15 + w * sum_{l in aux_layers} MSE_l * (s_15 / s_l),

where MSE_l = mean over units of (m_l - truth_l)^2 against the organizer's 1e9-sample means for all 16 layers in fable_refit_v1/truth688,
and s_l = the card's own init MSE at layer l, averaged over the first --aux_calib TRAIN nets. So every auxiliary term starts equal to
the final MSE. VALID is still graded on the final MSE only. Outputs go to fable_covhunter_v1/<tag>. `--smoke` runs the test below and
exits.

**Smoke test** (TRAIN640 nets 0 and 1, groups full, lr 1e-4, 3 threads, 6 min):
- **w = 0:** the loss equals the original refit_s6b.py expression bit for bit on both nets.
- **w = 0.3:** the loss is 4.63e-8 / 4.69e-8. That is the final MSE (~1.86e-8) plus 0.3 x 5 commensurate terms, as designed.
- **Gradients:** all finite, per-tensor norms .19-42.6 at the 1e8 loss scale, before the clip at 1.
- **One Adam step:** it moves GNN head and head_b by 1.0e-4 (= lr) at most.

**Per-layer init MSE of the S6b card on 4 VALID nets** (post mean vs 1e9 truth; ratio to the final MSE in brackets):

| net | L4 | L8 | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|---|---|
| 37 | 2.0e-9 (.12) | 7.0e-9 (.40) | 1.01e-8 (.58) | 1.26e-8 (.72) | 1.36e-8 (.78) | 1.52e-8 (.87) | 1.69e-8 (.97) | 1.74e-8 |
| 39 | 2.0e-9 (.14) | 6.5e-9 (.45) | 8.0e-9 (.56) | 9.8e-9 (.68) | 1.09e-8 (.75) | 1.13e-8 (.78) | 1.23e-8 (.85) | 1.45e-8 |
| 42 | 2.0e-9 (.10) | 7.0e-9 (.35) | 1.09e-8 (.55) | 1.16e-8 (.59) | 1.34e-8 (.68) | 1.40e-8 (.71) | 1.70e-8 (.87) | 1.97e-8 |
| 48 | 2.1e-9 (.13) | 7.2e-9 (.42) | 1.10e-8 (.64) | 1.22e-8 (.71) | 1.52e-8 (.89) | 1.46e-8 (.85) | 1.55e-8 (.91) | 1.71e-8 |

The intermediate means are almost as wrong as the final one: 55-64% of it at L10 and 85-97% at L14. The mean error grows steadily
through the net, so there is a large, clean supervision signal at L10-14 that no refit has used so far. The TRAIN calibration (nets
0 and 1) gives the same shape: .44 / .53 / .64 / .78 / .93 at L10 / 11 / 12 / 13 / 14.

**Full run (waiting for CPU).** Compare w = 0.3 against a w = 0 control with identical settings and nets. The caveat to watch: the
GNN deltas at L10-14 may currently cancel errors that only appear at L15, so a large w can pull them away from the final-layer optimum.
A w in {0.1, 0.3} pair against the control, read on VALID16 final MSE at each epoch, answers it.
Command: `refit_s6b_aux.py --tag <T> --groups full --split <abs>/fable_analyst_v1/split_train640.json --aux_w 0.3 [--ntrain N] --threads 5`.

**Noise floor of the truth.** truth688 is a 1e9-sample mean, so each layer's MSE contains mean per-unit variance / 1e9. At L0 the
card is exact: a closed-form Gaussian ReLU with bias-free x ~ N(0,I) and the x @ W convention. Its L0 "MSE" of 6.4-7.1e-10 is
therefore pure truth noise (implied N 0.96-1.06e9 on nets 37/39/42/48), and L1-2 are 50-90% noise. The floor falls with depth because
the per-unit variance falls (32M MC, nets 37/39: 0.95-1.26e-10 at L10, 0.57-0.80e-10 at L15). That is at most 1% of the card's MSE at
L10-15, so the aux terms are clean signal. Never put L0-2 in --aux_layers. (`scratchpad l0_floor.py`; diag of mcall cov_post.)

**v2 of the driver (07:50, before any real run).** v1's evaluate() called loss(), so at w > 0 it would have graded VALID on the
aux-inclusive loss rather than the final MSE. The smoke test never called evaluate(), so the smoke results above stand. v2 grades VALID
on the final-layer MSE only and logs each aux layer's VALID MSE beside it. Each update also logs batch_final (the final-MSE part of the
batch) next to the total loss, plus grad_norm and lr. v1 is kept as refit_s6b_aux_v1_smoke.py.

**Matching run A (lead's config).** A is fable_physicist_v1/refit6b/refit_s6b.py: TRAIN160, full, lr 1e-4, warmup 3, clip 1, batch 4,
2 epochs, VALID16, warm start from S6b's params_epoch2. With `--ntrain 160` and no --split, my driver reproduces A's setup exactly:
- the same TRAIN list (inputs.json train[:160]);
- the same seed and the same net order in both epochs, with the rng drawn only by the per-epoch permutation (checked: ep0 starts
  49, 209, 214, 117; ep1 starts 83, 187, 60, 168);
- the same full parameter set, the same Adam, cosine schedule with warmup 3 and clip 1, and the same 1e8 / batch loss scaling;
- the same init (s6bcard.PARAMS);
- the same final-layer truths (truth688[15] equals public_g22 truthpost bit for bit on all TRAIN160 and VALID40 nets).
So w = 0 would replay A up to thread-count rounding, and A is an exact w = 0 control. Two free certificates in the run's own log:
valid_before must equal A's 1.7740244697409974e-08, and the update-1 batch_final must equal A's update-1 batch_mean
1.7751601290537665e-08 (same 4 nets at the same init).

**CORRECTION to the VALID4 table above (07:52).** The v1 smoke test restored only head/head_b after its one Adam step, so the table
was taken on the card with one lr-1e-4 step on A, cstar and the edge parameters. The v2 end-to-end test (aux_v2_e2etest.log) gives
net 37 at true init: L10-15 = 1.0146 / 1.2583 / 1.3658 / 1.5154 / 1.6834 / 1.7357e-8 (ratios .585 / .725 / .787 / .873 / .970).
Against the table: final +0.27%, L10-14 within 0.1%, so the conclusion stands. The exact init values for all VALID16 nets come from
the aux run's valid_before (below). The v2 --smoke restores every trained tensor.

**True-init table (aux run valid_before; S6b literals; post-mean MSE vs the 1e9 truth; ratio to the net's final MSE):**

| net | L10 | L11 | L12 | L13 | L14 | L15 |
|---|---|---|---|---|---|---|
| 37 | 1.015e-8 (.58) | 1.258e-8 (.72) | 1.366e-8 (.79) | 1.515e-8 (.87) | 1.683e-8 (.97) | 1.736e-8 |
| 39 | 8.06e-9 (.56) | 9.84e-9 (.68) | 1.094e-8 (.75) | 1.135e-8 (.78) | 1.231e-8 (.85) | 1.452e-8 |
| 42 | 1.086e-8 (.55) | 1.156e-8 (.59) | 1.344e-8 (.68) | 1.403e-8 (.71) | 1.704e-8 (.87) | 1.966e-8 |
| 48 | 1.096e-8 (.64) | 1.221e-8 (.71) | 1.523e-8 (.89) | 1.461e-8 (.85) | 1.557e-8 (.91) | 1.717e-8 |
| VALID16 mean | 9.45e-9 (.54) | 1.079e-8 (.62) | 1.266e-8 (.72) | 1.394e-8 (.79) | 1.556e-8 (.88) | 1.774e-8 |

VALID16 per-net ratio ranges: L10 .42-.64, L11 .46-.72, L12 .60-.89, L13 .66-.91, L14 .76-.97. The TRAIN160[:8] calibration
ratios (s_l / s_15) are .55 / .64 / .73 / .82 / .88, with s_15 = 1.732e-8.

**Certificates against A: rounding-level, not bit-exact.**
- valid_before: 1.773985e-8 (mine, 3 threads) vs A's 1.774024e-8 (5 threads), a -2.2e-5 relative difference in the mean.
- The per-net differences have random signs: sd 4.8e-5, max 1.2e-4, net 130 identical.
- Net 37 at 2 threads (the e2e test) matches A to 2e-9, while 3 threads is off by -8e-5. So this is the thread-count BLAS reduction
  order. The MSE is a small residual (error ~1.3e-4 on values ~0.3), so float32 rounding in the mean shows up at ~1e-5..1e-4 relative.
- Update-1 batch_final: 1.775109e-8 vs A's 1.775160e-8 (-2.9e-5, same 4 nets).
- So A is the w = 0 control up to a ~1e-5 floor on the 16-net mean. That floor is negligible against the lead's 0.2-point bar.

**RESULT (09-24): aux-mean supervision HURTS on the full-groups-only refit and HELPS with the GNN trunk.** Both runs use w .3 on
L10-14 and are compared on VALID16 final MSE with their exact w = 0 control.
- Without the trunk (aux_w03_A vs A, epoch 0): +0.61% pooled, 4/16 nets better. The L10-14 means improve only 1.0-0.3%, and the
  heads/edges buy that by moving off the final-layer optimum.
- With the trunk (aux6b_B2_w03 vs B2, same 5 threads, valid_before bit-equal): -0.27% at epoch 0 and -0.21% at epoch 1 pooled,
  11/16 both times, per-net se 0.11-0.13%. The L10-14 means improve 2.8-1.3% (15/16 nets better at L10).
- VALID40 grade of epoch 1 below.

**VALID40 grade of aux6b_B2_w03 epoch 1 (09-24; eval-only at 3 threads with refit6b_aux.py, whose eval path is the physicist's driver
verbatim; `valid40_compare.py`, paired bootstrap over nets).**

| baseline | VALID40 mean change | 95% CI | nets better |
|---|---|---|---|
| analyst's torch S6b rows (s6b_parity40.jsonl) | -1.326% | [-1.760, -0.906] | 33/40 |
| B2 epoch 1 VALID40 (3 threads, the w = 0 control) | -0.006% | [-0.154, +0.142] | 20/40 |

- Split: on the VALID16 nets it reproduces the training-run result (-0.212%, 11/16). On the other 24 nets it is +0.124% WORSE than B2
  (9/24).
- The VALID16 gain (1.7 se) was noise that does not generalize. The VALID40 value against S6b is entirely the trunk refit, which B2
  already has.
- **FAILS the lead's S12 bar** (≥ 0.15 pt pooled vs B2 epoch 1 with ≥ 22/40). B2 stays the refit.
- Thread check: the 3-thread eval matches the 5-thread training evals on VALID16 (mean -2e-5, max 3.3e-4 per net).
- Reading: the aux loss does pull the L10-14 means toward truth (2.8-1.3% lower MSE on VALID16), but along directions that do not carry to
  L15. Uniform intermediate MSE is the wrong target; only final-adjoint-weighted error matters, and the final loss already supplies that.
  Aux-mean supervision at w .3 is CLOSED.

## 21. Tier cut + trunk refit: old-tier frame rank 320 (lead's test; `s6bcard_r320.py`, run `r320_B2recipe_1ep`; 2026-09-24) — CLOSED
**Setup.**
- Plumbing: s6bcard_r320.py sets covstep.TIER_RANK = 320, the rank= that the HookCard passes to the every3 push.
- Certified: at RANK 384 it is bit-exact with s6bcard; at 320 the old-tier Q is 1024x320 from L8 on.
- Run: B2's recipe (full+P weights-only, lr 1e-4 / 5e-5, warmup 3/5, clip 1), 1 epoch TRAIN160, warm start from B2 epoch 1, 5 threads.

**VALID16 final MSE:**

| state | VALID16 mean | vs B2 epoch 1 at rank 384 (1.74614e-8) |
|---|---|---|
| no refit at rank 320 | 1.80469e-8 | +3.35% (15/16 nets worse; per-net -1.5% to +8.0%) |
| after 1 refit epoch | 1.80453e-8 | +3.34% |

- The refit absorbed -0.009%, and the L10-14 means moved only 0.1-0.2%.
- Bar was ≤ 1.78106e-8 (+2%): FAIL.
- Even at the ceiling bill cut of -4%, the result is about -0.8% adjusted before any wall or rounding risk. The lead closed it.

**Law:** the trunk refit re-adapts to its own card, but it does not re-learn information that a structural cut removes (as with rank 288 and drop9).

## 22. Capacity test: 2x-wide GNN trunk (lead's test; `make_wide_init.py`, `s6bcard_wide.py`, run `wide2x_B2recipe_1ep`; 2026-09-24) — CLOSED
**Setup.**
- The hidden layers go 128/128 -> 256/256 (trunk 44,144 -> 117,872 params). The messages (cm 16), the state (64) and the heads are unchanged.
- Init is function-preserving from B2 epoch 1: new rows ~ N(0, std), zero output blocks.
- Certified: at step 0 the forward is bit-exact with B2 epoch 1 on all 16 VALID nets. The zero blocks receive gradient, and the new
  input rows get exactly 0 at step 0.
- Run: B2's recipe, 1 epoch TRAIN160, 5 threads.

**Result:** VALID16 1.74614e-8 -> 1.74850e-8 (+0.135%), 7/16 nets better, per-net se 0.13%. No gain, against a -1.0% bar.

**Conclusion:** with the T640 trunk run landing flat (-0.13% VALID40, analyst), the learned parts are neither data- nor capacity-limited.
The remaining lever is physics (the j1 line).

## 23. Additivity ("shadow") test of the j1 fix against the learned parts (lead's gate; `shadow_test.py`, `shadow_test_x0.5_0.3_n16.json`; 2026-09-24)
**Setup.** Per net, S(X) = m(ON, 0) + [m(OFF, X) - m(OFF, 0)]: the learned-ON card (B2 ep1) plus the j1 fix's learned-OFF effect, with no
refit. OFF means head, head_b and the edge output/bias are zeroed (the physicist's definition). Run on VALID16 at 3 threads.
**Controls reproduce:** learned-OFF is +24.1% vs ON; j1 learned-OFF is -6.33% at X 0.5 (15/16) and -3.45% at X 0.3 (13/16).

| X | shadow vs ON (β = 1) | nets better | per-net mean (se) | best scale β* | shadow at β* |
|---|---|---|---|---|---|
| 0.5 | -0.80% | 10/16 | -0.76% (1.04%) | 0.54 | -2.89% |
| 0.3 | -1.34% | 11/16 | -1.34% (0.84%) | 0.62 | -2.18% |

β* is one pooled scalar fitted on the same 16 nets (in-sample).

**Reading: PARTIALLY ADDITIVE.**
- About half of j1's learned-OFF effect is already inside the learned parts (β* 0.54-0.62). The additive half is worth about 2-3%, not 6%.
- The lead's gate for L2 was ≤ -2% at β = 1. It FAILS as pre-registered (-0.8% / -1.3%) and passes only with the fitted scale.
- Implication for the j05/j03 refits: a realistic ceiling is about -2..-3% vs B2 ep1, unless the refit finds more than the linear shadow.

## 24. L3: read-aware old-tier frame (lead's test; `s6bcard_readframe.py`, eval `rf320_read_v16`; 2026-09-24) — KILLED
**Setup.**
- At each every3 join, the frame sketch Omega = W[prev, :R] (the first R units) becomes W[prev, S], where S = the R current-layer units
  with the largest min(gate, 1-gate). Those are the rows the covariance plane read sees, and gate comes from diag(W^T C W) of the input
  state.
- Built from s2card.make_push's own source with only the Omega line replaced (exact-once).
- Certified: MODE 'first' at 384 is bit-exact with s6bcard.

**Results (B2 ep1 params, no refit):**

| frame | net 37 at 384 | net 37 at 320 | VALID16 at 320 vs B2 ep1 @384 |
|---|---|---|---|
| card's own (first units) | 0 (bit-exact) | +7.93% | +3.35% (15/16 worse) |
| read-aware (kink units) | +1.57% | +4.63% | +4.10% (15/16 worse; better than first-units on only 6/16 nets) |

Bar was < +1%: KILLED.

**Reading.** An old-tier cohort must serve reads 1-8 layers downstream, through products of W. A frame tuned to the next layer's kink rows
drops content those later reads need. The card's quasi-random sketch (the first units of W), which keeps the energy, is the better
choice. Sketch quality is not a bill lever either: a power iteration would double the join sketch (about +8% bill) to recover at most the
0-4% truncation loss.

## 25. L1: joint learned-OFF Gauss-Newton fit of the physics dials on the j05 card (lead's test; `s6bcard_joint.py`, `gn_joint_fit.py`, `gnjoint_j05ep0_off.json`; 2026-09-24) — PASS learned-OFF
**Setup.**
- Card: the physicist's j05 card (s6bcard + make_bd 'phys' j1 0.5) with the j05 refit's epoch-0 params, learned OFF (head, head_b and
  the edge output/bias zeroed).
- 12 dials, each skipped at its base value, so the base is bit-exact with j05 (certified on net 35):
  - X1: j1 strength;
  - X0 / X2 / X3: physical j0/j2/j3 directions added to the learned orth terms;
  - bscale: carried b;
  - s_carry, s_new: lam of carried columns per layer, and of newborn columns;
  - c0 / c1: the K31 row factor;
  - k4a / k4b / k4c: per-unit k4 at pair layers 1-7 / 8-11 / 12-15.
- Method: one forward-difference run per dial on 8 TRAIN nets gives the Jacobian of the final means. The Gauss-Newton step is
  Levenberg-Marquardt damped and checked leave-one-net-out.

**Fit nets (8 TRAIN, learned OFF):**
- In-sample -11.86%, LOO -10.83% (8/8 held-out nets better, lam 0.01).
- A REAL run at the step gives -11.89% (8/8). Half step: -8.76%.

**Single-dial slopes (forward differences):** k4c +9.5%/0.1, k4b +5.1%, k4a +2.1%, s_new +3.3%/0.02, s_carry -1.75%/0.005,
c0 +1.6%/0.05, bscale +0.9%/0.02. The j1/j0/j2/j3 slopes are all within 0.05%.

**Fitted dials (full step):**

| dial | base | fitted | change |
|---|---|---|---|
| X1 | 0.5 | 0.683 | |
| X0 | 0 | -0.061 | |
| X2 | 0 | +0.295 | |
| X3 | 0 | -0.415 | |
| bscale | 1 | 1.026 | |
| s_carry | 1 | 1.0076 | per layer, about +11% after 14 layers |
| s_new | 1 | 0.987 | |
| c0 | 0.85 | 0.458 | |
| c1 | 0.50 | 0.396 | |
| k4a | 1 | 0.954 | -4.6% |
| k4b | 1 | 0.953 | -4.7% |
| k4c | 1 | 0.915 | -8.5% |

**VALID16, learned OFF** vs the j05 learned-off control (2.02995e-8; matches the physicist's 2-thread control to -1.05e-5):
1.82580e-8 = **-10.06% pooled, 16/16 nets better** (per-net mean -10.04%, se 1.07%). The kill line was -2%: PASS by 5x.

**Caveat (shadow law, §23):** learned-OFF gains carry over to learned-ON at only about 0.3-0.5x. Next steps are the shadow test of these
dials, then the j05+joint refit from j05 epoch 0 (B2 recipe) in the next slot.

**j03 (X=0.3) refit, epoch 0 (hedge):** VALID16 1.71734e-8.
- vs B2 epoch 1: -1.65%, 11/16 nets better, per-net se 0.68%. Its bar was 1.7374e-8: PASS.
- vs j05 epoch 0 (1.71227e-8): +0.30%, 6/16. So j05 is the better strength, and both refit to about -1.7..-1.9%.

**Shadow test of the fitted L1 dials (learned ON; `shadow_joint.py`, `shadow_joint_n16.json`; base = j05 epoch 0 learned ON, 1.71226e-8):**
- Certificate: m(OFF, fit) equals the fit's VALID16 learned-OFF values on 16/16 nets bit-exactly, so s6bcard_jfit1 is certified.
- Learned ON with the fitted dials, no refit: **+7.87%, 0/16 nets better.**
- Shadow at β = 1: +5.70% (1/16). Best pooled scale β* = 0.30 gives -1.30% (in-sample).

Reading: most of the -10% learned-OFF gain is REDUNDANT with the learned parts, which already implement those corrections, so adding
the dials on top double-counts. A refit must recover about 8 points to break even and about 9 to reach its -1% bar. For comparison, the
j05 refit recovered 6.9 points in one epoch. The j05+joint refit (jfit1_j05ep0_1ep) was launched 18:28 with that bar pre-registered.

## 26. Learned-ON joint Gauss-Newton fit of the same 12 dials (lead's GO; `gn_joint_fit.py ... on`, `gnjoint_j05ep0_on.json`; 2026-09-24) — PASS, a pure literal change
**Setup.** The j05 card with the j05 epoch-0 params, learned ON (the deployable objective), the same dials, steps and 8 TRAIN nets as
§25. Base certified bit-exact.

**Fit nets:** in-sample -4.07%, LOO -3.01% (7/8). Real run at the full step: -4.14%; half step: -2.99%.

**VALID16 learned ON vs j05 epoch 0 (1.71226e-8):** 1.67409e-8 = **-2.23% pooled, 13/16 nets better** (per-net mean -2.30%, se 0.58%).
The bar was -1%: PASS, with no refit.

**Fitted dials, learned ON next to learned OFF (§25)** — the difference shows what the learned parts absorb:

| dial | base | ON fit | OFF fit |
|---|---|---|---|
| j1 X1 | 0.5 | 0.531 | 0.683 |
| j0 X0 | 0 | +0.001 | -0.061 |
| j2 X2 | 0 | +0.209 | +0.295 |
| j3 X3 | 0 | -0.117 | -0.415 |
| b scale | 1 | 1.051 | 1.026 |
| carried lam per layer | 1 | 1.0042 | 1.0076 |
| newborn lam | 1 | 0.9925 | 0.987 |
| K31 c0 | 0.85 | 0.703 | 0.458 |
| K31 c1 | 0.5 | 1.123 | 0.396 |
| k4 L1-7 | 1 | 0.991 | 0.954 |
| k4 L8-11 | 1 | 0.952 | 0.953 |
| k4 L12-15 | 1 | 0.954 | 0.915 |

**What the learned parts absorb:**
- Most of the extra j1 (0.683 -> 0.531) and most of the j3 cut.
- About 60% of the K31 c0 cut.
- The K31 slope c1 moves the other way (1.12 ON vs 0.40 OFF). Its single-dial slope is tiny (+0.03% per 0.1), so it is set by
  correlations with the other dials.

**Common to both fits:**
- k4 at L8-15 down about 5%.
- K31 c0 down.
- j2 physical added (+0.2..+0.3).
- Carried content up 0.4-0.8% per layer, newborn down 0.8-1.3%.

**Next:** re-validate these ON dials on the S18 card (the low-rank star hook) on VALID40 before shipping.

**Mechanism check: K31 row-factor scan (`c0_scan.py`, `c0_scan.json`).** Learned OFF, the same B2 epoch-1 literals on both cards,
8 TRAIN nets. Change in final MSE vs c0 = 0.85:

| card | c0 1.0 | c0 0.65 | c0 0.46 | quadratic optimum |
|---|---|---|---|---|
| S6b (j1 = learned c1·orth) | +3.27% | -1.21% | +0.89% | 0.68 |
| j05 (j1 physical 0.5) | +5.58% | -4.07% | -4.49% | 0.53 |

**CONFIRMED.**
- With the physical j1 feed, the best K31 c0 moves down (0.68 -> 0.53), and cutting K31 becomes 3.7x more valuable (-4.5% vs -1.2%).
  So K31 was partly standing in for the missing j1 feed.
- Part of the cut is independent of j1: even on S6b the learned-OFF optimum is about 0.68, not the 0.85 fitted on S2 with learned ON.

## 27. S18 re-validation of the learned-ON dials and the native-consistent re-fit (`s18_revalidate.py`, `s6bcard_joint2.py`, `s6bcard_jfitON2.py`; 2026-09-24)
**Porter's catch.** In torch, two dials behaved differently from native:
- The k4 band scaled only the main pair call; the GN v3 cov_pm recompute kept the unscaled ctx['k4'].
- The GN r31 remainder kept c0/c1 = 0.85/0.5.
Natively, both share one k4 and one _K31R_TABLE.

**Lead's choice (a):** make the torch card native-consistent. s6bcard_joint2.py (CH-60) scales k4 at its definition (the pair call and
ctx['k4']) and re-configures gn3 with the dialed c0/c1. At base it is bit-exact with the j05 card. The learned-ON dials were re-fit on
it with the same nets and steps.

**Re-fit on 8 TRAIN nets:** in-sample -4.12%, LOO -2.98% (7/8); real run -4.01% (7/8), half step -2.95%. VALID16 learned ON, unhooked:
-2.20% (14/16).

**Dials (physical units), old ON fit -> shared re-fit:**
- K31: c0 .703 -> .653, c1 1.123 -> 1.608.
- k4 bands: .991 / .952 / .954 -> .988 / .962 / .963.
- j3: -.117 -> -.228; j1 .531 -> .533; j2 .209 -> .213; j0 about 0.
- b: 1.051 -> 1.049.
- Carried lam per layer: 1.00425 -> 1.00439; newborn .9925 -> .9932.

**S18-hooked VALID40**, learned ON, j05 epoch-0 params, dials vs base (the hook alone is +0.18% vs no-hook S17; native gives +0.23%):

| dials (semantics) | VALID40 pooled [95% CI] | nets better | VALID16 | other 24 |
|---|---|---|---|---|
| old ON fit (mirrored; upper-bound reference) | -2.05% [-2.67, -1.44] | 34/40 | -2.34% | -1.87% |
| shared re-fit (S20 candidate; exact parity rows) | -1.98% [-2.59, -1.37] | 34/40 | -2.32% | -1.76% |

Both pass the bar (≤ -1.5%, CI excluding 0).
- S19 = the old dials under native SHARED semantics (porter's build).
- S20 = the shared re-fit. It is only about 0.07 points different from the old dials in torch, within noise. The porter's native
  S20-vs-S19 rows decide between them.

**jfit1 refit (learned-OFF §25 dials frozen, B2 recipe, 1 epoch from j05 epoch 0), VALID16:**
- 1.84707e-8 before -> 1.72399e-8 after. The refit recovered 7.2 points in one epoch.
- +0.69% vs j05 epoch 0 (7/16) and +1.47% vs j05 epoch 2 (6/16). The bar was 1.6951e-8: FAIL. j05 epoch 2 stays the learned-parts base.

Reading: the physics-first route (learned-OFF optimum, then refit) does not beat the co-adapted route. Fitting the learned-ON dials
gave -2.2% with no refit. The alternating loop continues from the learned-ON side.

## 28. Alternating loop, round 1: learned-ON dial re-fit on the S18-HOOKED card with the physicist's j05 epoch-2 params (`gnjoint_r1_ep2_on_shared_s18.json`, `s6bcard_jfitR1.py`; 2026-09-24)
**Setup.** s6bcard_joint2 (shared semantics) plus the lrstar hook. Params: j05 epoch 2 (VALID40 -0.69% vs epoch 0, physicist).
Linearized at BASE, same 8 TRAIN nets and steps. Certified bit-exact against the hooked j05 card.

**Fit:** in-sample -4.20%, LOO -3.20% (7/8); real full step -4.12% (7/8), half step -3.09%.
**VALID16, hooked, learned ON:** 1.70220e-8 -> 1.66645e-8 = -2.10% (14/16).

**Round-1 dials:** j1 .559, j2 +.358, j3 +.152, j0 -.006, b 1.0509, carried lam 1.00469 per layer, newborn .99312, K31 c0 .709 / c1 .503,
k4 .989 / .963 / .960.

**Stable across the three learned-ON fits** (epoch 0 mirrored, epoch 0 shared, epoch 2 hooked):
- j1 .53-.56, j2 +.21-.36;
- b about 1.05, carried about +0.45% per layer, newborn about -0.7%;
- c0 .65-.71;
- k4 about .99 / .96 / .96.

**Weakly determined (flat):** j3 (-.12 / -.23 / +.15) and c1 (1.12 / 1.61 / .50).

**Hand-off:** s6bcard_jfitR1.py went to the physicist for the next refit epoch with these dials frozen, on the S18 card with
gn_linear_hook. VALID40 of round 1 vs S19 is chained.

**Round 1 vs S19, VALID40** (`valid40_dials.py`, `r1_ep2dials_vs_S19.json`). Round 1 = j05 epoch-2 params + round-1 dials; S19 = j05
epoch-0 params + S19 dials. Both S18-hooked, shared semantics, learned ON:

| nets | pooled [95% CI] | nets better |
|---|---|---|
| VALID40 | **-0.89% [-1.13, -0.65]** | 35/40 (per-net mean -0.88%, se 0.12%) |
| VALID16 | -0.84% | 13/16 |
| other 24 | -0.92% | 22/24 |

The lead's per-round bar was ≥ 0.5% on VALID40: PASS. The rows are exact torch parity targets for the porter's S21 (minus d6x).

## 29. The per-unit k3+k4 lever on the CURRENT card (option (b); `k34_supply_price.py` v2, `k34_supply_price.json`; 2026-09-24)
**Setup.**
- Card: the S22 torch counterpart = s6bcard_joint2 + round-1 dials + lrstar r32 + gn_linear_hook + fine_ckpt=False + pair_series loop
  form; loopR1 params, learned ON.
- Supply: per-unit pre-activation k3 (diag K21) and k4 in the pair map, k = card + eps·(truth − card), at a layer band. The L12-14
  learned mean correction (GNN delta) runs at s = 1 or 0.5.
- Truths: 32M on r35/r100; 8M on the analyst's 7 held-out nets. The final MSE is scored against the independent organizer truth.
- Certificate: the same injection on the analyst's card reproduces their net-150 values bit-exactly.
- (v1 crashed at net 250 on the truth path, which only covers 200 nets; kept as k34_supply_price_v1*.)

**Change in final MSE vs the current card:**

| supply | 7 held-out (geo) | nets better | r35 | r100 |
|---|---|---|---|---|
| control: correction at 0.5, no supply | +2.98% | 1/7 | +1.12% | +3.94% |
| L12-14, eps 0.5, s 1 | -12.3% | 7/7 | -6.9% | -8.9% |
| L12-14, eps 1, s 1 | -13.3% | 7/7 | -2.9% | -8.4% |
| L12-14, eps 0.5, s 0.5 | -15.0% | 7/7 | -12.6% | -10.3% |
| L12-14, eps 1, s 0.5 | -19.6% | 7/7 | -12.6% | -12.9% |
| L2-14, eps 0.5, s 1 | -19.2% | 7/7 | -14.7% | -14.0% |
| L2-14, eps 1, s 1 | -16.9% | 7/7 | -9.5% | -9.2% |
| L2-14, eps 0.5, s 0.5 | -22.6% | 7/7 | -21.2% | -16.3% |
| L2-14, eps 1, s 0.5 | -23.7% | 7/7 | -20.5% | -14.8% |

**Reading.**
- §34's lever holds on the deployed physics, and is slightly larger. On the analyst's S6b + B2 ep1 frame, L12-14 at eps 1 gave
  -11.0% (s 1) and -18.4% (s 0.5).
- L2-14 adds 4-7 points. Per the analyst (§35 iv), that surplus is each layer's local pair-map effect, not a repair of the carried
  error, so a source must be accurate at all 13 layers.
- On the kill rule (eps 0.5 at s 0.5 must beat -4%): -15% and -23% PASS. But eps only removes error along the card's own direction.
  The real accuracy spec comes from random source error; see the noise rows below.

**Noise rows** (`k34_noise_price.py`, `k34_noise_price.json`): supply = truth + r·sigma_l·z per channel, where sigma_l is the card's own
error rms at layer l and z uses the analyst's seeds. s = 0.5. The control re-run equals v2 exactly.

| random source error r | L12-14 (7 held-out geo; r35 / r100) | L2-14 (7 held-out geo; r35 / r100) |
|---|---|---|
| 0 (truth) | -19.6% (-12.6 / -12.9%) | -23.7% (-20.5 / -14.8%) |
| 0.25 | -12.6%, 7/7 (-8.5 / -1.3%) | -14.4%, 7/7 (-17.9 / +4.9%) |
| 0.5 | +7.3%, 1/7 (+5.8 / +33.4%) | +13.2%, 0/7 (+0.4 / +58.6%) |

**Quadratic model**, ratio(r) = a + b·r²:
- L12-14: b ≈ 1.12 per r², break-even r* ≈ 0.42.
- L2-14: b ≈ 1.49 per r², r* ≈ 0.40. The model predicts r = 0.5 within 1.1 points in both bands.
- Both match the analyst's frame (-11.7% / +6.2% at L12-14). Noise hurts about 1.3x more over 13 pair maps than over 3.

**SPEC for any k3+k4 source:** its random (unaligned) per-unit error must be at most about 25% of the card's current error rms at
EVERY layer it feeds, which captures about 60% of the lever (-12..-14%). Break-even is about 40%.
- The card's current errors are k3 3-7% and k4 11-27% relative rms.
- So a source needs k3 within about 1-2% and k4 within about 3-7% relative rms, per unit, about 4x better than the card.
- r100 is fragile (+33% / +59% at r 0.5), so single-net screens will mislead. Price on the 7 held-out nets.

## 30. Dial loop rounds 2-3: CONVERGED (2026-09-24/25)
- **Round 2** (loopR1 params, S21 structure, 8 TRAIN nets, linearized at round 1; `gnjoint_r2_loopR1_on_shared_s18.json`): in-sample
  -0.55%, LOO +0.44%, VALID16 +0.106% (7/16). The write gate stopped it; no S22 dials file was written.
- **The physicist's loopR2** (loop3 = s1=0 star, round-1 dials): VALID16 -0.33% against a -0.5% bar, FAIL. The refit side has saturated too.
- **Round 3** (S23 STRUCTURE = Nystrom join + star s1=0 + gn_linear_hook + pair series; the deployed loopR1 params; 16 TRAIN nets;
  `gnjoint_r3b_loopR1_on_shared_s23.json`): in-sample -0.24%, LOO +0.49%, real step -0.15% on the fit nets, VALID16 +0.032% (8/16).
  The write gate stopped it; no S24 file.
- **Conclusion:** the round-1 dials are the optimum of this 12-dial family on both structures and both param sets. S23's structural change did
  not move it. The next dial gain can only come from joint training of dials and learned parts (the lead's loopR3 proposal: trainable
  dials s6bcard_joint2t.py, certified bit-exact against joint2; a backward over all 12 dials is finite) or from new dials.

## 31. Join schedule re-check under Nystrom joins (lead's item (1); `join_sched_s23.py`, `join_sched_s23_n16.json`; 2026-09-25)
**Setup.** S23-structure torch card: joint2 + round-1 dials + star s1=0 + Nystrom join with the schedule under test + gn_linear_hook +
pair series. loopR1 params, learned ON, no refit. The every3 arm reproduces the round-3b VALID16 base (1.65618e-8) to the last digit.

**Raw VALID16 vs every3 (S23):**
- even (L6/8/10/12/14): **+0.57%** [CI +0.04, +1.05], 4/16 nets better.
- four7 (L7/9/11/13): **+1.18%** [CI +0.51, +1.88], 3/16 nets better.

**Reading.** Under Nystrom joins, more joins still cost accuracy: each join re-truncates the old content, the 09-22 law. For the lead's
bar (adjusted ≤ -0.8% vs S23), even would need a native bill of about -1.4% just to break even and -2.2% to pass. four7 would need
about -2.0% and -2.8%. The porter's native bills decide; unless even reaches ≤ -2.2%, KILL.
**Native bill (porter, in-process, loopR1 + round-1 dials):** even vs every3 is -0.07% (r35) / -0.47% (r100), about -0.3%. The adjusted
change is about +0.57% raw - 0.3% bill = **+0.3%**, the wrong side of the -0.8% bar: **KILLED**. four7 was not built; its raw cost is 2x
even's and its bill saving can only be smaller. every3 stays. Law: under Nystrom the join schedule is still set by accuracy (each join
re-truncates), and the young-column saving from joining earlier is worth only about 0.3% bill.

## 32. loopR3 joint refit (physicist; learned parts + dials trained together) — VALID40 pre-gate (`valid40_dials.py`, `loopR3_joint_vs_S23.json`; 2026-09-25)
**Setup.** loopR3 params plus its exported dials (dials_loopR3_joint.json = dials_epoch0.json) vs the S23 torch state (loopR1 + round-1
dials). Both arms on the S23-structure card (joint2 + Nystrom + star s1=0 + gn_linear_hook + pair series), CPU float64, 3 threads.

**Result:**
- VALID40 **-0.543%** [CI -0.699, -0.387], 34/40 nets better.
- VALID16 -0.52% (13/16); other 24 -0.56% (21/24).

**Dial moves vs round 1:**
- k4 bands .989 / .963 / .960 -> .981 / .966 / .953; c0 .709 -> .694.
- b 1.0509 -> 1.0486; s_new .9931 -> .9939; s_carry 1.00469 -> 1.00427.
- X1 .559 -> .558; X2 .358 -> .352. X3 and c1 were frozen.
All small; the joint training found a slightly different learned-part / dial balance.

**Capacity test v3** (from loopR3, S24 card s6bcard_capS24, A control vs B wide2x, 2 epochs on MPS) was relaunched 03:19 after v2 died on
two chain bugs, now fixed: a lost REC line, and an exit check that read date's $?.

## 33. Capacity test v3: 2 epochs on MPS, from loopR3 on the S24 card (`capacity_chain_v3.sh`; 2026-09-25) — WIDTH CLOSED, arm A -> S27
**Setup.** Card s6bcard_capS24 = S23 structure + the loopR3 joint dials, frozen. Loop recipe, 2 epochs, MPS float32 through the fixed
FIFO lock.
- A = control from loopR3.
- B = function-preserving wide2x GNN from loopR3 (hidden 256/256), bit-exact with A at step 0.

**VALID16 (MPS), vs the shared start 1.64753e-8:**

| arm | epoch 0 | epoch 1 |
|---|---|---|
| A | 1.64228e-8 (-0.32%) | 1.63990e-8 (**-0.463%**) |
| B | 1.64093e-8 (-0.40%) | 1.63866e-8 (-0.538%) |

B beats A by only 0.08% at both epochs, against the 0.5% bar. **Width is CLOSED.** With 2 epochs and a matched control, "under-trained"
no longer explains it; the 09-24 1-epoch null stands.

**Arm A itself is a gain:** 2 learned-only epochs from loopR3 give -0.463% on VALID16, more than a joint epoch did (lead). Its
params_epoch1.pt (sha 4e9be88d) becomes S27 = S26 + arm-A literals. The dials are unchanged, but arm A trained WITH the j3 leg, which S26
cuts.
**Arm A VALID40 pre-read** (torch CPU, S24-structure card, arm-A ep1 vs the S24 torch state = loopR3 params + dials; `capA_ep1_vs_S24.json`):
- VALID40 **-0.377%** [CI -0.505, -0.245], 35/40 nets better.
- VALID16 -0.468% (14/16), matching the MPS VALID16 -0.463%. Other 24 -0.318% (21/24).
The gain holds on held-out nets but is smaller there, as usual for VALID16-monitored refits.
**Native parity (porter):** S27 over S26 VALID40 (both with j3cut + k-split v4 + treuse) is -0.296% [-0.421, -0.168], 31/40, against torch
-0.377%. Per-net correlation 0.80, difference sd 0.26%.
- Native comes out about 0.08 pt less negative, consistent with arm A having trained WITH the j3 leg that S26 cuts.
- The largest gap is net 42 (-0.32% native vs -1.43% torch). Net 42 is the engineer's pair-series knife-edge net.
- S27 vs S24: -1.235% adjusted, 40/40 (porter).

## 34. R320 router signals, per VALID40 net (lead's request, for the analyst's LOO; `r320_signals.py`, `r320_signals_n40.json`; 2026-09-25)
**Setup.** S24-structure torch card (loopR3), instrumented at the joins L7/10/13 without changing it: the rank-384 run is bit-exact with
the loopR3 VALID40 rows on 40/40 nets. The target is S25's native per-net R320 effect (S25r cand / S24c cand, porter).

**The torch-simulated R320 effect** (the same card at rank 320, scored against the truth) matches native almost exactly:
- corr **0.992** with the native raw effect, 0.991 with native adjusted;
- mean +4.38% vs native +4.40%; sd 2.98% vs 3.03%.
So torch is a faithful instrument for this router, but that number uses the truth and cannot be a runtime signal.

**Truth-free candidates, correlation with the native per-net raw effect** (the best 6 of 38):

| signal | corr |
|---|---|
| L10 kink fraction | -0.37 |
| L13 old-plane share of the read | +0.36 |
| L10 old share | +0.23 |
| L13 tail spectrum | -0.20 |
| L7 kink fraction | -0.19 |
| L13 raw energy loss | -0.17 |

- (a) old-plane share: weak, at most +0.36.
- (b) read-weighted truncation loss (join layer and the next 2 layers, mask or d-weighted): at most 0.14. It is almost identical to the raw
  energy loss, because the cut content is spread across kink rows like the kept content. It FAILS, like §39's energy signal.
- (c) kink fraction at L10: -0.37. Tail spectrum, column counts and the base MSE are all at most 0.2.

**Reading.** No cheap per-join statistic sees the per-net penalty. The penalty is a property of how the cut propagates through the later
layers to the final mean, not of how much or where content is cut. The next test, truth-free but full-cost, is whether |m320 - m384|²
(the change in the final prediction) predicts the penalty. If it does, a cheap linearized propagation of the cut to L15 is the router
signal to build.

**Addendum: the truth-free prediction change** (`r320_dm.py`, `r320_dm_n40.json`; logged at the lead's request, routing is now moot).
dm2 = mean((m320 - m384)²) / MSE384.
- Correlation with the native per-net raw effect: **0.27**. In the analyst's LOO harness it captures 0.10 as a threshold router and 0.30
  as a logistic one, against resp_L10's 0.59. It FAILS.
- The exact split of the penalty, MSE320 - MSE384 = mean(dm²) + 2 mean(e384 · dm), shows why: the CROSS term correlates **0.97** with
  the native effect.
- So the per-net R320 penalty is the cut's change lining up with the existing error, not the size of the change. It is unknowable
  without the truth, which is why every magnitude-type signal fails.
- The torch-simulated effect (which uses the truth) captures 0.98, as expected.

## 35. Cheap compensation for the rank-320 joins at the fixed schedule 384/320/320 (lead's task 09:4x; `r320_comp.py` v1-v7 (v7 is the reference), `r320_comp_table.py`, rows `r320_comp_<tag>.jsonl`; 2026-09-25) — FAIL, no cheap stand-in exists
**Setup.**
- Card: the analyst's torch card (s6bcard_loop4j = S23 structure, loopR3 params and dials, learned ON, float64 CPU, 3 threads) with the
  analyst's energy-ordered Nystrom push and per-join ranks.
- My push carries the old tier's Gram split by birth pair. Each birth adds two atoms p+- = u +- v/2 with weights +c and -c; they differ only
  at the birth unit. So c^2 (p+ p+^T + p- p-^T) = 2c^2 u u^T + (c^2/2) v v^T, carried as two blocks; weight 1 on both is the card.
- This is the card up to rounding: the analyst's rows move by at most 0.02% (net 37: all-384 +0.022%, schedule +0.005%). All
  comparisons below use my own all-384 and schedule rows from the same code.
- Screen: TRAIN24 = every 6th TRAIN160 net, panel nets 35/100/200 excluded. Measure: VALID40.
- Costs:
  - (a), the frame reweighting and the consumer scales cost nothing measurable;
  - (b) costs about 0.1% (one n x r^2 product per join);
  - the timing probe costs bill (below).

**Where the cut's content lives** (TRAIN24, share of the old tier's Gram energy missed by the frame, mean over nets):

| join | common part u, missed at 320 / 384 | difference part v, missed at 320 / 384 | v share of the Gram |
|---|---|---|---|
| L10 | 0.011% / 0.002% | 0.53% / 0.12% | 2.7% |
| L13 | 0.004% / 0.001% | 0.36% / 0.07% | 1.5% |

The cut multiplies the missed energy about 5x in both parts, and the total stays tiny. The penalty (TRAIN24: raw +2.02% vs all-384,
19/24 nets worse) is out of all proportion to the energy.

**Candidates, TRAIN24** (change vs the uncorrected schedule, in points of the all-384 pool; the 95% CI is a paired bootstrap over nets):

| candidate | knob | change, pt | 95% CI | share of the penalty recovered |
|---|---|---|---|---|
| (a) scalar on the kept block | F x 1.003 at L10 and L13 | +0.111 | [-0.047, +0.280] | -6% |
| (a) per join | F x 1.003 / 0.997 at L10 | +0.002 / +0.117 | | |
| (a) per join | F x 1.003 / 0.997 at L13 | +0.061 / +0.024 | | |
| (a) fitted (quadratic through the 5 points) | s10 = 1.0022, s13 = 0.9981 | predicted -0.027 | | 1% |
| (b) nearest version: restore each row's dropped Gram energy | rows x (d/dhat)^(1/2) | +0.002 (4 nets) | [-0.012, +0.017] | 0% |
| (c) frame weight toward u | v part x 0.5 | +0.047 | [-0.044, +0.139] | -2% |
| (c) frame weight toward v | v part x 3 / 10 / 30 (4 nets) | +0.07 / +0.79 / +1.10 | | worse |
| (c) timing probe (costs bill) | 384 for the join layer, 320 from the next layer | **-0.923** | [-1.485, -0.353] | 46% (18/24) |
| no-refit stopgap | learned mean correction x 1.03 at L10-15 | +0.057 | [-0.018, +0.135] | -3% |
| (c) consumer: the tier's per-unit k3 into the pair map (mean path), L10-14 | +1% / -1% | -0.034 / +0.160 | [-0.165, +0.105] / [+0.008, +0.316] | fitted +0.76%: predicted -0.037 (2%) |
| (c) consumer: the tier's plane read, L10-14 | +1% | +0.102 | [+0.029, +0.177] | -5% |
| (c) consumer: the tier's per-unit k3 in the final layer | +1% / -1% | -0.004 / +0.032 | [-0.039, +0.031] / [-0.003, +0.068] | fitted: predicted -0.006 (0%) |

**VALID40** (the requested measurement):
- The uncorrected schedule: raw **+2.889%** vs all-384 (37/40 worse; CI [+2.23, +3.53]). This reproduces the analyst's +2.88% on my code.
- (a) at its per-join fit (s10 = 1.0024, s13 = 0.9992, from the first 21 TRAIN nets): **+0.078 pt** [-0.001, +0.160], 15/40 better. It
  recovers -2.7% of the penalty, i.e. nothing.
- The other candidates were not taken to VALID40, because each fitted value predicts at most 0.04 pt on TRAIN24 with CIs that exclude
  0.5 pt. The timing probe costs bill, which puts it outside the task.

**Reading.**
- (a) The kept tier's scale already sits at its optimum. The bowl is steep: 1% on F costs +1 to +2.5 pt. The fitted per-join scales
  predict -0.03 pt, about 1% of the penalty.
- (b) As written, (b) has nothing to act on. The old tier stores third-order atoms (kappa3 = sum lam p x p x p). Its Gram, sum lam^2 p p^T,
  only ranks the rows for the Nystrom frame, and nothing in the covariance or the per-unit variance consumes it. Restoring each row's
  dropped Gram energy inside the tier changes nothing.
- (c) Value flow:
  - Reweighting the frame between the common and difference parts finds no better rank-320 frame. Toward v is worse; toward u is flat.
    The same holds at 384 (v x 10: -0.07% vs all-384, 4 nets).
  - The timing probe shows where the damage is consumed. The join layer's own consumers carry 46% [18, 74]%, and the next two layers
    carry the rest.
  - Per unit of bill that is uniform. On VALID40 the join layers hold 17% (L10) and 19% (L13) of the rank-dependent bill, against 17% and
    24% for the two layers after each. Cutting later therefore stays on the same raw-vs-bill exchange rate:
    - its bill saving falls from -3.38% to about -1.79% (the analyst's unit costs);
    - it needs about 58% recovery to break even, and it gets 46%.
  - No per-layer rank timing beats the per-join schedule.
  - Consumer-specific scales (my value-flow idea) find no consumer that wants more of the tier content:
    - per-unit k3 into the mean path: +1% gives -0.03 pt; the fit predicts -0.04;
    - the plane read: +1% is worse, +0.10 pt;
    - the final-layer per-unit k3: flat.
    Each costs nothing in a native build (a partial sum or a scale of a sub-block the card already computes), but nothing moves.
  - Code note: v5/v6's final-k3 knob REPLACED the card's own k3_15 hook (s6bcard.l15_rowcut), so its zero-strength control was not the
    card. v7 chains the hook, and zero-strength identity is bit-exact for every knob (the read, pair map, final k3 and push knobs, net 36).
    Those v5/v6 rows are archived as r320_comp_train24c_INVALIDkf.*.
- **Law:** the cut's penalty sits in specific tail directions of the frame (low energy, aligned with the existing error, §34). Nothing
  that lacks those directions stands in for them: not a scalar, not a diagonal, not a reweighting, not the learned correction's gain.
  Carrying them costs what the cut saves. The only lever is the rank schedule itself (the analyst's grid; 384/384/320 has the best rate).
  The same holds at L13 alone (the L13 scale fit is flat, above).

**Refit survival (shadow beta\*).** Not run, because no fix passed the screen, so there is nothing to test for absorption. For loopR7s,
the evidence says a refit cannot rebuild the cut content:
- §21: a trunk refit at all-320 absorbed -0.009% of +3.35%.
- Today, a 1.03 gain on the learned correction (the cheapest thing a refit could learn) gives +0.06 pt.
Expect the co-adaptation to recover little of the +1.07% (L13-only) or +2.88% (384/320/320) raw penalty.

**Addendum (11:38): the L10 cut and its timing, torch VALID40** (`r320_comp8.py`, rows `r320_comp_v40l10.jsonl`; the same code and all-384 rows as above).
Why: the porter's native S27s (L13-only) saved only -0.69% util, against -1.87% modelled; S27r (384/320/320) saved -3.17%. If the joins
add, the L10 cut holds about -2.5 of the 3.2 bill points, the reverse of the torch unit-cost split.

| schedule | raw vs all-384 | wins vs all-384 |
|---|---|---|
| 384/320/320 | +2.889% | 3/40 |
| 384/320/384 (L10 only) | **+1.837%** (matches the analyst's additive +1.82%) | 8/40 |
| L10 only, the join layer at 384 and the cut from L11 | **+1.096%** | |

- The L11 cut recovers -0.741 pt [-1.066, -0.432] against L10 only (31/40 nets), which is 40% of the L10 penalty.
- Estimated adjusted with the inferred native bill:
  - L10 only: raw +1.84% and bill about -2.48% give about -0.69%, against S27r's -0.34% and S27s's +0.35%.
  - L11 cut: it beats L10 only if its native bill saving stays above 1.77%, i.e. if the L10 join layer holds less than 29% of the L10
    segment's native saving (the unit-cost model says 50%, but that model already missed L13 by 2.7x).
- Both need native builds before any decision: S27t = {7:384, 10:320, 13:384}, and a per-layer truncation at L11.
- Correction to the table: L10 only is better than all-384 on **6/40** nets, not 8/40 (typed before the count). The L11-cut variant is
  better on 7/40.

## 36. Value-weighted row selection at the L10 Nystrom join (lead's task 11:4x; `r320_vsel.py`, `r320_vsel2.py`, rows `r320_comp_vs24.jsonl` and `r320_comp_vsdiag.jsonl`; 2026-09-25) — FAIL at the TRAIN24 screen
**Setup.**
- Card and code as in §35. The new script matches the l10 rows bit-exactly: net 37, plus the 12 diagnostic re-runs match the screen.
- Schedule 384/320/384 (the L10 cut; VALID40 raw +1.837% on this code).
- At the L10 join only, the Nystrom rows are ranked by d_i * v_i^2 instead of the Gram diagonal d_i (energy). The value factors:
  - code 1: v = the gate of the row unit (layer 9);
  - code 2: v^2 = |W_10[i, :]|^2, how strongly the unit feeds layer 10;
  - code 3: v^2 = sum_j W_10[i,j]^2 |z_j| phi(z_j) / var_j, the loading on the layer-10 units whose mean is most sensitive to their k3
    (dE[ReLU]/dk3 = -z phi(z) / (6 sigma^2));
  - code 4: v^2 = sum_j W_10[i,j]^2 min(g_j, 1-g_j), the loading on kink units;
  - weighted Nystrom (code 3 plus weighted coordinates): basis D_v^-1 orth(D_v A[:, rows]) and F = orth^T D_v J. The frame then
    minimizes the v-weighted atom error.
- All are truth-free and cost O(n^2) at the join (codes 3 and 4 reuse the layer's Cpre diagonal in a native build), so the bill is zero.

**TRAIN24** (the L10 penalty here is raw +0.964% vs all-384, 19/24 nets worse). Change vs energy selection, in points of the all-384 pool,
paired bootstrap:

| selection at the L10 join | change, pt | 95% CI | better on |
|---|---|---|---|
| code 1: gate | +0.174 | [-0.302, +0.708] | 7/24 |
| code 2: next-layer row norm | +0.118 | [-0.241, +0.473] | 10/24 |
| code 3: k3-sensitivity loading | +0.348 | [-0.029, +0.761] | 9/24 |
| code 4: kink loading | +0.282 | [-0.149, +0.651] | 8/24 |
| code 3 + weighted coordinates | +0.350 | [-0.035, +0.763] | 10/24 |

The bar (at least 0.3 pt better, CI excluding 0) fails: every arm is worse. VALID40 and the held-out 40 were not run, per the lead's
rule for decisive TRAIN24 nulls.

**Why** (diagnostic on TRAIN nets 36, 45 and 52):
- The value factors barely move the selection. The value-ranked top 320 rows share 297-313 of their 320 rows with the energy top 320
  (out of 832-864 live rows).
- The energy rows already hold 99-100% of the value-weighted score that the value rows hold (e.g. 0.738 vs 0.739).
- The Gram diagonal spans orders of magnitude across rows, while every truth-free value factor varies about 2x. So energy decides the
  selection. The 7-23 rows that do change carry none of the tail content, and changing them costs +0.1 to +0.35 pt raw.

**Reading.** This is the same law as §24 and §35:
- The frame must serve every consumer downstream of the join: the join layer pays only 40% of the L10 penalty, and L11-15 pay the rest.
  Energy is the right proxy for that.
- Next-layer loading is nearly flat across rows. Where it does move rows, it drops content that later layers need.
- The error-aligned value of the tail directions is invisible to any truth-free per-row score.
- Not run: the lead's (b), value selection at all-384. At rank 320 it changes only 2-7% of the rows and hurts; at 384 it would change
  even fewer.

**§35 addendum resolved (12:15; native pricing by the lead and the analyst, not by me).**
- The porter's per-layer bill map for the L10 cut: L10 -0.79, L11 -0.40, L12 -0.38, L13 -0.39, L14 -0.33, L15 -0.19 (total -2.48%).
- So the join layer holds 31-33% of the saving (analyst: net 35 31%, net 100 33%; analyst RESULT section 44), above the 29% break-even.
- The L11 cut therefore loses, whatever the truncation costs:
  - with free truncation: bill -1.69%, adjusted -0.61% vs -0.69% for L10 only (VALID40);
  - with the rotation it would actually need (+0.38% bill, analyst's native product model): about -0.23%.
- On the held-out nets: L10 only about -1.23%, the free-truncation variant about -0.93%.
- The native truncation build is not needed. S28 uses 384/320/384.

## 37. Truncating only the L15 k3 read of the old tier (lead's task 12:0x; `r320_l15split.py` v1-v3, rows `r320_comp_{l15v10,l15o10,l15r30,uz24}.jsonl`; 2026-09-25) — the plain cut FAILS; a split read is about -0.28% adjusted on the lead's bill model (borderline; the native bill decides)
**Setup.**
- Torch card with the **loopR3** params (not loopR7), S28 schedule 384/320/384, on VALID40.
- Baseline: my l10 rows (raw +1.837% vs all-384). Only the L15 read of the old tier changes; the tier itself stays.
- An L15 change moves only the final k3. No unit selection sits downstream, so there is no knife-edge noise, and 10 nets already give
  tight CIs. The r = 384 split identity reproduces the baseline to rounding (+0.000 pt).
- Bill uses the lead's model: the old part of `_l15_rowcut_k3` is about 1.8% of the bill at r = 384, linear in the product's inner
  dimension.

**Results** (raw change vs 384/320/384, in points of the all-384 pool):

| L15 read of the old tier | nets | raw change, pt | 95% CI | bill (model) | adjusted (est.) |
|---|---|---|---|---|---|
| (a) plain top-256 frame directions | 40 | +0.908 | [+0.653, +1.175] | -0.60% | +0.29% |
| (a) plain top-192 | 10 | +3.65 | [+2.89, +4.31] | -0.90% | +2.7% |
| (a) plain top-128 | 10 | +10.7 | [+8.8, +12.7] | -1.20% | +9% |
| (b) old tier dropped from the final k3 | 10 | +145 | [+133, +157] | -1.80% | MSE x2.4 |
| split: u on the first 256, v on all 384 | 40 | **+0.022** | [-0.009, +0.052] | -0.30% | **-0.28%** |
| split: u on 256 ranked by its own L15 energy | 40 | +0.021 | [-0.007, +0.050] | -0.30% | -0.28% |
| split: u on 192 | 40 | +0.187 | [+0.129, +0.249] | -0.45% | -0.27% |
| split: u on 192 ranked by its own L15 energy | 10 | +0.253 | [+0.142, +0.370] | -0.45% | -0.20% |
| split: u on 128 | 10 | +0.953 | [+0.555, +1.389] | -0.60% | +0.34% |
| split: u on 192 and v on 320, both ranked | 10 | +0.415 | [+0.176, +0.645] | -0.60% | -0.19% |

**The split read.**
- Each birth pair is p+- = U(f_u +- f_v/2). Its k3 is about 3c (U f_u)^2 (U f_v) + (c/4)(U f_v)^3.
- f_u, the columns the pair shares, is energy-concentrated in the frame's leading directions. f_v, the birth-unit difference, carries
  the k3 and needs every direction.
- Native form: form F_u and F_v from F by pair sums and differences, compute P_u = U[:, :r_u] F_u and P_v = U F_v, and set
  pp+- = P_u +- P_v/2.
- The old part then costs (r_u + r) / (2r) of today's.

**Split tier at every join** (TRAIN24; the u part zeroed beyond r_u after each join, so every later consumer sees it):
- r_u = 256: **+1.650 pt** [+1.171, +2.150], better than energy-only on 3/24 nets. r_u = 192: +4.81 pt.
- r_u = 384 reproduces the baseline to rounding (-0.001 pt).
- So the u tail that the final read can drop costs 75x more once it propagates. The saving would come from every layer's old push, but
  this costs more raw than the L10 rank cut for any plausible bill.

**Reading.**
- The lead's hypothesis (the final read tolerates a deep cut because nothing propagates) fails for the plain cut. The final k3 feeds
  the final mean directly, the old tier dominates it (without it the MSE is 2.4x), and a plain rank-256 read costs +0.9 pt.
- What the final read does tolerate is truncating the pairs' common part. That brings it to about -0.28% adjusted, just short of the
  -0.3% bar.
- The native per-direction cost of this read is uncertain. In the porter's map (fable_porter_v1/stack/billmap/REPORT_v3.txt), L15
  `_l15_rowcut_k3` moves -0.037..-0.047% when L13 drops 32 directions (384-384-352), but -0.19..-0.27% in schedules that cut L10.
  So a native 2-net bill of the split read has to decide.
- Law: rank-cut damage from the u part is a propagation effect (L15 only: +0.02 pt; every join: +1.65 pt). Damage from the v part is
  direct (it is the k3).
