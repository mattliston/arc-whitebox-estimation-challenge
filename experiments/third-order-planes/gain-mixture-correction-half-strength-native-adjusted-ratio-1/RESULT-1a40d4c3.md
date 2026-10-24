# Analyst lane: validation of the gain-mixture b fix (2026-09-22)

**Cold read, sections 16-21 (09-22 22:45 to 09-23 02:15; S2/S4 torch cards and native census):**
1. Bill dials do not pay: no early-layer (16) or deep-layer (18) dial saves 3% of the bill for 0.5% raw error, and native tier rank 352 (19) is +0.1% / +0.9% adjusted (r35 / r100) because the fast-multiply kernels absorb its saving.
2. Spending more at the last layers buys nothing (18): a larger or exact old tier, more births and more response rows move raw error by -0.2% to +4%, and the rank-384 frame sits exactly at the knee.
3. Retraining the learned heads on S2's physics gains nothing (17): +0.1% to +0.2% on 16 validation nets.
4. The plane-to-distinct feed (20) is real but priced out: its best form cuts raw error 2.4% on all 4 nets for +61% bill, since each dense 1024^3 product costs 0.53% of S2's bill; the k3-only form is flat.
5. The component audit (21) finds no fat: every auxiliary part is worth 3-500x its bill, and 92% of the bill sits in the carriers (young push, atoms read, old tier, star, covariance push), so any bill cut must come from carrier redesign.

The fix fails at the level of the physical state, not because the learned closure is co-adapted. Every variant of the b fix makes the
per-unit k3 or the per-unit variance of the torch card worse against Monte Carlo truth, with or without the learned parts. The (2,1)
plane is not a lever while the star (the plane-only carrier) transports it: even the TRUE plane makes the card 5-9x worse.
Recommendation: close the b-fix line in all modes and spend no refit cycles on it.

All numbers come from the torch research card. The snapshot `model_bank_snapshot.py` is a copy of `fable_refit_v1/model_bank_bfix.py`
taken 18:20; `capture.py` wraps it without edits, with parity exact to 0.0. The 8M Monte Carlo planes are in
`public_g19_attribution_v1/mps_planes_rank{0035,0100}.npz`, the 8M post planes in `mps_planes_rank0035_post*.npz`, and the per-unit
truths in `mps_truth_rank*_P32M.npz`. "Final MSE" is the layer-15 MSE against truthpost.

## 0. Provenance correction: the hub planes are a model, not truth
`hub_planes_rank*_honest.npz` holds the hub-carrier MODEL planes, the output of `hub_carrier_v*` whose run log is `hub_colcarry.log`.
They are not Monte Carlo truth. On r35 they differ from the 8M MC planes by 26-38% off-diagonal and 16-26% on the diagonal. The
difference between two independent 8M MC draws (`mps_planes_rank0035_noise2.npz`) is only this large for one draw:

| layer | hub vs MC, off-diagonal | hub vs MC, diagonal | 8M noise, off-diagonal | 8M noise, diagonal |
|---|---|---|---|---|
| 2 | .377 | .258 | .078 | .054 |
| 5 | .303 | .196 | .037 | .023 |
| 9 | .261 | .162 | .020 | .013 |
| 12 | .280 | .188 | .015 | .010 |

Every plane number in `plane_bias_map*`, `bfix_additive_test`, `k21off` and `offb` was graded against the hub model. Section 1 regrades
against the MC truth. The Sym(C,mu) structure survives the regrading, but the early-layer "40% error" belongs to the hub model.

## 1. Q1: does the fix make the card's state more accurate?
Baseline card against MC truth. The kappa and the Sym share come from an LSQ fit of the off-diagonal error on
Sym(C,mu) = v_a mu_c + 2 C_ac mu_a. Files: `state_accuracy_variants_v1.json` and `state_accuracy_variants_v1.log`.

| r35 layer | plane off-diag rel err | Sym share of err energy | kappa | top-1 share | per-unit k3 rel err | v rel err |
|---|---|---|---|---|---|---|
| 2 | .082 | .00 | - | - | .032 | .00025 |
| 4 | .353 | .15 | 1.0e-3 | .45 | .035 | .00031 |
| 6 | .485 | .33 | 2.3e-3 | .55 | .040 | .00046 |
| 8 | .571 | .45 | 3.6e-3 | .61 | .046 | .00066 |
| 10 | .601 | .50 | 4.3e-3 | .61 | .048 | .00095 |
| 12 | .623 | .58 | 4.9e-3 | .66 | .055 | .00127 |
| 14 | .638 | .64 | 5.6e-3 | .68 | .058 | .00173 |

For r100 at L8, L12 and L14 the plane error is .576, .614 and .618 and the Sym share is .50, .60 and .65. The top singular pair has
its left vector along v and sd (cos .92-.97) and its right vector along mu (cos .86-.97).

The per-unit diagonal already carries the gain term. The true k3 is 70-94% explained by 3*beta*v*mu, and the card has the same beta:

| layer (r35) | radial/6 | card b along m | truth beta | card beta | R2 of the truth fit |
|---|---|---|---|---|---|
| 2 | 2.6e-3 | 0.6e-3 | 4.49e-3 | 4.50e-3 | .71 |
| 8 | 6.6e-3 | 2.3e-3 | 8.43e-3 | 8.47e-3 | .89 |
| 14 | 9.4e-3 | 2.9e-3 | 1.02e-2 | 1.04e-2 | .92 |

On the diagonal, most of the gain content is carried by the atoms and the star, not by b. Any fix therefore must not touch the diagonal.
The true POST plane holds Sym(C_y,m_y) with coefficient 1.1-1.7x radial/6: 5.9e-3 at L2 and 1.0e-2 at L13. The card's post target
lacks about half of that. So the gain structure is real in the truth at both levels.

Variants against MC truth, at L14 or the final layer:

| variant | final MSE r35 / r100 | plane off-diag err L14 r35 / r100 | k3 err L14 r35 / r100 | v err L14 r35 / r100 |
|---|---|---|---|---|
| base | 1.00 / 1.00 | .638 / .618 | .058 / .064 | .0017 / .0017 |
| partition .5 | 1.02 / 1.14 | .598 / .566 | .105 / .139 | .0022 / .0022 |
| partition .85 | 1.20 / 1.53 | .578 / .542 | .144 / .197 | .0028 / .0030 |
| partition 1.0 | 1.32 / 1.79 | .571 / .536 | .162 / .222 | .0031 / .0034 |
| additive .5 | 14.7 / 22.5 | .404 / .372 | .406 / .464 | .0166 / .0204 |
| offb .3 | 4.4 / 5.3 | .508 / .477 | .067 / .073 | .0064 / .0084 |
| k21off_pair .5 | 156 / 176 | 1.57 / 1.37 | 1.67 / 1.47 | .044 / .044 |

Verdict for Q1: the plane improves by 6-13% relative under partition and by 20-40% under additive and offb; k21off overshoots. The
per-unit k3 (partition, additive) or v (all variants) gets 2-25x worse, and the final MSE rises. That is the physical state getting worse, not
co-adaptation. The mechanism is incomplete. After partition, the remaining plane error keeps its rank-1 Sym shape (top-1 .64-.67,
kappa down only 7-26%). After the additive fix the Sym shape is gone, but a non-Sym residual of relative size .38-.43 stays in every
variant.

## 2. Why the Sym term alone hurts: the covariance read
`pair_response.py` feeds the card's pair map at layer l a modified plane and checks v_{l+1} = diag(W^T Cpost W) against the 32M truth
(r35, relative rms error of v_{l+1}):

| input plane | L8 | L10 | L12 | L13 |
|---|---|---|---|---|
| card | .081% | .105% | .146% | .166% |
| card with off-diagonal = MC truth | .073% | .094% | .128% | .153% |
| card + kappa*Sym(C,mu) off-diagonal | .276% | .353% | .410% | .427% |

`read_decomp.py` explains the gap. The pair map reads the plane exactly as dC[a,c] = (E[a,c] d_a g_c + E[c,a] g_a d_c)/2, where d is
the kink density and g the gate (cos 1.000 against the pair map). Weighted by that read, the true error has NO Sym component: the
kappa is 3e-5, 5e-5 and -3e-6 at L6, L9 and L12, against a Frobenius kappa of 2.3e-3 to 4.9e-3. Split by the gate of the row
(r35 L12):

| row group | rows | kappa | share of error energy | share of the covariance read |
|---|---|---|---|---|
| always off (g < .02) | 239 | 5.8e-3 | .38 | .03 |
| .02-.2 | 144 | 3.3e-3 | .07 | .11 |
| kink .2-.8 | 256 | 3.4e-3 | .12 | .72 |
| .8-.98 | 144 | 3.2e-3 | .07 | .10 |
| always on (g > .98) | 241 | 5.9e-3 | .36 | .04 |

The Sym error lives in rows that the covariance map cannot see. The Sym part's read alone moves v_{l+1} by .16-.39% rms, and the
remaining error cancels it: the read of the full error moves v_{l+1} by only .04-.07%.

## 3. The true plane is harmful through the star
In `block_oracle.py`, the off-diagonal of the plane fed to the pair map is replaced at L2-14. Learned consumers, births and the
partition keep the card's plane.

| plane fed to the pair map | final MSE r35 | final MSE r100 | k3 err L12 r35 |
|---|---|---|---|
| MC truth, all rows | 5.3x | 9.3x | .334 |
| MC truth, kink rows (abs(mu/sd) < 1.5) | 1.01x | 1.10x | .055 |
| MC truth, always-on rows | 13.4x | 20.5x | .457 |
| kappa_fit*Sym(C,mu), fitted to truth (any rows) | 21-22x | 34-35x | .47-.56 |

In `transport_test.py` the input is the true 8M post plane at layer l, pushed with the card's `_flms_star` and compared with the truth
at l+1 (r35). The result has 43-49% off-diagonal error and 26-33% per-unit k3 error even with perfect input. The card's own plane at
l+1 has only 3-6% k3 error because atoms and b carry full tensors. Carrying the Sym(C_y,m_y) part as an exact full tensor (the
hybrid) helps only a little: at L13 the off-diagonal error goes from .43 to .36 and the k3 error from .27 to .23. So the card's plane
error partly compensates the star's transport defect on the diagonal. Correcting the plane without replacing the transport breaks
the per-unit k3.

The ceiling of the plane's covariance channel alone is in `cov_oracle.py`: the MC truth plane is used ONLY for the post covariance,
and every other output comes from the card's plane.

| covariance oracle rows | from L2 | from L6 | from L9 |
|---|---|---|---|
| all, r35 / r100 | 1.11 / 1.19 | 1.08 / 1.11 | 1.05 / .95 |
| kink, r35 / r100 | 1.05 / 1.13 | 1.04 / 1.06 | 1.00 / .94 |

The L2-5 rows are contaminated by 8M noise: the noise is 4-8% while the card's error is 8-46%. Even a perfect plane is worth about
+-5% through this channel.

## 4. Q2: which learned consumer carries the degradation?
Partition .85 against gamma 0 on r35, r100, r37 and r39 (`consumer_ablation_g085.log`):

| configuration | MSE ratio gamma .85 / 0 per net | geo-mean |
|---|---|---|
| all on | 1.20 1.53 1.26 1.38 | 1.34 |
| GNN head off | 1.42 1.79 1.54 1.58 | 1.58 |
| covariance edge off | 1.21 1.57 1.27 1.34 | 1.34 |
| message context off | 1.20 1.54 1.26 1.38 | 1.34 |
| birth hook static | 1.10 1.36 1.19 1.26 | 1.23 |
| all learned off | 1.33 1.69 1.50 1.49 | 1.50 |

No learned consumer carries the degradation. The GNN head already cancels part of it, since the ratio rises when the head is off. The
birth-coefficient hook amplifies it slightly. The degradation sits in the physics: the per-unit k3 channel for partition, and the
covariance read for the injections. A refit would have to re-learn a correction for a state that has become less accurate.

## 5. Q3: refinements of the partition fix
`refine.py`, final MSE relative to base (r35 / r100):

| variant | r35 | r100 | note |
|---|---|---|---|
| partition .5 | 1.018 | 1.137 | reference |
| partition .5, mean = host | 1.018 | 1.136 | host and m are equivalent |
| partition .5, diagonal kept at the gamma-0 value | .986 | 1.054 | off-diagonal effect alone; v gets worse |
| partition 1.0, diagonal kept | 1.155 | 1.310 | |
| partition .5, off-diagonal kept at the gamma-0 value | 1.200 | 1.270 | diagonal effect alone, the most harmful part |
| partition .5 at layers >= 8 only | .998 | 1.017 | neutral |
| partition .85 at layers >= 11 only | 1.002 | 1.023 | neutral |
| partition ramp .85*l/14 | 1.007 | 1.061 | |

No gamma, layer range, mean choice or diagonal treatment beats the baseline by more than the 1-2% noise on both nets. The diagonal
part (per-unit k3) should NOT be included: alone it costs +20-27%. The off-diagonal part alone is neutral at .5 and harmful at 1.0,
through the covariance read of section 2.

## 6. Q4: heads-only refit proxy
`refit_proxy.py` trains only P head, head_b and edge output, bias (1,072 values). It uses 8 TRAIN nets (35, 36, 38, 40, 41, 43, 44,
45) for 2 epochs with Adam at lr 1e-3, and grades on 8 VALID nets (37, 39, 42, 48, 59, 63, 66, 67). The control arm is gamma 0.

| arm | VALID8 mean before | after epoch 1 | after epoch 2 | against the refit control |
|---|---|---|---|---|
| gamma 0 (control) | 1.890e-8 | 1.905e-8 | 1.893e-8 | 1.000 |
| partition gamma .85 | 2.490e-8 | 2.412e-8 | 2.386e-8 | 1.260 |

The heads cannot improve the unchanged card: the control drifts +0.2%. With the fix, two epochs lower the MSE by 4.3%, which leaves
it 26% above the refit control; it started 32% above. Nothing suggests the fix can reach the baseline, let alone beat it. That fits sections 1 and 4: the physical state is
worse, and the heads can only paper over it.

## 7. Controls and lever ranking (added at the end)
The harm from the true plane is physics, not Monte Carlo noise. `block_oracle.py --layers=9,12` modifies r35 at L9 and L12 only:

| plane change at L9, L12 | final MSE ratio | k3 err L12 / L15 |
|---|---|---|
| true plane, always-on rows | 5.98 | .215 / .319 |
| one 8M draw's noise only (draw A - B)/sqrt(2), on rows | 1.007 | .055 / .064 |
| true plane, all rows | 4.76 | .217 / .275 |
| noise only, all rows | .997 | .055 / .064 |
| true plane, kink rows | .958 | .055 / .064 |
| noise only, kink rows | .991 | .055 / .063 |

The true plane in the KINK rows at L9 and L12 helps by about 4%. This is the covariance channel of section 3, consistent with the
L9+ covariance oracle at -0.3% (r35) and -6% (r100).

One-layer lever ranking (`pair_levers.py`, r35 and r100, L6/9/12/13). The pair map is called with one input swapped to truth, and the
outputs are graded against the 32M truth: the post mean before the GNN delta, and v_{l+1}.

| input swapped to truth | post-mean rmse change | v_{l+1} error change |
|---|---|---|
| pre mean mu (inherited from upstream) | -47 to -54% | 0 |
| per-unit v | -5 to -7% | 0 |
| per-unit k3 | -3 to -6% | 0 |
| per-unit k4 | -2 to -4% | 0 |
| off-diagonal K21 plane | 0 | -8 to -13% |
| K22 plane, K31 plane | 0 | 0 to -2% |
| all of them | -92 to -94% | -11 to -17% |

With exact inputs the pair map's mean is 12-16x more accurate. Three quarters of a layer's post-mean error variance is inherited
through mu. The next-layer variance error sits mostly in the carried covariance C, which cannot be swapped because no full-covariance
truth exists; the plane, K22, K31 and k4 together explain 11-17% of it.

## Recommendation
1. Close the gain-mixture b fix in all four modes: partition, additive, offb and k21off. The failures are physical and appear with
   the learned parts off.
2. Stop grading planes against `hub_planes_*_honest`. Use `mps_planes_rank{0035,0100}.npz` (8M MC, L2-15) and quote the 8M noise
   floor: .078 at L2, .037 at L5, .020 at L9, .015 at L12, .012 at L15.
3. The (2,1) plane is not a lever in this card. Its only mean-relevant channels are the covariance read at kink rows, worth about
   +-5% with the TRUE plane, and propagation through the star, where the true plane is worse. Improving the plane is useful only
   together with a transport that carries the always-on rows' content as a full tensor. The star transports the true post plane
   with 43-49% off-diagonal and 26-33% k3 error.
4. The levers that remain are the per-unit supplies and the carried covariance, not the plane. The per-unit v, k3 and k4 each cut
   one layer's post-mean error by 2-7%, and together with mu by about 93%. The v_{l+1} error sits mostly in the carried covariance C.
   The plane is worth at most about +-5% as a correction to C at the kink rows. Section 7 has the ranking.

Scripts, all in fable_analyst_v1: `capture.py`, `state_accuracy.py`, `dump_state.py`, `pair_response.py`, `read_decomp.py`,
`block_oracle.py`, `transport_test.py`, `consumer_ablation.py`, `refine.py`, `cov_oracle.py`, `refit_proxy.py`.

## 8. Follow-up (lead's re-scope, 19:20-19:40): Gaussian-born K31 tree terms
The one-step Gaussian-born fourth cumulant at layer l-1, pushed to the (3,1) plane at layer l, has two tree classes. Here
B = W^T (g o C_off), D = diag(phi/sigma) (E f''), D3 = diag(-alpha phi/sigma^2) (E f''') and P = (W^T D) o B.

| term | formula | cost per layer |
|---|---|---|
| STAR | 3((W^T o B o B) D3) B^T + ((B o B o B) D3) W | 3 n^3 MACs including B |
| CHAIN | 6[((W^T D) o (P C_off)) B^T] + 6[(B o (P C_off)) D W] | 3 n^3 MACs beyond B |

The matrix formulas equal a brute-force sum of the tree values over all index quadruples to 1e-15 (`k31_tree_cert.py`). The per-quadruple
Monte Carlo check at rho = .12 was too noisy to add information. The tree values follow from the standard Hermite/Wick expansion.

Truth check (`k31_tree_truthcheck.py`): the trees are built from the Monte Carlo-true state at l-1 (`fable_mc_v1/mc2_rank*_P32M.npz`)
and regressed with the card's iso form v_a C_ac against the 8M MC K31 at l.

| rank, layer | R2 iso | R2 iso+star+chain | star coef | chain coef | star rms / truth | chain rms / truth |
|---|---|---|---|---|---|---|
| r35 L5 | .428 | .462 | .91 | 1.13 | .146 | .121 |
| r35 L8 | .473 | .507 | .90 | 1.29 | .158 | .106 |
| r35 L11 | .502 | .533 | .95 | 1.22 | .155 | .093 |
| r35 L14 | .549 | .576 | 1.13 | 1.33 | .132 | .070 |
| r100 L5 | .463 | .496 | .94 | 1.15 | .140 | .120 |
| r100 L8 | .502 | .531 | .88 | 1.26 | .152 | .098 |
| r100 L11 | .524 | .553 | .99 | 1.43 | .145 | .077 |
| r100 L14 | .658 | .678 | 1.10 | 1.71 | .110 | .051 |

The noise of the 8M truth K31 at L5 is .345 relative (independent draw). Both trees are correct in scale, but they add only 2-3% of R2
over the iso form. They are small next to the true plane.

Closed loop (`k31_tree_closedloop.py`): trees built from the card's own state at l-1, fed through the physicist's
PHYS_SUPPLY fn as remainder31 on top of t4*rho, pair map only, learned parts ON, no refit. Change of the final MSE:

| supply | r35 | r100 | r37 | r39 | mean |
|---|---|---|---|---|---|
| star + chain, L1-14 | -1.33% | -4.10% | -0.69% | -0.91% | -1.76% |
| star, L1-14 | -0.87% | -3.15% | -0.99% | -0.77% | -1.45% |
| chain, L1-14 | -0.42% | -0.73% | -0.22% | +0.13% | -0.31% |
| star + chain, L2-6 | +0.51% | -1.29% | +0.28% | +0.07% | -0.11% |
| star + chain, L11-14 | -0.68% | -0.96% | -1.16% | -1.03% | -0.96% |

Each tree costs about 1.6% of the bank bill per layer: n^3 MACs is 2.1e9 flops against 4.06e11 flops at util .1845. Star + chain at
L11-14 costs about 13% of the bill for about 1% of MSE. At all layers it costs about 44% for about 1.8%. Neither can pay.
The -12% truth-K31 oracle must come from K31 structure outside the one-step Gaussian trees. Candidates are the gain-on-skewed-base
k3 (x) mu term the physicist found, coincident-index terms, loops, and transported pre-existing non-Gaussianity.

## 9. Rank-100 one-step budget (replication of the lead's r35 budget)
`fable_mc_v1/mc_second_order_mps.py 100 32 4,7,10,13` produced `mc2_rank0100_P32M.npz`: 584 s, 1.2 GB RSS, cov noise .0022/.0018/.0014.
`onestep_budget.py 100 7,10,13` wrote `fable_mc_v1/onestep_budget_r100.log` (34 s, 6.7 GB RSS). v_{l+1} rel-rms error (%) and
post-mean rms at L7 / L10 / L13:

| inputs | v_{l+1} L7 | v_{l+1} L10 | v_{l+1} L13 | post-mean rms L7 / L10 / L13 |
|---|---|---|---|---|
| all truth (closure floor) | .0172 | .0160 | .0167 | 8.3e-6 / 8.5e-6 / 8.8e-6 |
| truth state, card K21 off-diagonal | .0400 | .0531 | .0666 | same as floor |
| truth state, card K31 form | .0249 | .0295 | .0360 | same as floor |
| truth state, card K22 off-diagonal | .0195 | .0184 | .0196 | same as floor |
| truth state, card k3 / card k4 | .0172 / .0173 | .0166 / .0163 | .0189 / .0172 | 3.1e-5 / 3.0e-5 / 3.5e-5 (k3); 2.9e-5 / 3.3e-5 / 3.3e-5 (k4) |
| card state | .0725 | .1122 | .1576 | 1.14e-4 / 1.29e-4 / 1.37e-4 |
| card state + truth C | .0419 | .0568 | .0738 | 1.08e-4 / 1.23e-4 / 1.30e-4 |
| card state + truth K21 off-diagonal | .0639 | .1003 | .1425 | unchanged |
| card state + truth K31 | .0712 | .1114 | .1562 | unchanged |
| card state + truth mu | .0725 | .1122 | .1577 | 5.3e-5 / 5.7e-5 / 6.6e-5 |
| card state + truth k3 and k4 (fixed) | .0722 | .1120 | .1583 | 1.06e-4 / 1.22e-4 / 1.25e-4 |

BUG in `fable_mc_v1/onestep_budget.py`: the row `card_state_truth_k3k4` calls `mix(K21_c, K21_t, 'card', 'truth')` with its arguments
swapped. It therefore feeds the TRUTH off-diagonal K21 with the CARD k3, and its v_{l+1} simply repeats the truth-K21off row.
`fable_analyst_v1/onestep_budget_fixed.py` corrects it (last row above): truth k3+k4 leave v_{l+1} unchanged and cut the post-mean rms by
5-9%. The r100 budget reproduces the r35 picture. Half of the v_{l+1} error is the inherited covariance C. The K21 off-diagonal adds
10-12%, and K31 and K22 are small on the card state. Half of the post-mean error is the inherited mean. Card k3 and card k4 each
inject about 3e-5 per layer against a 1e-5 floor.

## 10. Per-unit k4 line (the lead's measurements, recorded at the lead's request)
k4 ~ a v^2 + 0.7 k3^2/v (8M R2 .71-.89 against .58-.76 isotropic) scores -3.9% with learned parts OFF, +0.7% with them ON, and
+2.9% with decoupled features. At the final layer only it scores +0.3..+1.4% coupled and +0.9..+3.8% decoupled. The GNN heads already
absorb per-unit k4 errors. CLOSED.

## 11. Zero-cost row factors for the radial-rebuilt per-unit k4 and K22 plane (lead's task, 19:45-20:50)
No row factor for k4 or K22 adds to the physicist's K31 row factor. The best zero-cost combination remains the K31 row factor alone:
-1.27 / -1.61 / -0.19 / -3.03% on r35 / r100 / r37 / r39. It misses the 1% bar on r37, and nothing new was sent to the porter.

Monte Carlo fit (`k4k22_truthfit.py`, card state vs truth on r35 and r100, both TRAIN nets), with q = mu_a k3_a / v_a^2:

| quantity | card error vs truth (L8 / L14) | best simple form | error after | truth noise |
|---|---|---|---|---|
| per-unit k4 (32M) | .186 / .271 (r35) | radial v^2 (0.93 + 0.35 q), c1 .25-.45 at L6-15 | .166 / .220 | .015-.058 |
| K22 off-diagonal (8M) | .196 / .272 (r35) | row factor sqrt(r_a r_c), r = 0.95 + 0.35 q | .187 / .253 | .045-.136 |

The k4 form matches the physicist's k4 = radial (v^2 + c mu k3) with c = 0.36-0.46. The K22 gain-skew term
(radial/2)(mu_a K21_ca + mu_c K21_ac) with the card's K21 reduces the error by about the same amount as the row factor.

Closed loop (`rowfactor_closedloop.py`, physicist harness, learned parts ON, no refit, layers 2-14 plus the final map for k4). Constants
were picked on TRAIN nets 40/41/43/44 only (logs `rf_train_v1..v4.log`), with geo-mean change on those nets:

| variant | TRAIN geo-mean |
|---|---|
| K31 row factor (physicist, reference) | -1.79% |
| k4 (0.93 + 0.35 q), K31 held | +0.72% |
| k4 (0.93 + 0.35 q), K31 follows k4 | -0.41% |
| k4 (1.0 + 0.35 q) | +3.33% |
| k4 x0.96 / x0.93 / x0.90 with K31 row | -1.71 / +0.31 / +4.12% |
| K22 row factor (0.95 + 0.35 q) | +9.22% |
| K22 gain-skew term x0.6 | +170% |
| K22 scale x0.95 / x0.97 / x1.03, all outputs | +42 / +13 / +28% |
| K22 x0.9 in the covariance output only, + K31 row | -2.52% |
| K22 (0.9 + 0.35 q) covariance only, + K31 row | -2.54% |

Any K22 change that reaches the radial (through the post (2,2) plane) is a cliff. The covariance-only split keeps the radial and the post
planes of the unmodified call. It is implementable in `_pair_series_fused` as a scale of the e22 term in output channel 0 only.

Evaluation on the four standard nets (`rf_eval4_v1.log`):

| variant | r35 | r100 | r37 | r39 | geo-mean |
|---|---|---|---|---|---|
| K31 row factor (physicist) | -1.27% | -1.61% | -0.19% | -3.03% | -1.53% |
| K22 x0.9 covariance only, alone | +1.35% | -1.09% | -0.05% | +3.42% | +0.89% |
| K22 x0.9 covariance only + K31 row | +0.25% | -2.75% | -0.65% | -0.16% | -0.83% |
| K22 (0.9 + 0.35 q) covariance only + K31 row | +0.07% | -2.64% | -0.65% | -0.40% | -0.91% |
| k4 (0.93 + 0.35 q), K31 follows | +0.80% | +0.63% | +1.03% | -0.13% | +0.58% |
| k4 x0.96 + K31 row | -1.01% | +0.81% | +1.25% | -2.63% | -0.41% |

The TRAIN-chosen K22 covariance scale helps r100 and r37 but reverses on r35 and r39. It is a per-net compensation, not physics. The k4 row
factor is more accurate against truth (15-20% less k4 error) and still loses in the closed loop, consistent with the lead's finding that
the GNN heads absorb per-unit k4 errors. CLOSED: per-unit k4 and K22 row factors.

## 12. Row package on VALID40, and the structure of the carried-covariance error (lead's tasks, 20:50-21:45)

### Row package
The TRAIN grid ran on 40/41/43/44 in the deployable form (`rowpackage.py`): K31 rows in all pair-map outputs, K22 rows reaching only
the covariance output. Geo-mean on those nets:
- K31 (0.8, 0.5) alone: -2.03%.
- + K22c (0.9, 0.5): -2.74%. With (0.9, 0.35): -2.72%. With (0.95, 0.6): -2.53%. With (0.85, 0.35): -2.46%.
- The K31 c0 = 0.85 variants are 0.25-0.3 points weaker.

VALID40 (torch, learned parts ON, no refit, base from the physicist's `valid40_rowmod.jsonl`; `rowpackage_valid40.jsonl`):

| variant vs the unchanged card | mean-MSE ratio (95% CI) | wins | worst net |
|---|---|---|---|
| K31 (0.85, 0.5), the physicist's, as ported | .9856 (.9827-.9885) | 40/40 | r199 .9990 |
| K31 (0.8, 0.5) | .9847 (.9809-.9884) | 37/40 | r79 1.0034 |
| K31 (0.8, 0.5) + K22 covariance-only (0.9, 0.5) | .9901 (.9855-.9946) | 31/40 | r135 1.0188 |

The K22 increment is 1.0055 (CI 1.0015-1.0097), 12/40 wins: HARMFUL, so the TRAIN gain was a 4-net fluke. K31 c0 0.8 vs 0.85 is
.9991 (CI .9979-1.0001), a wash. The porter was told not to build the K22 edit; the anchor is recorded in that message.

### Carried covariance error
dC = the card's carried post C minus the 32M truth (mc2), at L4/7/10/13, r35 and r100 (`dC_structure.py`, `dC_propagation.py`,
`dC_eigsubspace.py`, `dC_swap_closedloop.py`).

| layer | off-diagonal dC / truth | truth noise | perp share of error energy | nSV > noise top | top-30 SVD share | top-100 one-sided eigen share (random) |
|---|---|---|---|---|---|---|
| 4 | .46-.48% | .31% | .86 | 12-13 | .38-.39 | - |
| 7 | .62-.65% | .22% | .62-.64 | 38-49 | .47-.48 | .57-.58 (.10) |
| 10 | .80-.83% | .18% | .49-.55 | 55-66 | .54 | .62-.63 (.10) |
| 13 | 1.02-1.06% | .14% | .34-.38 | 71-81 | .58-.60 | .66 (.10) |

- **Candidate structures:** C, C^2, gate-weighted C, v-scaled C, kink-weighted C, sigma*phi outer, (Cm)(Cm)^T and Cm x sigma*phi. Projected
  by P, each explains under 1% of offdiag(P dC P), and jointly at most 1.2%.
- **Three-layer inheritance:** the linear response g_a g_c (W^T dC W) explains 23% (L4 to 7), 31-39% (L7 to 10) and 37-47% (L10 to 13) of
  dC_{l+3}, at scale 1.0-1.15. The remaining 55-75% is injected within those three layers.

Closed-loop swaps: the card continues after a structured part of E = offdiag(P dC P) is fixed. Final-MSE ratio, r35 / r100:

| fixed at L13 | r35 | r100 | share of the perp gain |
|---|---|---|---|
| all of E | .889 | .833 | 1 |
| top-10 / 30 / 100 SVD part | .966 / .935 / .913 | .966 / .886 / .843 | .20-.31 / .58-.68 / .78-.94 |
| kink x kink / kink x on / on x on block | .980 / .959 / .976 | .991 / .899 / .953 | .06-.18 / .37-.61 / .21-.28 |
| top-30 eigen-subspace of the card's P C P, one-sided | .919 | .889 | .67-.73 |
| top-100 eigen-subspace, one-sided | .910 | .835 | .81-.99 |
| top-100 eigen-subspace, both sides | .953 | .913 | .42-.52 |
| noise control (adds one 32M draw to E) | 1.017 | 1.029 | - |
| L7+10+13: all of E / top-100 one-sided / top-30 SVD | .766 / .783 / .958 | .707 / .718 / .851 | 1 / .93-.96 / .18-.51 |

Where to aim: the carried-C lever is the cross-covariance between the top ~30-100 principal directions of the card's mean-orthogonal
post covariance and all units. That is k x n numbers in a subspace the card knows without truth. A carrier that corrects only those
rows would capture 70-95% of the lever, which is about -22% at four layers.

## 13. Is the plane-driven covariance correction predictable? (lead's task, 21:40-22:05)
Target, per layer 2-14 (`dP_data.py`, the lead's planes_covonly construction):
- dP = offdiag(P [C_pair(truth K21, K22, K31) - C_pair(card planes)] P), with P = I - hh^T/|h|^2 and h the pair map's host.
- The truth planes are the 8M MC planes. K21 keeps the card's diagonal k3.

| quantity | L4 | L7 | L10 | L13 |
|---|---|---|---|---|
| dP / off-diagonal pair covariance (r35 / r100) | .34 / .36% | .33 / .33% | .36 / .33% | .37 / .32% |
| dP / offdiag(P dC P), the carried error (r35 / r100) | .79 / .82 | .66 / .65 | .61 / .56 | .56 / .53 |
| cos(dP, -P dC P) | .47-.50 | .54 | .49-.53 | .48-.50 |
| dP SVD energy, top-10 / top-30 / top-100 | .11 / .26 / .58 | .17 / .36 / .67 | .20 / .42 / .73 | .23 / .46 / .77 |

MC noise share of dP (`dP_noise.py`, independent 8M draw, r35): 85% at L2, 40% at L5, 14% at L9, 7% at L12. The target is mostly
signal from L9 on.

Predictability, cross-net R2 (fit on one net, graded on the other, both directions):

| model | L4 | L7 | L10 | L13 | pooled L2-14 |
|---|---|---|---|---|---|
| ridge on the exact covariance reads of 17 candidate planes + 2 covariance features (`dP_fit.py`) | .004-.008 | .012-.024 | .034-.046 | .066 | .025 |
| per-pair MLP (23 local dimensionless features, `dP_mlp.py`), projected dP | .005-.010 | .014-.019 | .022-.023 | .034-.040 | - |
| same MLP, raw unprojected D | .021-.030 | .075-.081 | .069-.125 | .15-.17 | - |

The candidate planes span the card's K21 and K21^T, Sym parts, k3-regression forms, the card's K22 and its vv / CC parts, gain-skew,
k3k3 and mumuC terms, and the card's K31 in both orientations with the q-row, gain-skew and P1 terms. The best single feature is P1
(pooled .021), then the K31 gain-skew term (.010). The ceiling of local predictability is about 5-7% of dP at L10-14, far below the
50% bar. The MLP finds more per-pair structure in the raw D, but mostly in the mean-aligned part, which the projection removes.

Closed loop (`dP_closedloop.py`, linear prediction with cross-fitted coefficients, added to the covariance output at strength s, L2-14,
learned parts ON). Final-MSE ratio:

| baseline card | s | r35 | r100 | r37 | r39 | geo-mean |
|---|---|---|---|---|---|---|
| unchanged | .25 | .9888 | .9943 | .9932 | .9920 | .992 |
| unchanged | .5 | .9796 | .9917 | .9902 | .9876 | .987 |
| unchanged | 1 | .9713 | .9940 | .9912 | .9892 | .986 |
| with the K31 row package (.85, .5) | .25 | .9901 | .9969 | .9930 | .9980 | .9945 |
| with the K31 row package | .5 | .9844 | .9982 | .9893 | 1.0011 | .993 |
| with the K31 row package | 1 | .9831 | 1.0123 | .9939 | 1.0233 | 1.003 |

Most of the prediction's value overlaps the K31 row factor. On top of it, the prediction adds about -0.55% at s = .25 (4/4), for one
extra pair-map call per layer. That is not a lever worth porting without a VALID40 check. The mean-orthogonal covariance lever needs
the non-local plane information itself, meaning a carrier, not a local closure.

## 14. Predicting the correction inside the top-k subspace (lead's task, 22:10-22:30)
Target (`R_data.py`, nets 35/100/37/39, truth = covhunter's 32M `mcall_rank*_P32M.npz` at every layer): R_l = P (C_card - C_truth) P U_k,
with U_k the top-k eigenvectors of the card's own P C P at layer l, k = 30 and 100.

Features (n x k, rows projected by P) are the card's plane-vector products, the terms through which plane errors enter the covariance
output's k x n slice:
- d o K21(g o U) and g o K21^T(d o U)
- d3 o K31(g o U) and g o K31^T(d3 o U), with the card's K31 = t4 rho
- d o K22(d o U)
- U, U Lambda, g o U, v o U, m o U and d o U

Here d = pdf/sd = E f'' and d3 = -alpha pdf/v = E f'''.

| layer | signal share of R, k=30 (1 - noise^2/R^2, from 32M A/B halves) | cross-net R2, k=30, train 35 / train 100 | k=100, train 35 / train 100 |
|---|---|---|---|
| 4 | .43 | -.001 / -.001 | .000 / .000 |
| 7 | .77 | .003 / .006 | .004 / .005 |
| 10 | .89 | -.003 / .001 | .001 / .001 |
| 13 | .95 | .006 / .009 | .004 / .007 |
| 14 | .95 | -.010 / .016 | .002 / .012 |

Band fits over L2-6, L7-10 and L11-14 reach at most .011. In-sample R2 is at most .05. The card's plane-vector products carry no
information about R, even though R is almost pure signal from L7 on.

Closed loop (`R_closedloop.py`, k = 30, cross-fitted band coefficients, C -= eps (R_hat U^T + U R_hat^T - U sym(U^T R_hat) U^T) at L2-14,
learned parts ON), final-MSE ratio:

| eps | r35 | r100 | r37 | r39 |
|---|---|---|---|---|
| .25 | 1.0015 | 1.0011 | .9952 | 1.0245 |
| .5 | 1.0115 | 1.0322 | 1.0010 | 1.0820 |
| 1 | 1.0607 | 1.1832 | 1.0418 | 1.2965 |

The prediction is fitted noise and hurts at every strength. CLOSED: the top-k subspace pinpoints where the value lies, but the card's
own planes do not predict the error there. Any correction must bring new information (a better plane carrier or more samples), not
re-read the card's planes.

## 15. The one-sided top-k block of the plane-driven correction (lead's narrowed follow-up, 22:30-22:45)
Target (`Rp_data.py`, r35 and r100, L2-14): R_k = P D P U_k, with:
- D = cov_pair(truth K21, K22, K31) - cov_pair(card planes), the lead's planes_covonly construction
- h = the pair map's host
- U_k = the top-k eigenvectors of the card's P cov_pair P, k = 10 and 30

Features (n x k, rows projected by P), 36 in all:
- seven plane-vector products: K21(g o U), K21^T(d o U), K31(g o U), K31^T(d3 o U), K22(d o U), P1(g o U) and P1^T(d3 o U), each raw
  and left-weighted by d, g and d3
- C U and U
- v, k3, k4, mu, g and d times U

Cross-net R2 (ridge, 35 to 100 and back; `Rp_fit.py`):

| band | k = 10 | k = 30 | signal share of R (r35, independent 8M draw) |
|---|---|---|---|
| L2-6 | .016-.019 | .014-.015 | .13 (L2), .61 (L5) |
| L7-10 | .049-.052 | .045-.048 | .86 (L9) |
| L11-14 | .085-.093 | .077-.080 | .93-.94 (L12) |
| L2-14 pooled | .044-.046 | .036-.037 | |

Per layer the best is .09-.11 at L13-14 (k = 10). The leading terms are the per-unit multipliers (v, g, k4, k3 times U) and the P1
contraction weighted by d3. The block is slightly more predictable than the full dP (5-7%) but far below the 25% bar, so I ran no
closed-loop test. LANE CLOSED: no local predictor, carrier needed.

## 16. Layer-resolved bill reallocation (lead's new lane, 22:45-23:30)
Target: at least 3% bill for at most 0.5% raw, by cheapening cost centers only at early layers.

### Native bill by layer (`bill_by_layer.py`, bank, r35, callers attribution; share of the 4.14e11-flop bill)

| function | L1 | L2 | L3 | L4 | L5 | L6 | L1-4 | L1-6 | all layers |
|---|---|---|---|---|---|---|---|---|---|
| newborn push (young columns) | .30 | .93 | 1.51 | 1.90 | 2.15 | 1.82 | 4.64 | 8.61 | 20.4 |
| old384 read (atoms' plane) | .62 | 1.21 | 1.77 | 2.13 | 1.99 | 1.84 | 5.72 | 9.55 | 18.1 |
| star push (3 products) | .87 | .87 | .87 | .87 | .87 | .87 | 3.48 | 5.22 | 12.2 |
| covariance push | .61 | .58 | .58 | .58 | .58 | .58 | 2.35 | 3.51 | 8.1 |
| tier (old384 push, slr) | 0 | 0 | 0 | 0 | 1.52 | 2.22 | 0 | 3.74 | 34 |
| all functions | 2.64 | 3.85 | 5.00 | 5.77 | 7.41 | 7.66 | 17.3 | 32.3 | 100 |

### Torch closed loop (learned ON, no refit), final-MSE change
`early_dials.py` records the per-layer alive, response, birth and column counts for each run.

| dial | layers | r35 | 4 nets (35/100/37/39) | effect on counts |
|---|---|---|---|---|
| birth floor .01 -> .02 | 1-4 | +1.4% | | births at L2-4 -3..-7% |
| birth floor .03 | 1-4 | +3.1% | | births -6..-10% |
| birth floor .1 | 1-4 | +19.7% | | |
| birth floor .1 | 0-3 | +5.7% | | columns at L2-8 -7..-12% |
| birth floor .2 | 0-3 | +98% | | |
| response floor .03 / .1 / .2 | 1-4 | +1.8 / +44 / +274% | | response rows -6..-50% |
| response floor .1 | 1-6 | +117% | | |
| star dropped / diagonal only | 1-4 | +27 / +23% | | |
| atoms' plane read truncated to rank 64 / 128 / 256 | 1-4 or 1-6 | +272 / +188-260 / +91% | | |
| star truncated to rank 64 (exact SVD) | 1-4 | -.61% | -.61 / -.37 / +.31 / -.38, geo -.26% | |
| star rank 128 (exact SVD) | 1-6 | -.63% | -.63 / -.37 / -.06 / -.61, geo -.42% | |
| star rank 128 (exact SVD) | 1-14 | -.37% | -.37 / -.41 / -.13 / -.00, geo -.23% | |
| star rank 128, deployable one-pass Gaussian sketch | 1-14 | +2.47% | +2.47 / +.49 / +3.79 / +3.14, geo +2.46% | |
| star rank 128, sketch + one power iteration | 1-14 | -.63% | -.63 / -.32 / +.52 / +.24, geo -.05% | |

### Findings
- **Birth and response floors are cliffs or near-free only when they cut almost nothing.** At L0-2 nearly every unit has
  min(g, 1-g) >= .03, so the floors barely cut columns until they reach .1, and by then the raw cost is +6-20%.
- **The star and the atoms' plane carry essential early content.**
- **The star's content is effectively rank <= 128 at every layer.** Exact SVD truncation to rank 128 is free, and even helps slightly on
  all four nets. The bill does not follow, for two reasons:
  - the star push's middle product (pushed o W^T) W stays n^3 for any low-rank A;
  - a deployable range finder needs a power iteration to match the exact SVD (a one-pass sketch costs +2.5%). The finder then costs
    about 8 n^2 r, about 1.1e9 flops at r = 128, which cancels the two products it saves (about 1.2e9).
  Net: about 0 to 1% of the bill.
- **Joining the first cohort earlier with a smaller frame (not tested):** by the per-layer table, the tier's fixed cost is about
  1.5%/layer (L5). Cohort 0's young push and read at L3-4 are worth about 2.1% over both layers. So the net is about 0.5% even at half
  rank, and the young_rank result says age-3 joins need rank >= 512 for 99.99% of the energy.

VERDICT: no early-layer or layer-resolved dial reaches at least 3% bill for at most 0.5% raw. The cheap-looking cost centers at early
layers (young columns, star, reads) carry content that later layers depend on. The only content that is structurally compressible
(the star at rank 128) has a transport whose dominant product does not shrink with the rank.

## 17. Retraining the learned closure on S2's physics (lead's task, 22:50-00:00)
S2 torch card (`s2card.py`): fable_refit_v1/model_bank_bfix.py with BFIX_GAMMA .1 (partition, mean m), plus the K31 row source patch (.85, .5)
at pair layers 1-14 (as physics_stack_test.py), plus the every3 join push (the make_push of join_schedule_test.py). Parity with native S2
(#331947): r35 1.61200e-8 vs 1.61237e-8 (-0.02%), r100 1.57727e-8 vs 1.57638e-8 (+0.06%). Nested non-reentrant checkpoints (pair map,
_bd_update, star, push, read) hold one forward+backward at 6.93 GB peak, against 7.97 GB without them.

Heads-only refit (`refit_s2.py --groups heads`: GNN head/head_b + edge output/bias, TRAIN[:32], VALID[:16], 2 epochs, lr 1e-3, batch 4):

| checkpoint | VALID16 mean MSE | vs the no-retrain control |
|---|---|---|
| S2 before retraining (control) | 1.80887e-8 | 1 |
| epoch 0 | 1.81263e-8 | +0.21% |
| epoch 1 | 1.81126e-8 | +0.13% |

There is no gain, which is under the 1% bar, so I did not start the full-group run. The lead started their own overnight full-group refit
(refit_s2_full_lr2e4) with this script. For reference, the bank's VALID16 is 1.8853e-8, so S2 is -4.1% in torch on these 16 nets.

Memory bug: `refit_s2.py` v1 cached every net's weights (64 MB per net, never freed). My 48-net run peaked at 9.87 GB. The file is fixed;
the v1 copy is kept as `refit_s2_v1_cachebug.py`. The lead's running job keeps v1 until it is restarted.

## 18. Deep-layer dials (L9-14), compensation knobs, and spending more at the end (lead's lane, 00:05-00:40)
All runs use the S2 torch card (`s2card.py`), learned parts ON, no refit (`deep_dials.py`); ratios are final-layer MSE against the unchanged
S2 card.

Native S2 bill by layer (r35, `bill_by_layer.py`): L9-14 is 47.3% of the bill and L13-15 is 20.1%. At L9-14:
- newborn push 9.9%
- slr_reconstruct 7.9%
- old384 read 7.2%
- slr_inner 5.5%
- star 5.3%
- old384 push 3.9%
- covariance 3.5%

The costs that scale with the tier frame rank at L10-15 come to about 21% of the bill.

| dial | r35 | r100 | r37 | r39 | geo | est. bill |
|---|---|---|---|---|---|---|
| tier rank 352 at joins L10+L13 | +.60% | +1.52% | +1.37% | -.10% | +.84% | -1.8% |
| tier rank 320 at L10+L13 | +.63% | +5.19% | +5.06% | -.25% | +2.63% | -3.5% |
| rank 384 at L10, 320 at L13 | +.25% | +2.55% | +3.54% | -.79% | +1.37% | -1.8% |
| rank 288 / 256 at L10+L13 | +8.2% / +13.9% | | | | | -5.3% / -7.0% |
| star rank 128 (exact SVD) at L9-14 | +.49% | +.03% | -.09% | +.07% | +.12% | about 0 deployable (section 16) |
| response floor .03 / .1 at L9-14 | +10.9% / +158% | | | | | about -1.9% (-15% rows) |
| alive floor .03 / .05 at L9-14 | +9.7% / +35% | | | | | -4..-8% columns |
| SPEND MORE: tier rank 512 at L13 | -.33% | -.10% | +.54% | +.06% | +.04% | +3.6% |
| SPEND MORE: rank 512 at L10+L13 | -.18% | +.27% | -.09% | -.56% | -.14% | +7% |
| SPEND MORE: all alive rows as response rows at L13-14 | +4.35% | +4.65% | +7.10% | +1.37% | +4.35% | +1.6% |
| SPEND MORE: birth floor .002 at L12-14 (+64-96 births/layer) | -.44% | -.04% | -.28% | +.66% | -.02% | about +1% |

Compensation knobs (single scalar, fitted on TRAIN 40/41; `deep_dials_train_comp.jsonl`, `deep_dials_train_floorcomp.jsonl`):
- old-tier read x(1+d) with rank 320: r40 / r41 at d = 0 are -1.26 / +4.96%; at d = -.05, -.91 / +6.27%; at d = +.05, +.52 / +5.97%. The
  best d is 0.
- rank 288 with d = .1 / .2: +4.1 / +18% (r40), +10 / +23% (r41).
- whole read x(1+d) with the response floor .03: d = -.05 gives +12.5 / +7.0%.
- with the alive floor .03, d = -.05 gives +1.8 / +7.8%.

No scalar recovers a truncation, because the loss is net-specific. At rank 320, r41 and r100 lose about 5% while r40 and r39 gain.

VERDICT: no deep-layer dial reaches at least 3% bill for at most 0.5% raw, and none gets within +3% raw after compensation at -5% bill or
better. Spending more of the existing machinery at L13-15 buys nothing: a larger frame, more births and more response rows give +-0.1% to
+4% worse. So the rivals' final-layer edge is not "more of what we already compute at the end"; it has to be a different final-layer
computation.

Addendum (00:50): an EXACT old tier, with the frame rank equal to all active rows (832-960, so no truncation), on the S2 card:
- at L13: -.38 / -.19 / +.47 / +.01% (geo -.02%)
- at L10+L13: -.34 / +.28 / -.15 / -.46% (geo -.17%)
- at every join (L7+L10+L13): +1.15 / -.10 / -.38 / +.36% (geo +.25%)

The rank-384 frame sits exactly at the knee: removing the truncation adds nothing, while 352 costs +.84% and 320 costs +2.6%. The final
error is not born in the tier truncation.

## 19. Native tier rank 352 at the L10/L13 joins (lead's task, 00:15-00:45)
Builder: `fable_join_v1/build_join_schedule_v2_rank.py`, a versioned copy of `build_join_schedule_v1_fired_S2.py` (the file that built the
fired S2; not edited). A schedule may carry '@r' (first join keeps 384, later joins use r) or '@r1,r2,..'. Nine count-asserted edits make
the rank per-join:
- the push's rank reads `_JOIN_RANK[layer]`
- `state['rank']` is set before the factors are finished
- the lazy-rotation shape checks use `_slr_rank(state)` and the per-group `rank_old` / `rank_new`
- the overflow envelope width follows the contracted rank

Parity:
- plain every3 on the porter's S3 pre-join intermediate (`stack_S3_build/6_eng.py`) is byte-identical to `stack_S3.py`
- every3@384,384,384 gives identical MSE and flops to S3 on r35 and r100

| native, on S3 | util | final MSE | adjusted vs S3 |
|---|---|---|---|
| r35, S3 | .18411 | 1.61237e-8 | 1 |
| r35, rank 352 at L10/L13 | .18337 (-0.40%) | 1.62072e-8 (+0.52%) | +0.12% |
| r100, S3 | .18179 | 1.57638e-8 | 1 |
| r100, rank 352 at L10/L13 | .18070 (-0.60%) | 1.60030e-8 (+1.52%) | +0.90% |

The raw change matches torch (r35 +0.60%, r100 +1.52%), but the bill saving is a third of the estimate (-1.8%). The per-function census on
r35 shows why: _bh_leaf +1.77%, _bh_backward -0.98%, _bh_forward -0.56%, old384 push -0.36%, slr_inner -0.33%. S3's fast-multiply kernels
split 384 = 3 x 128 cleanly, while 352 = 11 x 32 grows the leaf work. Adjusted is worse on both nets, so this fails the -0.5% bar: no
porter hand-off and no VALID40. The builder stays available for other ranks. A rank that splits as well as 384 does (320 or 256) costs
+2.6% raw or more in torch.

## 20. The plane-to-distinct feed, one-step, on the S2 torch card (lead's task, 00:44-01:02)
Verdict: CLOSED on S2. The feed is real and the hook reproduces the 09-02 feed exactly, but on S2 it buys at most a 2.4% raw cut for a
bill increase of about 61%. Every variant misses the bar (adjusted at most -1.5% geo, no net above +1%) by 10 to 60 points.

Hook: `feed_s2.py` (pattern (0,1,0) only; formulas IMPORTED from public_g29_bond_kernel_v1: `plane_feed_oracle.columns` / `feed_reads`
for the k3 read, `plane_feed_plane_read.plane_read_first` for the plane read). Closed loop: at source layer l the feed is computed from the
card's own pre-state (mean, pre-covariance, the full pre-plane = atoms read + star push + b terms, diagonal zeroed) and W_{l+1}; strength
times the delta is added at read layer l+1 to the star push, which enters the card only through the pre-plane. Windows are READ layers.
Learned parts ON, no refit; final MSE against truthpost mc_post[15]. Receipts: feed_s2_grid.jsonl, feed_s2_smoke.jsonl.

Checks:
- base r35 1.61200e-8 = the s2card parity value; the other bases are r100 1.57727e-8, r37 1.78979e-8, r39 1.46906e-8
- diag(plane read) = k3 read to 1e-15 at all 13 layers (FEED_CHECK=1)
- the S2 feed equals the 09-02 oracle's (0,1,0) read: k3 rms 1.235e-4 at L2 in both, within 4% at depth; plane read against the saved
  09-02 read (feed_plane_inject_rank0035_p010.npz): correlation 0.994-1.000 at every layer

Raw final MSE change against the S2 torch card, and adjusted with the bill from the GEMM count:

| form, read layers, strength | r35 | r100 | r37 | r39 | geo | GEMMs | bill | adjusted geo | worst net |
|---|---|---|---|---|---|---|---|---|---|
| k3 read, L2-14, x1 | -1.3% | +3.7% | +2.2% | +0.0% | +1.2% | 39 | +20.3% | +21.6% | +25.2% |
| k3 read, L2-14, x0.5 | -1.9% | +0.9% | +0.2% | -1.1% | -0.5% | 39 | +20.3% | +19.7% | +21.8% |
| k3 read, L9-14, x1 | -1.1% | +1.3% | +3.5% | -0.2% | +0.9% | 18 | +9.3% | +10.3% | +13.2% |
| k3 read, L9-14, x0.5 | -1.6% | -0.1% | +1.0% | -1.0% | -0.4% | 18 | +9.3% | +8.9% | +10.5% |
| k3 + plane read, L2-14, x1 | -0.4% | +1.9% | -1.0% | -1.2% | -0.2% | 117 | +60.8% | +60.4% | +65.2% |
| k3 + plane read, L2-14, x0.5 | -3.0% | -1.3% | -2.5% | -3.0% | -2.4% | 117 | +60.8% | +56.9% | +60.1% |
| k3 + plane read, L9-14, x1 | +0.8% | -0.7% | +4.8% | +2.2% | +1.8% | 54 | +28.0% | +30.3% | +34.2% |
| k3 + plane read, L9-14, x0.5 | -1.6% | -1.7% | +1.2% | -0.4% | -0.6% | 54 | +28.0% | +27.2% | +29.6% |

Also on r35 (feed_s2_smoke.jsonl): the final-layer k3 read alone (read layer 15) RAISES raw MSE by 0.44% at x1 and 3.0% at x2; adding it to
the L2-14 k3 read gives -0.8% instead of -1.3%.

Bill: flopscope bills one 1024^3 f32 matmul at 2.146e9 flops, which is +0.000976 util. S2's native util is .1858 / .1835 / .1880 / .1948
(r35 / r100 / r37 / r39), so each GEMM adds about 0.53% to the bill. A GEMM must therefore buy at least a 0.53% raw cut; the best form
buys 0.02% per GEMM. Restricting the GEMMs to the card's own alive (w, s) and response (u) units cuts the bill only 25% (S2 keeps 800-1024
units alive), e.g. the best form still costs about +47%.

Why the 09-02 gain does not transfer (feed_s2_diag.py, feed_s2_diag2.py; r35 and r100 against the 32M per-unit and 8M plane truths):
- GEMMs are 3.3x dearer on S2 (util .18 against about .60 on the 09-02 card, 1.32e12 flops).
- The k3 part points along S2's own k3 error: correlation .36-.59 at L3-14, slope 1.2-1.5, 17-31% of the error energy explained at
  strength 1. It does not convert: the learned parts compensate the k3 error, so the k3 read at x1 raises raw MSE by 1.2% geo.
- The off-diagonal plane read is ANTI-aligned with S2's plane error at L4-14 (slope -1.4 to -2.7), even after absorbing S2's plane
  amplitude defect in a joint fit (slope -1.2 to -2.2). Both the S2 and the 09-02 reads are anti-correlated with the truth plane itself
  (-0.05 to -0.14). The 09-02 plane (residual 9-13% of truth) over-carried that pattern, so adding the feed fixed it. S2's off-diagonal
  pre-plane is too small (card/truth rms .91 at L4, .73 at L7, .56-.66 at L10-14, correlation .82-.94), so it under-carries the pattern.
- The final-layer read hurts on S2 (above); the 09-02 in-card test included it.

Addendum (lead's follow-up: k3-read-only first, at strength 1 / .5 / .25; receipt feed_s2_grid_eps025.jsonl, same bases):

| form, read layers, strength | r35 | r100 | r37 | r39 | geo | GEMMs | bill | adjusted geo | worst net |
|---|---|---|---|---|---|---|---|---|---|
| k3 read, L2-14, x0.25 | -1.3% | +0.2% | -0.2% | -0.8% | -0.5% | 39 | +20.3% | +19.6% | +21.0% |
| k3 read, L9-14, x0.25 | -1.1% | -0.3% | +0.4% | -0.7% | -0.4% | 18 | +9.3% | +8.9% | +9.8% |
| k3 + plane read, L2-14, x0.25 | -2.3% | -1.2% | -1.6% | -2.0% | -1.8% | 117 | +60.8% | +57.9% | +60.3% |
| k3 + plane read, L9-14, x0.25 | -1.3% | -1.1% | +0.4% | -0.6% | -0.7% | 54 | +28.0% | +27.2% | +28.6% |

The k3-only read is flat at every strength: -0.4% to -0.5% geo at .25 and .5, +0.9% to +1.2% at 1. The plane read does overshoot at
strength 1, as the physicist's star-transport result predicts, but it beats the k3-only read at .25 and .5 on every window: its raw
optimum is near .5 (-2.4% geo on L2-14, 4/4 nets). On S2 the covariance channel carries the feed's value, not the mean channel. The
verdict is unchanged: the cheapest form (k3 L9-14) is still +8.9% adjusted at every strength.

## 21. Component value audit on the S4 physics (lead's task, 01:15-02:15)
Verdict: nothing to remove. Every component on the lead's list is worth 3x to 500x its bill in raw accuracy. The only removals that pay
are slices (the star at L3, the GNN trunk at L0-4) and lower pair-series orders; each saves at most 0.9% of the bill for -0.1% to -0.2%
adjusted with mixed nets, and they do not combine: the three together RAISE adjusted by 4.3%.

Card: covhunter's HookCard (fable_covhunter_v1/covstep.py) + gn3_knobs.configure(V3) = S2 + v3 knobs, the S4 math (the HEB kernel does
not change predictions). `audit_s4.py` rebuilds the hook card's step from covstep.py's own source with seven count-asserted edits that add
removal knobs. Parity: max abs difference 0.0 against the hook card over all 16 layers; r35 1.572086e-8 = covhunter's S2+v3 value; the
torch rewrite of the pair series at the source orders is identical. Learned parts ON, no refit, final MSE against truthpost mc_post[15].
One variant per process: peak physical footprint about 1.05 GB (macOS max-RSS reads 6.6 GB because freed pages stay mapped).

Bill shares: `bill_by_layer_v2.py` (v2 of AN-32: also skips the HE/HEB fast-multiply helpers, so their cost lands on the calling
physics function) on the fired file fable_fire_v1/fire_candS4/candS4_S3HEB_gn.py, r35, util .18377. The carriers take about 92% of the
bill: young-cohort push 25.6%, atoms plane read 21.4%, old tier 25.4% (slr_reconstruct 10.9, old384 push 7.7, slr_inner 6.6, join .3),
star push 12.3% (0.88% per layer at L1-14), covariance push 8.5%. Everything this audit could remove is about 4.3% together.

Removal definitions: GNN trunk off = gnn_step and features not run at L0-14, hidden state frozen at its initial zeros (birth
coefficients then fall back to cstar; the edge and the final hook see the zero state); delta0 = only the mean correction zeroed (no bill
saved). Birth update off = its coefficients 0; static = coefficients cstar (no learned node head). b-vector family off = pre b terms 0 and
post b 0 before trace_post, so its plane content falls to the star residual. Star off at L = the star push dropped at L (L15: the star
part of the final k3). Series R:E:EL = Gaussian Hermite order, plane-term bond order, local t3/t4 bond order (source 6:2:2); their bill is
an op count on the fused series (about 123 float32 n^2 ops per pair layer, 0.45% of the bill in all).

Raw final MSE change when removed (r35 / r100 / r37 / r39, geo), and adjusted = raw x (1 - bill share):

| removed | bill share | r35 | r100 | r37 | r39 | raw geo | adjusted geo | flag |
|---|---|---|---|---|---|---|---|---|
| gnn=delta0 | 0.00% (value only) | +11.3% | +18.8% | +7.0% | +17.3% | +13.5% | +13.5% |  |
| gnn=off | 1.23% (census) | +30.8% | +31.9% | +11.9% | +33.8% | +26.8% | +25.2% |  |
| final_off | 0.08% (census) | +6.8% | +4.2% | +4.8% | +6.0% | +5.5% | +5.4% |  |
| edge_off | 0.79% (census) | +5.5% | +5.1% | +10.7% | +1.4% | +5.6% | +4.8% |  |
| bd=off | 0.35% (census) | +13.3% | +14.9% | +8.0% | +8.9% | +11.2% | +10.8% |  |
| bd=static | 0.00% (census) | +2.5% | +3.3% | -1.4% | +2.8% | +1.8% | +1.8% |  |
| omitted_off | 0.38% (census) | +29.3% | +26.8% | +35.9% | +30.4% | +30.6% | +30.1% |  |
| b_off | 0.05% (op count ~10 n^2/layer) | +24.6% | +41.0% | +30.8% | +10.1% | +26.1% | +26.1% |  |
| star_off=1:2 | 1.76% (census) | +8.1% | +2.3% | +2.9% | +8.6% | +5.4% | +3.6% |  |
| star_off=14 | 0.88% (census) | +10.3% | +6.4% | +7.2% | +5.0% | +7.2% | +6.3% |  |
| star_off=15 | 0.29% (census) | +3.7% | +2.3% | +2.3% | +2.1% | +2.6% | +2.3% |  |
| star_off=14:15 | 1.17% (census) | +12.0% | +7.5% | +8.7% | +6.4% | +8.6% | +7.3% |  |
| knobs_off | 0.15% (census) | +2.5% | -0.1% | +1.8% | +0.0% | +1.1% | +0.9% |  |
| series=5:2:2 | 0.03% (op count) | -0.0% | -0.0% | +0.0% | -0.0% | -0.0% | -0.0% | REMOVAL PAYS |
| series=4:2:2 | 0.07% (op count) | +0.0% | -0.1% | +0.0% | -0.0% | -0.0% | -0.1% | REMOVAL PAYS |
| series=6:1:2 | 0.05% (op count) | -0.1% | +0.1% | +0.0% | +0.1% | +0.0% | -0.0% | REMOVAL PAYS |
| series=6:1:1 | 0.08% (op count) | +0.2% | +0.0% | +0.5% | +0.5% | +0.3% | +0.2% |  |
| series=4:1:1 | 0.15% (op count) | +0.2% | +0.1% | +0.5% | +0.5% | +0.3% | +0.2% |  |
| star_off=1 | 0.88% (census) | +3.1% | -0.0% | +0.5% | +3.6% | +1.8% | +0.9% |  |
| star_off=2 | 0.88% (census) | +6.1% | +2.0% | +2.2% | +6.5% | +4.2% | +3.3% |  |
| edge_off@8:10 | 0.34% (census) | +4.8% | +1.5% | +5.9% | -1.8% | +2.6% | +2.2% |  |
| edge_off@11:12 | 0.23% (census) | -0.1% | +2.2% | +2.6% | +1.2% | +1.5% | +1.2% |  |
| edge_off@13:14 | 0.23% (census) | +0.7% | +1.2% | +2.1% | +1.1% | +1.3% | +1.1% |  |
| gnn=off@0:4 | 0.35% (census) | -0.8% | -1.2% | +2.5% | +0.2% | +0.2% | -0.2% | REMOVAL PAYS |
| gnn=off@0:7 | 0.61% (census) | -1.4% | +4.0% | +0.2% | +2.8% | +1.4% | +0.8% |  |
| gnn=off@8:14 | 0.61% (census) | +30.2% | +31.7% | +15.3% | +30.7% | +26.8% | +26.0% |  |
| series=4:1:2 | 0.12% (op count) | -0.0% | +0.1% | +0.0% | +0.1% | +0.1% | -0.1% | REMOVAL PAYS |
| series=3:1:2 | 0.15% (op count) | +0.0% | +0.2% | +0.0% | +0.1% | +0.1% | -0.1% | REMOVAL PAYS |
| series=4:0:2 | 0.16% (op count) | +20.5% | +28.6% | +33.7% | +16.7% | +24.7% | +24.5% |  |
| star_off=3 | 0.88% (census) | +0.6% | +2.3% | -1.2% | +1.1% | +0.7% | -0.2% | REMOVAL PAYS |
| star_off=4 | 0.88% (census) | +0.6% | +9.5% | +0.3% | -1.9% | +2.0% | +1.1% |  |
| star_off=5 | 0.88% (census) | +6.6% | +12.1% | +4.4% | -1.3% | +5.3% | +4.4% |  |
| star_off=6 | 0.88% (census) | +12.6% | +13.5% | +9.7% | +0.6% | +9.0% | +8.0% |  |
| star_off=7 | 0.88% (census) | +12.0% | +17.0% | +19.2% | +7.3% | +13.8% | +12.8% |  |
| star_off=8 | 0.88% (census) | +13.2% | +21.0% | +18.5% | +14.6% | +16.8% | +15.8% |  |
| star_off=9 | 0.88% (census) | +21.7% | +22.2% | +15.4% | +17.0% | +19.0% | +18.0% |  |
| star_off=10 | 0.88% (census) | +26.2% | +17.8% | +20.7% | +12.5% | +19.2% | +18.1% |  |
| star_off=11 | 0.88% (census) | +25.8% | +14.3% | +19.0% | +10.6% | +17.3% | +16.2% |  |
| star_off=12 | 0.88% (census) | +18.0% | +11.0% | +16.6% | +9.2% | +13.6% | +12.6% |  |
| star_off=13 | 0.88% (census) | +16.7% | +7.7% | +15.6% | +9.3% | +12.3% | +11.3% |  |
| gnn=off@0:4+star_off=3+series=4:1:2 | 1.35% (census + op count) | +3.8% | +5.8% | +1.7% | +12.1% | +5.8% | +4.3% |  |

Reading the table:
- Largest value per bill: the omitted-radial source (+30.6% raw for 0.38%), the b-vector family (+26.1% for about 0.05%), the final hook
  (+5.5% for 0.08%), the birth update (+11.2% for 0.35%; its learned node head alone +1.8% for 0.004%), the GNN trunk (+26.8% for 1.23%;
  the late trunk L8-14 carries all of it, the mean correction alone is +13.5%). The v3 knobs are +1.1% for 0.15%.
- Message edge: +5.6% for 0.79%, and every band pays for itself (L8-10 +2.6% for .34%, L11-12 +1.5% for .23%, L13-14 +1.3% for .23%).
- Star push by layer: worth little at L1, L3 and L4 (+1.8%, +0.7%, +2.0% raw for 0.88% each), then +4% to +19% from L5 to L14 (peak
  L9-L10). The native push is three dense products weighted by W squared, so a low-rank star gives no cheaper form.
- Flags (removal pays): pair series orders down to 3-4 Gaussian and bond order 1 (-0.1% adjusted, saves about 0.12%; bond order 0 is a
  cliff at +24.7%), the GNN trunk at L0-4 (-0.2% adjusted, r37 +2.5% raw) and the star at L3 (-0.2% adjusted, r100 +2.3% raw).
- Together those three flags give +5.8% raw (r39 +12.1%) and +4.3% adjusted: the card sits at a co-adapted optimum and the near-free
  trims interact. Nothing here clears a porting bar.
Receipts: audit_s4_v1-v4.jsonl (per net, per variant), bill_by_layer_v2_candS4_S3HEB_gn_r35.json; table built by audit_table.py.

## 22. s6bcard.py: the fired S6b physics in torch, with a refit entry point (lead's task, 05:30-06:14)
Verdict: READY. Over the 40 VALID nets the torch card matches native S6b with a pooled gap of -0.009% (median -0.017%, per-net sd 0.15%,
36/40 within 0.2%, 40/40 within 0.5%), and it reproduces the refit's native gain (torch -0.51% mean, 34/40 nets; native VALID40 -0.50%, 33/40).

Card (`s6bcard.py`): covhunter's HookCard + gn3_knobs.configure(V3) (= S2 physics + v3 knobs, bit-exact to S2+v3) + the L15 k3 row cut as
a 'k3_15' hook (the CP and star parts only on rows select(3.5 - |mu/sigma|, 0), rounded up to 32, stable order; the 3 v b term on every
row) + the params_epoch2 literals of refit_s2_full_lr2e4_v2. The HookCard and s2card give identical outputs with the refit literals too
(max diff 0.0, r100). Kernel layers are not modelled.

| net | torch S6b | native S6b | gap | torch no-refit card | native S5e | gap |
|---|---|---|---|---|---|---|
| r35 | 1.564284e-8 | 1.565366e-8 | -0.07% | 1.571390e-8 | 1.572180e-8 | -0.05% |
| r100 | 1.573344e-8 | 1.570229e-8 | +0.20% | 1.582406e-8 | 1.583176e-8 | -0.05% |

The per-net gap is a precision-path effect, not a model difference (native_checks/, built with build_stack.sh without kernel flags):
- the S6b physics built natively without the kernel layers (N2, same refit and cut stages by sha) scores r35 1.560159e-8 and r100
  1.577639e-8, so the fired file's kernel layers alone move native MSE by +0.33% and -0.47%; with the bank literals the same kernels move
  it by +0.08% and +0.14% (N0 against the fired S5e)
- the torch card's MSE moves by about +-0.04% under 1e-7 relative weight noise (about one float32 ulp) and by -0.3% to +0.5% under 1e-6
  noise, for the refit and the bank literals alike; float32 kernels and float64 torch paths differ at that level
- so single-net comparisons below about 0.3% are noise; judge refits on pooled VALID nets.
Parameter jitter found one ill-conditioned input: the GNN standardizes three constant feature columns (2l/15-1, l/16, ones) with scales
down to 1e-6. They are exact and identical in torch and native, but any trained change to 'centers' would be catastrophic (1e-6 relative
moves MSE +6%). The refit entry point never trains centers, scales, rms or first.

Refit entry point (`refit_s6b.py`, sibling of refit_s2.py, same loss/schedule/.pt format, no weight cache): starts from params_epoch2;
groups heads | full | full+trunk (trunk = msg_w/b, enc0/2/4 w/b, gain at --trunk_lr); defaults lr 1e-4, trunk_lr 5e-5, 160 TRAIN nets,
16 VALID, 3 epochs. Checked: every trained tensor gets a finite gradient (r40); one fwd+bwd is 10 s + 38 s; the smoke test
(refit_s6b_smoketest/) trains and saves, and only the trained tensors change.
Memory, one fwd+bwd: physical footprint 1.96 GB, but macOS max-RSS reads 8.55 GB (freed pages stay mapped). With MallocLargeCache=0 in
the environment the max-RSS falls to 4.16 GB (footprint 3.3 GB) at +45% wall (about 70 s per net instead of 48 s); results identical.

## 23. Where the S6b final-layer error mass sits (lead's task, 06:16-06:20)
Verdict: the error is only moderately concentrated, and it is MEAN-PROPAGATION error in active units. The top 10% of units by |error|
hold 55-59% of the final squared error (an iid Gaussian error puts 44% there). They share no kink, skew or variance signature. They are
the strongly active units (large outputs, none dead), and on r35/r100 their error follows the card's own pre-activation mean error at L15.

Method (`err_mass_s6b.py`; receipts err_mass_s6b.json/.npz, err_mass_s6b_regimes.json): the torch S6b card (section 22, unbiased against
native) on six nets, 35 and 100 (which have the 32M per-unit truth at L15) and VALID 37, 39, 42, 48; one forward each, no new native runs.
e_u = prediction - truthpost mc_post[15]. The truthpost truth is precise: its noise is at most about 3% of the MSE per unit (from its
difference with the independent 32M draw).

| net | final MSE | top 1% share | top 5% | top 10% | error share: active units (z > 1) | kink units (\|z\| <= 1) | dead (z < -1) |
|---|---|---|---|---|---|---|---|
| 35 | 1.564e-8 | 11.9% | 39.1% | 58.6% | 67% (35% of units) | 29% | 3% |
| 100 | 1.573e-8 | 12.1% | 37.8% | 58.2% | 66% (39%) | 31% | 3% |
| 37 | 1.736e-8 | 13.5% | 38.4% | 57.3% | 71% (40%) | 26% | 3% |
| 39 | 1.452e-8 | 12.9% | 37.2% | 55.4% | 68% (40%) | 29% | 4% |
| 42 | 1.966e-8 | 11.9% | 38.7% | 57.8% | 76% (41%) | 21% | 3% |
| 48 | 1.717e-8 | 12.9% | 37.6% | 56.3% | 72% (40%) | 26% | 2% |

For an iid Gaussian error the top 1% / 5% / 10% would hold 8.5% / 28% / 44%. z = mu/sigma is the card's L15 pre-activation z.

Signature of the top 10% against all units (six nets, ranges):
- dead units (z < -3): 0% against 14-20%; active (z > 1): 70-82% against 35-41%; their median true output is 0.64-1.06 against
  0.11-0.17. The rank correlation of e^2 with the true output is +0.62 to +0.72, the strongest of all.
- kink: |z| < 1 holds 18-30% of them against 20-25% of all units; |z| < 0.5 holds 9-19% against 9-14% (at most 1.5x, on 3 of 6 nets).
- skew |k3|/v^1.5: 5-12% of them sit in the top skew decile (base rate 10%); median skew equal to all units. No signature.
- variance: 7-15% in the top v decile (base 10%); median v equal. No signature.
- the error sign is mixed (positive in 42-69% of them), so it is not one shared bias.
Input errors at L15 on r35 / r100 (card against the 32M per-unit truth): the pre-activation mean error is 0.0008 truth-sigma rms in the
top 10% against 0.0004 in the rest; the variance error (0.2%) and the k3 error (0.05 sigma^3) are the same in both groups. The final
error follows gate x (mean error): correlation 0.82 / 0.79, slope 0.85 / 0.79, 67% / 62% of the error energy explained. That is a lower
bound: the 32M truth's own noise in the mean is 13-15% of the error energy and attenuates the fit.

Reading: the remaining final-layer error is the carried mean error (mu at L15 = W^T m at L14) passed straight through by active units.
It is not local non-Gaussianity at the final layer. This matches the exact-state oracle on the 331824 card (truth m at L13 cut the final MSE by 43-48%).
A final-layer fix of k3, v or the kink rows cannot reach it; it has to come from the carried mean (and the covariance that feeds it).

## 24. Platform-timeout risk model from the public eval pages (lead's task, 06:21-06:28; JSON only)
Verdict: S6b's timeout risk is tiny from platform noise (0.03% at its own profile, 2.6% at +10 s), and S7's +3.4 s only moves it to
0.14%. J2W's profile (worst expected roundtrip 112 s) carries a 76% risk, which matches their observed loss. The one unresolved risk is
design-independent: J2W's stall was cut off by the 120 s kill, so it may have been a hang that no headroom avoids (1 episode in 10 runs).

Data and method (`timeout_risk.py`, receipt timeout_risk.json): 10 unique submissions x 100 MLPs = 1000 MLP runs, ours #331947 (S2),
#331961 (S4), #331992 (S5e), #332005 (S6b) and the rivals' marius_binner, J2W, reds, a_s6, ben3, suliman_tadros. The cap applies to
participant_roundtrip_wall_s. For each submission, predict wall = OLS on 1 + flops + cold (the MLP right after a worker setup); the residual
is the in-predict excursion. Flops explain little of the per-MLP wall (within-run correlation -0.16 to 0.74), so the excursions are mostly
platform timing.
- In-predict excursions, pooled over 1000 MLPs: p50 +0.03 s, p90 +3.7 s, p99 +8.7 s, p99.9 +13.5 s, max +13.8 s (suliman); relative
  p99 9.0%, p99.9 14.7%. The per-run spread ranges from 0.7% (reds) to 5.6% (suliman); ours 2.4-4.6%. They barely repeat across our runs
  (correlation -0.38 to +0.63); a small fixed per-MLP part remains (sd 1.75 s, at most +6.3 s on MLP 39).
- Outside predict: median 0.39 s, p99 0.45 s. Stalls over 1 s: 3 of 1000 (0.3%), all J2W worker w-3, MLPs 66, 71 and 76 in a row. The
  worker re-ran setup after each failure and was fine from MLP 81. The stalls measure 10.16-10.41 s, but all three roundtrips are
  pinned at 120.24-120.28 s while predict varies 109.84-110.12 s, so the platform killed them at the cap: the true stall was at least
  10.2 s, possibly a hang.
- Risk model: a design's expected roundtrips (its own fit + 0.39 s) + the uniform added wall + pooled relative noise (empirical below the
  95th percentile, exponential tail above; it predicts about 1 excursion beyond +14.8% in 1000 runs, as observed) + stall episodes at the
  pooled rate (1 per 10 runs, one worker, 3 consecutive MLPs) as either a fixed 10.35 s delay or a hang until the kill.
- Calibration: over the 9 runs without stalls the model predicts 0.16 in-predict timeouts in total; none occurred. For J2W it predicts 1.3
  (P 74%); J2W had 0 in-predict timeouts (a quiet run, 2.4% spread) and 3 stall timeouts.

P(at least one of 100 MLPs over 120 s):

| added wall | S6b worst expected roundtrip | S6b: noise + 10.35 s stall | S6b: if the stall is a hang | J2W worst expected | J2W: noise + stall | J2W: if hang |
|---|---|---|---|---|---|---|
| +0 s | 94.7 s | 0.03% | 10.0% | 112.2 s | 76% | 76% |
| +3.4 s (S7 push5p6) | 98.1 s | 0.14% | 10.1% | 115.6 s | 99.4% | 99.4% |
| +5 s | 99.7 s | 0.29% | 10.2% | 117.2 s | ~100% | ~100% |
| +10 s | 104.7 s | 2.6% | 11.8% | 122.2 s | 100% | 100% |

The S6b numbers at +0 to +5 extrapolate beyond the largest excursion ever seen: its worst expected wall plus the largest excursion plus a
J2W-size stall is 94.7 + 13.8 + 10.35 = 118.9 s, still under the cap. Treating the fixed per-MLP part as known (MLP 39 +6.3 s) shifts
every S6b row down by about one row. The hang column's 10% floor rests on a single episode (a 95% interval of roughly 0.3% to 45% per run).

Rule for the private-eval designation: designate the best-adjusted entry whose worst expected per-MLP roundtrip is at most ~105 s. Every
S-series entry qualifies (S6b 94.7 s, S7 about 98 s, both at 0.3% or less), and the risk climbs steeply above that (a_s6 at 108.6 s: 13%;
J2W at 112 s: 76%).

## 25. TRAIN640 split for the next refit round (lead's task, 06:29-06:35; no training)
Verdict: the split file is ready and every net checks out. The split has 641 nets, not 640, because panel net 150 is also a VALID40 net.
All 681 nets (641 train + 40 valid) load. Their weights match the manifest hashes, and their truths match their mlp_ids. The new nets
come from the same distribution as TRAIN160. One 641-net epoch costs about 8.5 h at 48 s/net. s6bcard.py needs no change;
refit_s6b.py now takes the split with --split.

Split (`train640_split.py` -> `split_train640.json`): every observer_fit role rank 0-687 except VALID40 (inputs.json 'valid') and the
panel8 gate nets (35 100 150 200 250 300 400 500, from public_g22_operator_features_v1/panel_candidate.sh). The virgin6 gate nets
(3016500-3016505) are not observer_fit ranks. That is 157 kept TRAIN160 nets + 484 new ranks = 641.
- TRAIN160 contains three panel nets, 35, 100 and 200, so every refit trained on inputs.json's TRAIN160 (including params_epoch2 in
  S6/S6b) was partly in-sample on panel8. VALID40 and virgin6 are clean; panel8 verdicts on refit candidates are 3/8 in-sample.
- Truths: fable_refit_v1/truth688 (organizer all-layer means, 1e9 samples, all 688 ranks). They are bit-identical to
  public_g22 truthpost on all 200 nets that directory holds.

Checks (`train640_check.json`, one net at a time, 2 min, 0.46 GB):
- 0 failures over 681 nets: weights.npy sha256 = the row manifest's, shape (16,1024,1024) float32, finite; truth (16,1024) finite, and
  its mlp_id = the row manifest's mlp_id (so weights and truth are the same MLP).
- The MLPs have no biases: the decoded columns are mlp_id, mlp_name, mlp_seed and weights only.

| group | nets | weight std per layer (L0 / L7 / L15) | net-to-net sd | max\|w\| range | final-layer truth mean | truth std | dead fraction |
|---|---|---|---|---|---|---|---|
| new ranks | 484 | .04419 / .04420 / .04419 | 3e-5 | .223-.280 | .557 (sd .064) | .772 | 4.1% |
| TRAIN160 kept | 157 | .04420 / .04419 / .04420 | 3e-5 | .228-.276 | .557 (sd .071) | .771 | 4.1% |
| VALID40 | 40 | .04419 / .04420 / .04419 | 3e-5 | .226-.260 | .567 (sd .057) | .785 | 4.3% |

The weight std is sqrt(2/1024) = .044194 (He initialization) in every layer of every net. Two-sample KS statistics, new ranks against the
kept TRAIN160 nets, are .065-.115 for the mean weight std, the L0 and L15 weight std, and the final truth mean, std and dead fraction,
all below the 5% critical value .125. No new net is more than 4 sd from TRAIN160 on any of them.

Cost of one epoch over 641 nets:
- 641 x 48 s (10 s fwd + 38 s bwd, 3 threads, quiet box) = 30,800 s = 8.5 h, plus about 7 min per VALID40 evaluation.
- The physicist's runs manage about 38-42 s/net at 5 threads under today's load, which gives 6.8-7.5 h.
- Two epochs is 14-17 h per job.
- At batch 4, one epoch is 161 optimizer updates against 40 on TRAIN160. So one TRAIN640 epoch moves the weights about as far as four
  TRAIN160 epochs at the same lr: consider one epoch, or a lower lr.

Code: s6bcard.py needs no change, since it does not know about splits. refit_s6b.py needed two changes, now made behind a flag (v1 kept
as refit_s6b_v1_train160.py):
- --split <json> reads the training ranks from the file's 'train' list; without it TRAIN is inputs.json's 160 as before. An assert
  rejects any VALID40 overlap.
- truth() reads fable_refit_v1/truth688, since public_g22 has only 200 nets. This is bit-identical for the old nets.
The weights already load by rank from the store manifest for all 688. The physicist's driver (fable_physicist_v1/refit6b/refit_s6b.py)
needs the same two edits.

## 26. Trainable band knobs for the TRAIN640 refit (lead's task, 06:37-06:47; smoke test only)
Verdict: READY. refit_s6b.py now takes a '+knobs' group: 20 band strengths (iso, k21n, k31, edge, rc x L2-5, L6-9, L10-12, L13-14) plus
the K31 row constants c0, c1, 22 float64 scalars in their own Adam group. At init the card is bit-identical to s6bcard. Every active
scalar gets a finite gradient, and one Adam step moves all 21 of them. Nothing longer was launched.

Code:
- `s6bknobs.py`: the S6b card with the knob strengths as tensors. The formulas and their operation order are copied from
  gn3_knobs.configure: the cov_pm hook re-runs the pair map with the modified inputs, edge is x (1 + t) and recentering is alpha = 1 + t,
  and the row factor c0 + c1 mu k3 / v^2 is used both in the pair map (layers 1-14) and in the k31 knob. The cov knobs act at L2-14, edge
  at L8-14 (so edge@L2-5 is structurally inert) and rc at L1-14 (L1 keeps the shipped rc). Layers outside the bands keep the shipped values.
- `refit_s6b.py` (v2 kept as refit_s6b_v2_split.py): --groups accepts knobs alone or heads/full/full+trunk + knobs; --knob_lr (default
  1e-3); the knobs get their own grad-norm clip, so the learned groups train exactly as before. Knob values go to knobs_epochN.json and
  log.json, not to the .pt, which keeps build_refit_candidate_v2.py's p/a/c/e contract.

Smoke test (`smoke_knobs.py`, `refit_s6b.py --tag smoke_knobs_refit`, TRAIN640 ranks 0 and 1, VALID 37 and 39):
- Init parity: s6bknobs.card at the shipped values against s6bcard.card, all 16 layers: max abs difference 0.0 on both nets, final MSE
  1.79678e-8 / 1.93111e-8 on both cards. The refit's VALID-before mean (1.593941454575072e-8) equals s6bcard's values for 37 and 39 to
  every digit.
- Gradients (one fwd+bwd, fine checkpoints, net 0, loss x 1e8): all 21 active scalars are finite and nonzero; edge@L2-5 has none (unused).

| knob | L2-5 | L6-9 | L10-12 | L13-14 |
|---|---|---|---|---|
| iso | +0.100 | +0.0078 | -0.130 | -0.045 |
| k21n | -0.0032 | +0.083 | +0.082 | -0.021 |
| k31 | -0.045 | -0.098 | +0.066 | +0.0091 |
| edge | (inert) | -0.0068 | -0.023 | +0.0014 |
| rc | -0.0003 | -0.0075 | +0.015 | +0.0098 |

  c0 -0.178, c1 -0.010. On this one net, raising c0 by .01 cuts the loss by about 0.1%.
- One Adam step (--groups full+knobs, batch 2): 21 knobs moved by exactly +-3.33e-4 (knob_lr x the warmup factor 1/3); edge@L2-5 did
  not move; every tensor of the full group moved; VALID mean -0.0013%. The update of 2 nets took 117 s at load ~30 (~58 s/net). Memory:
  footprint 3.0 GB, macOS max-RSS 7.3 GB (default allocator, as agreed).

Notes for the 07:45 decision:
- Adam moves each knob by about +-lr per update whatever the gradient size. One TRAIN640 epoch is 161 updates, so knob_lr 1e-3 could
  move a consistently pushed knob by up to ~0.08, about 40% of the shipped k31 (-.212). knob_lr 5e-4 halves that, in line with halving
  the main lr.
- Suggested command if A or B2 passes: `refit_s6b.py --tag <T> --groups full+knobs (or full+trunk+knobs) --split split_train640.json
  --lr 5e-5 [--trunk_lr 2.5e-5] --knob_lr 5e-4 --epochs 1 --nvalid 40`. Judge on VALID40 and virgin6 only.
- Porting: c0, c1 port directly (build_physics.py takes c0,c1@layers or a table). build_gn_knobs.py already stores edge and rc per layer,
  so per-band values need only its json reader. Per-band iso/k21n/k31 need _gn_dcov to read the strength per layer (a porter edit). The
  native knobs are covhunter's closed-form first-order delta, not a pair-map re-run, so trained values carry over to first order (v3:
  torch -2.48% against native -2.36% on r35).

## [section removed for publication: competitor per-network telemetry]
























## 28. Row-drop physics variants from E-48, no refit (lead's task, 2026-09-24 11:00-11:16)
Verdict: (a), the dead-side row cut, PASSES: pooled raw +0.09% against a bill cut of 2.3%, so projected adjusted -2.2%, and all 16 nets
are better on adjusted. (b) and (c) are cliffs (+12% and +22% raw).

Card (`rowdrop_s6b.py`): the torch S6b card (s6bcard: HookCard + v3 knobs + L15 row cut) with the B2 epoch-1 params, learned ON, no refit,
VALID16, 3 threads. A unit's rows of the pushed columns feed only its own CP k3 diagonal and the omitted-radial source: the plane read uses
response rows and the frame U, and the carried atoms are masked by alive. So each variant is one row mask on the float64 columns at L1-14,
inserted by a count-asserted edit of covstep.HookCard.step. With no mask the card equals s6bcard.card (max abs diff 0.0). My baseline
VALID16 is 1.746113e-8, against 1.74614e-8 from the physicist's evaluation (-0.002%).

| variant (E-48) | rows dropped at L1-14 (r37) | bill (E-48) | pooled raw | raw wins | projected adjusted | adjusted wins | per-net raw |
|---|---|---|---|---|---|---|---|
| (a) dead-side cut: all columns off on rows with a < -3.5 (kept = select(a + 3.5), rounded to 32) | 0 at L1-6, 32-128 at L7-14 | -2.3% | +0.09% | 8/16 | **-2.21%** | **16/16** | -0.63% .. +0.94% |
| (b) old-tier rebuild on response rows only (old columns off on dead + saturated rows) | 32-480 | -3.6% | +12.0% | 0/16 | +8.0% | 3/16 | +1.1% .. +26.7% |
| (c) all columns on alive rows only | 32-224 | -4.5% | +21.9% | 0/16 | +16.5% | 0/16 | +16.1% .. +30.3% |

Reading: the k3 of units far on the dead side (a < -3.5) carries no weight in the final MSE, but the k3 of near-dead units (a between -3.5
and the alive floor near -2.3) and of saturated units does, through the skew term of their means (E-48: 1.7x the Gaussian mean at a = -3).
(a) is a porter/engineer build: the same one-sided rule as the L15 cut, applied at L1-14. Its bill (-2.3%) is E-48's estimate and has to be
confirmed by the native census; its raw cost (+0.09%) is at the precision noise, so judge the native build on pooled VALID40.
Receipts: rowdrop_s6b.jsonl (per net and mode, rows dropped per layer), rowdrop_s6b_summary.json.

Addendum: threshold sweep (lead's follow-up, 11:19-11:30; `rowdrop_s6b.py a3.5,a3.0,a2.75,a2.5`; receipts rowdrop_sweep.jsonl and
rowdrop_sweep_summary.json). Same harness: B2 ep1, no refit, VALID16, 3 threads, the same baseline. a3.5 reproduces the first (a) run
exactly. Rows cut are means over the 16 nets. "Rounded" means the card's select keeps a multiple of 32 rows, as the native L15 rule does;
"exact" counts every row below the threshold.

| dead-side threshold | pooled raw | raw wins | per-net raw | row-layers cut at L1-14 (rounded / exact) |
|---|---|---|---|---|
| a < -3.5 | +0.09% | 8/16 | -0.63% .. +0.94% | 620 / 783 |
| a < -3.0 | +0.97% | 5/16 | -1.07% .. +2.85% | 966 / 1144 |
| a < -2.75 | +2.79% | 1/16 | -2.89% .. +4.61% | 1180 / 1368 |
| a < -2.5 (run although -2.75 is past +0.5%) | +8.03% | 0/16 | +2.63% .. +14.67% | 1436 / 1638 |

Rows cut per layer, rounded (exact), means over 16 nets; L1-2 cut no rows at any threshold:
- a < -3.5: L3 0 (2), L4 0 (7), L5 0 (16), L6 6 (26), L7 30 (43), L8 42 (55), L9 54 (69), L10 74 (86), L11 82 (100), L12 98 (114), L13 106 (124), L14 128 (141)
- a < -3.0: L5 18 (34), L6 38 (53), L7 54 (71), L8 78 (90), L9 88 (107), L10 110 (124), L11 120 (137), L12 138 (154), L13 154 (166), L14 168 (183)
- a < -2.75: L4 12 (30), L5 32 (49), L6 50 (69), L7 78 (91), L8 96 (112), L9 114 (128), L10 128 (145), L11 144 (159), L12 158 (177), L13 178 (191), L14 190 (204)
- a < -2.5: L4 30 (43), L5 52 (65), L6 78 (91), L7 98 (112), L8 118 (135), L9 140 (152), L10 150 (167), L11 170 (183), L12 182 (201), L13 204 (215), L14 214 (228)

The knee is between -3.5 and -3.0. A rough projection scales E-48's -2.3% for -3.5 by the rounded rows cut; the engineer's per-row
pricing replaces it:
- -3.0: about -3.6% bill, so about -2.6% adjusted (5/16 raw wins, one net +2.9%)
- -2.75: about -4.4% bill, so about -1.7% adjusted
-3.5 is the safe choice (raw at the noise, 16/16 adjusted). -3.0 may buy about 0.4 points more adjusted at a visible per-net raw risk, and
needs a pooled VALID40 check before it is chosen.

## 29. Dead units out of the non-column work too, no refit (lead's task, 2026-09-24 11:35-12:21)
Verdict: the bar FAILS. All three exclusions together cost +2.74% raw (1/16), against a bar of +0.5%. Only the star exclusion is cheap:
(a) + star is +0.39% raw in total, under the bar on its own. The covariance and pair/learned exclusions cost about 1.7% each.

Card (`rowdrop_nc_s6b.py`): variant (a) (column work off on a < -3.5 at L1-14, rowdrop_s6b.py) plus hooks on the same dead set D_l,
recomputed at the covariance hook from mu and the diagonal of W^T C W with (a)'s rounded select. Torch s6bcard + B2 ep1 params, no
refit, VALID16, 3 threads. The masks are applied after the v3 knob hooks. Mode (a) alone reproduces §28 exactly. D has 620 rows summed over
L1-14 (per layer: L6 6, L7 30, L8 42, L9 54, L10 74, L11 82, L12 98, L13 106, L14 128; none at L1-5).
- (i) star: the star push's output rows and columns of D are zeroed, diagonal included (its W^T A rows and second-stage rows/columns).
- (ii) covariance: the off-diagonal entries of D's rows and columns in W^T C W are zeroed; D's own variances are kept.
- (iii) pair map and learned parts: D's off-diagonal pair terms are zeroed in the post covariance and post plane (the per-unit diagonals
  are kept), D's rows and columns of the message-edge increment are zeroed, and D's GNN rows (mean correction and hidden state) are set to 0.

| mode | pooled raw vs B2 ep1 | over (a) alone | raw wins | per-net raw |
|---|---|---|---|---|
| (a) | +0.09% | 0 | 8/16 | -0.63% .. +0.94% |
| (a) + (i) star | **+0.39%** | +0.30% | 4/16 | -0.17% .. +2.20% |
| (a) + (ii) covariance off-diagonal | +1.80% | +1.70% | 3/16 | -2.67% .. +6.51% |
| (a) + (iii) pair map + edge + GNN rows | +1.79% | +1.70% | 3/16 | -1.28% .. +4.40% |
| (a) + (i) + (ii) + (iii) | +2.74% | +2.64% | 1/16 | -0.60% .. +6.54% |

Reading:
- The dead units' own outputs barely matter, but (ii) and (iii) also change what the learned parts see for every unit. The features
  average over all units (mean rho, rho @ alpha / n, the post-rho rms), and the GNN messages read every unit's hidden state. So the no-refit
  cost is partly a distribution shift that a refit might absorb; as tested, (ii) and (iii) are not viable.
- The star exclusion only removes D's rows and columns from a plane that the pair map reads with D's small weights. It is the one piece
  worth pricing: (a) + star at +0.39% raw, if the engineer's census puts the star's share of the dead-unit non-column work well above 0.4%.
Receipts: rowdrop_nc.jsonl, rowdrop_nc_summary.json.

## 30. Is the TRAIN640 trunk refit learning something different from B2, or converging slower? (lead's check, 2026-09-24 12:23-12:32)
Verdict: it is converging SLOWER ALONG THE SAME PATH. At mid-epoch it covers 76-92% of B2 epoch 1's per-layer gain at L5-15, with the
same shape by layer band, and its parameter moves point the same way. The one difference: it leaves the early layers (L1-3) flat or better,
where B2 ep1 makes them slightly worse.

(`t640_vs_b2_layers.py`, receipt t640_vs_b2_layers.json. The three parameter sets are the S6b literals (the shared start), B2 ep1 (TRAIN160,
80 updates at lr 1e-4 / trunk 5e-5) and T640 params_mid80 (TRAIN640, 80 updates at lr 5e-5 / trunk 2.5e-5).)
- Parameter space (moves from the shared start): cosine(T640, B2) is .84 for the trunk (T640 displacement .90 of B2's), .71 for all
  heads (.86), .74 for the birth coefficients (.82) and .62 for the edges (.70). Head rows by band: L13-14 .93, L15 .88, L10-12 .81,
  L5-9 .58, L1-4 .28.
- Function (torch s6bcard, VALID16, MSE of the post means against truth688 per layer, pooled by band, relative to the S6b literals):

| band | B2 ep1 | T640 mid80 | T640 share of B2's gain |
|---|---|---|---|
| L5-9 | -0.83% (14/16) | -0.63% (16/16) | 76% |
| L10-12 | -1.76% (16/16) | -1.54% (15/16) | 88% |
| L13-14 | -1.68% (14/16) | -1.55% (15/16) | 92% |
| L15 (final) | -1.57% (14/16) | -1.28% (14/16) | 82% |

Per layer at L1-3: B2 ep1 is +0.48 / +0.68 / +0.26% worse, T640 is -0.48 / -0.02 / +0.04%. So on the same path, the second half of the
T640 epoch (cosine decay) should close most of the remaining 8-24%. At the end, compare pooled VALID40 against B2 ep1 (compare_t640.py).

## 31. TRAIN640 refit results, and my top-3 levers (2026-09-24 17:15)
T640 ends (VALID40 vs each run's own before, and vs B2 ep1 VALID40 1.798667e-8; compare_t640.py; runs A/s6b_T640_*):

| run | VALID40 | vs before | vs B2 ep1 (VALID40) | vs B2 ep1 (VALID16) |
|---|---|---|---|---|
| full + trunk (lr 5e-5, trunk 2.5e-5, warmup 10) | 1.796375e-8 | -1.45% (37/40), PASS | -0.13% (24/40) | +0.005% (9/16) |
| full + knobs (knob_lr 5e-4) | 1.809499e-8 | -0.73% (33/40), PASS | +0.60% (9/40) | +0.83% (2/16) |

Mid-epoch the trunk run covered 76-92% of B2 ep1's per-layer gain on the same path (section 30); the second half closed the gap and
passed B2 ep1 by 0.13% on VALID40. The trained knobs barely moved from v3: k31 -0.19..-0.23, rc from -0.10 (L2-5) to -0.05 (L13-14),
c0 .855, c1 .510. So 4x the data and the band knobs add about 0.1%, and the trunk refit lever is saturating.

Levers sent to the lead (paper, section references for the evidence):
1. Final-layer-born error (J2W final/all 2.24 vs our 2.5; the carried mean explains only 62-67% of our final error, section 23): test by
   swapping 32M-truth mu15 / v15 / k3_15 / k4_15 into the final scalar map on r35/r100.
2. A per-layer least-squares b* fitted to the post plane: it moves star content into the Sym(C x b) carrier, which has distinct entries
   (b is +26% for ~0.05% bill, section 21).
3. A learned correction in the top-32 eigendirections of the card's C15, which hold 21-26% of the final error energy (section 27).
Crazy idea: a knob hypernetwork (per-net knob strengths from global statistics). Checked and dead: float16 products, because flopscope
bills f16 matmul exactly like f32.

## 32. Lever 1: how much final error is born at L15? (lead's GO test, 2026-09-24; `l15_oracle.py`, receipts l15_oracle*.json)
Verdict: NO-GO as a final-layer lever. 70-72% of the final MSE is inherited through mu15 and v15 from the L14 mean and covariance. The
part born at L15 (its own k3/k4) is 7-8%, right at the bar, but it sits at a co-adapted optimum: no zero-cost handle exists, and a
per-unit fix needs the closed higher-order carriers.

Torch s6bcard + B2 ep1 params, learned ON, no refit, final MSE against truthpost; truths are the 32M per-unit L15 inputs (mps_truth) and
the 32M L14 post mean and covariance (mcall). Change in final MSE against the unmodified card, r35 / r100:

| swap at the final layer | r35 | r100 |
|---|---|---|
| truth mu15 | -64.5% | -57.5% |
| truth v15 | -4.7% | -11.9% |
| truth k3_15 | -3.1% | -2.2% |
| truth k4_15 | -3.2% | -3.1% |
| truth k3 + k4 (L15-born) | **-8.1%** | **-7.0%** |
| truth v + k3 + k4 (GO test, bar 8%) | -15.3% | -19.9% |
| truth mu + v | -72.0% | -69.6% |
| truth m14 + C14 feeding L15 (makes mu15, v15 exact) | -72.0% | -69.6% |
| truth m14 + C14 + truth k3/k4 | -80.9% | -74.4% |
| all four L15 inputs from truth | -80.9% | -74.4% |
| zero-cost k4 x1.5 / x0.9 / x0.8 | +15.6 / +0.9 / +3.6% | +15.9 / +1.5 / +4.7% |
| zero-cost k3 x1.1 / x0.95 / x0.9 | +9.2 / +3.7 / +12.8% | +12.0 / +2.2 / +9.9% |

Reading:
- The inherited part, 70-72%, is exactly mu15 + v15, i.e. the L14 mean and covariance: truth m14 + C14 reproduces truth mu + v.
- The part born at L15 is the card's own k3/k4 there: 7-8% with the inherited errors present, 5-9% more once they are removed.
- About 19-26% remains with every L15 input from truth. Most of it is the 32M truth's own noise in mu15 (13-15% of the error energy,
  section 23); the final map and hook are at most about 5-10%.
- The v + k3 + k4 swap passes the 8% bar only because v15 is in it, and v15 is inherited from C14.
- The card's L15 k3/k4 sit at their optimum under +-5-50% rescales (the final hook is co-adapted), so the L15-born 7-8% is per-unit
  structure. Reaching it needs per-unit-accurate k3/k4 at L15, which is the closed carrier territory.
- J2W's final-layer edge (final/all 2.24 vs our 2.5) is therefore most likely better inherited m14/C14, i.e. deep-layer propagation, and
  not a better final map.

## 33. Lever 3: a learned correction in the top covariance directions of the final layer (lead's GO test, 2026-09-24; NO-GO)
Verdict: NO-GO. The card's own top-8 and top-32 covariance directions at L15 hold 7.7% and 22.9% of the final error energy (40 TRAIN640
nets), but the error coefficients there are unpredictable from any deployable quantity: leave-one-net-out R^2 is about 0 (bar 0.3).

`lever3_subspace.py` (receipts lever3_subspace.json/.npz): torch s6bcard + B2 ep1, no refit, 40 nets spread over TRAIN640 (every 16th).
The directions are the top-32 eigenvectors of the card's linear-response post covariance at L15, G (W15^T C14 W15) G with G = gates, each
signed toward the prediction. Coefficients c_k = u_k . e with e = prediction - truth688. On average they hold 7.7% (top-8) and 22.9%
(top-32) of the error energy, the same as the TRUE-covariance subspace in section 27 (6-15% / 21-26%). Deployable features per
direction: the prediction's own coefficient, the scalar map's (host) and the final hook's (delta), and the log eigenvalue.

| model (leave-one-net-out) | R^2 top-8 | R^2 top-32 | final-MSE change using top-8 | using top-32 |
|---|---|---|---|---|
| A: per-direction shrinkage c_k = beta_k p_k | -0.03 | -0.02 | +0.23% | +0.54% |
| B: pooled ridge on [p, host, delta, p log lambda] | +0.004 | 0.000 | -0.01% | +0.02% |

This is the fourth granularity at which the final error proves unpredictable from the card's own state: per unit (section 3's heads, the
closed per-unit channel), per net scalar (under 4%, section 27), per covariance direction (here), and by aux mean supervision. So the final
error is not a learnable residual of the card's outputs. It is carried error from the physics upstream (mostly the L14 mean and
covariance, section 32), and only better propagation reduces it.

## [section removed for publication: competitor per-network telemetry]









































































































































## 35. Spec for a per-unit k3+k4 source (lead's task after section 34, 2026-09-24 22:00-23:15)
Verdict: the lever is real but hard to collect. A source must supply per-unit k3 AND k4 at L12-14 itself (L13-14
first), on essentially all units, with random error well under the card's own, and ship with the L12-14 mean correction at 0.5.
- (i) Accuracy. Along the card's own error direction the response is concave: removing half the error gives 83% of the value, and the
  optimum is near 85%. But a source's own random error hurts about 3.5x more than the card's error of the same size. With independent error
  at 25% of the card's error rms a source keeps 69% of the value (-11.7%); it breaks even with the deployed card at about 42%, and at 50%
  it loses (+6.2%).
- (ii) Units. There is no deployable concentration. A truth-free sensitivity ranking (the Jacobian norm of the final prediction) gets 15% /
  31% / 57% of the value from 64 / 128 / 256 units per layer, against 12% / 26% for random 128 / 256. Ranking by |adjoint| looks
  concentrated (128 units beat all 1024: -21.4% vs -18.4%), but it picks units with the final-layer error, which is unpredictable from the
  card (section 33). The first-order-gain ranking is target leakage: a random-sign fix chosen the same way gets -42.7%.
- (iii) Layers. L13 and L14 carry 85% of the value (-6.3% and -5.8% alone), L12 the rest (-2.7%). The single layers add up to the
  joint value. L10-11 is worth at most 1%.
- (iv) Origin. An earlier fix neither persists nor regrows, because it never reaches the later layers. Per-unit k3/k4 feed only their
  own layer's pair map (its mean and covariance), not the third-order carriers. After truth at L6-8, L9-11 or L12, the card's k3/k4 error
  at later layers moves by at most 0.7% of itself. Earlier fixes buy 2-5%.
- Implied spec: per-unit k3 within about 1.5% and k4 within about 7% relative rms (25% of the card's current 6% / 25% error), on all
  units at L12-14, with the L12-14 correction at 0.5, is worth about -12% (-18% if exact). Break-even is about 2.6% / 11%. That is a 4x
  accuracy gain over the card. Sampling would need about 2 million forward samples (about 175x our bill), and a targeted estimator on 128
  truth-free-ranked units per layer would collect only about -4% even if exact.

Harness (`k34_spec.py`, launcher `k34_chain.sh` / `k34_chain_E.sh`): torch s6bcard + B2 ep1, learned ON, no refit, final MSE against the
organizer truth; 7 nets held out from B2's training (150, 250, 300, 400, 500, 600, 650); per-unit truth k3/k4 from our 8M files; the
learned mean correction at 0.5 at L12-14 in every row unless noted. "vs deployed" is the geometric mean over the 7 nets of the final MSE
ratio to the deployed card. "Share" is the fraction of the full-injection gain, measured against the correction-0.5 control.
Checks before the numbers:
- The 8M truth's own noise is under 1% of |truth| for k3 and about 3% for k4 at L12-14, i.e. 1-3% of the measured card-vs-truth gap's
  energy. The truth is effectively exact here.
- Section 34's injection changed only the pair map's inputs; the message-edge source, the GNN features and the birth update kept the
  card's k3. Writing the new k3 into every consumer gives the same result: -10.8% vs -10.9% as deployed, -18.1% vs -18.4% with the
  correction at 0.5. Everything below uses the pair-map form.
- Per-net precision noise is about +-0.5% (a variant identical up to rounding moved single nets by up to 0.4%), so read the 7-net means.
- The first-order (adjoint) prediction of the joint gain is 41-55% per net against the measured 21%, so the response is strongly curved.

(i) Accuracy (all 1024 units, L12-14):

| source | vs deployed | share | nets better |
|---|---|---|---|
| control: card's own k3/k4, correction 0.5 | +3.0% | 0 | 1/7 |
| card + 0.25 x (truth - card) | -7.7% | .50 | 7/7 |
| card + 0.5 x (truth - card) | -14.7% | .83 | 7/7 |
| card + 0.75 x (truth - card) | -18.3% | 1.00 | 7/7 |
| truth | -18.4% | 1 | 7/7 |
| card + 1.25 x (truth - card) | -14.7% | .83 | 7/7 |
| card + 0.5 x (truth - card), correction left at 1 | -11.6% | .68 | 7/7 |
| truth + independent noise, rms 0.25 x the card's error | -11.7% | .69 | 7/7 |
| same, 0.5 x | +6.2% | -.15 | 3/7 |
| same, 1 x | +79.2% | -3.6 | 0/7 |

- Along the card's own error direction the response is concave, with its optimum near 0.85: half the error removed gives 83% of the value.
  A quadratic through the five points has a linear term of -48% of the control MSE (the adjoint predicts 41-55% per net) and a
  curvature of +27%, which puts the optimum at 0.88 and fits every point within 0.1%.
- Independent error is far more harmful than the card's own error. A quadratic fit gives about +100% of the MSE per (noise / card error)^2,
  against about +28% for the card's own error direction. So a source must beat the card by a wide margin: at 25% of the card's error rms it
  keeps 69% of the value, and it breaks even with the deployed card at about 42%.
- In absolute terms, the card's error at L12-14 is 5.7-6.8% (k3) and 23-29% (k4) of |truth| (relative rms). A 25% source means k3 within
  about 1.5% and k4 within about 7%; break-even is about 2.6% and 11%.
- A half-accurate source still wants the correction retuned: 0.5 gives -14.7%, 1 gives -11.6%.

(ii) Units (truth only on the top-K units per layer at L12-14):

| ranking (units per layer) | 64 | 128 | 256 |
|---|---|---|---|
| truth-free: Jacobian norm x the layer's rms error (16 random projections) | -0.2% (.15), 3/7 | -3.7% (.31), 5/7 | -9.2% (.57), 7/7 |
| random | | +0.5% (.12), 3/7 | -2.6% (.26), 7/7 |
| \|adjoint\| x the layer's rms error (uses the final-layer truth) | -15.7% (.88), 7/7 | -21.4% (1.14), 7/7 | -25.2% (1.32), 7/7 |
| first-order gain -(g3 d3 + g4 d4) (uses final and per-unit truth) | -40.7% | -50.2% | -55.8% |

Each cell is vs deployed, then the share of the full-injection gain in parentheses, then nets better.

| control or combination (128 or 256 units) | vs deployed | nets better |
|---|---|---|
| \|adjoint\| units, random-sign fix (s x (truth - card)) | +8.9% | 0/7 |
| gain-style selection and injection of a random-sign fix | -42.7% | 7/7 |
| Jacobian 128, half accurate / truth + 0.25 noise | -2.2% / -1.9% | 4/7, 3/7 |
| Jacobian 256, half accurate / truth + 0.25 noise | -6.4% / -5.8% | 7/7, 6/7 |

Reading:
- The first-order-gain ranking is target leakage. Choosing units by the sign of the final-MSE gradient lets a random fix reach -42.7%, so
  its -50% is not a price of k3/k4.
- The |adjoint| ranking passes the sign control: a random-sign fix at its units costs +8.9%, so the truth direction matters there. But
  |adjoint| = |J^T e| needs the final error e. The truth-free |J| picks different units (top-128 overlap 33-38%) and gets about a third of
  the effect. Because the final error is unpredictable from the card's state (section 33), |adjoint| has no deployable stand-in.
- Truth-free targeting beats random by only 2-2.6x. The Jacobian ranking is not cheap either: 16 backward passes through L12-15 here.
- Partial accuracy and targeting compound as expected: at 256 units, half accuracy keeps 77% of the targeted value and 25% noise 72%.

(iii) Layers (truth at the named layers only; correction 0.5 at L12-14):

| layers | vs deployed | vs control | share | nets better |
|---|---|---|---|---|
| L12 | -2.7% | -5.5% | .27 | 6/7 |
| L13 | -6.3% | -9.0% | .44 | 7/7 |
| L14 | -5.8% | -8.5% | .41 | 7/7 |
| L12-14 | -18.4% | -20.7% | 1 | 7/7 |
| L10-11 | +0.9% | -2.0% | .10 | 4/7 |
| L10-11, correction 0.5 at L10-14 | -1.0% | -3.9% | .19 | 4/7 |

The single layers add up (23.0% of the control MSE against 20.7% joint). L13 and L14 carry 85% of the value; L10-11 is worth little.

(iv) Origin (truth at earlier layers only):

| layers | vs deployed | vs control | share | largest change in the card's k3 / k4 error at L12-14 |
|---|---|---|---|---|
| L6-8, correction 0.5 at L12-14 | +0.2% | -2.7% | .13 | 0.5% / 0.1% of itself |
| L9-11, correction 0.5 at L12-14 | +1.1% | -1.8% | .09 | 0.6% / 0.1% |
| L6-8, correction 0.5 at L6-8 and L12-14 | -0.9% | -3.8% | .18 | |
| L9-11, correction 0.5 at L9-14 | -2.0% | -4.9% | .23 | |

An earlier fix does not persist, and the card does not regrow anything: the fix never reaches the later layers. Per-unit k3/k4 feed only that
layer's pair map (its mean and covariance). They do not enter the third-order carriers (bank columns, star) that produce the next layer's
k3/k4, whose error moves by at most 0.7% of itself after truth at L6-8, L9-11 or L12. So a per-unit source must act at each of L12-14 itself;
acting once mid-network buys only that layer's mean/covariance effect, 2-5%.

The spec: per-unit k3 and k4 at L12-14 (L13-14 carry 85%), all units, random error at most about 25% of the card's current error
(k3 within about 1.5%, k4 within about 7% relative rms; break-even about 2.6% / 11%), with the L12-14 learned mean correction retuned to
about 0.5. Delivered, it is worth about -12% raw (-18% if exact). Nothing on our books reaches that accuracy: it is 4x the card's, the
physicist found no cheap per-unit k4 form, and sampling needs about 2 million samples.

Receipts: k34_A_semantics.jsonl, k34_adjoint.jsonl + k34_adj_rank*.npz (gradients and per-unit errors), k34_B_accuracy.jsonl,
k34_C_layers.jsonl, k34_D_units.jsonl, k34_jacobian.jsonl + k34_jac_rank*.npz, k34_E_targeting.jsonl (+ .log each). Script versions:
k34_spec_v1.py (legs A-D), k34_spec.py (adds jac/adjflip/fake and the e/n modifiers; legs E).

Addendum (2026-09-25 00:20): the current-card frame from covhunter. Card: s6bcard_joint2 + round-1 dials + low-rank star +
gn_linear_hook + pair series, loopR1 params; same 7 held-out nets, my noise seeds, correction 0.5; fable_covhunter_v1/k34_noise_price.json,
their RESULT section 29. The random-error tolerance holds on the deployed card:

| band fed | truth (covhunter's a) | truth + noise at 25% of the card's error rms | at 50% | fitted curvature b | break-even |
|---|---|---|---|---|---|
| L12-14, current card | -19.6% | -12.6% (7/7) | +7.3% (1/7) | 1.12 | 42% |
| L2-14, current card | -23.7% | -14.4% (7/7) | +13.2% (0/7) | 1.49 | 40% |
| L12-14, my card (above) | -18.4% | -11.7% (7/7) | +6.2% (3/7) | about 1.0 | 42% |

Feeding L2-14 buys a few more points than L12-14 but tightens the tolerance about 1.3x. On r100, noise at 50% costs +33% (L12-14) and
+59% (L2-14), so price candidates on the held-out nets, not on r35/r100.
Joint spec: random per-unit error at most about 25% of the card's current error rms at EVERY layer the source feeds, for about 60-65% of
the lever (-12..-14% raw); break-even about 40%. Over L2-14 the card's error is 3-7% (k3) and 11-27% (k4) relative rms, so k3 within about
1-2% and k4 within about 3-7%.

## 36. Component ablation audit on today's card: no second s1 (lead's task, 2026-09-24 23:05-00:00)
Verdict: no listed term is harmful, and exactly one is neutral. That one is K21 leg j3 of the birth update: +0.002% pooled raw on VALID16,
90% interval -0.05..+0.05%, 9/16 nets better. Its bill is a few n^2 elementwise operations per layer, so it is a free drop worth at most
about 0.1% of the bill (engineer2 to price). Every other term costs +0.6% to +2100% raw when removed.
- The nearest bill candidate is the message edge at L12-14. It costs +0.76% raw against about 0.5% of the bill (S21 census: edge 1.86 U
  over 7 layers), so removing it loses about 0.25% adjusted.
- The one extra I added, the bank part of the L15 k3 (the ~3% L15 bank pass), is essential: +579%.

Card: s6bcard_loop3 (LOOP_DIALMOD s6bcard_jfitR1: round-1 dials; S18 low-rank star with s1 = 0; gn_linear_hook; E-67 pair-series loop
form, fine_ckpt off) with the loopR1 params. loopR2 failed its bar at 23:29, so loopR1 stays deployed. Learned ON, no refit, float64 CPU,
3 threads, VALID16, organizer truth; `ablate_loop3.py` (`ablate_loop3_v2.py` adds '+' combinations, carry_s / carry_n and bank_l15).
Each term is removed by wrapping the card's own hooks, so the dial module's hooks stay underneath.
Parity: the base on net 37 is 1.6313824e-8 against the physicist's loopR2 valid_before on this card, 1.6313830e-8 (3.5e-7, thread-count
rounding). My exec'd copy of the dial module's birth update is bit-exact, and both scripts give identical bases on all 16 nets.

| term set to 0 (or its simplest form) | pooled raw | 90% interval | nets better | class |
|---|---|---|---|---|
| K21 leg j3 of the birth update (learned c3 orth(dir3) and its physical add X3 = .15) | +0.00% | -0.05% .. +0.05% | 9/16 | neutral |
| carry / newborn CP-weight dials (s_carry 1.005, s_new .993) -> 1 | +0.61% | -0.09% .. +1.38% | 9/16 | essential (free dials: keep) |
| message edge at L12-14 | +0.76% | +0.46% .. +1.04% | 3/16 | essential |
| learned mean correction at L0-7 | +1.37% | +0.64% .. +2.25% | 3/16 | essential |
| message edge at L8-11 | +1.42% | +0.53% .. +2.29% | 2/16 | essential |
| learned mean correction at L8-11 | +1.90% | +0.73% .. +3.09% | 3/16 | essential |
| GN v3 covariance increment (gn_linear_hook) | +1.99% | +1.32% .. +2.63% | 2/16 | essential |
| message edge, all of L8-14 | +2.21% | +1.38% .. +3.04% | 3/16 | essential |
| gain b-fix (gamma .1 -> 0) | +3.66% | +3.19% .. +4.11% | 0/16 | essential |
| K21 leg j0 | +3.98% | +2.84% .. +5.13% | 0/16 | essential |
| recentering (knob .92 -> 0) | +4.00% | +3.13% .. +4.91% | 0/16 | essential |
| learned mean correction at L15 (the GNN head) | +4.17% | +3.55% .. +4.80% | 0/16 | essential |
| K21 leg j2 | +4.36% | +3.26% .. +5.43% | 0/16 | essential |
| learned mean correction at L12-14 | +7.67% | +6.00% .. +9.37% | 1/16 | essential |
| omitted-radial source | +19% | +14% .. +24% | 0/6 | essential |
| star part of the L15 k3 | +22% | +17% .. +26% | 0/6 | essential |
| K21 leg j1 (the physical X1 = .56 leg) | +22% | +20% .. +25% | 0/16 | essential |
| K31 input to the pair map | +32% | +28% .. +37% | 0/6 | essential |
| carried b, every layer | +34% | +29% .. +40% | 0/6 | essential |
| star push off-diagonal (s2; s1 is already 0) | +91% | +74% .. +107% | 0/7 | essential |
| star push diagonal | +160% | +136% .. +185% | 0/7 | essential |
| bank part of the L15 k3 (extra: the ~3% L15 bank pass) | +579% | +547% .. +611% | 0/16 | essential |
| radial k4 everywhere (also zeroes the K31 input, built from t4) | +1040% | +935% .. +1152% | 0/6 | essential |
| K22 input to the pair map | +2141% | +1971% .. +2305% | 0/6 | essential |

Notes:
- The rows with 6-7 nets come from a set I stopped after five complete nets at 23:52, because the box was fully loaded (load average 16
  on 16 cores, 14.5 s per net and variant). Every one of those nets was above +7.8%.
- There is no separate "carry dressing" term on this card: the physicist's distinct-index dressing was closed on the dense replica
  (TEAM_LOG 14:36). The nearest terms are the carry / newborn CP-weight dials, reported above.
- Zeroing radial k4 also zeroes the K31 input, because T31 is built from t4.
- Net 37 alone suggested four candidates: the carry dials -1.5%, the mean correction at L12-14 -0.9% and at L0-7 -0.6%, and leg j3.
  On 16 nets only leg j3 survives. A single-net screen is unreliable at this card's +-0.5% per-net precision noise.
- The S4 audit's only flags were lower pair-series orders (at most 0.15% of the bill); not re-run.
Receipts: ablate_loop3_r37.jsonl, ablate_loop3_v16a/b/c.jsonl (+ .log), ablate_summary.json; launchers ablate_chain.sh, ablate_chain2.sh.

Addendum (2026-09-25 01:36): engineer2 measured the two survivors natively (per-request bill trace of S23 on net 100, 311.08 G/net;
fable_engineer2_v1/diet5/bill_attrib_S23_edge_bd.txt, bill_sites_bd_update_S23.txt):
- K21 leg j3: 18.9 M per layer, 0.283 G per net = 0.091% of the bill. Removing it is about -0.089% adjusted, a small clean gain. The full
  saving needs the persistent (4,n,n) direction stack in _e2_bd_state shrunk to 3; otherwise only 0.028% of the bill goes.
- Message edge at L12-14: 1.71 G = 0.550% of the bill (a flat 0.183% per layer over L8-14; all of L8-14 is 1.283%). Removing it is about
  +0.76% raw - 0.55% bill = +0.21% adjusted, a loss. Keep it.

## 37. Is the fourth-order content reaching per-unit k4 at L13-14 compressible into columns? (lead's task with the physicist, 2026-09-25 00:00-01:15)
Verdict: the "columns x layers x 2n^2" framing fails on physics, before cost.
- The fourth-order content born at a layer, pushed forward with average-gate linear transport (the column-carrier model), reaches only
  5% (births at L0), 7-8% (L1) and 6-16% (L2-9) of the same births' true contribution to per-unit k4 at L14, on both nets. In the
  per-unit spread (what the 7% spec is about) its pattern does not align with the true one for births at L0-7 (cosine -0.20..+0.15).
  So no number of CP4 columns with gate transport reproduces the early births.
- The early contribution arrives through the true ReLU transport at every layer, in two channels of similar size. The born fourth-order
  content, carried with coincident-index dressing (5-10x the linear-gate value), gives 50-72% of the early spread. The born third-order
  content, converting to k4 at later ReLUs, gives 28-50% (both nets).
- The carrier the spec needs is therefore a dressed fourth-order recursion (diagonal + (2,2)/(3,1) planes with pattern-dependent ReLU
  transfer at every layer) plus a k3 -> k4 conversion fed by the full third-order tensor. Its plane floor alone is several n^3 products
  per layer, roughly +20-40% of the bill. The lever is -12% (a source at 25% of the card's error) to -18% (exact). Cost goes on the
  physicist's page.
- The birth split also changes the target. The NORM of k4 at L13-14 is early-born (L0 27%, L0-5 about 70%), but the per-unit SPREAD
  beyond the isotropic v^2 term, 23-26% of |T| and what the card misses, is born evenly: every two-layer band holds 10-18%, L10-13 about
  25% together.

(a) Birth-layer split (`birth_split_k4.py`, from the physicist's Gaussian-restart runs CR-19, r35 / r100, 8M per half; the increment of
births in a band is restart(l) - restart(next); shares are projections on the independent 32M truth T, and "spread" is the part beyond
span{1, v^2}). restart(0) reproduces T to 3.5% (the MC floor).

| births at | norm share L13 | norm share L14 | spread share L13 | spread share L14 |
|---|---|---|---|---|
| L0 | .27 / .27 | .27 / .27 | .13 / .15 | .15 / .18 |
| L1 | .13 / .13 | .12 / .12 | .06 / .05 | .06 / .04 |
| L2-3 | .16 / .21 | .16 / .21 | .11 / .16 | .11 / .15 |
| L4-5 | .16 / .16 | .15 / .15 | .13 / .14 | .11 / .12 |
| L6-7 | .13 / .10 | .13 / .10 | .14 / .12 | .16 / .11 |
| L8-9 | .08 / .09 | .09 / .09 | .16 / .16 | .15 / .13 |
| L10 | .04 / .02 | .05 / .03 | .10 / .08 | .07 / .07 |
| L11 | .02 / .02 | .01 / .02 | .10 / .07 | .07 / .07 |
| L12 | .01 / .01 | .02 / .02 | .06 / .04 | .04 / .05 |
| L13 | | -.00 / -.02 | | .06 / .06 |

(b) Linear reach (`k4_reach_mc.py`, MPS, 8M samples per birth layer): the same Gaussian restart at pre-l, y = relu(z_l), dy = y - E[y]
(exact relu mean), pushed with M = W[l+1] D W[l+2] ... D W[L], D = diag(Phi(m/s)) from the 32M truth. The per-unit k4 of dy @ M is split
into index classes (certified against the full tensor on a synthetic 6-unit case to 2e-11). Compared with the restart increment of the
same births (`k4_reach_compare.py`). Two checks: births at L13 reach L14 through one exact linear step, and the linear reach recovers
0.92-0.93 of the increment there (MC-noise-limited); one ReLU in between (L12 -> L14) already drops to 0.54-0.57.

| births at | linear reach carries (coef along the increment), r35 / r100 | spread-space cosine, r35 / r100 | classes of the linear reach (r35): diag / (2,2) / (3,1) / distinct |
|---|---|---|---|
| L0 | .045 / .047 | -.05 / +.15 | .71 / .23 / .03 / .03 |
| L1 | .083 / .065 | +.09 / -.10 | .49 / .35 / .02 / .13 |
| L2-3 | .090 / .080 | -.20 / -.13 | .31 / .48 / -.01 / .21 |
| L4-5 | .090 / .074 | -.04 / -.05 | .13 / .41 / .05 / .41 |
| L6-7 | .098 / .055 | +.24 / +.14 | .06 / .26 / .06 / .62 |
| L8-9 | .163 / .103 | +.35 / +.29 | .02 / .13 / .08 / .77 |
| L10 | .295 / .166 | +.49 / +.36 | .02 / .11 / .09 / .78 |
| L11 | .304 / .403 | +.43 / +.46 | .01 / .05 / .07 / .86 |
| L12 | .542 / .568 | +.63 / +.62 | .01 / .06 / .06 / .86 |
| L13 | .927 / .924 | +.86 / +.84 | .01 / .04 / .08 / .87 |

At this MC noise the largest possible spread-space cosine is about .83-.95. The full-vector cosine for L0 is .94-.95, but only because
of the shared level. Rescaling the linear reaches by birth band and fitting the truth on held-out units (r35) moves the error only from
.265 to .246 (L14) and from .250 to .233 (L13), against a spec of .07.

(c) Which born order carries the contribution (`sym_restart_mc.py` / `sym_restart_analyze.py`, 8M per half, common random numbers):
after the birth ReLU, "full" keeps y and "even" uses E[y] + s (y - E[y]) with a random sign s per sample. That keeps the covariance and
the whole fourth-order tensor and removes every odd order. odd = full - even; even = even - restart(next). My full runs match the
physicist's restart(l) within MC noise (4-9% of |restart(l)| for l <= 6). Births at L12 give exactly 0 odd channel at L13 (one linear
step), as they must. Shares of each band's increment at L14, norm / spread:

| births at | odd channel (born third order), norm, spread: r35 / r100 | even channel (born fourth order), norm, spread: r35 / r100 |
|---|---|---|
| L0 | .50, .41 / .45, .33 | .50, .59 / .55, .67 |
| L1 | .73, .49 / .63, .47 | .27, .51 / .37, .53 |
| L2-3 | .51, .33 / .45, .28 | .49, .67 / .55, .72 |
| L4-5 | .46, .40 / .55, .39 | .54, .60 / .45, .61 |
| L6-7 | .44, .42 / .45, .41 | .56, .58 / .55, .59 |
| L10 | .65, .60 / .73, .66 | .35, .40 / .27, .34 |
| L12 | .39, .38 / .30, .30 | .61, .62 / .70, .70 |

L13 agrees within about .05. The odd channel's noise is .07-.21 of the increment. For the two-layer bands only the first layer's odd
channel is measured (the even run symmetrizes post-l only), so their odd share is a lower bound.

Reading:
- A column carrier with gate transport is the wrong object. Each ReLU transfers born fourth-order content far more strongly than the
  average gate says: E[f'^m] = Phi for m coincident paths through a unit, instead of Phi^m. The early content's influence is rebuilt at
  every layer.
- About half of the early spread comes through the born third-order content (k3 x k3 -> k4 at later ReLUs). A k4 carrier alone cannot
  collect it; it needs the full K3, including the distinct content the card's bank misses (sections 34-35, the physicist's J-K).
- So the spec (k3 within ~1.5%, k4 within ~7% at L12-14) calls for the closed-family object: a full-depth, nonlinear fourth-order
  recursion fed by an accurate third-order tensor. The physicist's page carries the cost.
Closing notes (2026-09-25 01:05):
- k4 alone. The physicist ran it on this harness (refit6b/carry/k34_spec_ch.py, a versioned copy that reproduces my deployed and joint rows
  bit-exactly; 7 held-out nets, correction 0.5 at L12-14): k3 + k4 -18.4%, k3 alone -8.7%, k4 alone -4.9% exact and -3.2% at the spec,
  control +3.0%. That is under the lead's 5% line, so the k4 question is closed.
- Planes only. A planes-only fourth-order recursion also fails on content, not just cost: the 09-02 cross-review found plane-only pushes
  lose about half the (2,2)/(3,1) content per layer to the distinct classes, which matches the distinct remainder of 54-87% of the linear
  reach from L4 on in table (b). Cost rows are on the physicist's page (leadergap RESULT.md section N): the dressed planes-only recursion
  is 2.3-3.5 U/layer, +21-33% of the bill, break-even raw 18-25%.
Receipts: birth_split_k4.json, k4_reach_r35_all8M.npz/.json, k4_reach_r100_all8M.npz/.json, k4_reach_compare_r35/r100.json,
sym_restart_r35_h8M.npz, sym_restart_r100_h8M.npz, sym_restart_analyze_r35/r100.json.

## 38. Per-net adaptive allocation, step 1: the oracle bound is too small (lead's task, 2026-09-25 01:05-01:17)
Verdict: STOP at step 1. For every pair and multi-way set of card variants with native VALID40 rows, choosing the better variant per net
beats the best single variant by only 0.00-0.19% adjusted. That is under the lead's 0.5% bar and close to the noise floor, so no router
over these variants can pay. The only per-net heterogeneity large enough on record is in cost dials: the S2-era old-tier rank 320 at the
L10/L13 joins gives a 4-net oracle 0.6-0.7% better than its best single setting. That is an old card and 4 nets, so it is a pointer, not
evidence.

The score is the mean over MLPs of MSE_i x util_i (lead, checked on S23's page), so each net can take its own variant. For each variant
set, from the porter's native rows (fable_porter_v1/paired_valid40_*.json, 'cand' rows, adjusted = MSE x max(util, .1); `oracle_router.py`):
best single = the variant with the lowest mean adjusted; oracle = the mean over nets of each net's best adjusted.

| variant set | mean adjusted per variant | best single | oracle gain | nets whose winner differs | sd of per-net log ratio |
|---|---|---|---|---|---|
| S21 vs S21jn (join) | 2.6424e-9 / 2.5944e-9 | S21jn | -0.07% | 5/40 | 1.35% |
| S21s1 vs S21s1 + s1 = 0 | 2.5953e-9 / 2.5197e-9 | s1 = 0 | 0.00% | 0/40 | 0.62% |
| S21s1 vs S22 | 2.5953e-9 / 2.5626e-9 | S22 | -0.11% | 6/40 | 1.49% |
| S21s1 + s1 = 0 vs S22 | 2.5197e-9 / 2.5626e-9 | s1 = 0 | -0.16% | 7/40 | 1.72% |
| S23 vs S23 + Nystrom star (q1) | 2.4422e-9 / 2.4344e-9 | Nystrom | -0.08% | 10/40 | 0.52% |
| S21, S21jn, S21s1, s1 = 0, S22 (5-way) | | s1 = 0 | -0.19% | 8/40 | |
| S22, S23, S23 + Nystrom (3-way) | | Nystrom | -0.08% | 10/40 | |

Noise floor: two equal cards whose per-net rounding-level differences have sd s give an oracle "gain" of about s/sqrt(2 pi). With s about
0.29% (the porter's native-vs-torch parity) that is about 0.12%, which is the size of every gain above.

Why the gains are small: these pairs are successive physics improvements, so one side wins on most nets (s1 = 0 wins on all 40), and the
per-net spread (sd 0.5-1.7%) is small next to the mean gaps.

Where routing value could live: a cost dial whose raw price varies by net, so that cheap settings pay on tolerant nets only. The one record
is my S2-era deep-dial table (section 18, torch, 4 nets, bills estimated):

| dial | adjusted per net, r35 / r37 / r39 / r100 | best single | oracle gain |
|---|---|---|---|
| old-tier rank 320 at L10 and L13 (bill -3.5%) | .971 / 1.014 / .963 / 1.015 | .991 (rank 320) | -0.73% |
| rank 384 at L10, 320 at L13 (bill -1.8%) | .985 / 1.017 / .974 / 1.007 | .996 | -0.60% |
| rank 352 at L10 and L13 (bill -1.8%) | .988 / .996 / .981 / .997 | .990 | 0.00% (it wins on all 4) |

Testing that idea on today's card would take one cost dial's rows on all 40 VALID nets, plus a runtime rule. The natural rule is an
adaptive join rank chosen from the energy the join's own sketch captures, which is a deterministic rule rather than a learned router.
Under the method-router rule it would still need the cross-seed held-out check. Nothing was run for step 2.
Receipts: oracle_router.py, oracle_router.json.

## 39. Energy-adaptive old-tier rank at the joins: CLOSED (lead's task, 2026-09-25 01:18-01:34)
Verdict: it fails the bar. At matched estimated bill the energy-adaptive rank never beats a fixed rank by 0.3%.
- Where the threshold makes per-net choices at all (tau .99976, at the L10 join), it is 0.28 pt worse than the fixed-rank curve and
  0.40 pt worse than the same per-join schedule with a fixed rank.
- The other thresholds make no per-net choices; they reproduce fixed per-join schedules exactly.
- The reason: a join's captured energy barely varies across nets at a given join (+-10-20%) and does not predict which nets tolerate the cut.
  The correlation between a net's missed energy and its raw penalty is -0.58..+0.18.
- The per-net penalty itself varies a lot (fixed 320: -0.7% to +8.6% raw). So per-net routing between ranks 320 and 384 has an oracle of
  about -0.3 to -0.8 pt adjusted, but this signal cannot find it.

Card: the physicist's S23-structure torch card refit6b/s6bcard_loop4j.py, loopR1 params + round-1 dials (S24's verdict was not in), loaded
as refit_mps_v3j2.py --eval-only does; learned ON, no refit, float64 CPU, VALID16. `adaptive_join.py` rebuilds the Nystrom join with its
384 landmark rows in ENERGY order, so the QR nests: the first r basis vectors span exactly the rank-r join. Every truncation's captured
energy comes from quantities the join computes anyway:
  E(r) = sum_{k<r} [sum_c w_c (Qn^T J)_kc^2 + (Qn^T Qp core Qp^T Qn)_kk] / trace(Gram)
Arms: fixed<r> (every join), sched<a>-<b>-<c> (L7 / L10 / L13), and adapt<tau> (per join and net, the smallest r in {256, 320, 384} with
E(r) >= tau).
Parity on net 37: the card as built gives 1.67397e-8 (the physicist's certificate is 1.67403e-8; thread-count rounding), and energy-ordered
fixed384 is +0.001% from it.
Bill: the rank-dependent work per net from the push logs, with the porter's S21 unit costs at rank 384 scaled by r/384. That is 0.174 U
basis push plus 0.323 U per 1000 old columns per layer, and 0.718 U per 1000 joined columns per join, as a share of ~145 U.
(`adaptive_join_analyze.py`)

| arm | mean rank, L7 / L10 / L13 | est. bill vs fixed384 | pooled raw vs fixed384 | fixed curve at that bill | arm minus curve | nets better |
|---|---|---|---|---|---|---|
| fixed320 | 320 / 320 / 320 | -5.35% | +4.04% | | 0 | 1/16 |
| adapt .9997 (= sched 384-320-320 exactly) | 384 / 320 / 320 | -4.10% | +2.62% | +2.73% | -0.11 pt | 1/16 |
| sched 384-352-320 | 384 / 352 / 320 | -3.19% | +1.66% | +1.78% | -0.12 pt | 1/16 |
| adapt .99976 (per-net split at L10) | 384 / 352 / 320 | -3.18% | +2.06% | +1.78% | +0.28 pt | 2/16 |
| fixed352 (not a tree size natively) | 352 / 352 / 352 | -2.67% | +1.25% | | 0 | 2/16 |
| adapt .9998 (= sched 384-384-320 exactly) | 384 / 384 / 320 | -2.27% | +1.26% | +1.06% | +0.20 pt | 2/16 |
| fixed384 | 384 / 384 / 384 | 0 | 0 | | 0 | |

A 50/50 mixture of 320 and 384 (mean rank 352, both tree sizes) is +2.02% raw, worse than fixed352's +1.25%: the raw cost is convex in
rank. That same convexity is why a per-net 320/384 split at a join loses to a fixed 352 there.

Missed energy 1 - E(320), over the 16 nets:

| join | range | correlation with the per-net raw penalty of fixed320 | of fixed352 |
|---|---|---|---|
| L7 | 1.18-1.45e-3 | +0.18 | -0.24 |
| L10 | 2.18-2.84e-4 | -0.30 | -0.58 |
| L13 | 0.70-1.24e-4 | -0.25 | +0.10 |

Side note for the cost ledger (not the tested question; it depends only on the native bill): uniform rank 320 costs +4.04% raw (1/16).
Its estimated bill change is -5.35% with the unit costs, or -4.3% with engineer2's 26% rank-dependent share. That makes it -0.5% to -1.6%
adjusted, and the per-net 320/384 oracle is -1.3% to -1.9%. Only a native census of a tree-sized rank-320 S23 can say whether the uniform
cut is a small real gain; the torch estimate cannot.
Receipts: adaptive_join_smoke.jsonl, adaptive_join_fixed.jsonl, adaptive_join_pass2.jsonl, adaptive_join_summary.json.

## 40. H1 falsifier, the subspace-Gaussianized restart: FAILS on k3 (lead's task from the physicist's rivals page section O, 2026-09-25 01:36-02:41)
Verdict: H1 fails its falsifier. At l0 = 6, keeping the TRUE state in the top-64 covariance subspace (58% of the variance) and Gaussianizing
only the complement brings per-unit k4 at L13-14 to .086-.092 (r35 / r100, L13 / L14), which passes the <= 11% bar. It leaves per-unit k3 at .072-.082, unchanged
from the full restart and far from the <= 3% bar. Even 256 modes (91% of the variance) leave k3 at .031-.038.
- The top modes hold k3's LEVEL (its scale vs truth rises from .50 to .80), which the isotropic term already absorbs, but not its per-unit
  spread.
- k4 alone is worth about -5% (the physicist's arm), so a top-mode subspace carrier on top of the card would collect little. H1 closes.
- The complement test (r35, l0 = 6) shows where the content lives. Keeping everything TRUE except the top modes, and Gaussianizing only
  those, is as bad as the full restart:
  - Gaussianize the top 64 (42% of the variance kept true): k4 .153 / .149, k3 .083 / .072;
  - Gaussianize the top 256 (9% kept true): k4 .154 / .148, k3 .086 / .077;
  - the full restart: k4 .153 / .146, k3 .085 / .077.
  So the bulk's own non-Gaussianity contributes nothing to k3/k4 at L13-14. The top modes' own non-Gaussianity carries about half of k4 and
  k3's level. The per-unit k3 spread, and the rest of k4, live in MIXED cumulants coupling the top modes with the bulk. A carrier would
  need those top x bulk terms, e.g. (top, top, bulk) and (top, bulk, bulk): 64^2 x 1024 and 64 x 1024^2 entries at R = 64. A top-mode
  subspace carrier misses them by construction.
- A bottom-R test as first posed is degenerate: the bottom 64 eigen-directions of cov_pre_6 carry 0% of the variance (clipped negative
  eigenvalues of a near-singular covariance) and the bottom 256 carry 0.08%. I stopped that run and replaced it with the complement
  test above.

Method (`subspace_restart_mc.py`, MPS with the lock in a with-block, behind machine_gate.sh; 8M samples per half, common random numbers
across R). The exact network is run to pre-activation l0 and d = z - mean_pre_l0 is split in the eigenbasis U of the 32M cov_pre_l0
(covhunter's mcall; descending eigenvalues). Then
  z'_R = m + (d U)[:, :R] U_R^T + (eps sqrt(lam))[:, R:] U_perp^T
and the true network runs on to L15. Mean and covariance are exact for every R; the joint law inside the top-R subspace is the true one;
every higher cumulant involving the complement is removed. Scored with the physicist's gauss_restart_eval.py, unchanged: "hyb" is the
restart plus the best isotropic term (the lead's pass test), "restart" is the restart alone, noise-debiased; per-unit relative error vs
the independent 32M truth.
Controls: R = 1024 (the exact network) scores 0.000 everywhere, and R = 0 reproduces the physicist's restart (l0 = 6: k4 .153/.146,
k3 .085/.077 against their .152/.146, .085/.077).

Net 35 (variance held by the top R at l0 = 2: 12 / 36 / 80% for R = 16 / 64 / 256; at l0 = 6: 24 / 58 / 91%):

| l0 | R | k4 hyb L13 / L14 | k4 restart L13 / L14 | k3 hyb L13 / L14 | k3 restart L13 / L14 |
|---|---|---|---|---|---|
| 2 | 0 | .079 / .076 | .406 / .400 | .027 / .023 | .290 / .278 |
| 2 | 16 | .075 / .074 | .394 / .386 | .027 / .023 | .280 / .268 |
| 2 | 64 | .052 / .052 | .306 / .301 | .026 / .022 | .216 / .208 |
| 2 | 256 | .025 / .022 | .121 / .115 | .019 / .017 | .083 / .077 |
| 6 | 0 | .153 / .146 | .729 / .715 | .085 / .077 | .517 / .488 |
| 6 | 16 | .117 / .104 | .579 / .546 | .091 / .086 | .383 / .359 |
| 6 | 64 | .092 / .087 | .385 / .357 | .082 / .073 | .222 / .212 |
| 6 | 256 | .047 / .037 | .121 / .111 | .038 / .031 | .063 / .058 |
| both | 1024 | 0 | 0 | 0 | 0 |

Net 100 (the same variance shares within 0.3 pt):

| l0 | R | k4 hyb L13 / L14 | k4 restart L13 / L14 | k3 hyb L13 / L14 | k3 restart L13 / L14 |
|---|---|---|---|---|---|
| 2 | 0 | .070 / .073 | .405 / .401 | .023 / .023 | .285 / .280 |
| 2 | 16 | .066 / .069 | .390 / .384 | .023 / .023 | .274 / .269 |
| 2 | 64 | .045 / .046 | .298 / .291 | .023 / .023 | .208 / .204 |
| 2 | 256 | .023 / .029 | .110 / .110 | .017 / .017 | .077 / .076 |
| 6 | 0 | .156 / .162 | .777 / .771 | .080 / .070 | .535 / .514 |
| 6 | 16 | .114 / .113 | .596 / .578 | .092 / .086 | .390 / .371 |
| 6 | 64 | .086 / .089 | .399 / .378 | .078 / .072 | .231 / .219 |
| 6 | 256 | .038 / .040 | .106 / .096 | .035 / .032 | .054 / .049 |
| both | 1024 | 0-.007 | 0-.007 | 0-.002 | 0-.002 |

Receipts: subrestart_r{35,100}_R{0,16,64,256,1024}_h8M.npz with _eval.json beside each, subrestart_collect.json, subrestart_r35_bottomR960_h8M.npz and subrestart_r35_bottomR768_h8M.npz (the complement test, with _eval.json).
Scripts: subspace_restart_mc.py (v1 kept as subspace_restart_mc_v1.py; the current version adds the bottom option), subrestart_collect.py.

## 41. S25 per-part attribution (lead's request; watcher run 2026-09-25 02:58, recorded 2026-09-25 08:33)
`s25_attrib.py` on the porter's native VALID40 rows (paired_valid40_stack_S25r_mem.json vs paired_valid40_stack_S24c_mem.json, same 40 nets).
S25 = S24 + treuse (bit-identical, bill -0.09%) + j3 cut (raw ~0, bill -0.091%) + R320. So S25's raw change is all R320's, and its bill
is R320's share plus about -0.18%.

| | pooled | per net |
|---|---|---|
| raw (= R320's cost) | +4.455% | -1.19 / +4.63 / +10.46% (min / median / max); better on 3/40 |
| util | -4.583% (R320 -4.403%, treuse + j3 -0.18%) | -4.69..-4.37% |
| adjusted | -0.343% | better on 21/40 |
| R320 alone, adjusted, vs S24 + treuse + j3 | -0.164% | |

This matches the porter's split: j3 + treuse -0.136% [-0.235, -0.033] 31/40, R320 part -0.207% [-1.095, +0.653] 19/40. S25 failed its
bar (-0.5%, CI excluding 0) and is held.
Per-net R320 on/off oracle: keeping R320 only on the 20/40 nets where it lowers adjusted would be -1.10% vs S25. That is the per-net
routing headroom section 39 estimated in torch (-0.3 to -0.8 pt), now measured natively. The join's energy signal cannot find those nets
(section 39), and no other signal has been tried.

Router check (2026-09-25 08:36; step 2 of the per-net study, for the one dial with native headroom). Per net, R320's own adjusted effect (S25 vs
S24 x (1 - .0018)) has mean -0.20%, sd 2.92%, and an oracle of -1.31% (20/40). Correlations with observable per-net quantities:

| feature | correlation with the per-net effect |
|---|---|
| R320's own bill saving (the old tier's share of the net's bill; known from column counts at the join) | +0.40 |
| S24 util | -0.18 |
| S24 wall | -0.22 |
| the net's raw MSE level (not observable at runtime) | +0.12 |

A leave-one-out threshold router on the bill saving puts 29/40 nets on R320 for -0.55%, against -0.20% always-on. That captures 31% of the
-1.1 pt headroom, below the lead's 50% bar, and it is one borderline correlation (p about .01) on 40 nets. A router on util overfits
(+0.55%). A weak lead only: it would need held-out nets and the cross-seed check before any use.

## 42. Learning-curve projection: training alone cannot close the gap (lead's task, 2026-09-25 08:30)
Verdict: from arm A's end, continued training on today's card projects only 0.06-0.12% more on VALID16 (0.2% at the most optimistic
decay), against a podium gap of 3.1%. Arm A already holds about 80% of what training can reach on this card, and 90% comes at about
3 epochs from the loopR3 end. Learned-only with a multi-epoch cosine beats joint 1-epoch rounds per epoch, and the dials have converged,
so a joint continuation is not the better bet. More width (+0.08 pt) and 4x more nets (TRAIN640, section 31) do not change that: the
learned parts are information-limited here, as the physicist's head audit found.

Every refit on record (VALID16 against its own start; lr 1e-4 learned / 5e-5 trunk, cosine over the run, batch 4, TRAIN160):

| run | card | VALID16 start -> end (1e-8) | change | schedule |
|---|---|---|---|---|
| B2 | S6b | 1.77402 -> 1.75080 -> 1.74614 | -1.31%, cumulative -1.57% | 2 epochs, one cosine |
| j05, 1st epoch | j05 physics change (no-refit +4.97%) | 1.83290 -> 1.71227 | -6.58% | 1 |
| j05, 2nd epoch | same | 1.71227 -> 1.69896 | -0.78% | 1 (restart) |
| loopR1 | + round-1 dials | 1.66631 -> 1.65054 | -0.95% | 1 |
| loopR2 | + s1 = 0, E-67 | 1.64783 -> 1.64237 | -0.33% | 1 |
| loopR3 (joint) | S23 structure | 1.65645 -> 1.64757 | -0.54% | 1 |
| loopR4 (joint) | same | 1.64757 -> 1.64460 | -0.18% | 1 (restart) |
| arm A (learned only) | S24 structure, own dials | 1.64753 -> 1.64228 -> 1.63990 | -0.32%, cumulative -0.46% | 2 epochs, one cosine |
| arm B (2x width) | same | 1.64753 -> 1.64093 -> 1.63866 | -0.40%, cumulative -0.54% | 2 epochs, one cosine |

Fit: on a fixed card, each successive epoch gains q times the previous one. q is .21 (B2), .34 (loopR3 -> R4) and .45 (arm A, within
one cosine). From arm A's end the remaining gain is .145% x q / (1 - q) = .06-.12%, or .22% at q = .6. Arm A holds 1 - q^2 = 80-88% of
the asymptotic gain from the loopR3 end, and 90% needs ln .1 / ln q = 2-3 epochs.
Joint vs learned-only, from the same start (loopR3's end, 1.6475e-8):
- loopR4 (joint, a full 1-epoch cosine) -0.18%;
- arm A's first epoch (learned only, the first half of a 2-epoch cosine, not annealed yet) -0.32%.
Arm A wins per epoch even before annealing. The dials drifted at most 0.3 GN steps in loopR3/R4, so they have converged.
Capacity: 2x width adds 0.08 pt after 2 epochs, all in the first epoch; the second-epoch gains are equal (-.138% vs -.145%).
Caveat: VALID16 pooled differences under about 0.1% sit near the card's per-net noise (+-0.3-0.5% per net), so the projected
0.06-0.12% is at the noise floor. Only VALID40 can confirm a gain that small.
Recommendation: if loopR5 runs, use learned-only with a 2-3 epoch cosine from arm A, expecting about -0.1% VALID16 (about -0.05..-0.1% on
the board). The remaining ~3% has to come from structure or bill.
CORRECTION (2026-09-25 09:10): the projection above is WRONG for joint training. The physicist's loopR5 (JOINT, 10 dials trainable including c0,
a fresh 3-epoch cosine, from arm A's endpoint 1.63984e-8) is at -0.172% after epoch 0 and -0.473% after epoch 1, with epoch 2 still running.
That is about 4x my projected -0.06..-0.12%.
- My error: I treated the dials as converged from loopR3/R4's drift of at most 0.3 GN steps. From arm A's learned params, c0 keeps moving
  (.694 -> .659 -> .643), so learned parts and dials are still co-adapting, and the geometric-decay fit on learned-only / 1-epoch rounds does
  not describe a longer joint schedule.
- Revised reading: joint training is NOT exhausted. "Learned-only beats joint" held only for loopR4's 1-epoch schedule. Judge further
  training by loopR5's own epochs, not by this projection.
- loopR5 epoch 2 (09:18): 1.63206e-8, -0.474% vs its start, so the third epoch added nothing (-0.0015 pt). The joint continuation from
  arm A saturates at about -0.47%, 4x my projection, and is now exhausted.

## 43. Per-net R320 routing: no router generalizes; the lever is a FIXED schedule, R320 at the L13 join only (lead's top priority, 2026-09-25 08:40-10:33)
CORRECTION (2026-09-25 11:36): the pick below (384/384/320, L13 only) is WRONG. Its bill column comes from a torch unit-cost model
that assumed the L13 join rotates every old column into the L13 frame. The native L13 join is lazy, so the L13 cut saves only -0.688%
natively, not -1.87%, and L13-only is harmful (+0.35% measured). The router verdict and the raw columns stand. See section 44.
Verdict: no router ships. The one that passed the VALID40 leave-one-out bar (L10 kink rows) does worse than always-on on 40 held-out nets.
The robust lever is a FIXED schedule, and the finer schedules put all of it in the last join.
- R320 at the L13 join ONLY (384/384/320) is the best fixed schedule on VALID40: raw +1.07% (se .17), bill -1.87%, adjusted -0.832%, 30/40
  nets better.
- The L7 and L10 cuts are net harmful alone (+0.44%, +0.31%), and rank 256 at L13 is too deep (+0.71%).
- The cuts add up: L10 alone +1.82% plus L13 alone +1.07% gives 2.89% raw, against 2.88% for both.
- Only the last join has a good exchange rate: it carries 43% of the rank-dependent bill and damages only L13-15. It also avoids the
  L7 cut, which costs a third of R320's raw penalty for 23% of its bill.
- Held out (first three schedules): 384/320/320 -1.37%, all 320 -1.47%. The L13-only held-out rows are queued behind the memory gate.
VALID40 is the less tolerant draw for R320 (raw +4.38% vs +3.16% held out, about 1.8 se), so a VALID40-only judgment understates it.

Target: R320's exact per-net effect from the porter's native rows, S25 (paired_valid40_stack_S25r_mem.json) vs the same card without R320
(paired_valid40_s25r_noR320_mem.json): always-on -0.207%, per-net oracle -1.274% (19/40). The torch S25-like card (s6bcard_loop4j,
loopR3 params and dials; `r320_features.py`) reproduces the native per-net raw penalty at slope .996, corr .995 (means +4.38 / +4.36%). So
the per-net tolerance is a deterministic property of the net, and torch rows can stand in for native ones on other nets.

Router search on VALID40 (`r320_router.py`; LOO over 40 nets; 34 runtime features from the torch card: join sizes, live rows, missed
energy 1 - E(r) per join, alive / response / birth counts and kink fractions per layer, rank cost; plus the native bill share):

| router | pooled adjusted | nets routed | share of the oracle |
|---|---|---|---|
| nested LOO (feature and threshold chosen inside each fold; resp_L10 in 40/40 folds) | -0.753% | 26/40 | .59 |
| R320 iff resp_L10 >= 672 (kink rows at L10) | -0.753% | 26/40 | .59 |
| bill share + w x resp_L10 | -0.799% | 24/40 | .63 |
| bill share alone | -0.531% | 31/40 | .42 |
| any L7-time feature | -0.28..-0.51% | | |

Neither this router nor its causal forms carry over to held-out nets: resp_L10 is known only at L10, after the L7 join.

Per-join pricing (`r320_perjoin.py`, `r320_schedules.py`; schedule = ranks at the L7 / L10 / L13 joins; raw from the torch card; bill =
the native all-320 bill change x each segment's share of the rank-dependent cost, L7-9 .234 / L10-12 .341 / L13-15 .425):

| schedule (L7 / L10 / L13) | VALID40 raw | bill | VALID40 adjusted | better | held-out raw | held-out adjusted |
|---|---|---|---|---|---|---|
| 384 / 384 / 320 (L13 only) | +1.07% (se .17) | -1.87% | -0.832% | 30/40 | pending | pending |
| 384 / 320 / 320 | +2.88% (se .35) | -3.38% | -0.588% | 24/40 | +2.12% (se .31) | -1.373% (32/40) |
| 320 / 320 / 320 (S25's R320) | +4.38% (se .48) | -4.41% | -0.183% | 18/40 | +3.16% (se .47) | -1.469% (25/40) |
| 384 / 320 / 384 (L10 only) | +1.82% (se .30) | -1.51% | +0.305% | 19/40 | pending | pending |
| 320 / 384 / 384 (L7 only) | +1.46% (se .24) | -1.03% | +0.444% | 19/40 | +0.98% (se .27) | -0.084% (22/40) |
| 384 / 384 / 256 | +4.77% (se .39) | -3.75% | +0.707% | 15/40 | pending | pending |
| 384 / 320 / 256 | +6.54% (se .42) | -5.25% | +0.832% | 15/40 | pending | pending |

The torch-derived all-320 effect on VALID40 is -0.183%, against the native -0.207% (per-net corr .995).

Held-out check: 40 TRAIN640 nets outside TRAIN160 and VALID40 (every 12th of the split's new ranks, 0-672), torch card, VALID40 mean bill.
Every VALID40-fitted router does worse than always-on:

| rule (VALID40 thresholds) | held-out pooled adjusted | vs always-on all joins (-1.469%) |
|---|---|---|
| all joins, R320 iff resp_L10 >= 672 (not causal) | -1.288% | worse |
| L7 384, L10+L13 R320 iff resp_L10 >= 672 | -0.990% | worse (vs always 384/320/320 -1.373%) |
| L7 320, L10+L13 routed | -1.069% | worse |
| L7 routed by miss320_L7, L10+L13 by resp_L10 | -0.909% | worse |

The held-out oracles are -2.00% (all joins) and -2.31% (best of 4 schedules), so per-net headroom exists but VALID40 thresholds do not
find it. On a spread of 2-3% raw per net, 40 nets cannot pin a threshold.
Recommendation: a native build of the fixed schedule 384/384/320 (R320 at the L13 join only) on S27, judged on VALID40 plus native (or torch) held-out nets, not VALID40
alone: its 95% CI is about +-0.7%. Also a porter census for the per-join bill split, since the L7 share of 23% is the torch unit-cost model.
Pick (2026-09-25 10:36, provisional; sent to the lead, the porter and the physicist for loopR7s): 384/384/320. The 95% bootstrap intervals over nets:

| schedule | VALID40 adjusted | held-out 40 adjusted | pooled 80 adjusted |
|---|---|---|---|
| 384/384/320 | -0.832% [-1.15, -0.52], 30/40 | pending (memory gate) | about -0.96% (estimated) |
| 384/320/320 | -0.588% [-1.23, +0.07], 24/40 | -1.373% [-1.96, -0.78], 32/40 | -0.987% [-1.43, -0.54] |
| 320/320/320 | -0.183% [-1.06, +0.69], 18/40 | -1.469% [-2.32, -0.58], 25/40 | -0.837% [-1.47, -0.21] |

384/320/320 passes the lead's pre-set held-out bar (<= -0.30%, CI excluding 0). The held-out estimate for L13-only (-1.09%) scales the
measured held-out 384/320/320 raw penalty by L13's VALID40 share (.37), using the per-join additivity measured on VALID40. The lead's rule
(L13-only wins unless 384/320/320 is more than 0.2 pt better on 80 nets) would flip only if L13-only's held-out came out worse than
-0.73%, which needs a held-out raw penalty of +1.16% or more. VALID40's is +1.07%, and held-out penalties ran 25-30% lower for every
schedule measured.
Receipts: r320_router.py (v1 kept), r320_features.py (v1 kept), r320_feature_table.py, r320_features.json, r320_perjoin.py,
r320_schedules.py, r320_features_v40*.jsonl, r320_features_ho40*.jsonl, r320_router.json.

## 44. Native bills reverse the R320 schedule pick to L10 only (384/320/384); the torch bill model missed the lazy L13 join (lead's task, 2026-09-25 11:22-11:36)
Verdict: 384/320/384 (R320 at the L10 join only; the porter's S27t) is the best schedule on VALID40 with native bills, at -0.689%.
The next best is 384/320/320 at -0.370%. L13-only is harmful at +0.37% (native measured +0.351%).

Why my bill model misattributed L13 vs L10. The native L13 join is lazy (`_SLR_LAZY_LAYERS = (13,)` in fable_porter_v1/stack/stack_S27s_mem.py).
It never rotates the old factors into the L13 frame. The columns joined at L7 and L10 (about 11.8k of 16.4k old columns) stay in the L10
frame for all of L13-15. Their reconstruct (1024 x r10 x C10 per layer) and their read (resp x C10 x r10) therefore scale with the L10
rank. Only the L13 cohort (about 4.6k columns), the old-basis push, the L13 join itself and a small map product (1024 x r13 x r10)
scale with the L13 rank. My torch model rotated everything into the L13 frame, so it credited the whole L13-15 segment to r13. The
lead's guess was right in substance: the tier arriving from L10 is billed at the L10 rank.

Evidence, two independent checks:
- **Model.** `r320_lazy_model.py` prices each native old-tier product from the per-net shapes in the torch logs. One constant per component
  is fitted to engineer2's S27 census totals only (reconstruct 17.50, read 11.52, basis push 1.46, joins 8.80 U). It reproduces his segment
  split, 9.66 / 14.44 / 15.18 U against 9.59 / 14.82 / 14.87. With one shape factor of 0.87 (engineer2 measured shape effects of about 2
  of 16.7 points), it predicts both native schedule bills, which were not used in the fit:

| schedule | lazy L13 join (native code) x 0.87 | native VALID40 | materialized L13 join (my old assumption) |
|---|---|---|---|
| 384/384/320 | -0.72% | -0.688% | -1.87% |
| 384/320/320 | -3.16% | -3.167% | -3.70% |

- **Native census.** The porter's bill map (fable_porter_v1/stack/billmap; nets 35 and 100; no setup) shows where the L10-only cut saves.
  On net 35 it saves -2.445%. Of that, -0.909 pt (37%) falls in L13-15. The prefix reconstruct gives -0.22 / -0.20 / -0.19 pt at L13 /
  L14 / L15, and the lazy prefix read gives -0.15 / -0.13 pt at L13 / L14. All of it is L10-rank work inside the L13 segment.

Native per-join bills (vs S27): L7 -1.105%, L10 -2.483%, L13 -0.688% per 64 ranks. That is a .26 / .58 / .16 split, not the torch
model's .23 / .34 / .43. The singles sum to -4.28% against -4.41% for all-320, so the joins are nearly additive in bill as well as raw.
Rank 352 is bill-inefficient natively (352 = 11 x 32 limits the tree kernel's depth). 384/352/384 saves -0.620% (half its rank share),
384/384/352 -0.190% and 352/384/384 -0.183%. At L13 the reconstruct even costs more at 352 than at 384 on net 35. Even at zero raw
penalty, no 352 half-cut can beat 384/320/384's -0.69%.

Schedule table (`r320_schedules.py` v2 = torch raw per net x NATIVE bill; v1 with the unit-cost bill kept as r320_schedules_v1.py):

| schedule (L7 / L10 / L13) | VALID40 raw | native bill (source) | VALID40 adjusted | better | held-out raw | held-out adjusted |
|---|---|---|---|---|---|---|
| 384 / 320 / 384 (L10 only) | +1.82% (se .30) | -2.483% (bill map) | -0.689% [-1.23, -0.13] | 25/40 | +1.15% (se .23) | -1.368% [-1.81, -0.91], 32/40 |
| 384 / 320 / 320 | +2.88% (se .35) | -3.167% (S27r, VALID40) | -0.370% | 23/40 | +2.12% (se .31) | -1.155% |
| 320 / 320 / 320 | +4.38% (se .48) | -4.411% (S25r, VALID40) | -0.183% | 18/40 | +3.16% (se .47) | -1.469% |
| 320 / 384 / 384 (L7 only) | +1.46% (se .24) | -1.105% (bill map) | +0.370% | 19/40 | +0.98% (se .27) | -0.160% |
| 384 / 384 / 320 (L13 only) | +1.07% (se .17) | -0.688% (S27s, VALID40) | +0.366% | 15/40 | not run | |
| 384 / 352 / 384 | +0.55% (se .14; the difference estimate was +0.56%) | -0.620% (bill map) | -0.059% [-0.33, +0.19] | 19/40 | | |
| 384 / 384 / 352 | +0.44% (se .08) | -0.190% (bill map) | +0.249% | 13/40 | | |
| 352 / 384 / 384 | +0.59% (se .16; the difference estimate was +0.59%) | -0.183% (bill map) | +0.432% [+0.14, +0.72] | 13/40 | | |
| 384 / 352 / 320 | +1.63% (se .22) | -1.337% (bill map) | +0.276% | 18/40 | | |

All rows are 40 VALID40 nets (grid finished 11:36). Torch raw at 352 is 0.30-0.41 of the raw at 320 for every join. The raw is convex in the cut, but the native bill at 352 is also far
below linear, so the half-cuts lose. The held-out raw penalties run about 0.7 of VALID40's. That puts 384/320/384 near -1.2% held out
(estimate) and about -0.95% on the 80-net pool. All-320 is -0.84% on the pool and 384/320/320 about -0.76%.
Receipts: r320_lazy_model.py, r320_schedules.py (v2), r320_features_v40_sched3.jsonl, fable_porter_v1/stack/billmap/census_*.json,
fable_engineer2_v1/bill27/ (census).
Addendum (11:39): covhunter's timing variant loses natively. It keeps the L10 join layer at 384 and cuts from L11, with torch raw +1.096% against
+1.837% for L10 only (covhunter RESULT section 35 addendum). The porter's per-layer map puts 31% (net 35) and 33% (net 100) of the L10-only
saving in the L10 layer itself, so keeping that layer at 384 leaves a bill of about -1.69%.
- With a free truncation, adjusted is -0.61% on VALID40 against -0.69%, and about -0.93% held out against -1.23% (held-out raw about 0.7x).
- A real truncation at L11, rotating the 384-frame factors of about 11.7k columns to 320, costs about +0.38% of the bill by the fitted
  model. That gives about -0.23%.
- A lazy map instead keeps L11-12's prefix work at 384.
No build is needed.
Held-out check (11:42, run once, on the winner only; r320_chain7.sh; `r320_schedules.py pool80` v3 = both pools, bootstrap 95% CI over
nets stratified by pool; v1 and v2 kept): 384/320/384 passes the lead's bar, held-out adjusted at most -0.30% with a CI excluding 0.

| schedule | VALID40 adjusted | held-out 40 adjusted | pooled 80 adjusted |
|---|---|---|---|
| 384/320/384 | -0.689% [-1.23, -0.13], 25/40 | -1.368% [-1.81, -0.91], 32/40 | -1.034% [-1.39, -0.67], 57/80 |
| 320/320/320 | -0.183% [-1.03, +0.65], 18/40 | -1.469% [-2.33, -0.61], 25/40 | -0.837% [-1.47, -0.21], 43/80 |
| 384/320/320 | -0.370% [-1.02, +0.25], 23/40 | -1.155% [-1.73, -0.56], 30/40 | -0.769% [-1.21, -0.33], 53/80 |
| 320/384/384 | +0.370% [-0.10, +0.82], 19/40 | -0.160% [-0.66, +0.34], 22/40 | +0.101% [-0.24, +0.45], 41/80 |

Held-out raw is 0.63x VALID40's for 384/320/384, in line with the other schedules (0.67-0.74x). All-320 is 0.10 pt better on held-out
alone but 0.51 pt worse on VALID40, and 0.20 pt worse on the pool, so the pick stands. The porter's native S27t rows are the verdict for S28.

## 45. The schedule at the S28 params: control for co-adaptation, and layer-13 raise arms (lead's tasks, 2026-09-25 11:53-12:28)
Setup: the S28 params are loopR7 epoch 1, the start point of the physicist's loopR7s. Torch float64 CPU, VALID40, card as in section 43
(`r320_features.py` v4, env R320_PARAMS / R320_EPOCH). A one-net smoke of loopR7 epoch 2 matched the physicist's MPS VALID16 value to 0.001%.
v4 lets a join rank above 384 take that many Nystrom rows; before, the card clipped it at 384. A smoke confirmed 384/320/384 reproduces exactly.

Control (`r320_chain8b.sh`): 384/320/384 costs +1.934% raw (se .306) at loopR7 epoch 1, against +1.825% at loopR3. The shift is +0.109 pt
(se .060) with per-net correlation .98, so the refit at all-384 leaned slightly harder on the full rank. loopR7 epoch 1 is -1.31% vs loopR3
at all-384.

Raise arms, raw at the S28 params x the porter's native bills (fable_porter_v1/stack/billmap/REPORT_v3.txt):

| schedule | raw vs all-384 | native bill | VALID40 adjusted | vs 384/320/384 |
|---|---|---|---|---|
| 384/320/384 | +1.934% | -2.483% | -0.570% [-1.12, +0.01], 24/40 | |
| 384/320/448 | +1.766% | -1.764% | -0.004% [-0.56, +0.59], 23/40 | +0.566 pt [+0.45, +0.68] |
| 384/320/512 | +1.716% | -1.058% | +0.667% [+0.11, +1.27], 20/40 | +1.237 pt [+1.13, +1.34] |

Each 64 ranks at the lazy L13 join costs about +0.72% of the bill for about -0.16% of raw (se .06), so 384/320/384 stands.
Co-adaptation (loopR7s, trained AT 384/320/384 from loopR7 epoch 1): recovery = (C - B) / (C - A), with A and C the control above and B
loopR7s at 384/320/384. Its epochs are evaluated on VALID40 as they land (`r320_chain9.sh`); results are appended below.
- loopR7s epoch 0 (12:46): B = +1.858% raw vs A, against C = +1.934%. B vs C = -0.072% (se .070), 27/40 better, recovery 4%.
  The physicist's MPS VALID16 shows 1%.
- loopR7s epoch 1 (12:56): B = +1.748% raw vs A. B vs C = -0.179% (se .072), 25/40 better, recovery 9% [2, 16%].
  The physicist's MPS VALID16 shows -0.206% (15%). With the same bill, S28b vs S28 is about -0.18% adjusted on VALID40. Part of this may be
  the fresh cosine cycle, not co-adaptation to the cut, but the fire question only needs B vs C.
- Separation (13:03, `r320_chain10.sh`): loopR7s epoch 1 on the ALL-384 card beats loopR7 epoch 1 by -0.143% (se .047), against
  -0.179% (se .072) on the cut card. The cut-specific part is -0.036 pt (se .051), so training at the cut does not co-adapt to it.
  loopR7s's own cut penalty is +1.894%, against +1.934%. The gain is the extra fresh-cosine block, general to both cards. This is covhunter's
  law again: the penalty is information the cut discards, and no learned part can rebuild it from what remains.
- loopR7s epoch 2 (13:12): B vs C = -0.154% (se .059), 26/40 better. Epoch 1 is the best checkpoint on VALID40 (-0.179%) and on
  VALID16. For S28b, use loopR7s epoch 1: about -0.18% adjusted vs S28 at an equal bill.

Native confirmation (13:13, the porter's held-out v3, fable_porter_v1/heldout40_v3.json): S27t (384/320/384) vs S27.

| pool | native adjusted | native raw | torch prediction (section 44) |
|---|---|---|---|
| VALID40 | -0.612% [-1.20, -0.00], 25/40 | +1.861% | -0.689%, raw +1.825% |
| held-out 40 | -1.318% [-1.76, -0.88], 31/40 | +1.150% | -1.368%, raw +1.15% |
| pooled 80 | -0.973% [-1.35, -0.59], 56/80 | +1.499% | -1.034% [-1.39, -0.67] |

Torch raw x native bill predicted the native result to within 0.08 pt in every pool. S27t beats S27r (384/320/320) by -0.190% pooled
[-0.41, +0.03], so the L13 cut is harmful, as priced.

Raise task, completed (13:18; the lead's arms (a)-(c); `r320_chain11.sh` adds (c)). VALID40, torch raw at the S28 params x the
porter's native bills (REPORT_v3; util change vs S27t):

| arm | raw vs its base | native bill | adjusted vs base |
|---|---|---|---|
| 384/320/448 vs 384/320/384 | -0.164% (se .058), 25/40 | +0.737% | +0.57% [+0.45, +0.68] |
| 384/320/512 vs 384/320/384 | -0.213% (se .056), 25/40 | +1.461% | +1.24% [+1.13, +1.34] |
| 384/384/448 vs all-384 (reference) | -0.165% (se .060), 24/40 | about +0.74% (assumed equal) | about +0.57% |

Why a raise cannot pay. At the L13 join, the VALID40 mean missed Nystrom energy is 9.1e-5 at 320 and 1.65e-5 at 384. Everything beyond 384
holds only 22% of the 320-384 slice. The 384 to 320 cut cost +1.07% raw, so if raw tracks missed energy (0.14% per 1e-5), the whole tail
beyond 384 is worth at most about 0.24% raw. The raises capture 0.165% (448) and 0.213% (512) of it, while each 64 ranks costs about +0.74%
of the bill. The raw-vs-rank curve is steep below 384 and nearly flat above it. The raise gain is the same with the L10 rank at 384 or 320,
so the joins stay additive.
Cleanliness checks:
- Torch card: all 40 nets ran at the intended ranks, with 768-864 live rows at L13 and finite MSEs. The card takes max(384, r) Nystrom rows,
  and 384/320/384 reproduces bit-for-bit.
- Tree kernel: the native bills are linear in the raise (+0.737% per 64, +1.461% per 128), so 448 and 512 carry no padding penalty.
- Rotation overflow envelope: native float32 only, widened by the porter's builder. The torch card is float64 and has no such bound.
Pick unchanged: 384/320/384. No raise beats it, so no held-out check.
