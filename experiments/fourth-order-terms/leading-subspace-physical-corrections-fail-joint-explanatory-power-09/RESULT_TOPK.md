# A physical third/fourth-order rule for the top-k covariance block: none found (fable_physicist_v1/topk, 2026-09-22 night)

Target (the analyst's Rp_data construction): at every pair-map call L2-14 of the torch card, R = P (cov_pair(8M truth K21 off-diagonal +
card k3, truth K22, truth K31) - cov_pair(card)) P U_k, with U_k = top-k eigenvectors of P cov0 P and k = 30. Noise share of R from the
independent draw (r35): 86% (L2), 40% (L5), 14% (L9), 7% (L12). Only L8-14 carry signal.

## 1. Oracle-grade screen (`topk_screen.py`, r35)

Each candidate's exact pair-map covariance response is projected and contracted like R. Values are rho with R (per layer):

| candidate | L5 | L9 | L12 | L13 | L14 | best scale |
|---|---|---|---|---|---|---|
| P1 = 0.45 * 3 (k3_a/v_a)(K21_ac - C_ac k3_a/v_a), card K21 | .136 | .201 | .233 | .260 | .249 | about 2 (coefficient 0.9) |
| P1 with the TRUTH K21 (oracle) | .081 | .139 | .181 | .202 | .195 | - |
| gain-skew K31 (radial/4)(3 mu_a K21_ac + mu_c k3_a), card K21 | .119 | .153 | .174 | .182 | .178 | about 2 |
| gain-skew K31 with the truth K21 | .057 | .085 | .107 | .113 | .116 | - |
| gain-skew K22 partner | -.048 | -.114 | -.153 | -.147 | -.142 | wrong sign |
| tree-level K31 (star + path, card state at l-1) | .063 | .108 | .119 | .131 | .164 | 0.4-1.0 (physically right) |

Joint fits: P1 + gain-skew explains 2-7% of R's energy (L5-14). Adding the tree term gives 2-9%. The truth-K21 versions give 1-4%.

Building the product terms from the truth K21 makes them WORSE. So their alignment comes from the card's own K21 error structure, not
from missing physics.

## 2. Closed loop in the top-k block (`topk_closedloop.py`; learned ON, no refit, L2-14)

cov <- cov0 + eps X, with X the symmetric one-sided block of the supply's exact covariance response:

| supply | eps | 35 / 100 / 37 / 39 | mean |
|---|---|---|---|
| P1 (0.9), top-k block | 0.25 | -0.46 / -0.48 / -0.21 / +0.49% | -0.16% |
| P1 (0.9), top-k block | 0.5 | -0.81 / -0.94 / -0.22 / +1.26% | -0.18% |
| gain-skew (2x), top-k block | 0.25 | - | +0.46% |
| gain-skew (2x), top-k block | 0.5 | - | +1.33% |
| P1 (0.9), full plane, no block | 0.5 | - | +0.45% |

Bar: at least -3% at eps 0.5 for at most +3% bill. Not met.

## 3. The lead's other two candidates, from measurements already made

- **lam4 per column from each birth's own fourth-order partner.** The birth's quartic partner at tree level is the star with centre
  i and three covariance leaves, (e_i, s_i, s_i, s_i), i.e. the star part of the tree term above. Two columns per birth can hold only 2
  of its 5 binary-quartic coefficients. Its top-k rho is therefore at most the tree term's (at most 0.16). With FREE per-column weights
  the family carried only 51% of the full K31 error and 27-38% of the net covariance change (fourth/RESULT_FOURTH.md).
- **The star's content transported exactly.** The card already reads the pushed star's (2,1) slice exactly at the next layer. Only the
  distinct part matters later, and carrying it exactly (star-as-columns oracle, all rows) is worth at most -1.4% at eps 0.5
  (audit/RESULT_AUDIT.md). A k-direction version cannot do better than the exact carry, and it needs the future layers' directions.

## 4. Why no cheap rule exists within this card

A k-direction read of the carried third-order objects (bank columns, star, b) adds NO information. The card already forms the full K21
plane from exactly those carriers, so contracting them with u_i reproduces a slice of what it already has; only the cost changes. New
information must come from content the card never carries:

- **the non-isotropic fourth cumulant (K31/K22).** It is the whole oracle lever. Tree-level physics, fresh plus transported, explains
  12-13% of it.
- **the non-isotropic fourth cumulant's feed into the next layer's third cumulant** (the (2,1,1) channel).
- **the star's discarded distinct part.** Worth at most 1.4%.

None of these has a derivable, cheap form accurate at the 0.35+ correlation needed. The derived terms are physically right in scale
(tree coefficient about 1) but carry only 1-9% of R.
