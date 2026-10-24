# Where the card discards third-order information, and what making it exact buys (fable_physicist_v1, 2026-09-22 night)

Torch card copy `../model_bank_audit.py` (switches for the select thresholds, the birth-pair clamps and the star-as-columns oracle;
defaults bit-exact: r35 1.6484657558908164e-08). Learned parts ON, no refit. Final MSE change vs the shipped card on
35 / 100 / 37 / 39. Scripts `../audit_run.py`, `../star_cols_oracle.py`; rows `../audit_sel*.jsonl`.

## Result

Only one bookkeeping discard pays back: the old-tier frame truncation, which the join schedule already harvests. Every other place
the card drops third-order information is lossless, sits at its co-adapted optimum, or is worth under 1% at a large cost.

| discard | exact / relaxed version | closed loop (35 / 100 / 37 / 39) | cost of exactness | verdict |
|---|---|---|---|---|
| old-tier frame, re-truncated to rank 384 at every join | no frame (every cohort pushed exactly) | +1.7 / -4.0 / -2.7 / -1.5%, mean -1.6% | old tier becomes young: far above the bill | the join schedule (fewer truncations, about bill-neutral) is the cheap way in |
| birth units (min(g, 1-g) >= .01) | threshold .001 (+12-13% born columns) | -0.69 / -3.77 / -0.24 / +0.48%, mean -1.06% | +12% columns, about +3-4% bill | inconsistent; no net gain |
| response rows (plane read only where min(g, 1-g) >= .01; same set as births) | threshold .001 (+8% read rows) | -0.25 / +5.99 / +4.98 / -2.48%, mean +2.06% | +8% read | worse |
| births + response | both at .001 | mean +0.73% | - | worse |
| alive rows (gate >= .01; others zeroed) | threshold .001 (+4-5% alive rows) | +8.6 / +19.0 / +15.5 / +9.8%, mean +13.2% | - | much worse |
| all three thresholds | .001 / 1e-5 | mean +16.9% / +90% | - | much worse (co-adapted to .01) |
| thresholds raised (discard more) | births .02 / .03; births+response .02 | mean +3.5% / +15.1% / +10.5% | -5 to -10% born columns | worse: .01 is the optimum both ways |
| birth-pair clamps (u within 8v, gap floor 0.1v, gap cap 16v) | stats | u clamp never binds; gap floor binds for 10-19% of births at L4-14; gap cap never | - | lossless: the unmatched born k3 goes into the star's diagonal, whose transport is exact |
| star: the residual plane is pushed exactly, but only the (2,1) slice and diagonal of the pushed tensor are kept, so its distinct-index part is lost every layer | residual carried as exact CP columns (3 Sym(e_a, e_a, r_a) = [(eps r+e)^3 + (eps r-e)^3 - 2(eps r)^3]/(2 eps), plus the diagonal; 4 columns per alive row), no frame in either arm | -2.3 / +0.2 / 0.0 / -0.6% vs no-frame, mean -0.7% | about 3.6k extra columns per layer (2.7x the births): about +60% of the bill | dead |
| newborn / shared push | exact algebra (W^T(b + t e) = W^T b + t W_i) | - | - | no loss |
| pair-map Hermite series | converged (lead: <= 0.08%); exact on true inputs at the MC noise floor | - | - | no loss |

## The next bookkeeping lever

The frame is the only discard with payback. Two frame-side refinements are worth pricing with the join harness (lead's lane):

- **Better truncation at each join.** The range finder is one pass with the fixed test matrix Omega = W[previous, :384]. The lead's
  frame_test_eig showed exact eigen-frames worth -2 to -3% (r100). A warm start (Omega built from the previous frame Qp plus a few
  W columns), or a single power step, costs about 1 unit per join.
- **Joins versus truncation quality.** The two stack, because fewer joins means fewer truncations.

## Partial-strength grading (after covhunter's overshoot finding; `../audit_blend.py`, `../gain_train_sweep.py`)

Each "make it exact" change is blended at eps 0.1 / 0.25 / 0.5 / 1. The eps = 0.1 run is the first-order probe. The eps = 1 blends
reproduce the threshold runs above exactly. Closed loop on 35 / 100 / 37 / 39, learned ON, no refit:

| change | eps 0.1 | 0.25 | 0.5 | 1 | cost |
|---|---|---|---|---|---|
| star residual carried as exact columns (vs frame-free card) | -0.54% (4/4) | -1.07% (4/4) | **-1.42% (4/4)** | -0.67% (2/4) | about 3.6k columns per layer at any eps |
| extra births (units .001..0.01) at weight eps | -0.27% (4/4) | -0.63% (4/4) | -1.04% (4/4) | -1.06% (3/4) | +12% born columns at any eps |
| extra response rows | -0.19% (2/4) | +0.01% (2/4) | +1.98% (2/4) | +2.06% | - |
| extra alive rows | +13.3% | +13.3% | +13.3% | +13.2% | structural: the rows become full-weight response candidates |

| b-partition gain fix, b_post += gamma (radial_new - radial)/6 m (zero cost) | gamma 0.05 | 0.1 | 0.2 | 0.3 |
|---|---|---|---|---|
| 35 / 100 / 37 / 39 | -0.64% (4/4) | **-1.10% (4/4)** | -0.76% (3/4) | +0.78% (2/4) |

For reference, the lead's earlier full-strength runs were +8% (gamma 0.5), +34% (0.85) and +52% (1.0).

TRAIN-only choice (ranks 40/41/43/44/45/46/47/49; VALID untouched):

| variant | mean | wins | worst net |
|---|---|---|---|
| gamma 0.05 | -0.67% | 7/8 | +0.05% |
| gamma 0.075 | -0.90% | 7/8 | +0.14% |
| gamma 0.1 | -1.03% | 7/8 | +0.30% |
| gamma 0.125 | -1.14% | 7/8 | +0.63% |
| gamma 0.15 | -1.17% | 7/8 | +0.92% |
| K31 row factor (0.8, 0.5) alone | -1.54% | 5/8 | +2.36% |
| K31 row factor + gamma 0.075 | -2.00% | 6/8 | +1.49% |
| K31 row factor + gamma 0.1 | -2.01% | 5/8 | +1.41% |

Reading: the star's lossy handling of the gain-type residual is a real discard. Half-strength exactness pays (-1.4%) but costs a
fortune as columns. The gain-type part can be carried free by b: gamma about 0.1, about -1% alone and about -0.5% on top of the K31
row factor. Pick gamma 0.1 (flat optimum 0.1-0.15, smallest worst case).
