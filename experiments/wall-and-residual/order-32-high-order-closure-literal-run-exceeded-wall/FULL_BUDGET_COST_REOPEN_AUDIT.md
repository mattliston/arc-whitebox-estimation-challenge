# Full-budget closure cost-reopen audit

Status: **target-free architecture audit in progress**.  This document separates
branches that were closed only by the former low-cost operating point from
branches that remain structurally or arithmetically closed at the official
Phase-2 limit

```text
B = 2^41 = 2,199,023,255,552 metered FLOPs per MLP.
```

Nothing in this note is an accuracy result, execution release, target-open
authority, or submission decision.

## Reopened and selected

| mechanism | current Phase-2 cost | decision | reason |
|---|---:|---|---|
| full-birth CP3, cap 16,384 | 620,090,560,514 measured | selected | No CP birth is truncated through the 16-layer network.  The measured recursive-mean form is the current pure-closure leader at `9.340196551564291e-7` final raw MSE. |
| fused full-rank CP K21 plane | 450,956,886,016 projected | selected | Reads the complete `K21[i,j]=K3[i,i,j]` plane and its transpose from the carried CP state through z15; W16 needs only the marginal K3 read because the requested output is a mean. |
| dense cubic residual star | 122,346,799,104 projected | selected | Resets the K21 plane that the CP state does not match, then transports one source-only all-distinct completion through the next known weight matrix.  W16 specializes to its marginal action. |
| coherent quartic `Star4(H)+alpha P_C` | 182,446,981,120 projected | selected | Carries K31 and K22 in one same-moment gauge.  It replaces both the incomplete diagonal-only sidecar and the empirically damaging independently scheduled synthetic K31 plane; W16 computes only its marginal action. |
| diagonal-preserving spectral PSD repair | 330,683,000,000 conservative | selected | The old nearest-PSD work was scientifically positive but cost-screened.  Phase 2 can afford fourteen float64 eigenspectrum repairs plus reconstruction and exact diagonal renormalization. |
| full pair Edgeworth consumer through `L4 + 1/2 L3^2` | 196,478,000,000 conservative | selected | The oracle pair consumer requires `k3`, both K21 orientations, `k4`, K31, and K22.  The incumbent omitted both the complete fourth-order action and the `K3^2` term. |

The consolidated target-free design projection is about `1.9092T` FLOPs,
`86.82%` of `B`, with about `289.82B` unassigned headroom.  The estimate
includes a `6B` reserve for weights, reset assembly, and elementwise work.  Literal
FlopScope metering, peak memory, residual time, and output finiteness remain
mandatory gates.

## Reopened but not selected in the leading composition

| mechanism | disposition under Phase 2 |
|---|---|
| diagonal CP4 rolling action | Reopened on cost and used as evidence, but superseded as a state: it recovered 77.89% of the oracle K4 mean gain while failing mixed-plane accuracy and reaching covariance minimum eigenvalue `-0.008273`.  The coherent quartic action retains its useful diagonal channel and repairs the omitted planes. |
| standalone nearest-PSD / eigenvalue-floor family | Reopened and absorbed into the selected diagonal-preserving spectral repair.  Global off-diagonal shrink remains rejected. |
| TT3/TT4 rank-24-style transport | Cost is no longer an automatic rejection: the preserved width-1024 projection is about `1.463T` for fifteen pushes.  It is not stacked on the selected CP/K21 host because it would exceed the budget and the standard source-blind TT3 action gate already missed its `0.15` threshold (`0.1944` at rank 23).  It may reopen only as a replacement with a stronger action certificate. |
| fifth-order consumer/state | Not selected.  Preserved evidence gives only modest nonlinear capacity and no closed K41/K32 supplier; the fourth-order joint-state defect is earlier and better evidenced. |

## Still closed at `2^41`

Dense exact third-cumulant transport remains arithmetically closed.  At width
1024, the naive dense affine contraction `6 n^4` costs
`6,597,069,766,656` FLOPs for **one** transition, already `3.0 B`.  Even the
best preserved pairwise-symmetric spelling scales to more than one complete
Phase-2 budget per transition and would have to run fifteen times.  The full
tensor would also collide with the 8-GB solution-memory limit once live
temporaries and the remaining state are included.

This does not close structured third-order transport.  CP factors, dense
action planes, tensor trains that replace an existing carrier, and
action-specific source-only completions remain valid candidates when their
complete meter fits.

## Evidence boundary

- Phase-2 CP ladder and meter:
  `cp3_full_birth_cap_ladder/SYNTHETIC_RESOURCE_LADDER.json`.
- Phase-2 raw score:
  `fullbudget_joint_score_v2/INDEPENDENT_POST_OPEN_AUDIT_V2.json`.
- Diagonal CP4 evidence:
  `../x3_r0c1_kappa3_ceiling/diagonal_cp4_rolling_action/RESULTS.json`.
- Standard TT3 action gate:
  `../x3_r0c1_kappa3_ceiling/carried_tt3_standard_rank23/RESULTS.json`.
- Pair-consumer and PSD evidence are graph nodes `Q2`, `P4`, `P6`, `P6C2`,
  and `P6D` in `problem_graph/graph.json`.
- Dense exact-transport economics are graph node `E1`.

The next binding artifact must freeze the exact update order, tensor
conventions, Hermite derivative table, connected-moment reset formulas, dtype
policy, and literal resource ledger before a synthetic meter or accuracy read.
