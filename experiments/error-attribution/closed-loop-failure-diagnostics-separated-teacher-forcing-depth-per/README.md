# Closed-loop failure diagnostics separated teacher-forcing depth and per-layer correction effects

2026-09-04 · experiment · rejected · measured

PER-LAYER REMOVAL IN THE CLOSED LOOP (closedloop_layers.py, r35, operator_1): one-step error -> after its correction: fourth-cumulant Edgeworth operator (L4) 5.8e-9 -> 4.1e-9 (29%), L8 2.07e-8 -> 1.47e-8 (29%), L12 4.56e-8 -> 3.27e-8 (28%), L15 4.51e-8 -> 3.16e-8 (30%) - the operator removes ~30% at EVERY layer in its own loop vs 60% teacher-forced: off-distribution inputs (drifted tape + recurrent state) halve its effectiveness uniformly, and the uncorrected 70% drives the one-step errors to 2x the oracle's. Iteration 2 trains on this distribution (the cure for the input mismatch); the residual question is the in-distribution removal fraction. VIRGIN REFERENCES (deployed card, score_candidate.py, 4M truths): 3016500 1.89e-8, ..501 2.37e-8, ..502 4.10e-8, ..503 2.40e-8, ..504 2.45e-8, ..505 3.07e-8; mean raw 2.712e-8, util 0.608, adjusted 1.645e-8.

Recorded measurement:

    - T-108 FAILURE-ANALYSIS PROTOCOL FOR THE CLOSED LOOP (Matt 00:30: if it fails, find WHY and attack the cause; decompose into sub-problems; never discard). Pre-registered diagnostics, all with existing instruments: (1) DRIFT PROFILE: per-layer one-step error under operator_k's own feedback (rows_closedloop host.npy) vs the oracle's (rows_meanleak) -> early- or late-born mismatch. (2) PER-LAYER REMOVAL in closed loop: one-step error before vs after the correction at each layer (rows = pre vs the carried means) -> which layers/units the operator fails on (split units by firing rate, sigma, |m|). (3) TEACHER-FORCING LADDER: oracle feedback at layers <= L0, operator feedback after (RETRO_MEANLEAK_LAYERS=1-L0 + RETRO_CLOSED_LOOP_LAYERS=(L0+1)-15) -> where compounding starts and how much of the final error is accumulated vs freshly born. (4) MEAN vs COVARIANCE drift: oracle covariance feedback with operator means (and the converse) -> which state carries the exposure gap. (5) ITERATION DYNAMICS: rows_closedloop one-step errors per iteration -> contraction or oscillation. FIXES mapped to causes: early drift -> exposure-mixed rows (blend oracle/own feedback); late removal cap -> add the operator's own cumulative correction (the closed-loop state) and late-layer mechanistic features (G28 dk3 R^2 0.6) as inputs; oscillation -> payload averaging / damped iteration; covariance-carried drift -> a learned covariance write (Keenan's rank-16); all else -> the differentiable card.
    
    - T-108a TEACHER-FORCING LADDER, operator_1 (r35, F2J-off row): oracle means fed back through L4 then own feedback -> 3.41e-8; through L8 -> 3.15e-8; through L12 -> 2.31e-8; through L14 (teacher-forced) -> 6.6e-9; none (pure closed loop) -> 3.28e-8. The exposure gap is BORN IN THE LAST THREE LAYERS: with exact upstream through L12, operator_1's own corrections at L13-14 leave residuals that derail L15 (2.31e-8 vs 6.6e-9). Sub-problem = the 40% one-step residual at L13-14 propagating into L15; iteration 2 trains exactly on that distribution; if it does not close, the cause is the per-layer removal cap at L13-15 -> specialize/enrich the operator there (mechanistic late-layer inputs, a second late-layer pass, its own cumulative correction as input).
    
    - T-108b PER-LAYER REMOVAL IN THE CLOSED LOOP (closedloop_layers.py, r35, operator_1): one-step error -> after its correction: L4 5.8e-9 -> 4.1e-9 (29%), L8 2.07e-8 -> 1.47e-8 (29%), L12 4.56e-8 -> 3.27e-8 (28%), L15 4.51e-8 -> 3.16e-8 (30%) - the operator removes ~30% at EVERY layer in its own loop vs 60% teacher-forced: off-distribution inputs (drifted tape + recurrent state) halve its effectiveness uniformly, and the uncorrected 70% drives the one-step errors to 2x the oracle's. Iteration 2 trains on this distribution (the cure for the input mismatch); the residual question is the in-distribution removal fraction. VIRGIN REFERENCES (deployed card, score_candidate.py, 4M truths): 3016500 1.89e-8, ..501 2.37e-8, ..502 4.10e-8, ..503 2.40e-8, ..504 2.45e-8, ..505 3.07e-8; mean raw 2.712e-8, util 0.608, adjusted 1.645e-8.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g22_operator_features_v1/closedloop_layers.py](code/public_g22_operator_features_v1/closedloop_layers.py)
- [code/public_g22_operator_features_v1/score_candidate.py](code/public_g22_operator_features_v1/score_candidate.py)
- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/closedloop_layers_gnn.py](code/public_g22_operator_features_v1/closedloop_layers_gnn.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L793](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L793)
- [LANE_g3_tracker.md#L794](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L794)
- [LANE_g3_tracker.md#L795](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L795)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-closed-loop-failure-diagnostics-separated-teacher-forcing-depth-per) · [Copy provenance](../../../PROVENANCE.json)

