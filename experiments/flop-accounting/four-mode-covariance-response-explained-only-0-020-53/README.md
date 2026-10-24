# Four-mode covariance response explained only 0.02–0.53 of sampled fourth-cumulant variance

2026-09-04 · experiment · informational · measured

SSC REPRESENTATION TEST (ssc_k4_test.py, r35, 131k Monte Carlo (MC) samples, true post states): the 4-mode covariance-response form (9 features incl. the six Q terms, 45 products, ridge, held-out units) explains the next layer's per-unit k4 with R^2 only 0.02 / 0.26 / 0.43 / 0.53 at fourth-cumulant Edgeworth operator (L4)/8/12/14 (3 canonical modes 0.06-0.44; diagonal transport negative). At width 1024 the fourth cumulant is NOT a low-mode covariance response; Puffi's phase-1 gain (depth 32, width 256) does not transfer as a representation. An SSC-style replacement of our fourth-order machinery (worth 4x at the host) would recover at most half.

Recorded measurement:

    - T-134 DESIGN FROM THE PUBLIC RECORD (Team Puffi, rank 4 in phase 2; phase-1 algorithmic PDF "Separate-Universe Response Closure", sub 327725): keep a K=3-class carrier (mean, covariance, CP third-cumulant columns) but REPLACE the fourth-order machinery by a covariance-RESPONSE closure: k4_ijkl = sum_ab Lambda_ab Sym(B_a (x) B_b) from p<=4 live modes B_1 = C, B_2 = Q/vbar (Q = P_off[c0 vbar C + c1 C∘C + c2/2 (D_m K^T + K D_m) + c3 (t m^T + m t^T) + c4 v v^T + c5 vbar^2 11^T], K = K21 plane, t = k3 diag), B_3 = vbar I, B_4 = 1/2 (g_i+g_j) C_ij; per-unit next pre-k4 = 3 q_i^T Lambda q_i with q_ai = w_i^T B_a w_i; Lambda (4x4, indefinite allowed) and the six c's fitted offline per layer by ridge to the measured next-layer marginal k4 (they: 30-net panel; we have vlabels mc_pre_k4 for 200 nets), frozen; plus a 31->32->32->1 late-layer drift corrector and a one-step response memory (beta=-0.3). Phase-1 result: layer-32 MSE 77% below factorized K=3 and 48% below FULL factorized K=4 at ~0.5B (K=4 costs 10^2 B). For OUR card: the fourth-order machinery is 21% of util (quartic push 13.7 + CP4 reads 13 + CP4 push 4.7 of 0.6); an SSC supply of pre_k4 / K31 / K22 costs ~6 GEMMs/layer (~10% util) or less. If raw holds: util 0.6 -> ~0.45; if Puffi's -48% vs K=4 transfers: raw 2.9 -> ~1.5e-8 => adjusted ~7e-9 = COMPETITIVE. First read (zero cost): can the 4-mode form represent the true next-layer k4(z_i) at width 1024 (R^2 per layer, true states, torch)? Then fit on CARD states, emulate in the instrument (host, F2J-off) at CP4_CAP=1 + NO_QUARTIC vs base 7.85e-8, then co-adapt the closure, package.
    
    - T-134a SSC REPRESENTATION TEST (ssc_k4_test.py, r35, 131k MC samples, true post states): the 4-mode covariance-response form (9 features incl. the six Q terms, 45 products, ridge, held-out units) explains the next layer's per-unit k4 with R^2 only 0.02 / 0.26 / 0.43 / 0.53 at L4/8/12/14 (3 canonical modes 0.06-0.44; diagonal transport negative). At width 1024 the fourth cumulant is NOT a low-mode covariance response; Puffi's phase-1 gain (depth 32, width 256) does not transfer as a representation. An SSC-style replacement of our fourth-order machinery (worth 4x at the host) would recover at most half. Design T-134 downgraded (not dead: fitting to the MEAN effect rather than k4 was their better variant, +13.6%).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/ssc_k4_test.py](code/ssc_k4_test.py)
- [code/diff_card.py](code/diff_card.py)

## Evidence

- [LANE_g3_tracker.md#L943](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L943)
- [LANE_g3_tracker.md#L944](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L944)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-four-mode-covariance-response-explained-only-0-020-53) · [Copy provenance](../../../PROVENANCE.json)

