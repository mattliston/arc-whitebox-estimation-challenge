# Exact third-order pair deficit through its kernel does not predict next-layer variance error

2026-10-20T10:02:04Z · experiment · rejected · measured

The exact third-order pair-plane deficit had correlation -0.06..+0.07 with the required next-layer pre-variance correction. An 18-function pair-geometry model explained 0.0-0.8% of the other network variance error.

Recorded measurement:

    [fable-lead] 2026-10-20T10:02:04Z K21 DEFICIT IS NOT THE VARIANCE ERROR (lead/runs/fable_oct19_kc_v1/k21_geom_fit_v1.py; nets 35/100, L9-13): the EXACT deficit D = K21_true - K21_card pushed through the pair map's K21 kernel ((d_i g_j D_ij + g_i d_j D_ji)/2) to the next layer's per-unit variance has corr -0.06..+0.07 with the needed correction (share 0.000-0.005; applied at size 1 it makes dv 20-30% WORSE), and an 18-function pair-geometry model of D (s_i^2 s_j x polynomials in rho x phi/Phi/a of the pair, K x geometry) fitted on one net explains 0.0-0.8% of dv^2 on the other. LAW: the late per-unit variance error (the -18/-26% oracle) is not produced by any single object at the previous layer through its kernel; single-layer exact-Cpre oracles (-7..-16% each, -26/-34% for L10-14 jointly, non-additive) say the covariance error is a SELF-CONSISTENT CHAIN: the card's means, variances, cumulants and covariance are mutually consistent at ~1e-4 relative accuracy and jointly off the truth; fixing any one quantity moves the chain partway. Only a uniformly better closure (the leaders') helps; no patch exists. This ends the structural hunt for this deadline.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/k21_geom_fit_v1.py](code/k21_geom_fit_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6171](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6171)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-exact-third-order-pair-deficit-through-kernel-does-not) · [Copy provenance](../../../PROVENANCE.json)

