# Edgeworth corrections to births worsen error with learned corrections both enabled and disabled

2026-09-24T19:08Z · experiment · rejected · measured

Adding skew and second-order birth corrections failed at full and half strength. The per-unit pre-activations were already nearly Gaussian, indicating that births were not the main plane-error source.

Recorded measurement:

    - 2026-09-24 12:08 [physicist] EDGEWORTH BIRTHS (lead Q). Birth code today (covstep step + zero.py _bd_update): vertex E[f''] = phi/sigma x (1+radial/3) (the elliptical kappa4 cross term, no per-unit skew); legs = Phi(alpha) x Cpre plus 4 LEARNED leg directions, each orthogonalized to the Gaussian leg: bivariate skew d/2 K21 (learned coefficient ~0.8 at every layer), a-side eta K21 (~0.2-0.35), Gaussian C^2 (~0.8), and the marginal gate skew (~0 at L9-14, noisy -0.65..+0.53 early); beta mixes Gaussian E[ff'] with the corrected host mean; the born diagonal is fitted to the pair map's corrected post k3. Measured net 37: pre-activation |t3| median .03-.04 (p90 .07-.09) at L9-14, t4 ~.05, t3^2 ~.001-.002, so the corrections are small (median |dg/g| ~.25%, |dE[f'']/d| ~1%, p90 3-4%). Formulas certified vs exact gamma (fable_physicist_v1/refit6b/edgeworth/cert.py); S=0 control bit-exact vs B2 ep1 on r37. Running VALID16 no-refit: order 1 (skew) and order 2 (t4 + t3^2) at S = 1 and .5 (edgeworth/run_arms.sh).
    - 2026-09-24 13:46 [physicist] EDGEWORTH BIRTHS: NEGATIVE, not a refit candidate. VALID16 vs B2 ep1, no refit: skew full 1.0129 (1/16), half 1.0054 (1/16); order 2 (t4+t3^2) full 1.0102 (2/16), half 1.0039 (4/16). Learned OFF vs the learned-off control: skew full 1.0100 (0/16), order 2 full 1.0080 (0/16), so it is not co-adaptation. Pre-activations are nearly Gaussian per unit (|t3| med .03-.04 at L9-14), and births are not the plane-error source. Truth floors ~2%/1.3% put the card's own plane error at ~7.3%/8.4% (L10/L14): real. Details: fable_physicist_v1/leadergap/RESULT.md section I.
    - 2026-09-24 13:55 lead: EDGEWORTH BIRTHS CLOSED (physicist, no refit VALID16): skew full x1.0129, half x1.0054; second order full x1.0102, half x1.0039; also negative with learned parts OFF (x1.008-1.010, 0/16). Pre-activations are nearly Gaussian (median |skew| .03-.04 at L9-14). The card's own plane error after removing the 8M truth noise is 7.3% (L10) / 8.4% (L14). CANDIDATE SOURCE: the distinct-index part of the second-order carry term d_i d_j C_ij k3(i,j,k) g_k, dropped every layer (the pair map keeps only its on-plane part in the star). Next: size it with the dense full-cum3 instrument on r35 (A = card transport, B = A + the term, vs 8M truth planes). It starts only when free memory is >=20 GB.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_physicist_v1/refit6b/edgeworth`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cert.py](code/cert.py)
- [code/run_arms.sh](code/run_arms.sh)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L376](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L376)
- [TEAM_LOG_2026-09-22_to_09-26.md#L397](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L397)
- [TEAM_LOG_2026-09-22_to_09-26.md#L398](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L398)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT.md](../../late-layer-exactness/layer-15-third-cumulant-response-cut-z-3-5/RESULT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-edgeworth-corrections-births-worsen-error-learned-corrections-both-enabled) · [Copy provenance](../../../PROVENANCE.json)

