# Second-order state audit: sub-percent scatter with no coherent covariance drift

2026-08-26 · experiment · informational · measured

Second-order MARGINALS on the supplied host are exact: pre_mean noise-corrected error <=0.0002 at every layer L2-15 (both ranks 35/201), pre_var grows ~1.3e-4 per layer to 0.0018 at L15 with median ratio 1.000 - unit-level scatter, no coherent drift; at final-output sensitivity ~0.5x that scatter is the same order as the entire remaining residual. Receipt written. COV VERDICT (rank 35, 2000 pairs, layer-1 gate clean): nc 0.002 to 0.0096 L2-L15, median ratio 1.000 at every layer - the covariance channel is exonerated; ALL second-order state is now sub-percent scatter with no coherent drift.

Recorded measurement:

    Second-order MARGINALS on the supplied host are exact: pre_mean noise-corrected error <=0.0002 at every layer L2-15 (both ranks 35/201), pre_var grows ~1.3e-4 per layer to 0.0018 at L15 with median ratio 1.000 - unit-level scatter, no coherent drift; at final-output sensitivity ~0.5x that scatter is the same order as the entire remaining residual. Receipt written. COV VERDICT (rank 35, 2000 pairs, layer-1 gate clean): nc 0.002 to 0.0096 L2-L15, median ratio 1.000 at every layer - the covariance channel is exonerated; ALL second-order state is now sub-percent scatter with no coherent drift. The attribution coheres: the deep k4 deficit (0.82@L15) times its consumption weight (~g4/24 c1[4]) matches the whole remaining residual scale, and the micro-head's l4_term beta is the learned cancellation of exactly that term. Remaining tasked: rank-201 cov confirmation, (2) the L16 final-layer MC stats for the sensitivity decomposition dE = dE/dm dm + dE/ds ds + dE/dk3 dk3 + dE/dk4 dk4 - the attribution that names which order pays at the output. DEEP-DECAY ATTRIBUTION (deriver, corrected window): the exact closure kappa5 = 5 k3 k4/v - (40/9) k3^3/v^2 (verified, residual slope 1.93; kappa6 NOT free) feeds the post-k4 emission at median 0.49 percent/layer vs the needed 1.56 (0.93@L7 to 0.82@L15) - 0.32x median, 0.82x at the g4-rich corner. The closure is a FLOOR on true pre-kappa5 (intrinsic chaos-4 content not expressible in carried k3/k4); the direct discriminator is MEASURING pre-k5_std on the MC state at 2-3 layers and multiplying by the closed-form bracket dk4_post/dk5 - if measured ~ closure, push-inexactness owns ~2/3; if measured >> closure, emission owns the ladder. Sign check: closure kappa5 crosses zero at k4 = (8/9) k3^2. Not wired (retrofit law); banked for oracle-frame accounting and clean-slate. SOLVED - THE ATTRIBUTION IS COMPLETE (FINAL_SENSITIVITY_RESULT.json + K5_MEASURED_RESULT.json, rank 35, 8M): at the L16 consumption the MEAN channel owns the output error (share 1.96 of dE MSE; var 0.21, k3 0.11, k4 only 0.05, remainder -1.33 anti-correlated), the consumption map at true inputs errs only 1.46e-5 RMS (2e-10-class MSE), and measured pre-k5 = 0.72-0.76x the closure - so the deep k4 decay is ~3/4 push-side projection loss but pays only ~5 percent at the output. THE REFRAME: the residual is the COMPOUNDED PER-STEP MEAN EMISSION BIAS (~0.1 percent/step), which retroactively explains all five co-adaptation refutations (they fixed a 5-percent-share channel while perturbing the mean channel). The paths that attack the mean channel directly: the micro-head (learned, running), the adjoint derivation (exact mean functional without forward-state perturbation), the exact ledger, quadrature.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_oracle_kappa_probe_v1/analyze_second_order.py](code/public_oracle_kappa_probe_v1/analyze_second_order.py)
- [code/public_k211_supply_v1/run_state_audit_deep.py](code/public_k211_supply_v1/run_state_audit_deep.py)
- [code/public_oracle_kappa_probe_v1/pairs_spec.py](code/public_oracle_kappa_probe_v1/pairs_spec.py)
- [code/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/public_trivariate_derivation_v1/kappa5_verify.py](code/public_trivariate_derivation_v1/kappa5_verify.py)
- [code/public_trivariate_derivation_v1/l5_verify.py](code/public_trivariate_derivation_v1/l5_verify.py)
- [code/public_trivariate_derivation_v1/verify.py](code/public_trivariate_derivation_v1/verify.py)

## Evidence

- [ORACLE_SECOND_ORDER_RESULT.json#L12](ORACLE_SECOND_ORDER_RESULT.json#L12)
- [KAPPA5_CLOSURE.json#L546](KAPPA5_CLOSURE.json#L546)
- [FINAL_SENSITIVITY_RESULT.json#L56](FINAL_SENSITIVITY_RESULT.json#L56)
- [K5_MEASURED_RESULT.json#L1](K5_MEASURED_RESULT.json#L1)

<details>
<summary>All result files and records (4)</summary>

- [ORACLE_SECOND_ORDER_RESULT.json](ORACLE_SECOND_ORDER_RESULT.json) · result
- [KAPPA5_CLOSURE.json](KAPPA5_CLOSURE.json) · result
- [FINAL_SENSITIVITY_RESULT.json](FINAL_SENSITIVITY_RESULT.json) · result
- [K5_MEASURED_RESULT.json](K5_MEASURED_RESULT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-26-second-order-state-audit-sub-percent-scatter-no-coherent) · [Copy provenance](../../../PROVENANCE.json)

