# Complete tangent injections with true supply reached raw 7.09e-9 and 9.89e-9 on two networks

2026-08-30 · experiment · informational · measured

The measurement (rank 35, fixed universal betas k4 .16/k3 .33/v .82, Monte Carlo (MC)-noise-corrected): the additive-delta consumption harness converts ORACLE state deltas (true dv,dk3,dk4 per layer) to 7.09e-9 (-73.5%; rank 100: 9.89e-9, -64.0%) - the harness itself carries a sub-1e-8 result, so INGREDIENT ACCURACY IS THE WHOLE GAP. Ladder at rank 35: deployed 2.676e-8; v4-tensor supply through the OLD tangent (dK21-only injections) 2.53e-8 (-5.3%); TRUE third-cumulant repeated-index plane (K21) planes, old injections 1.72e-8 (-35.8%); TRUE planes + FULL injections 1.21e-8 (-54.9%); oracle deltas 7.09e-9. TWO NEW TANGENT CHANNELS did the -35.8 -> -54.9 jump: (1) univariate variance-map injection dVar/dk3, dVar/dk4 (closed forms verified vs quadrature to 6 decimals), (2) dK22/dK31 pair-plane channels (Gker(2,2), Gker(3,1)+(1,3)) - true fourth-cumulant two-pair plane (K22)/fourth-cumulant triple-single plane (K31) planes exist in mps_planes npz layers 2-15 both nets.

Recorded measurement:

    THE NIGHT'S CENTRAL MEASUREMENT (rank 35, fixed universal betas k4 .16/k3 .33/v .82, MC-noise-corrected): the additive-delta consumption harness converts ORACLE state deltas (true dv,dk3,dk4 per layer) to 7.09e-9 (-73.5%; rank 100: 9.89e-9, -64.0%) - the harness itself carries a sub-1e-8 result, so INGREDIENT ACCURACY IS THE WHOLE GAP. Ladder at rank 35: deployed 2.676e-8; v4-tensor supply through the OLD tangent (dK21-only injections) 2.53e-8 (-5.3%); TRUE K21 planes, old injections 1.72e-8 (-35.8%); TRUE planes + FULL injections 1.21e-8 (-54.9%); oracle deltas 7.09e-9. TWO NEW TANGENT CHANNELS did the -35.8 -> -54.9 jump: (1) univariate variance-map injection dVar/dk3, dVar/dk4 (closed forms verified vs quadrature to 6 decimals), (2) dK22/dK31 pair-plane channels (Gker(2,2), Gker(3,1)+(1,3)) - true K22/K31 planes exist in mps_planes npz layers 2-15 both nets. With all four injection channels + true supply the tangent chain is near-EXACT: dv_corr 0.98, dv_beta 0.996, eps_offdiag 0.986 (was capped 0.83 = supply-limited, NOT transport-limited). Remaining gaps: v4 plane supply 8% rel collapses capture to -5.3% (supply-sensitivity extreme); fixed beta over-applies estimated-quality vectors (v-channel beta_opt 0.50 vs 0.82; retune recovers 12.7->21.6% solo); residual MC-noise fraction of ||resid||^2 = 0.239 (quote with all end-frame corrs).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g19_attribution_v1/run_tangent_v2.py](code/public_g19_attribution_v1/run_tangent_v2.py)
- [code/public_g19_attribution_v1/run_estimated_consumption.py](code/public_g19_attribution_v1/run_estimated_consumption.py)
- [code/public_g19_attribution_v1/build_gate1_supplies.py](code/public_g19_attribution_v1/build_gate1_supplies.py)
- [code/public_g19_attribution_v1/consumption_betas.py](code/public_g19_attribution_v1/consumption_betas.py)
- [code/public_g19_attribution_v1/run_attribution.py](code/public_g19_attribution_v1/run_attribution.py)
- [code/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

## Evidence

- [run_tangent_v2.py#L1](code/public_g19_attribution_v1/run_tangent_v2.py#L1)
- [run_estimated_consumption.py#L1](code/public_g19_attribution_v1/run_estimated_consumption.py#L1)

[Research graph](../../../graph/index.html#node=2026-08-30-complete-tangent-injections-true-supply-reached-raw-7-09e) · [Copy provenance](../../../PROVENANCE.json)

