# Deep correlation matrices required 6–19 spectral modes for 50–70% Frobenius mass

2026-08-30 · experiment · informational · measured

HUB v2 DESIGN SPEC (final piece, 12:00): deep rho needs only 6-19 spectral modes for 50-70% Frobenius mass (L13: 6/13/37 for 50/70/90%) - the KRHO=4 transport null was ~30% coverage, not a dead mechanism. v2 = KRHO~16 bond splitting (49 children/triad) + retention ~8 layers + PLANE-READ-AWARE compression to ~96 columns (objective: match current + W-pushed future plane reads - the only consumed functionals; Frobenius-ALS refuted), total bill e10-class (well under 0.35B). Target: q 0.62 -> 0.75+ (the measured conversion threshold). This is the complete, numbers-closed build plan for the fireable card's supply.

Recorded measurement:

    HUB v2 DESIGN SPEC (final piece, 12:00): deep rho needs only 6-19 spectral modes for 50-70% Frobenius mass (L13: 6/13/37 for 50/70/90%) - the KRHO=4 transport null was ~30% coverage, not a dead mechanism. v2 = KRHO~16 bond splitting (49 children/triad) + retention ~8 layers + PLANE-READ-AWARE compression to ~96 columns (objective: match current + W-pushed future plane reads - the only consumed functionals; Frobenius-ALS refuted), total bill e10-class (well under 0.35B). Target: q 0.62 -> 0.75+ (the measured conversion threshold). This is the complete, numbers-closed build plan for the fireable card's supply.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v1.py](code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v1.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v0.py](code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v0.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v2b.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v2b.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [hub_carrier_v1.py#L1](code/p2_width1024_depth16/public_g19_attribution_v1/hub_carrier_v1.py#L1)

[Research graph](../../../graph/index.html#node=2026-08-30-deep-correlation-matrices-required-619-spectral-modes-5070-frobenius) · [Copy provenance](../../../PROVENANCE.json)

