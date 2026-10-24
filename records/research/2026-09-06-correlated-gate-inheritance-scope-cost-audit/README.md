# Correlated-gate inheritance: scope and cost audit

2026-09-06T22:20Z · finding · informational · measured

The comparisons are against the accumulated old card's scalar k3 residual, using archived 32M moments. This rejects the proposed one-step scalar explanation. After an arbitrary next W, each A needs its own congruence.

Recorded measurement:

    The comparisons are against the accumulated old card's scalar k3 residual, using archived 32M moments. This rejects the proposed one-step scalar explanation. For example rank100 at read15 has D1 RMS3.0948e-5 versus residual RMS9.6061e-4, correlation .057, and natural-strength explained energy .0026. These are not current-bank final raw measurements. The CURRENT public_g19_attribution_v1/run_tensor_recursion_v4.py constructs terms with di,dj,dk all positive, then uses zero-bond plus those terms in F3. Thus it excludes single-bond patterns. The T-68 standalone oracle and certificate enumerate those patterns without this filter. Their statement that they reproduce the engine's full bond0/1/2 loop is not source parity with the current file. - Full-source unweighted plane-error MSE ratio1.00254042. - Leading-consumer-weighted ratio1.01336446, cosine -.04847, source norm 7.69% of the corresponding weighted baseline error. - Fixed top4 covariance-edge approximation: weighted source relative error 84.76%, unweighted83.30%, versus the prescribed15% limits. Other later receipts are adjacent, not substitutes: INHERITED_DISTINCT_GAIN_ V2_CLOSED.md tested a genuinely accumulated per-layer scalar gain on inherited distinct CP content and obtained pooled final raw ratio .99544356, failing .90. It did not add correlation bonds. PROJECTED_PLANE_FEED_ROUND_V1_CLOSED.md tested the different repeated-plane source projected into births; it failed transfer across35/100. Neither should be mislabeled a full bond1 ablation. After an arbitrary next W, each A needs its own congruence. The shared-rho Hadamard form is not preserved. Storing full per-atom matrices at the recorded R=20,352 and n=1024 requires R*n^2 float32 entries:79.5GiB, before anything else. Factoring rho instead expands each atom across its covariance modes; full-rank factors do not avoid that growth after transport. The tested q4 truncation is neither accurate nor free, and was explicitly closed. The existing exact query code uses two dominant n-by-n by n-by-R products per queried output row, about4*n^2*R ordinary FLOPs. Reading all1024 rows at R=20,352 is approximately4*n^3*R, or39.75 budgets of2^41, for ONE source cell, before remaining terms. The 32-row oracle is consequently an offline information diagnostic, not a near-deployable plane generator. A dense correction tensor is a different complete form:4GiB float32 state, with three classical mode pushes costing6*n^4 =3 full budgets per transition, before nonlinear source construction and the original card. These are costs of the specified ordinary complete implementations, not lower bounds against all fast algebra or hypothetical compression. Neither is a viable additive carrier for an adjusted-score target near1e-9.

## Programs

- [../../../experiments/error-attribution/distinct-third-cumulant-times-covariance-edge-fixed-round-closed/code/astra_phase2_reexamination_v1/third_covariance_edge_v1.py](../../../experiments/error-attribution/distinct-third-cumulant-times-covariance-edge-fixed-round-closed/code/astra_phase2_reexamination_v1/third_covariance_edge_v1.py)
- [../../../experiments/error-attribution/distinct-third-cumulant-times-covariance-edge-fixed-round-closed/code/astra_phase2_reexamination_v1/check_third_covariance_edge_wide_v1.py](../../../experiments/error-attribution/distinct-third-cumulant-times-covariance-edge-fixed-round-closed/code/astra_phase2_reexamination_v1/check_third_covariance_edge_wide_v1.py)
- [../../../experiments/third-order-planes/projected-plane-feed-closed-no-transferable-raw-gain/code/astra_phase2_reexamination_v1/check_projected_plane_feed_v1.py](../../../experiments/third-order-planes/projected-plane-feed-closed-no-transferable-raw-gain/code/astra_phase2_reexamination_v1/check_projected_plane_feed_v1.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [../../../experiments/third-order-planes/projected-plane-feed-closed-no-transferable-raw-gain/code/astra_phase2_reexamination_v1/projected_plane_feed_v1.py](../../../experiments/third-order-planes/projected-plane-feed-closed-no-transferable-raw-gain/code/astra_phase2_reexamination_v1/projected_plane_feed_v1.py)
- [../../../experiments/third-order-planes/projected-plane-feed-closed-no-transferable-raw-gain/code/astra_phase2_reexamination_v1/projected_plane_feed_v2.py](../../../experiments/third-order-planes/projected-plane-feed-closed-no-transferable-raw-gain/code/astra_phase2_reexamination_v1/projected_plane_feed_v2.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_binary_compact.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_cubic.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_cubic.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_mehler.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_mehler.py)
- [../../../experiments/third-order-planes/projected-plane-feed-closed-no-transferable-raw-gain/code/astra_phase2_reexamination_v1/screen_projected_plane_feed_v1.py](../../../experiments/third-order-planes/projected-plane-feed-closed-no-transferable-raw-gain/code/astra_phase2_reexamination_v1/screen_projected_plane_feed_v1.py)

<details>
<summary>Additional programs and dependencies (8)</summary>

- [../../../experiments/third-order-planes/projected-plane-feed-closed-no-transferable-raw-gain/code/astra_phase2_reexamination_v1/screen_projected_plane_feed_v2.py](../../../experiments/third-order-planes/projected-plane-feed-closed-no-transferable-raw-gain/code/astra_phase2_reexamination_v1/screen_projected_plane_feed_v2.py)
- [../../../experiments/learned-correction/global-context-exact-weight-suffix-state-0-102505-gain/code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](../../../experiments/learned-correction/global-context-exact-weight-suffix-state-0-102505-gain/code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)
- [../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py](../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py)
- [../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [bond_kernel_oracle.py](bond_kernel_oracle.py)
- [../../../submissions/328326/estimator.py](../../../submissions/328326/estimator.py)
- [../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

</details>

## Evidence

- [T68_CORRELATED_GATE_SCOPE_AUDIT_V1.md#L1](T68_CORRELATED_GATE_SCOPE_AUDIT_V1.md#L1)
- [T68_CORRELATED_GATE_SCOPE_AUDIT_V1.md#L98](T68_CORRELATED_GATE_SCOPE_AUDIT_V1.md#L98)

<details>
<summary>All result files and records (3)</summary>

- [T68_CORRELATED_GATE_SCOPE_AUDIT_V1.md](T68_CORRELATED_GATE_SCOPE_AUDIT_V1.md) · record
- [THIRD_COVARIANCE_EDGE_DESIGN_V1.md](../../../experiments/error-attribution/distinct-third-cumulant-times-covariance-edge-fixed-round-closed/THIRD_COVARIANCE_EDGE_DESIGN_V1.md) · record
- [PROJECTED_PLANE_FEED_ROUND_V1_CLOSED.md](../../../experiments/third-order-planes/projected-plane-feed-closed-no-transferable-raw-gain/PROJECTED_PLANE_FEED_ROUND_V1_CLOSED.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-correlated-gate-inheritance-scope-cost-audit) · [Copy provenance](../../../PROVENANCE.json)

