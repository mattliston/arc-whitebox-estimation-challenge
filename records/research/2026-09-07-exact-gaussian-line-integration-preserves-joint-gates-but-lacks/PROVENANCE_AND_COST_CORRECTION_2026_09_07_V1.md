# Additive provenance and diagnostic-cost correction

September 7, 2026 PDT. Phase 2 only. Earlier receipts are preserved unchanged.
This is a source correction, not a new experiment or reopening authorization.

## Historical rank35 is the current development rank35

The different-network claim in
`EXACT_LINE_CONDITIONING_SOURCE_PREFLIGHT_V1.md` and
`CONTINUATION_2026_09_07_0540_PDT.md` is incorrect.

`public_quadrature_derivation_v1/measure_realnet.py:59` and current
`check_response_rows_adjusted_v1.py:124` both select
`sealed_target_blind_feature_store_v3/manifest.json` →
`roles.observer_fit.row_manifests[35]`. The canonical row manifest is
`rows/observer_fit/0035/row_manifest.json`, MLP89, seed2758000291903877946.

Historical `realnet_measurements.json` and that canonical manifest both
record raw weight SHA256
`d18ae98fb182d0941b0dc28905d2419ef2f1c8c281b6f904d8266fc38b9692a3`.
The canonical weights.npy FILE SHA256 is
`14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2`,
also pinned by the native response-row development acquisition and the
new input-only diagnostic. Do not confuse raw-array and file hashes.
Both actual forward sources use row-input `x @ W` orientation.

What does NOT change: the historical160-Jacobian summary measures
unconditional final POST-ReLU gradient energy. It does not measure
preactivation, g'-weighted Mehler-kernel noise after cyclic cross-layer
transfers. The source accumulated Jacobian/Jacobian-mean summaries but
did not persist the needed actual-gradient row arrays. Its variance
information cannot admit the proposed correction estimator.

## Exact conservative threshold replaces a leading-arithmetic threshold

The earlier full one-probe leading subtotal195,035,136 FLOPs/row is not an
exact installed-FlopScope lower bound: its ordinary matmul rule is
`m*n*(2*k-1)`, not `2*m*n*k`. The previous nominal threshold
1.127501075e-5 must not be presented as an exact-cost impossibility test.

The frozen paired-probe rejection protocol instead grants ALL work except
the two ordinary-dense actual forwards free. Their32 products require at
least67,076,096 arithmetic FLOPs/row under the cheaper FP32 convention.
Its threshold is therefore

    T = 1e-9 * 2^41 / 67,076,096 = 3.278400781631656e-5.

This deliberately conservative necessary condition concerns only the
declared fixed circuit and equally weighted development-two-net mean.
It is not a bound on cheaper future circuits, a complete candidate bill,
an official-score claim, or FP32 variance qualification from FP64 data.
The paired-probe screen uses exactly one aggregate Markov decision at20T;
low Y remains inconclusive.

Tiny qualification passed297 fixed scalar grid points and the fixed
width3/depth3 Jacobian/quadratic/affine witnesses. Its JSON SHA256 is
`5e69f79244f5afb8ea9abba357b365ad189737c02d5c1afbd370382f39f88c97`.
This is finite numerical qualification, not a uniform analytic error bound.

The confirmed official bank4.982231460183304e-9 and the retained independent
14-net response component4.83424098592912e-9 are unchanged by these corrections.
