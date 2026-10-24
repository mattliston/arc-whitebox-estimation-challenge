# Numerically guarded Gamma consumer, not a relaxed certificate

September 5, 2026. Universal eight-node Gamma v1 fails its target-free
synthetic r=.30 certificate: relative covariance disagreement 6.1251e-8
versus the fixed 1e-8 bound. The v2 input-boundary repair passed all exact
replay checks but retains this numerical failure. No accuracy target was
scored, and no node count will be swept or increased.

Separate bounded version3: use exactly the same eight-node consumer only
when 0<=r<=.10; elsewhere retain the parent map exactly. This is a
numerical-domain guard selected from the predeclared target-free checks,
not from accuracy labels or a layer/radial-strength fit. Negative r is
not clipped into a Gamma law. Keep the exact current parent if the
Gamma law is not in this numerically qualified domain.

The synthetic .30 comparison therefore must be byte-exact to the parent
in both nominal8/reference32 variants. The original synthetic0/.03/.10
and ALL seven current cells must pass the original numerical tolerances.
Current r=.0398..0574, with covariance8/32 disagreements at most1.095e-12
relative, so current-cell accuracy is not relying on a loosened gate.

This changes the v1 universal method; it does NOT turn v2's global passed
flag true. Version3 has its own result files and records the failed v2.
The original .95 covariance/.90 next-value/5-of7 wins information gates
are unchanged. One existing-data pass only, no rollout/fit/node/range/
strength sweep after a failed information gate.

