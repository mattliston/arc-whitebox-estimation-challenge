# Shared-energy pair matrix K = E[eps H H^T] via one VJP and one JVP: 134.15 MF per depth-16 row, no value witness

2026-09-08T07:19Z · finding · inconclusive · derived

By Gaussian covariance interpolation X_t = t X + sqrt(1-t^2) Z the raw energy pair matrix K = E[epsilon H H^T] (epsilon = ||H||^2 - eta ||X||^2) equals 2 integral_0^1 E[a_t H(X_t)^T + H(X_t) a_t^T] dt with a_t = J(X_t) v and v = J(X)^T H(X) - eta X, so one vector-Jacobian and one Jacobian-vector product per sample supply all entries without a Hutchinson trace probe. The tempting local identity with 3L products drops inherited ReLU facet measures and is not exact; no consumer variance or value was measured and no run was admitted.

Recorded measurement:

    literal row cost 4*L*n*(2n-1) = 134,152,192 F at L16, n1024 before RNG, gates, accumulation and consumers; the local 3L-product term alone is 100,614,144 F

## Evidence

- [SOURCE_DECISIONS_2026_09_08_V1.md#L97](../2026-09-08-moving-unsquared-query-leg-q-w-t-p-keeps/SOURCE_DECISIONS_2026_09_08_V1.md#L97)
- [SOURCE_DECISIONS_2026_09_08_V1.md#L115](../2026-09-08-moving-unsquared-query-leg-q-w-t-p-keeps/SOURCE_DECISIONS_2026_09_08_V1.md#L115)
- [SOURCE_DECISIONS_2026_09_08_V1.md#L125](../2026-09-08-moving-unsquared-query-leg-q-w-t-p-keeps/SOURCE_DECISIONS_2026_09_08_V1.md#L125)

<details>
<summary>All result files and records (1)</summary>

- [SOURCE_DECISIONS_2026_09_08_V1.md](../2026-09-08-moving-unsquared-query-leg-q-w-t-p-keeps/SOURCE_DECISIONS_2026_09_08_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-shared-energy-pair-matrix-k-e-eps-h-h) · [Copy provenance](../../../PROVENANCE.json)

