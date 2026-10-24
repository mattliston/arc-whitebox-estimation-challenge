# Reusing the numerical left Winograd transform across same-layer products: static ceiling 2.67 GF (0.573% of the bill)

2026-09-08T10:58Z · finding · rejected · derived

Within a layer the left operand L = W32[previous,:]^T is identical across the active covariance outer product, star push, old CP output tiles and shared-newborn push, but the frame cache stores views, not transformed leaves; caching one copied left leaf per depth and layout would save ((t-1) gamma_d - (7/4)^d) n a per t same-depth uses. A generous static bound over all such opportunities is below 0.6% of the panel mean bill before cache-copy costs, so no standalone implementation was selected.

Recorded measurement:

    <= 62 gamma5 n^2 = 2,338,072,576 F gross plus resident-Gram top-block duplicates <= 301,686,784 F and duplicate W gathers <= 31,457,280 F: <= 2,671,216,640 F = 0.5731502% of the current panel mean; maximum leaf owner 39,337,984 B (D4) or 68,841,472 B (D5)

## Evidence

- [SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md#L97](SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md#L97)
- [SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md#L115](SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md#L115)
- [SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md#L124](SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md#L124)

<details>
<summary>All result files and records (1)</summary>

- [SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md](SERIAL_TOP_WINOGRAD_AND_NUMERICAL_REUSE_PREFLIGHT_2026_09_08_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-reusing-numerical-left-winograd-transform-across-same-layer-products) · [Copy provenance](../../../PROVENANCE.json)

