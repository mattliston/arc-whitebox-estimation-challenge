# Reusing an 832-wide bottleneck map for 17,728 old factor columns raises the affected GEMM cost by 286,682,564 F

2026-09-08T10:26Z · finding · rejected · derived

In exact arithmetic a cohort's transport cannot exceed the rank of an earlier active-coordinate bottleneck, so an old factor group could keep a k-by-R coefficient matrix and a later a-by-k map (pp = (W_active^T U) Q). On saved rank-35 mask metadata the only growth interval is the post9 bottleneck of 832 active units before 864 at post10/11; pricing the 17,728-column group with the established kernel cost polynomials, the smaller inner dimension saves 694,051,848 F per transition but the paid map product and the separately executed young group make the affected GEMMs more expensive, and rank-100 active counts are nonincreasing, so no prototype was selected.

Recorded measurement:

    rank35 active counts 1024,1024,1024,1024,992,960,960,896,896,832,864,864,832,832,768,768; pre11 original 18,942,208,216 F vs proposed 19,077,588,986 F; pre12 20,374,326,064 F vs 20,525,627,858 F; corrected two-step GEMM increase 286,682,564 F (group-only figure 270,761,540 F superseded); archived rank35 first-context whole bill 490,645,310,609 F

## Evidence

- [EXACT_MASK_BOTTLENECK_PRICE_PREFLIGHT_2026_09_08_V1.md#L13](EXACT_MASK_BOTTLENECK_PRICE_PREFLIGHT_2026_09_08_V1.md#L13)
- [EXACT_MASK_BOTTLENECK_PRICE_PREFLIGHT_2026_09_08_V1.md#L44](EXACT_MASK_BOTTLENECK_PRICE_PREFLIGHT_2026_09_08_V1.md#L44)
- [EXACT_MASK_BOTTLENECK_PRICE_PREFLIGHT_2026_09_08_V1.md#L54](EXACT_MASK_BOTTLENECK_PRICE_PREFLIGHT_2026_09_08_V1.md#L54)
- [EXACT_MASK_BOTTLENECK_PRICE_PREFLIGHT_2026_09_08_V1.md#L98](EXACT_MASK_BOTTLENECK_PRICE_PREFLIGHT_2026_09_08_V1.md#L98)
- [EXACT_MASK_BOTTLENECK_PRICE_PREFLIGHT_2026_09_08_V1.md#L105](EXACT_MASK_BOTTLENECK_PRICE_PREFLIGHT_2026_09_08_V1.md#L105)

<details>
<summary>All result files and records (1)</summary>

- [EXACT_MASK_BOTTLENECK_PRICE_PREFLIGHT_2026_09_08_V1.md](EXACT_MASK_BOTTLENECK_PRICE_PREFLIGHT_2026_09_08_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-reusing-832-wide-bottleneck-map-17-728-old-factor) · [Copy provenance](../../../PROVENANCE.json)

